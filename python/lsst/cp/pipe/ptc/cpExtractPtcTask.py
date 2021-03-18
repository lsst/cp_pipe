# This file is part of cp_pipe.
#
# Developed for the LSST Data Management System.
# This product includes software developed by the LSST Project
# (https://www.lsst.org).
# See the COPYRIGHT file at the top-level directory of this distribution
# for details of code ownership.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
#
import numpy as np

import lsst.afw.math as afwMath
import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase
from lsst.cp.pipe.utils import arrangeFlatsByExpTime, arrangeFlatsByExpId

import lsst.pipe.base.connectionTypes as cT

from .astierCovPtcUtils import (CovFft, computeCovDirect)
from .astierCovPtcFit import makeCovArray

from lsst.ip.isr import PhotonTransferCurveDataset


__all__ = ['PhotonTransferCurveExtractConfig', 'PhotonTransferCurveExtractTask']


class PhotonTransferCurveExtractConnections(pipeBase.PipelineTaskConnections,
                                            dimensions=("instrument", "detector")):

    inputExp = cT.Input(
        name="ptcInputExposurePairs",
        doc="Input post-ISR processed exposure pairs (flats) to"
            "measure covariances from.",
        storageClass="Exposure",
        dimensions=("instrument", "exposure", "detector"),
        multiple=True,
        deferLoad=False,
    )

    outputCovariances = cT.Output(
        name="ptcCovariances",
        doc="Extracted flat (co)variances.",
        storageClass="PhotonTransferCurveDataset",
        dimensions=("instrument", "exposure", "detector"),
        multiple=True,
    )


class PhotonTransferCurveExtractConfig(pipeBase.PipelineTaskConfig,
                                       pipelineConnections=PhotonTransferCurveExtractConnections):
    """Configuration for the measurement of covariances from flats.
    """
    matchByExposureId = pexConfig.Field(
        dtype=bool,
        doc="Should exposures by matched by ID rather than exposure time?",
        default=False,
    )
    maximumRangeCovariancesAstier = pexConfig.Field(
        dtype=int,
        doc="Maximum range of covariances as in Astier+19",
        default=8,
    )
    covAstierRealSpace = pexConfig.Field(
        dtype=bool,
        doc="Calculate covariances in real space or via FFT? (see appendix A of Astier+19).",
        default=False,
    )
    binSize = pexConfig.Field(
        dtype=int,
        doc="Bin the image by this factor in both dimensions.",
        default=1,
    )
    minMeanSignal = pexConfig.DictField(
        keytype=str,
        itemtype=float,
        doc="Minimum values (inclusive) of mean signal (in ADU) above which to consider, per amp."
            " The same cut is applied to all amps if this dictionary is of the form"
            " {'ALL_AMPS': value}",
        default={'ALL_AMPS': 0.0},
    )
    maxMeanSignal = pexConfig.DictField(
        keytype=str,
        itemtype=float,
        doc="Maximum values (inclusive) of mean signal (in ADU) below which to consider, per amp."
            " The same cut is applied to all amps if this dictionary is of the form"
            " {'ALL_AMPS': value}",
        default={'ALL_AMPS': 1e6},
    )
    maskNameList = pexConfig.ListField(
        dtype=str,
        doc="Mask list to exclude from statistics calculations.",
        default=['SUSPECT', 'BAD', 'NO_DATA'],
    )
    nSigmaClipPtc = pexConfig.Field(
        dtype=float,
        doc="Sigma cut for afwMath.StatisticsControl()",
        default=5.5,
    )
    nIterSigmaClipPtc = pexConfig.Field(
        dtype=int,
        doc="Number of sigma-clipping iterations for afwMath.StatisticsControl()",
        default=1,
    )
    minNumberGoodPixelsForFft = pexConfig.Field(
        dtype=int,
        doc="Minimum number of acceptable good pixels per amp to calculate the covariances via FFT.",
        default=10000,
    )
    detectorMeasurementRegion = pexConfig.ChoiceField(
        dtype=str,
        doc="Region of each exposure where to perform the calculations (amplifier or full image).",
        default='AMP',
        allowed={
            "AMP": "Amplifier of the detector.",
            "FULL": "Full image."
        }
    )


class PhotonTransferCurveExtractTask(pipeBase.PipelineTask,
                                     pipeBase.CmdLineTask):
    """Task to measure covariances from flat fields.
    This task receives as input a list of flat-field images
    (flats), and sorts these flats in pairs taken at the
    same time (if there's a different number of flats,
    those flats are discarded). The mean, variance, and
    covariances are measured from the difference of the flat
    pairs at a given time. The variance is calculated
    via afwMath, and the covariance via the methods in Astier+19
    (appendix A). In theory, var = covariance[0,0]. This should
    be validated, and in the future, we may decide to just keep
    one (covariance).

    The measured covariances at a particular time (along with
    other quantities such as the mean) are stored in a PTC dataset
    object (`PhotonTransferCurveDataset`), which gets partially
    filled. The number of partially-filled PTC dataset objects
    will be less than the number of input exposures, but gen3
    requires/assumes that the number of input dimensions matches
    bijectively the number of output dimensions. Therefore, a
    number of "dummy" PTC dataset are inserted in the output list
    that has the partially-filled PTC datasets with the covariances.
    This output list will be used as input of
    `PhotonTransferCurveSolveTask`, which will assemble the multiple
    `PhotonTransferCurveDataset`s into a single one in order to fit
    the measured covariances as a function of flux to a particular
    model.

    Astier+19: "The Shape of the Photon Transfer Curve of CCD
    sensors", arXiv:1905.08677.
    """
    ConfigClass = PhotonTransferCurveExtractConfig
    _DefaultName = 'cpPtcExtract'

    def runQuantum(self, butlerQC, inputRefs, outputRefs):
        """Ensure that the input and output dimensions are passed along.

        Parameters
        ----------
        butlerQC : `~lsst.daf.butler.butlerQuantumContext.ButlerQuantumContext`
            Butler to operate on.
        inputRefs : `~lsst.pipe.base.connections.InputQuantizedConnection`
            Input data refs to load.
        ouptutRefs : `~lsst.pipe.base.connections.OutputQuantizedConnection`
            Output data refs to persist.
        """
        inputs = butlerQC.get(inputRefs)
        # Dictionary, keyed by expTime, with flat exposures
        if self.config.matchByExposureId:
            inputs['inputExp'] = arrangeFlatsByExpId(inputs['inputExp'])
        else:
            inputs['inputExp'] = arrangeFlatsByExpTime(inputs['inputExp'])
        # Ids of input list of exposures
        inputs['inputDims'] = [expId.dataId['exposure'] for expId in inputRefs.inputExp]
        outputs = self.run(**inputs)
        butlerQC.put(outputs, outputRefs)

    def run(self, inputExp, inputDims):
        """Measure covariances from difference of flat pairs

        Parameters
        ----------
        inputExp : `dict` [`float`,
                        (`~lsst.afw.image.exposure.exposure.ExposureF`,
                        `~lsst.afw.image.exposure.exposure.ExposureF`, ...,
                        `~lsst.afw.image.exposure.exposure.ExposureF`)]
            Dictionary that groups flat-field exposures that have the same
            exposure time (seconds).

        inputDims : `list`
            List of exposure IDs.
        """
        # inputExp.values() returns a view, which we turn into a list. We then
        # access the first exposure to get teh detector.
        detector = list(inputExp.values())[0][0].getDetector()
        detNum = detector.getId()
        amps = detector.getAmplifiers()
        ampNames = [amp.getName() for amp in amps]
        maxMeanSignalDict = {ampName: 1e6 for ampName in ampNames}
        minMeanSignalDict = {ampName: 0.0 for ampName in ampNames}
        for ampName in ampNames:
            if 'ALL_AMPS' in self.config.maxMeanSignal:
                maxMeanSignalDict[ampName] = self.config.maxMeanSignal['ALL_AMPS']
            elif ampName in self.config.maxMeanSignal:
                maxMeanSignalDict[ampName] = self.config.maxMeanSignal[ampName]

            if 'ALL_AMPS' in self.config.minMeanSignal:
                minMeanSignalDict[ampName] = self.config.minMeanSignal['ALL_AMPS']
            elif ampName in self.config.minMeanSignal:
                minMeanSignalDict[ampName] = self.config.minMeanSignal[ampName]
        tags = [('mu', '<f8'), ('afwVar', '<f8'), ('i', '<i8'), ('j', '<i8'), ('var', '<f8'),
                ('cov', '<f8'), ('npix', '<i8'), ('ext', '<i8'), ('expTime', '<f8'), ('ampName', '<U3')]
        dummyPtcDataset = PhotonTransferCurveDataset(ampNames, 'DUMMY',
                                                     self.config.maximumRangeCovariancesAstier)
        covArray = [np.full((self.config.maximumRangeCovariancesAstier,
                    self.config.maximumRangeCovariancesAstier), np.nan)]
        for ampName in ampNames:
            dummyPtcDataset.rawExpTimes[ampName] = [np.nan]
            dummyPtcDataset.rawMeans[ampName] = [np.nan]
            dummyPtcDataset.rawVars[ampName] = [np.nan]
            dummyPtcDataset.inputExpIdPairs[ampName] = [(np.nan, np.nan)]
            dummyPtcDataset.expIdMask[ampName] = [np.nan]
            dummyPtcDataset.covariances[ampName] = covArray
            dummyPtcDataset.covariancesModel[ampName] = np.full_like(covArray, np.nan)
            dummyPtcDataset.covariancesSqrtWeights[ampName] = np.full_like(covArray, np.nan)
            dummyPtcDataset.covariancesModelNoB[ampName] = np.full_like(covArray, np.nan)
            dummyPtcDataset.aMatrix[ampName] = np.full_like(covArray[0], np.nan)
            dummyPtcDataset.bMatrix[ampName] = np.full_like(covArray[0], np.nan)
            dummyPtcDataset.aMatrixNoB[ampName] = np.full_like(covArray[0], np.nan)
            dummyPtcDataset.ptcFitPars[ampName] = [np.nan]
            dummyPtcDataset.ptcFitParsError[ampName] = [np.nan]
            dummyPtcDataset.ptcFitChiSq[ampName] = np.nan
            dummyPtcDataset.finalVars[ampName] = [np.nan]
            dummyPtcDataset.finalModelVars[ampName] = [np.nan]
            dummyPtcDataset.finalMeans[ampName] = [np.nan]
        # Output list with PTC datasets.
        partialDatasetPtcList = []
        # The number of output references needs to match that of input references:
        # initialize outputlist with dummy PTC datasets.
        for i in range(len(inputDims)):
            partialDatasetPtcList.append(dummyPtcDataset)

        for expTime in inputExp:
            exposures = inputExp[expTime]
            if len(exposures) == 1:
                self.log.warn(f"Only one exposure found at expTime {expTime}. Dropping exposure "
                              f"{exposures[0].getInfo().getVisitInfo().getExposureId()}.")
                continue
            else:
                # Only use the first two exposures at expTime
                exp1, exp2 = exposures[0], exposures[1]
                if len(exposures) > 2:
                    self.log.warn(f"Already found 2 exposures at expTime {expTime}. "
                                  "Ignoring exposures: "
                                  f"{i.getInfo().getVisitInfo().getExposureId() for i in exposures[2:]}")
            expId1 = exp1.getInfo().getVisitInfo().getExposureId()
            expId2 = exp2.getInfo().getVisitInfo().getExposureId()
            nAmpsNan = 0
            partialDatasetPtc = PhotonTransferCurveDataset(ampNames, '',
                                                           self.config.maximumRangeCovariancesAstier)
            for ampNumber, amp in enumerate(detector):
                ampName = amp.getName()
                # covAstier: [(i, j, var (cov[0,0]), cov, npix) for (i,j) in {maxLag, maxLag}^2]
                doRealSpace = self.config.covAstierRealSpace
                if self.config.detectorMeasurementRegion == 'AMP':
                    region = amp.getBBox()
                elif self.config.detectorMeasurementRegion == 'FULL':
                    region = None
                # The variable `covAstier` is of the form: [(i, j, var (cov[0,0]), cov, npix) for (i,j)
                # in {maxLag, maxLag}^2]
                muDiff, varDiff, covAstier = self.measureMeanVarCov(exp1, exp2, region=region,
                                                                    covAstierRealSpace=doRealSpace)
                expIdMask = True
                if np.isnan(muDiff) or np.isnan(varDiff) or (covAstier is None):
                    msg = (f"NaN mean or var, or None cov in amp {ampName} in exposure pair {expId1},"
                           f" {expId2} of detector {detNum}.")
                    self.log.warn(msg)
                    nAmpsNan += 1
                    expIdMask = False
                    covArray = np.full((1, self.config.maximumRangeCovariancesAstier,
                                        self.config.maximumRangeCovariancesAstier), np.nan)
                    covSqrtWeights = np.full_like(covArray, np.nan)

                if (muDiff <= minMeanSignalDict[ampName]) or (muDiff >= maxMeanSignalDict[ampName]):
                    expIdMask = False

                partialDatasetPtc.rawExpTimes[ampName] = [expTime]
                partialDatasetPtc.rawMeans[ampName] = [muDiff]
                partialDatasetPtc.rawVars[ampName] = [varDiff]

                if covAstier is not None:
                    tupleRows = [(muDiff, varDiff) + covRow + (ampNumber, expTime,
                                                               ampName) for covRow in covAstier]
                    tempStructArray = np.array(tupleRows, dtype=tags)
                    covArray, vcov, _ = makeCovArray(tempStructArray,
                                                     self.config.maximumRangeCovariancesAstier)
                    covSqrtWeights = np.nan_to_num(1./np.sqrt(vcov))
                partialDatasetPtc.inputExpIdPairs[ampName] = [(expId1, expId2)]
                partialDatasetPtc.expIdMask[ampName] = [expIdMask]
                partialDatasetPtc.covariances[ampName] = covArray
                partialDatasetPtc.covariancesSqrtWeights[ampName] = covSqrtWeights
                partialDatasetPtc.covariancesModel[ampName] = np.full_like(covArray, np.nan)
                partialDatasetPtc.covariancesModelNoB[ampName] = np.full_like(covArray, np.nan)
                partialDatasetPtc.aMatrix[ampName] = np.full_like(covArray[0], np.nan)
                partialDatasetPtc.bMatrix[ampName] = np.full_like(covArray[0], np.nan)
                partialDatasetPtc.aMatrixNoB[ampName] = np.full_like(covArray[0], np.nan)
                partialDatasetPtc.ptcFitPars[ampName] = [np.nan]
                partialDatasetPtc.ptcFitParsError[ampName] = [np.nan]
                partialDatasetPtc.ptcFitChiSq[ampName] = np.nan
                partialDatasetPtc.finalVars[ampName] = [np.nan]
                partialDatasetPtc.finalModelVars[ampName] = [np.nan]
                partialDatasetPtc.finalMeans[ampName] = [np.nan]
            # Use location of exp1 to save PTC dataset from (exp1, exp2) pair.
            # expId1 and expId2, as returned by getInfo().getVisitInfo().getExposureId(),
            # and the exposure IDs stured in inoutDims,
            # may have the zero-padded detector number appended at
            # the end (in gen3). A temporary fix is to consider expId//1000 and/or
            # inputDims//1000.
            # Below, np.where(expId1 == np.array(inputDims)) (and the other analogous
            # comparisons) returns a tuple with a single-element array, so [0][0]
            # is necessary to extract the required index.
            try:
                datasetIndex = np.where(expId1 == np.array(inputDims))[0][0]
            except IndexError:
                try:
                    datasetIndex = np.where(expId1//1000 == np.array(inputDims))[0][0]
                except IndexError:
                    datasetIndex = np.where(expId1//1000 == np.array(inputDims)//1000)[0][0]
            partialDatasetPtcList[datasetIndex] = partialDatasetPtc
            if nAmpsNan == len(ampNames):
                msg = f"NaN mean in all amps of exposure pair {expId1}, {expId2} of detector {detNum}."
                self.log.warn(msg)
        return pipeBase.Struct(
            outputCovariances=partialDatasetPtcList,
        )

    def measureMeanVarCov(self, exposure1, exposure2, region=None, covAstierRealSpace=False):
        """Calculate the mean of each of two exposures and the variance
        and covariance of their difference. The variance is calculated
        via afwMath, and the covariance via the methods in Astier+19
        (appendix A). In theory, var = covariance[0,0]. This should
        be validated, and in the future, we may decide to just keep
        one (covariance).

        Parameters
        ----------
        exposure1 : `lsst.afw.image.exposure.exposure.ExposureF`
            First exposure of flat field pair.
        exposure2 : `lsst.afw.image.exposure.exposure.ExposureF`
            Second exposure of flat field pair.
        region : `lsst.geom.Box2I`, optional
            Region of each exposure where to perform the calculations (e.g, an amplifier).
        covAstierRealSpace : `bool`, optional
            Should the covariannces in Astier+19 be calculated in real space or via FFT?
            See Appendix A of Astier+19.

        Returns
        -------
        mu : `float` or `NaN`
            0.5*(mu1 + mu2), where mu1, and mu2 are the clipped means of the regions in
            both exposures. If either mu1 or m2 are NaN's, the returned value is NaN.
        varDiff : `float` or `NaN`
            Half of the clipped variance of the difference of the regions inthe two input
            exposures. If either mu1 or m2 are NaN's, the returned value is NaN.
        covDiffAstier : `list` or `NaN`
            List with tuples of the form (dx, dy, var, cov, npix), where:
                dx : `int`
                    Lag in x
                dy : `int`
                    Lag in y
                var : `float`
                    Variance at (dx, dy).
                cov : `float`
                    Covariance at (dx, dy).
                nPix : `int`
                    Number of pixel pairs used to evaluate var and cov.
            If either mu1 or m2 are NaN's, the returned value is NaN.
        """

        if region is not None:
            im1Area = exposure1.maskedImage[region]
            im2Area = exposure2.maskedImage[region]
        else:
            im1Area = exposure1.maskedImage
            im2Area = exposure2.maskedImage

        if self.config.binSize > 1:
            im1Area = afwMath.binImage(im1Area, self.config.binSize)
            im2Area = afwMath.binImage(im2Area, self.config.binSize)

        im1MaskVal = exposure1.getMask().getPlaneBitMask(self.config.maskNameList)
        im1StatsCtrl = afwMath.StatisticsControl(self.config.nSigmaClipPtc,
                                                 self.config.nIterSigmaClipPtc,
                                                 im1MaskVal)
        im1StatsCtrl.setNanSafe(True)
        im1StatsCtrl.setAndMask(im1MaskVal)

        im2MaskVal = exposure2.getMask().getPlaneBitMask(self.config.maskNameList)
        im2StatsCtrl = afwMath.StatisticsControl(self.config.nSigmaClipPtc,
                                                 self.config.nIterSigmaClipPtc,
                                                 im2MaskVal)
        im2StatsCtrl.setNanSafe(True)
        im2StatsCtrl.setAndMask(im2MaskVal)

        #  Clipped mean of images; then average of mean.
        mu1 = afwMath.makeStatistics(im1Area, afwMath.MEANCLIP, im1StatsCtrl).getValue()
        mu2 = afwMath.makeStatistics(im2Area, afwMath.MEANCLIP, im2StatsCtrl).getValue()
        if np.isnan(mu1) or np.isnan(mu2):
            self.log.warn(f"Mean of amp in image 1 or 2 is NaN: {mu1}, {mu2}.")
            return np.nan, np.nan, None
        mu = 0.5*(mu1 + mu2)

        # Take difference of pairs
        # symmetric formula: diff = (mu2*im1-mu1*im2)/(0.5*(mu1+mu2))
        temp = im2Area.clone()
        temp *= mu1
        diffIm = im1Area.clone()
        diffIm *= mu2
        diffIm -= temp
        diffIm /= mu

        diffImMaskVal = diffIm.getMask().getPlaneBitMask(self.config.maskNameList)
        diffImStatsCtrl = afwMath.StatisticsControl(self.config.nSigmaClipPtc,
                                                    self.config.nIterSigmaClipPtc,
                                                    diffImMaskVal)
        diffImStatsCtrl.setNanSafe(True)
        diffImStatsCtrl.setAndMask(diffImMaskVal)

        varDiff = 0.5*(afwMath.makeStatistics(diffIm, afwMath.VARIANCECLIP, diffImStatsCtrl).getValue())

        # Get the mask and identify good pixels as '1', and the rest as '0'.
        w1 = np.where(im1Area.getMask().getArray() == 0, 1, 0)
        w2 = np.where(im2Area.getMask().getArray() == 0, 1, 0)

        w12 = w1*w2
        wDiff = np.where(diffIm.getMask().getArray() == 0, 1, 0)
        w = w12*wDiff

        if np.sum(w) < self.config.minNumberGoodPixelsForFft:
            self.log.warn(f"Number of good points for FFT ({np.sum(w)}) is less than threshold "
                          f"({self.config.minNumberGoodPixelsForFft})")
            return np.nan, np.nan, None

        maxRangeCov = self.config.maximumRangeCovariancesAstier
        if covAstierRealSpace:
            covDiffAstier = computeCovDirect(diffIm.image.array, w, maxRangeCov)
        else:
            shapeDiff = np.array(diffIm.image.array.shape)
            # Calculate sizes of FFT dimensions
            s = shapeDiff + maxRangeCov
            tempSize = np.array(np.log(s)/np.log(2.)).astype(int)
            fftSize = np.array(2**(tempSize+1)).astype(int)
            fftShape = (fftSize[0], fftSize[1])

            c = CovFft(diffIm.image.array, w, fftShape, maxRangeCov)
            covDiffAstier = c.reportCovFft(maxRangeCov)

        return mu, varDiff, covDiffAstier
