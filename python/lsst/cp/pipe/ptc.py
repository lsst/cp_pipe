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
from collections import Counter

import lsst.afw.math as afwMath
import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase
from .utils import (fitLeastSq, fitBootstrap, funcPolynomial, funcAstier, makeExpPairs)
from scipy.optimize import least_squares

import lsst.pipe.base.connectionTypes as cT

from .astierCovPtcUtils import (fftSize, CovFft, computeCovDirect, fitData)
from .astierCovPtcFit import makeCovArray
from .photodiode import getBOTphotodiodeData

from lsst.pipe.tasks.getRepositoryData import DataRefListRunner
from lsst.ip.isr import PhotonTransferCurveDataset

from ._lookupStaticCalibration import lookupStaticCalibration

__all__ = ['PhotonTransferCurveExtractConfig', 'PhotonTransferCurveExtractTask',
           'PhotonTransferCurveSolveConfig', 'PhotonTransferCurveSolveTask',
           'MeasurePhotonTransferCurveTask', 'MeasurePhotonTransferCurveTaskConfig']


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

    camera = cT.PrerequisiteInput(
        name="camera",
        doc="Camera the input data comes from.",
        storageClass="Camera",
        dimensions=("instrument",),
        isCalibration=True,
        lookupFunction=lookupStaticCalibration,
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


class PhotonTransferCurveExtractTask(pipeBase.PipelineTask,
                                     pipeBase.CmdLineTask):
    """Task to measure covariances from flat fields.
    """
    ConfigClass = PhotonTransferCurveExtractConfig
    _DefaultName = 'cpPtcExtract'

    def runQuantum(self, butlerQC, inputRefs, outputRefs):
        """Ensure that the input and output dimensions are passed along.
        Parameters
        ----------
        butlerQC : `lsst.daf.butler.butlerQuantumContext.ButlerQuantumContext`
            Butler to operate on.
        inputRefs : `lsst.pipe.base.connections.InputQuantizedConnection`
            Input data refs to load.
        ouptutRefs : `lsst.pipe.base.connections.OutputQuantizedConnection`
            Output data refs to persist.
        """
        inputs = butlerQC.get(inputRefs)
        # Use the dimensions to set calib information.
        temp = makeExpPairs(inputs['inputExp'], log=self.log)
        inputs['inputExp'] = temp
        # Use the dimensions to set calib information.
        inputs['inputDims'] = [exp.getInfo().getVisitInfo().getExposureId()
                               for exp in np.array(list(inputs['inputExp'].values())).ravel()]

        outputs = self.run(**inputs)
        # Need to pad the output list so that it matches the length of the input references.
        padLength = len(inputRefs.inputExp) - len(outputs.outputCovariances)
        dummyDatasetPtc = PhotonTransferCurveDataset()
        for i in range(padLength):
            outputs.outputCovariances.append(dummyDatasetPtc)

        butlerQC.put(outputs, outputRefs)

    def run(self, inputExp, camera, inputDims=None):
        """Measure covariances from difference of flat pairs

        Parameters
        ----------
        inputExp : `dict` [`float`,
                        (`lsst.afw.image.exposure.exposure.ExposureF`,
                        `lsst.afw.image.exposure.exposure.ExposureF`)]
        Dictionary that groups flat-field exposures that have the same
        exposure time (seconds).

        camera : `lsst.afw.cameraGeom.Camera`
            Input camera.
        """
        exposurePairs = inputExp
        detector = list(exposurePairs.values())[0][0].getDetector()
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

        tags = ['mu', 'afwVar', 'i', 'j', 'var', 'cov', 'npix', 'ext', 'expTime', 'ampName']
        partialDatasetPtcList = []
        for expTime, (exp1, exp2) in exposurePairs.items():
            expId1 = exp1.getInfo().getVisitInfo().getExposureId()
            expId2 = exp2.getInfo().getVisitInfo().getExposureId()
            tupleRows = []
            nAmpsNan = 0
            partialDatasetPtc = PhotonTransferCurveDataset(ampNames, 'FULLCOVARIANCE')
            # self.config.ptcFitType)
            for ampNumber, amp in enumerate(detector):
                ampName = amp.getName()
                # covAstier: [(i, j, var (cov[0,0]), cov, npix) for (i,j) in {maxLag, maxLag}^2]
                doRealSpace = self.config.covAstierRealSpace
                muDiff, varDiff, covAstier = self.measureMeanVarCov(exp1, exp2, region=amp.getBBox(),
                                                                    covAstierRealSpace=doRealSpace)

                if np.isnan(muDiff) or np.isnan(varDiff) or (covAstier is None):
                    msg = (f"NaN mean or var, or None cov in amp {ampName} in exposure pair {expId1},"
                           f" {expId2} of detector {detNum}.")
                    self.log.warn(msg)
                    nAmpsNan += 1
                    continue

                if (muDiff <= minMeanSignalDict[ampName]) or (muDiff >= maxMeanSignalDict[ampName]):
                    continue

                partialDatasetPtc.rawExpTimes[ampName].append(expTime)
                partialDatasetPtc.rawMeans[ampName].append(muDiff)
                partialDatasetPtc.rawVars[ampName].append(varDiff)

                tupleRows += [(muDiff, varDiff) + covRow + (ampNumber, expTime,
                                                            ampName) for covRow in covAstier]
                tempRecArray = np.core.records.fromrecords(tupleRows, names=tags)
                singleAmpTuple = tempRecArray[tempRecArray['ampName'] == ampName]
                covArray, _, _ = makeCovArray(singleAmpTuple, self.config.maximumRangeCovariancesAstier)
                partialDatasetPtc.covariances[ampName].append(covArray)
            partialDatasetPtcList.append(partialDatasetPtc)
            if nAmpsNan == len(ampNames):
                msg = f"NaN mean in all amps of exposure pair {expId1}, {expId2} of detector {detNum}."
                self.log.warn(msg)

        return pipeBase.Struct(
            outputCovariances=partialDatasetPtcList,
        )

    def measureMeanVarCov(self, exposure1, exposure2, region=None, covAstierRealSpace=False):
        """Calculate the mean of each of two exposures and the variance and covariance of their difference.
        The variance is calculated via afwMath, and the covariance via the methods in Astier+19 (appendix A).
        In theory, var = covariance[0,0]. This should be validated, and in the future, we may decide to just
        keep one (covariance).
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

        maxRangeCov = self.config.maximumRangeCovariancesAstier
        if covAstierRealSpace:
            covDiffAstier = computeCovDirect(diffIm.getImage().getArray(), w, maxRangeCov)
        else:
            shapeDiff = diffIm.getImage().getArray().shape
            fftShape = (fftSize(shapeDiff[0] + maxRangeCov), fftSize(shapeDiff[1]+maxRangeCov))
            c = CovFft(diffIm.getImage().getArray(), w, fftShape, maxRangeCov)
            covDiffAstier = c.reportCovFft(maxRangeCov)

        return mu, varDiff, covDiffAstier


class PhotonTransferCurveSolveConnections(pipeBase.PipelineTaskConnections,
                                          dimensions=("instrument", "detector")):
    inputCovariances = cT.Input(
        name="ptcCovariances",
        doc="Tuple with measured covariances from flats.",
        storageClass="PhotonTransferCurveDataset",
        dimensions=("instrument", "exposure", "detector"),
        multiple=True,
    )
    camera = cT.PrerequisiteInput(
        name="camera",
        doc="Camera the input data comes from.",
        storageClass="Camera",
        dimensions=("instrument",),
        isCalibration=True,
        lookupFunction=lookupStaticCalibration,
    )
    outputPtcDataset = cT.Output(
        name="ptcDatsetProposal",
        doc="Output proposed ptc dataset.",
        storageClass="PhotonTransferCurveDataset",
        dimensions=("instrument", "detector"),
        multiple=False,
    )


class PhotonTransferCurveSolveConfig(pipeBase.PipelineTaskConfig,
                                     pipelineConnections=PhotonTransferCurveSolveConnections):
    """Configuration for fitting measured covariances.
    """
    ptcFitType = pexConfig.ChoiceField(
        dtype=str,
        doc="Fit PTC to Eq. 16, Eq. 20 in Astier+19, or to a polynomial.",
        default="POLYNOMIAL",
        allowed={
            "POLYNOMIAL": "n-degree polynomial (use 'polynomialFitDegree' to set 'n').",
            "EXPAPPROXIMATION": "Approximation in Astier+19 (Eq. 16).",
            "FULLCOVARIANCE": "Full covariances model in Astier+19 (Eq. 20)"
        }
    )
    maximumRangeCovariancesAstier = pexConfig.Field(
        dtype=int,
        doc="Maximum range of covariances as in Astier+19",
        default=8,
    )
    sigmaClipFullFitCovariancesAstier = pexConfig.Field(
        dtype=float,
        doc="sigma clip for full model fit for FULLCOVARIANCE ptcFitType ",
        default=5.0,
    )
    maxIterFullFitCovariancesAstier = pexConfig.Field(
        dtype=int,
        doc="Maximum number of iterations in full model fit for FULLCOVARIANCE ptcFitType",
        default=3,
    )
    polynomialFitDegree = pexConfig.Field(
        dtype=int,
        doc="Degree of polynomial to fit the PTC, when 'ptcFitType'=POLYNOMIAL.",
        default=3,
    )
    sigmaCutPtcOutliers = pexConfig.Field(
        dtype=float,
        doc="Sigma cut for outlier rejection in PTC.",
        default=5.0,
    )
    maxIterationsPtcOutliers = pexConfig.Field(
        dtype=int,
        doc="Maximum number of iterations for outlier rejection in PTC.",
        default=2,
    )
    initialNonLinearityExclusionThresholdPositive = pexConfig.RangeField(
        dtype=float,
        doc="Initially exclude data points with a variance that are more than a factor of this from being"
            " linear in the positive direction, from the PTC fit. Note that these points will also be"
            " excluded from the non-linearity fit. This is done before the iterative outlier rejection,"
            " to allow an accurate determination of the sigmas for said iterative fit.",
        default=0.12,
        min=0.0,
        max=1.0,
    )
    initialNonLinearityExclusionThresholdNegative = pexConfig.RangeField(
        dtype=float,
        doc="Initially exclude data points with a variance that are more than a factor of this from being"
            " linear in the negative direction, from the PTC fit. Note that these points will also be"
            " excluded from the non-linearity fit. This is done before the iterative outlier rejection,"
            " to allow an accurate determination of the sigmas for said iterative fit.",
        default=0.25,
        min=0.0,
        max=1.0,
    )
    doFitBootstrap = pexConfig.Field(
        dtype=bool,
        doc="Use bootstrap for the PTC fit parameters and errors?.",
        default=False,
    )


class PhotonTransferCurveSolveTask(pipeBase.PipelineTask,
                                   pipeBase.CmdLineTask):
    """Task to fit the PTC from flat covariances.
    """
    ConfigClass = PhotonTransferCurveSolveConfig
    _DefaultName = 'cpPhotonTransferCurveSolve'

    def runQuantum(self, butlerQC, inputRefs, outputRefs):
        """Ensure that the input and output dimensions are passed along.
        Parameters
        ----------
        butlerQC : `lsst.daf.butler.butlerQuantumContext.ButlerQuantumContext`
            Butler to operate on.
        inputRefs : `lsst.pipe.base.connections.InputQuantizedConnection`
            Input data refs to load.
        ouptutRefs : `lsst.pipe.base.connections.OutputQuantizedConnection`
            Output data refs to persist.
        """
        inputs = butlerQC.get(inputRefs)

        # Use the dimensions to set calib information.
        inputs['inputDims'] = [exp.dataId.byName() for exp in inputRefs.inputCovariances]
        outputs = self.run(**inputs)
        butlerQC.put(outputs, outputRefs)

    def run(self, inputCovariances, camera=None, inputDims=None, outputDims=None):
        """Fit measure covariances to different models.

        Parameters
        ----------
        inputCovariances : `list`:
            [dict[`(ampName, expid1, expId2)`], dict(),...]
            The interleaved empty dictionary is there to ensure that all
            input exposures/dataIds have corresponding outputs.
            The values in the non-empty dictionary are tuples with at
            least (mu, afwVar, cov, var, i, j, npix), where:
            mu : 0.5*(m1 + m2), where:
                mu1: mean value of flat1
                mu2: mean value of flat2
            afwVar : variance of difference flat, calculated with afW
            cov: covariance value at lag(i, j)
            var: variance (covariance value at lag(0, 0))
            i: lag dimension
            j: lag dimension
            npix: number of pixels used for covariance calculation.
        camera : `lsst.afw.cameraGeom.Camera`
            Input camera.
        inputDims : `list` [`lsst.daf.butler.DataCoordinate`]
            DataIds to use.
        outputDims : `list` [`lsst.daf.butler.DataCoordinate`]
            DataIds to use to populate the output calibration.
        Returns
        -------
        results : `lsst.pipe.base.Struct`
            The results struct containing:
            ``outputPtcDatset`` : `lsst.ip.isr.PhotonTransferCurveDataset`
                Final PTC dataset, containing information such as the means, variances,
                and exposure times.
        """
        # Re-arrange data in the form of the recarray that fitCovariancesAstier is expecting
        tupleRecords = []
        for dictionary in inputCovariances:
            # Dummy dictionary (empty)
            if not bool(dictionary):
                continue
            for key in dictionary:
                tupleRecords += dictionary[key]
        tags = ['mu', 'afwVar', 'i', 'j', 'var', 'cov', 'npix', 'ext', 'expTime', 'ampName']
        covariancesWithTags = np.core.records.fromrecords(tupleRecords, names=tags)

        # Create PTC dataset and fill raw means, expTime, and afwVariances
        ampNames = np.array(np.unique(covariancesWithTags['ampName']), dtype=str)
        datasetPtc = PhotonTransferCurveDataset(ampNames, self.config.ptcFitType)
        mask = (covariancesWithTags['i'] == 0) & (covariancesWithTags['j'] == 0)
        for ampName in ampNames:
            index = covariancesWithTags[mask]['ampName'] == ampName
            rawExpTimes = covariancesWithTags[mask][index]['expTime']
            rawMeans = covariancesWithTags[mask][index]['mu']
            rawVars = covariancesWithTags[mask][index]['afwVar']

            datasetPtc.rawExpTimes[ampName] = rawExpTimes
            datasetPtc.rawMeans[ampName] = rawMeans
            datasetPtc.rawVars[ampName] = rawVars

        if self.config.ptcFitType in ["FULLCOVARIANCE", ]:
            # Calculate covariances and fit them, including the PTC, to Astier+19 full model (Eq. 20)
            datasetPtc = self.fitCovariancesAstier(datasetPtc, covariancesWithTags)
        elif self.config.ptcFitType in ["EXPAPPROXIMATION", "POLYNOMIAL"]:
            # Fit the PTC to a polynomial or to Astier+19 exponential approximation (Eq. 16)
            # Fill up PhotonTransferCurveDataset object.
            datasetPtc = self.fitPtc(datasetPtc)

        detector = inputDims.getDetector()
        datasetPtc.updateMetadata(setDate=True, camera=camera, detector=detector)

        return pipeBase.Struct(
            outputPtcDataset=datasetPtc,
        )

    def fitCovariancesAstier(self, dataset, covariancesWithTagsArray):
        """Fit measured flat covariances to full model in Astier+19.
        Parameters
        ----------
        dataset : `lsst.ip.isr.ptcDataset.PhotonTransferCurveDataset`
            The dataset containing information such as the means, variances and exposure times.
        covariancesWithTagsArray : `numpy.recarray`
            Tuple with at least (mu, afwVar, cov, var, i, j, npix), where:
            mu: 0.5*(m1 + m2), where:
                mu1: mean value of flat1
                mu2: mean value of flat2
            afwVar: variance of difference flat, calculated with afw.
            cov: covariance value at lag(i, j)
            var: variance(covariance value at lag(0, 0))
            i: lag dimension
            j: lag dimension
            npix: number of pixels used for covariance calculation.
        Returns
        -------
        dataset: `lsst.ip.isr.ptcDataset.PhotonTransferCurveDataset`
            This is the same dataset as the input paramter, however, it has been modified
            to include information such as the fit vectors and the fit parameters. See
            the class `PhotonTransferCurveDatase`.
        """

        covFits, covFitsNoB = fitData(covariancesWithTagsArray,
                                      r=self.config.maximumRangeCovariancesAstier,
                                      nSigmaFullFit=self.config.sigmaClipFullFitCovariancesAstier,
                                      maxIterFullFit=self.config.maxIterFullFitCovariancesAstier)

        dataset = self.getOutputPtcDataCovAstier(dataset, covFits, covFitsNoB)

        return dataset

    def getOutputPtcDataCovAstier(self, dataset, covFits, covFitsNoB):
        """Get output data for PhotonTransferCurveCovAstierDataset from CovFit objects.
        Parameters
        ----------
        dataset : `lsst.ip.isr.ptcDataset.PhotonTransferCurveDataset`
            The dataset containing information such as the means, variances and exposure times.
        covFits: `dict`
            Dictionary of CovFit objects, with amp names as keys.
        covFitsNoB : `dict`
             Dictionary of CovFit objects, with amp names as keys, and 'b=0' in Eq. 20 of Astier+19.
        Returns
        -------
        dataset : `lsst.ip.isr.ptcDataset.PhotonTransferCurveDataset`
            This is the same dataset as the input paramter, however, it has been modified
            to include extra information such as the mask 1D array, gains, reoudout noise, measured signal,
            measured variance, modeled variance, a, and b coefficient matrices (see Astier+19) per amplifier.
            See the class `PhotonTransferCurveDatase`.
        """
        assert(len(covFits) == len(covFitsNoB))

        for i, amp in enumerate(dataset.ampNames):
            lenInputTimes = len(dataset.rawExpTimes[amp])
            # Not used when ptcFitType is 'FULLCOVARIANCE'
            dataset.ptcFitPars[amp] = np.nan
            dataset.ptcFitParsError[amp] = np.nan
            dataset.ptcFitChiSq[amp] = np.nan
            if amp in covFits:
                fit = covFits[amp]
                fitNoB = covFitsNoB[amp]
                # Save full covariances, covariances models, and their weights
                dataset.covariances[amp] = fit.cov
                dataset.covariancesModel[amp] = fit.evalCovModel()
                dataset.covariancesSqrtWeights[amp] = fit.sqrtW
                dataset.aMatrix[amp] = fit.getA()
                dataset.bMatrix[amp] = fit.getB()
                dataset.covariancesNoB[amp] = fitNoB.cov
                dataset.covariancesModelNoB[amp] = fitNoB.evalCovModel()
                dataset.covariancesSqrtWeightsNoB[amp] = fitNoB.sqrtW
                dataset.aMatrixNoB[amp] = fitNoB.getA()

                (meanVecFinal, varVecFinal, varVecModel,
                    wc, varMask) = fit.getFitData(0, 0, divideByMu=False, returnMasked=True)
                gain = fit.getGain()
                dataset.expIdMask[amp] = varMask
                dataset.gain[amp] = gain
                dataset.gainErr[amp] = fit.getGainErr()
                dataset.noise[amp] = np.sqrt(fit.getRon())
                dataset.noiseErr[amp] = fit.getRonErr()

                padLength = lenInputTimes - len(varVecFinal)
                dataset.finalVars[amp] = np.pad(varVecFinal/(gain**2), (0, padLength), 'constant',
                                                constant_values=np.nan)
                dataset.finalModelVars[amp] = np.pad(varVecModel/(gain**2), (0, padLength), 'constant',
                                                     constant_values=np.nan)
                dataset.finalMeans[amp] = np.pad(meanVecFinal/gain, (0, padLength), 'constant',
                                                 constant_values=np.nan)
            else:
                # Bad amp
                # Entries need to have proper dimensions so read/write with astropy.Table works.
                matrixSide = self.config.maximumRangeCovariancesAstier
                nanMatrix = np.full((matrixSide, matrixSide), np.nan)
                listNanMatrix = np.full((lenInputTimes, matrixSide, matrixSide), np.nan)

                dataset.covariances[amp] = listNanMatrix
                dataset.covariancesModel[amp] = listNanMatrix
                dataset.covariancesSqrtWeights[amp] = listNanMatrix
                dataset.aMatrix[amp] = nanMatrix
                dataset.bMatrix[amp] = nanMatrix
                dataset.covariancesNoB[amp] = listNanMatrix
                dataset.covariancesModelNoB[amp] = listNanMatrix
                dataset.covariancesSqrtWeightsNoB[amp] = listNanMatrix
                dataset.aMatrixNoB[amp] = nanMatrix

                dataset.expIdMask[amp] = np.repeat(np.nan, lenInputTimes)
                dataset.gain[amp] = np.nan
                dataset.gainErr[amp] = np.nan
                dataset.noise[amp] = np.nan
                dataset.noiseErr[amp] = np.nan
                dataset.finalVars[amp] = np.repeat(np.nan, lenInputTimes)
                dataset.finalModelVars[amp] = np.repeat(np.nan, lenInputTimes)
                dataset.finalMeans[amp] = np.repeat(np.nan, lenInputTimes)

        return dataset

    @staticmethod
    def _initialParsForPolynomial(order):
        assert(order >= 2)
        pars = np.zeros(order, dtype=np.float)
        pars[0] = 10
        pars[1] = 1
        pars[2:] = 0.0001
        return pars

    @staticmethod
    def _boundsForPolynomial(initialPars):
        lowers = [np.NINF for p in initialPars]
        uppers = [np.inf for p in initialPars]
        lowers[1] = 0  # no negative gains
        return (lowers, uppers)

    @staticmethod
    def _boundsForAstier(initialPars):
        lowers = [np.NINF for p in initialPars]
        uppers = [np.inf for p in initialPars]
        return (lowers, uppers)

    @staticmethod
    def _getInitialGoodPoints(means, variances, maxDeviationPositive, maxDeviationNegative):
        """Return a boolean array to mask bad points.
        A linear function has a constant ratio, so find the median
        value of the ratios, and exclude the points that deviate
        from that by more than a factor of maxDeviationPositive/negative.
        Asymmetric deviations are supported as we expect the PTC to turn
        down as the flux increases, but sometimes it anomalously turns
        upwards just before turning over, which ruins the fits, so it
        is wise to be stricter about restricting positive outliers than
        negative ones.
        Too high and points that are so bad that fit will fail will be included
        Too low and the non-linear points will be excluded, biasing the NL fit."""
        ratios = [b/a for (a, b) in zip(means, variances)]
        medianRatio = np.nanmedian(ratios)
        ratioDeviations = [(r/medianRatio)-1 for r in ratios]

        # so that it doesn't matter if the deviation is expressed as positive or negative
        maxDeviationPositive = abs(maxDeviationPositive)
        maxDeviationNegative = -1. * abs(maxDeviationNegative)

        goodPoints = np.array([True if (r < maxDeviationPositive and r > maxDeviationNegative)
                              else False for r in ratioDeviations])
        return goodPoints

    def _makeZeroSafe(self, array, warn=True, substituteValue=1e-9):
        """"""
        nBad = Counter(array)[0]
        if nBad == 0:
            return array

        if warn:
            msg = f"Found {nBad} zeros in array at elements {[x for x in np.where(array==0)[0]]}"
            self.log.warn(msg)

        array[array == 0] = substituteValue
        return array

    def fitPtc(self, dataset):
        """Fit the photon transfer curve to a polynimial or to Astier+19 approximation.
        Fit the photon transfer curve with either a polynomial of the order
        specified in the task config, or using the exponential approximation
        in Astier+19 (Eq. 16).
        Sigma clipping is performed iteratively for the fit, as well as an
        initial clipping of data points that are more than
        config.initialNonLinearityExclusionThreshold away from lying on a
        straight line. This other step is necessary because the photon transfer
        curve turns over catastrophically at very high flux (because saturation
        drops the variance to ~0) and these far outliers cause the initial fit
        to fail, meaning the sigma cannot be calculated to perform the
        sigma-clipping.
        Parameters
        ----------
        dataset : `lsst.ip.isr.ptcDataset.PhotonTransferCurveDataset`
            The dataset containing the means, variances and exposure times.
        Returns
        -------
        dataset: `lsst.ip.isr.ptcDataset.PhotonTransferCurveDataset`
            This is the same dataset as the input parameter, however, it has been modified
            to include information such as the fit vectors and the fit parameters. See
            the class `PhotonTransferCurveDatase`.

        Raises
        ------
        RuntimeError:
            Raises if dataset.ptcFitType is None.
        """
        if dataset.ptcFitType:
            ptcFitType = dataset.ptcFitType
        else:
            raise RuntimeError(f"None ptcFitType in PTC dataset.")
        matrixSide = self.config.maximumRangeCovariancesAstier
        nanMatrix = np.empty((matrixSide, matrixSide))
        nanMatrix[:] = np.nan

        for amp in dataset.ampNames:
            lenInputTimes = len(dataset.rawExpTimes[amp])
            listNanMatrix = np.empty((lenInputTimes, matrixSide, matrixSide))
            listNanMatrix[:] = np.nan

            dataset.covariances[amp] = listNanMatrix
            dataset.covariancesModel[amp] = listNanMatrix
            dataset.covariancesSqrtWeights[amp] = listNanMatrix
            dataset.aMatrix[amp] = nanMatrix
            dataset.bMatrix[amp] = nanMatrix
            dataset.covariancesNoB[amp] = listNanMatrix
            dataset.covariancesModelNoB[amp] = listNanMatrix
            dataset.covariancesSqrtWeightsNoB[amp] = listNanMatrix
            dataset.aMatrixNoB[amp] = nanMatrix

        def errFunc(p, x, y):
            return ptcFunc(p, x) - y

        sigmaCutPtcOutliers = self.config.sigmaCutPtcOutliers
        maxIterationsPtcOutliers = self.config.maxIterationsPtcOutliers

        for i, ampName in enumerate(dataset.ampNames):
            timeVecOriginal = np.array(dataset.rawExpTimes[ampName])
            meanVecOriginal = np.array(dataset.rawMeans[ampName])
            varVecOriginal = np.array(dataset.rawVars[ampName])
            varVecOriginal = self._makeZeroSafe(varVecOriginal)

            goodPoints = self._getInitialGoodPoints(meanVecOriginal, varVecOriginal,
                                                    self.config.initialNonLinearityExclusionThresholdPositive,
                                                    self.config.initialNonLinearityExclusionThresholdNegative)
            if not (goodPoints.any()):
                msg = (f"\nSERIOUS: All points in goodPoints: {goodPoints} are bad."
                       f"Setting {ampName} to BAD.")
                self.log.warn(msg)
                # The first and second parameters of initial fit are discarded (bias and gain)
                # for the final NL coefficients
                dataset.badAmps.append(ampName)
                dataset.expIdMask[ampName] = np.repeat(np.nan, len(dataset.rawExpTimes[ampName]))
                dataset.gain[ampName] = np.nan
                dataset.gainErr[ampName] = np.nan
                dataset.noise[ampName] = np.nan
                dataset.noiseErr[ampName] = np.nan
                dataset.ptcFitPars[ampName] = (np.repeat(np.nan, self.config.polynomialFitDegree + 1) if
                                               ptcFitType in ["POLYNOMIAL", ] else np.repeat(np.nan, 3))
                dataset.ptcFitParsError[ampName] = (np.repeat(np.nan, self.config.polynomialFitDegree + 1) if
                                                    ptcFitType in ["POLYNOMIAL", ] else np.repeat(np.nan, 3))
                dataset.ptcFitChiSq[ampName] = np.nan
                dataset.finalVars[ampName] = np.repeat(np.nan, len(dataset.rawExpTimes[ampName]))
                dataset.finalModelVars[ampName] = np.repeat(np.nan, len(dataset.rawExpTimes[ampName]))
                dataset.finalMeans[ampName] = np.repeat(np.nan, len(dataset.rawExpTimes[ampName]))
                continue

            mask = goodPoints

            if ptcFitType == 'EXPAPPROXIMATION':
                ptcFunc = funcAstier
                parsIniPtc = [-1e-9, 1.0, 10.]  # a00, gain, noise
                bounds = self._boundsForAstier(parsIniPtc)
            if ptcFitType == 'POLYNOMIAL':
                ptcFunc = funcPolynomial
                parsIniPtc = self._initialParsForPolynomial(self.config.polynomialFitDegree + 1)
                bounds = self._boundsForPolynomial(parsIniPtc)

            # Before bootstrap fit, do an iterative fit to get rid of outliers
            count = 1
            while count <= maxIterationsPtcOutliers:
                # Note that application of the mask actually shrinks the array
                # to size rather than setting elements to zero (as we want) so
                # always update mask itself and re-apply to the original data
                meanTempVec = meanVecOriginal[mask]
                varTempVec = varVecOriginal[mask]
                res = least_squares(errFunc, parsIniPtc, bounds=bounds, args=(meanTempVec, varTempVec))
                pars = res.x

                # change this to the original from the temp because the masks are ANDed
                # meaning once a point is masked it's always masked, and the masks must
                # always be the same length for broadcasting
                sigResids = (varVecOriginal - ptcFunc(pars, meanVecOriginal))/np.sqrt(varVecOriginal)
                newMask = np.array([True if np.abs(r) < sigmaCutPtcOutliers else False for r in sigResids])
                mask = mask & newMask
                if not (mask.any() and newMask.any()):
                    msg = (f"\nSERIOUS: All points in either mask: {mask} or newMask: {newMask} are bad. "
                           f"Setting {ampName} to BAD.")
                    self.log.warn(msg)
                    # The first and second parameters of initial fit are discarded (bias and gain)
                    # for the final NL coefficients
                    dataset.badAmps.append(ampName)
                    dataset.expIdMask[ampName] = np.repeat(np.nan, len(dataset.rawExpTimes[ampName]))
                    dataset.gain[ampName] = np.nan
                    dataset.gainErr[ampName] = np.nan
                    dataset.noise[ampName] = np.nan
                    dataset.noiseErr[ampName] = np.nan
                    dataset.ptcFitPars[ampName] = (np.repeat(np.nan, self.config.polynomialFitDegree + 1)
                                                   if ptcFitType in ["POLYNOMIAL", ] else
                                                   np.repeat(np.nan, 3))
                    dataset.ptcFitParsError[ampName] = (np.repeat(np.nan, self.config.polynomialFitDegree + 1)
                                                        if ptcFitType in ["POLYNOMIAL", ] else
                                                        np.repeat(np.nan, 3))
                    dataset.ptcFitChiSq[ampName] = np.nan
                    dataset.finalVars[ampName] = np.repeat(np.nan, len(dataset.rawExpTimes[ampName]))
                    dataset.finalModelVars[ampName] = np.repeat(np.nan, len(dataset.rawExpTimes[ampName]))
                    dataset.finalMeans[ampName] = np.repeat(np.nan, len(dataset.rawExpTimes[ampName]))
                    break
                nDroppedTotal = Counter(mask)[False]
                self.log.debug(f"Iteration {count}: discarded {nDroppedTotal} points in total for {ampName}")
                count += 1
                # objects should never shrink
                assert (len(mask) == len(timeVecOriginal) == len(meanVecOriginal) == len(varVecOriginal))

            if not (mask.any() and newMask.any()):
                continue
            dataset.expIdMask[ampName] = mask  # store the final mask
            parsIniPtc = pars
            meanVecFinal = meanVecOriginal[mask]
            varVecFinal = varVecOriginal[mask]

            if Counter(mask)[False] > 0:
                self.log.info((f"Number of points discarded in PTC of amplifier {ampName}:" +
                               f" {Counter(mask)[False]} out of {len(meanVecOriginal)}"))

            if (len(meanVecFinal) < len(parsIniPtc)):
                msg = (f"\nSERIOUS: Not enough data points ({len(meanVecFinal)}) compared to the number of"
                       f"parameters of the PTC model({len(parsIniPtc)}). Setting {ampName} to BAD.")
                self.log.warn(msg)
                # The first and second parameters of initial fit are discarded (bias and gain)
                # for the final NL coefficients
                dataset.badAmps.append(ampName)
                dataset.expIdMask[ampName] = np.repeat(np.nan, len(dataset.rawExpTimes[ampName]))
                dataset.gain[ampName] = np.nan
                dataset.gainErr[ampName] = np.nan
                dataset.noise[ampName] = np.nan
                dataset.noiseErr[ampName] = np.nan
                dataset.ptcFitPars[ampName] = (np.repeat(np.nan, self.config.polynomialFitDegree + 1) if
                                               ptcFitType in ["POLYNOMIAL", ] else np.repeat(np.nan, 3))
                dataset.ptcFitParsError[ampName] = (np.repeat(np.nan, self.config.polynomialFitDegree + 1) if
                                                    ptcFitType in ["POLYNOMIAL", ] else np.repeat(np.nan, 3))
                dataset.ptcFitChiSq[ampName] = np.nan
                dataset.finalVars[ampName] = np.repeat(np.nan, len(dataset.rawExpTimes[ampName]))
                dataset.finalModelVars[ampName] = np.repeat(np.nan, len(dataset.rawExpTimes[ampName]))
                dataset.finalMeans[ampName] = np.repeat(np.nan, len(dataset.rawExpTimes[ampName]))
                continue

            # Fit the PTC
            if self.config.doFitBootstrap:
                parsFit, parsFitErr, reducedChiSqPtc = fitBootstrap(parsIniPtc, meanVecFinal,
                                                                    varVecFinal, ptcFunc,
                                                                    weightsY=1./np.sqrt(varVecFinal))
            else:
                parsFit, parsFitErr, reducedChiSqPtc = fitLeastSq(parsIniPtc, meanVecFinal,
                                                                  varVecFinal, ptcFunc,
                                                                  weightsY=1./np.sqrt(varVecFinal))
            dataset.ptcFitPars[ampName] = parsFit
            dataset.ptcFitParsError[ampName] = parsFitErr
            dataset.ptcFitChiSq[ampName] = reducedChiSqPtc
            # Masked variances (measured and modeled) and means. Need to pad the array so astropy.Table does
            # not crash (the mask may vary per amp).
            padLength = len(dataset.rawExpTimes[ampName]) - len(varVecFinal)
            dataset.finalVars[ampName] = np.pad(varVecFinal, (0, padLength), 'constant',
                                                constant_values=np.nan)
            dataset.finalModelVars[ampName] = np.pad(ptcFunc(parsFit, meanVecFinal), (0, padLength),
                                                     'constant', constant_values=np.nan)
            dataset.finalMeans[ampName] = np.pad(meanVecFinal, (0, padLength), 'constant',
                                                 constant_values=np.nan)

            if ptcFitType == 'EXPAPPROXIMATION':
                ptcGain = parsFit[1]
                ptcGainErr = parsFitErr[1]
                ptcNoise = np.sqrt(np.fabs(parsFit[2]))
                ptcNoiseErr = 0.5*(parsFitErr[2]/np.fabs(parsFit[2]))*np.sqrt(np.fabs(parsFit[2]))
            if ptcFitType == 'POLYNOMIAL':
                ptcGain = 1./parsFit[1]
                ptcGainErr = np.fabs(1./parsFit[1])*(parsFitErr[1]/parsFit[1])
                ptcNoise = np.sqrt(np.fabs(parsFit[0]))*ptcGain
                ptcNoiseErr = (0.5*(parsFitErr[0]/np.fabs(parsFit[0]))*(np.sqrt(np.fabs(parsFit[0]))))*ptcGain
            dataset.gain[ampName] = ptcGain
            dataset.gainErr[ampName] = ptcGainErr
            dataset.noise[ampName] = ptcNoise
            dataset.noiseErr[ampName] = ptcNoiseErr
        if not len(dataset.ptcFitType) == 0:
            dataset.ptcFitType = ptcFitType
        if len(dataset.badAmps) == 0:
            dataset.badAmps = np.repeat(np.nan, len(list(dataset.rawExpTimes.values())[0]))

        return dataset


class MeasurePhotonTransferCurveTaskConfig(pexConfig.Config):
    extract = pexConfig.ConfigurableField(
        target=PhotonTransferCurveExtractTask,
        doc="Task to measure covariances from flats.",
    )
    solve = pexConfig.ConfigurableField(
        target=PhotonTransferCurveSolveTask,
        doc="Task to fit models to the measured covariances.",
    )
    ccdKey = pexConfig.Field(
        dtype=str,
        doc="The key by which to pull a detector from a dataId, e.g. 'ccd' or 'detector'.",
        default='ccd',
    )
    doPhotodiode = pexConfig.Field(
        dtype=bool,
        doc="Apply a correction based on the photodiode readings if available?",
        default=True,
    )
    photodiodeDataPath = pexConfig.Field(
        dtype=str,
        doc="Gen2 only: path to locate the data photodiode data files.",
        default=""
    )


class MeasurePhotonTransferCurveTask(pipeBase.CmdLineTask):
    """A class to calculate, fit, and plot a PTC from a set of flat pairs.
    The Photon Transfer Curve (var(signal) vs mean(signal)) is a standard
    tool used in astronomical detectors characterization (e.g., Janesick 2001,
    Janesick 2007). If ptcFitType is "EXPAPPROXIMATION" or "POLYNOMIAL",
    this task calculates the PTC from a series of pairs of flat-field images;
    each pair taken at identical exposure times. The difference image of each
    pair is formed to eliminate fixed pattern noise, and then the variance
    of the difference image and the mean of the average image
    are used to produce the PTC. An n-degree polynomial or the approximation
    in Equation 16 of Astier+19 ("The Shape of the Photon Transfer Curve
    of CCD sensors", arXiv:1905.08677) can be fitted to the PTC curve. These
    models include parameters such as the gain (e/DN) and readout noise.
    Linearizers to correct for signal-chain non-linearity are also calculated.
    The `Linearizer` class, in general, can support per-amp linearizers, but
    in this task this is not supported.
    If ptcFitType is "FULLCOVARIANCE", the covariances of the difference
    images are calculated via the DFT methods described in Astier+19 and the
    variances for the PTC are given by the cov[0,0] elements at each signal
    level. The full model in Equation 20 of Astier+19 is fit to the PTC
    to get the gain and the noise.

    Parameters
    ----------
    *args: `list`
        Positional arguments passed to the Task constructor. None used at this
        time.
    **kwargs: `dict`
        Keyword arguments passed on to the Task constructor. None used at this
        time.
    """

    RunnerClass = DataRefListRunner
    ConfigClass = MeasurePhotonTransferCurveTaskConfig
    _DefaultName = "measurePhotonTransferCurve"

    def __init__(self, *args, **kwargs):
        super().__init__(**kwargs)
        self.makeSubtask("extract")
        self.makeSubtask("solve")

    @pipeBase.timeMethod
    def runDataRef(self, dataRefList):
        """Run the Photon Transfer Curve (PTC) measurement task.
        For a dataRef (which is each detector here),
        and given a list of exposure pairs (postISR) at different exposure times,
        measure the PTC.
        Parameters
        ----------
        dataRefList : `list` [`lsst.daf.peristence.ButlerDataRef`]
            Data references for exposures.
        """
        if len(dataRefList) < 2:
            raise RuntimeError("Insufficient inputs to combine.")

        # setup necessary objects
        dataRef = dataRefList[0]
        camera = dataRef.get('camera')

        if len(set([dataRef.dataId[self.config.ccdKey] for dataRef in dataRefList])) > 1:
            raise RuntimeError("Too many detectors supplied")

        # Organize exposures by observation date.
        expDict = {}
        for dataRef in dataRefList:
            try:
                tempFlat = dataRef.get("postISRCCD")
            except RuntimeError:
                self.log.warn("postISR exposure could not be retrieved. Ignoring flat.")
                continue
            expDate = tempFlat.getInfo().getVisitInfo().getDate().get()
            expDict.setdefault(expDate, tempFlat)
        sortedExps = list({k: expDict[k] for k in sorted(expDict)}.values())
        # Get the pairs of flats indexed by expTime
        expPairsDict = makeExpPairs(sortedExps)
        expPairsList = list(expPairsDict.values())
        detector = expPairsList[0][0].getDetector()
        # Make a list of the Id's of the exp pairs
        expIds = []
        for (exp1, exp2) in expPairsList:
            id1 = exp1.getInfo().getVisitInfo().getExposureId()
            id2 = exp2.getInfo().getVisitInfo().getExposureId()
            expIds.append((id1, id2))
        self.log.info(f"Measuring PTC using {expIds} exposures for detector {detector.getId()}")

        # Get first exposure
        inputDims = expPairsList[0][0]
        result = self.extract.run(expPairsDict, camera)
        finalResults = self.solve.run(result.outputCovariances, camera=camera, inputDims=inputDims)

        for ampName in finalResults.outputPtcDataset.ampNames:
            finalResults.outputPtcDataset.inputExpIdPairs[ampName] = expIds

        # Fill up the photodiode data, if found, that will be used by linearity task.
        finalResults.outputPtcDataset = self._setBOTPhotocharge(dataRef, finalResults.outputPtcDataset,
                                                                expIds)

        self.log.info("Writing PTC data.")
        dataRef.put(finalResults.outputPtcDataset, datasetType="photonTransferCurveDataset")

        return

    def _setBOTPhotocharge(self, dataRef, datasetPtc, expIdList):
        """Set photoCharge attribute in PTC dataset

        Parameters
        ----------
        dataRef : `lsst.daf.peristence.ButlerDataRef`
            Data reference for exposurre for detector to process.

        datasetPtc : `lsst.ip.isr.ptcDataset.PhotonTransferCurveDataset`
            The dataset containing information such as the means, variances
            and exposure times.

        expIdList : `list`
            List with exposure pairs Ids (one pair per list entry).

        Returns
        -------
        datasetPtc: `lsst.ip.isr.ptcDataset.PhotonTransferCurveDataset`
            This is the same dataset as the input parameter, however,
            it has been modified to update the datasetPtc.photoCharge
            attribute.
        """
        if self.config.doPhotodiode:
            for (expId1, expId2) in expIdList:
                charges = [-1, -1]  # necessary to have a not-found value to keep lists in step
                for i, expId in enumerate([expId1, expId2]):
                    # //1000 is a Gen2 only hack, working around the fact an
                    # exposure's ID is not the same as the expId in the
                    # registry. Currently expId is concatenated with the
                    # zero-padded detector ID. This will all go away in Gen3.
                    dataRef.dataId['expId'] = expId//1000
                    if self.config.photodiodeDataPath:
                        photodiodeData = getBOTphotodiodeData(dataRef, self.config.photodiodeDataPath)
                    else:
                        photodiodeData = getBOTphotodiodeData(dataRef)
                    if photodiodeData:  # default path stored in function def to keep task clean
                        charges[i] = photodiodeData.getCharge()
                    else:
                        # full expId (not //1000) here, as that encodes the
                        # the detector number as so is fully qualifying
                        self.log.warn(f"No photodiode data found for {expId}")

                for ampName in datasetPtc.ampNames:
                    datasetPtc.photoCharge[ampName].append((charges[0], charges[1]))
        else:
            # Can't be an empty list, as initialized, because astropy.Table won't allow it
            # when saving as fits
            for ampName in datasetPtc.ampNames:
                datasetPtc.photoCharge[ampName] = np.repeat(np.nan, len(expIdList))

        return datasetPtc
