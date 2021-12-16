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
from lsst.cp.pipe.utils import (arrangeFlatsByExpTime, arrangeFlatsByExpId,
                                sigmaClipCorrection, CovFastFourierTransform,
                                computeCovDirect)

import lsst.pipe.base.connectionTypes as cT

from .astierCovPtcFit import makeCovArray

from lsst.ip.isr import PhotonTransferCurveDataset
from lsst.ip.isr import IsrTask

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
        deferLoad=True,
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
        doc="Should exposures be matched by ID rather than exposure time?",
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
        doc="Minimum values (inclusive) of mean signal (in ADU) per amp to use."
            " The same cut is applied to all amps if this parameter [`dict`] is passed as "
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
        default=['SUSPECT', 'BAD', 'NO_DATA', 'SAT'],
    )
    nSigmaClipPtc = pexConfig.Field(
        dtype=float,
        doc="Sigma cut for afwMath.StatisticsControl()",
        default=5.5,
    )
    nIterSigmaClipPtc = pexConfig.Field(
        dtype=int,
        doc="Number of sigma-clipping iterations for afwMath.StatisticsControl()",
        default=3,
    )
    minNumberGoodPixelsForCovariance = pexConfig.Field(
        dtype=int,
        doc="Minimum number of acceptable good pixels per amp to calculate the covariances (via FFT or"
            " direclty).",
        default=10000,
    )
    thresholdDiffAfwVarVsCov00 = pexConfig.Field(
        dtype=float,
        doc="If the absolute fractional differece between afwMath.VARIANCECLIP and Cov00 "
            "for a region of a difference image is greater than this threshold (percentage), "
            "a warning will be issued.",
        default=1.,
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
    numEdgeSuspect = pexConfig.Field(
        dtype=int,
        doc="Number of edge pixels to be flagged as untrustworthy.",
        default=0,
    )
    edgeMaskLevel = pexConfig.ChoiceField(
        dtype=str,
        doc="Mask edge pixels in which coordinate frame: DETECTOR or AMP?",
        default="DETECTOR",
        allowed={
            'DETECTOR': 'Mask only the edges of the full detector.',
            'AMP': 'Mask edges of each amplifier.',
        },
    )


class PhotonTransferCurveExtractTask(pipeBase.PipelineTask,
                                     pipeBase.CmdLineTask):
    """Task to measure covariances from flat fields.

    This task receives as input a list of flat-field images
    (flats), and sorts these flats in pairs taken at the
    same time (the task will raise if there is one one flat
    at a given exposure time, and it will discard extra flats if
    there are more than two per exposure time). This task measures
    the  mean, variance, and covariances from a region (e.g.,
    an amplifier) of the difference image of the two flats with
    the same exposure time.

    The variance is calculated via afwMath, and the covariance
    via the methods in Astier+19 (appendix A). In theory,
    var = covariance[0,0].  This should be validated, and in the
    future, we may decide to just keep one (covariance).
    At this moment, if the two values differ by more than the value
    of `thresholdDiffAfwVarVsCov00` (default: 1%), a warning will
    be issued.

    The measured covariances at a given exposure time (along with
    other quantities such as the mean) are stored in a PTC dataset
    object (`~lsst.ip.isr.PhotonTransferCurveDataset`), which gets
    partially filled at this stage (the remainder of the attributes
    of the dataset will be filled after running the second task of
    the PTC-measurement pipeline, `~PhotonTransferCurveSolveTask`).

    The number of partially-filled
    `~lsst.ip.isr.PhotonTransferCurveDataset` objects will be less
    than the number of input exposures because the task combines
    input flats in pairs. However, it is required at this mioment
    that the number of input dimensions matches
    bijectively the number of output dimensions. Therefore, a number
    of "dummy" PTC dataset are inserted in the output list.  This
    output list will then be used as input of the next task in the
    PTC-measurement pipeline, `PhotonTransferCurveSolveTask`,
    which will assemble the multiple `PhotonTransferCurveDataset`
    objects into a single one in order to fit the measured covariances
    as a function of flux to one of three models
    (see `PhotonTransferCurveSolveTask` for details).

    Reference: Astier+19: "The Shape of the Photon Transfer Curve of CCD
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
        # Ids of input list of exposure references
        # (deferLoad=True in the input connections)

        inputs['inputDims'] = [expRef.datasetRef.dataId['exposure'] for expRef in inputRefs.inputExp]

        # Dictionary, keyed by expTime, with tuples containing flat
        # exposures and their IDs.
        if self.config.matchByExposureId:
            inputs['inputExp'] = arrangeFlatsByExpId(inputs['inputExp'], inputs['inputDims'])
        else:
            inputs['inputExp'] = arrangeFlatsByExpTime(inputs['inputExp'], inputs['inputDims'])

        outputs = self.run(**inputs)
        butlerQC.put(outputs, outputRefs)

    def run(self, inputExp, inputDims):
        """Measure covariances from difference of flat pairs

        Parameters
        ----------
        inputExp : `dict` [`float`, `list`
                          [`~lsst.pipe.base.connections.DeferredDatasetRef`]]
            Dictionary that groups references to flat-field exposures that
            have the same exposure time (seconds), or that groups them
            sequentially by their exposure id.

        inputDims : `list`
            List of exposure IDs.

        Returns
        -------
        results : `lsst.pipe.base.Struct`
            The results struct containing:

            ``outputCovariances``
                A list containing the per-pair PTC measurements (`list`
                [`lsst.ip.isr.PhotonTransferCurveDataset`])
        """
        # inputExp.values() returns a view, which we turn into a list. We then
        # access the first exposure-ID tuple to get the detector.
        # The first "get()" retrieves the exposure from the exposure reference.
        detector = list(inputExp.values())[0][0][0].get(component='detector')
        detNum = detector.getId()
        amps = detector.getAmplifiers()
        ampNames = [amp.getName() for amp in amps]

        # Each amp may have a different min and max ADU signal
        # specified in the config.
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
        # These are the column names for `tupleRows` below.
        tags = [('mu', '<f8'), ('afwVar', '<f8'), ('i', '<i8'), ('j', '<i8'), ('var', '<f8'),
                ('cov', '<f8'), ('npix', '<i8'), ('ext', '<i8'), ('expTime', '<f8'), ('ampName', '<U3')]
        # Create a dummy ptcDataset. Dummy datasets will be
        # used to ensure that the number of output and input
        # dimensions match.
        dummyPtcDataset = PhotonTransferCurveDataset(ampNames, 'DUMMY',
                                                     self.config.maximumRangeCovariancesAstier)
        # Initialize amps of `dummyPtcDatset`.
        for ampName in ampNames:
            dummyPtcDataset.setAmpValues(ampName)
        # Output list with PTC datasets.
        partialPtcDatasetList = []
        # The number of output references needs to match that of input
        # references: initialize outputlist with dummy PTC datasets.
        for i in range(len(inputDims)):
            partialPtcDatasetList.append(dummyPtcDataset)

        if self.config.numEdgeSuspect > 0:
            isrTask = IsrTask()
            self.log.info("Masking %d pixels from the edges of all exposures as SUSPECT.",
                          self.config.numEdgeSuspect)

        for expTime in inputExp:
            exposures = inputExp[expTime]
            if len(exposures) == 1:
                self.log.warning("Only one exposure found at expTime %f. Dropping exposure %d.",
                                 expTime, exposures[0][1])
                continue
            else:
                # Only use the first two exposures at expTime. Each
                # elements is a tuple (exposure, expId)
                expRef1, expId1 = exposures[0]
                expRef2, expId2 = exposures[1]
                # use get() to obtain `lsst.afw.image.Exposure`
                exp1, exp2 = expRef1.get(), expRef2.get()

                if len(exposures) > 2:
                    self.log.warning("Already found 2 exposures at expTime %f. Ignoring exposures: %s",
                                     expTime, ", ".join(str(i[1]) for i in exposures[2:]))
            # Mask pixels at the edge of the detector or of each amp
            if self.config.numEdgeSuspect > 0:
                isrTask.maskEdges(exp1, numEdgePixels=self.config.numEdgeSuspect,
                                  maskPlane="SUSPECT", level=self.config.edgeMaskLevel)
                isrTask.maskEdges(exp2, numEdgePixels=self.config.numEdgeSuspect,
                                  maskPlane="SUSPECT", level=self.config.edgeMaskLevel)

            nAmpsNan = 0
            partialPtcDataset = PhotonTransferCurveDataset(ampNames, 'PARTIAL',
                                                           self.config.maximumRangeCovariancesAstier)
            for ampNumber, amp in enumerate(detector):
                ampName = amp.getName()
                # covAstier: [(i, j, var (cov[0,0]), cov, npix) for
                # (i,j) in {maxLag, maxLag}^2]
                doRealSpace = self.config.covAstierRealSpace
                if self.config.detectorMeasurementRegion == 'AMP':
                    region = amp.getBBox()
                elif self.config.detectorMeasurementRegion == 'FULL':
                    region = None
                # `measureMeanVarCov` is the function that measures
                # the variance and covariances from a region of
                # the difference image of two flats at the same
                # exposure time.  The variable `covAstier` that is
                # returned is of the form:
                # [(i, j, var (cov[0,0]), cov, npix) for (i,j) in
                # {maxLag, maxLag}^2].
                muDiff, varDiff, covAstier = self.measureMeanVarCov(exp1, exp2, region=region,
                                                                    covAstierRealSpace=doRealSpace)
                # Correction factor for bias introduced by sigma
                # clipping.
                # Function returns 1/sqrt(varFactor), so it needs
                # to be squared. varDiff is calculated via
                # afwMath.VARIANCECLIP.
                varFactor = sigmaClipCorrection(self.config.nSigmaClipPtc)**2
                varDiff *= varFactor

                expIdMask = True
                # Mask data point at this mean signal level if
                # the signal, variance, or covariance calculations
                # from `measureMeanVarCov` resulted in NaNs.
                if np.isnan(muDiff) or np.isnan(varDiff) or (covAstier is None):
                    msg = ("NaN mean or var, or None cov in amp %s in exposure pair %d, %d of detector %d.",
                           ampName, expId1, expId2, detNum)
                    self.log.warning(msg)
                    nAmpsNan += 1
                    expIdMask = False
                    covArray = np.full((1, self.config.maximumRangeCovariancesAstier,
                                        self.config.maximumRangeCovariancesAstier), np.nan)
                    covSqrtWeights = np.full_like(covArray, np.nan)

                # Mask data point if it is outside of the
                # specified mean signal range.
                if (muDiff <= minMeanSignalDict[ampName]) or (muDiff >= maxMeanSignalDict[ampName]):
                    expIdMask = False

                if covAstier is not None:
                    # Turn the tuples with teh measured information
                    # into covariance arrays.
                    tupleRows = [(muDiff, varDiff) + covRow + (ampNumber, expTime,
                                                               ampName) for covRow in covAstier]
                    tempStructArray = np.array(tupleRows, dtype=tags)
                    covArray, vcov, _ = makeCovArray(tempStructArray,
                                                     self.config.maximumRangeCovariancesAstier)
                    covSqrtWeights = np.nan_to_num(1./np.sqrt(vcov))

                # Correct covArray for sigma clipping:
                # 1) Apply varFactor twice for the whole covariance matrix
                covArray *= varFactor**2
                # 2) But, only once for the variance element of the
                # matrix, covArray[0,0] (so divide one factor out).
                covArray[0, 0] /= varFactor

                partialPtcDataset.setAmpValues(ampName, rawExpTime=[expTime], rawMean=[muDiff],
                                               rawVar=[varDiff], inputExpIdPair=[(expId1, expId2)],
                                               expIdMask=[expIdMask], covArray=covArray,
                                               covSqrtWeights=covSqrtWeights)
            # Use location of exp1 to save PTC dataset from (exp1, exp2) pair.
            # Below, np.where(expId1 == np.array(inputDims)) returns a tuple
            # with a single-element array, so [0][0]
            # is necessary to extract the required index.
            datasetIndex = np.where(expId1 == np.array(inputDims))[0][0]
            # `partialPtcDatasetList` is a list of
            # `PhotonTransferCurveDataset` objects. Some of them
            # will be dummy datasets (to match length of input
            # and output references), and the rest will have
            # datasets with the mean signal, variance, and
            # covariance measurements at a given exposure
            # time. The next ppart of the PTC-measurement
            # pipeline, `solve`, will take this list as input,
            # and assemble the measurements in the datasets
            # in an addecuate manner for fitting a PTC
            # model.
            partialPtcDatasetList[datasetIndex] = partialPtcDataset

            if nAmpsNan == len(ampNames):
                msg = f"NaN mean in all amps of exposure pair {expId1}, {expId2} of detector {detNum}."
                self.log.warning(msg)
        return pipeBase.Struct(
            outputCovariances=partialPtcDatasetList,
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
        exposure1 : `lsst.afw.image.exposure.ExposureF`
            First exposure of flat field pair.
        exposure2 : `lsst.afw.image.exposure.ExposureF`
            Second exposure of flat field pair.
        region : `lsst.geom.Box2I`, optional
            Region of each exposure where to perform the calculations
            (e.g, an amplifier).
        covAstierRealSpace : `bool`, optional
            Should the covariannces in Astier+19 be calculated in real
            space or via FFT?  See Appendix A of Astier+19.

        Returns
        -------
        mu : `float` or `NaN`
            0.5*(mu1 + mu2), where mu1, and mu2 are the clipped means
            of the regions in both exposures. If either mu1 or m2 are
            NaN's, the returned value is NaN.
        varDiff : `float` or `NaN`
            Half of the clipped variance of the difference of the
            regions inthe two input exposures. If either mu1 or m2 are
            NaN's, the returned value is NaN.
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
            self.log.warning("Mean of amp in image 1 or 2 is NaN: %f, %f.", mu1, mu2)
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

        # Variance calculation via afwMath
        varDiff = 0.5*(afwMath.makeStatistics(diffIm, afwMath.VARIANCECLIP, diffImStatsCtrl).getValue())

        # Covariances calculations
        # Get the pixels that were not clipped
        varClip = afwMath.makeStatistics(diffIm, afwMath.VARIANCECLIP, diffImStatsCtrl).getValue()
        meanClip = afwMath.makeStatistics(diffIm, afwMath.MEANCLIP, diffImStatsCtrl).getValue()
        cut = meanClip + self.config.nSigmaClipPtc*np.sqrt(varClip)
        unmasked = np.where(np.fabs(diffIm.image.array) <= cut, 1, 0)

        # Get the pixels in the mask planes of the difference image
        # that were ignored by the clipping algorithm
        wDiff = np.where(diffIm.getMask().getArray() == 0, 1, 0)
        # Combine the two sets of pixels ('1': use; '0': don't use)
        # into a final weight matrix to be used in the covariance
        # calculations below.
        w = unmasked*wDiff

        if np.sum(w) < self.config.minNumberGoodPixelsForCovariance:
            self.log.warning("Number of good points for covariance calculation (%s) is less "
                             "(than threshold %s)", np.sum(w), self.config.minNumberGoodPixelsForCovariance)
            return np.nan, np.nan, None

        maxRangeCov = self.config.maximumRangeCovariancesAstier
        if covAstierRealSpace:
            # Calculate  covariances in real space.
            covDiffAstier = computeCovDirect(diffIm.image.array, w, maxRangeCov)
        else:
            # Calculate covariances via FFT (default).
            shapeDiff = np.array(diffIm.image.array.shape)
            # Calculate the sizes of FFT dimensions.
            s = shapeDiff + maxRangeCov
            tempSize = np.array(np.log(s)/np.log(2.)).astype(int)
            fftSize = np.array(2**(tempSize+1)).astype(int)
            fftShape = (fftSize[0], fftSize[1])

            c = CovFastFourierTransform(diffIm.image.array, w, fftShape, maxRangeCov)
            covDiffAstier = c.reportCovFastFourierTransform(maxRangeCov)

        # Compare Cov[0,0] and afwMath.VARIANCECLIP covDiffAstier[0]
        # is the Cov[0,0] element, [3] is the variance, and there's a
        # factor of 0.5 difference with afwMath.VARIANCECLIP.
        thresholdPercentage = self.config.thresholdDiffAfwVarVsCov00
        fractionalDiff = 100*np.fabs(1 - varDiff/(covDiffAstier[0][3]*0.5))
        if fractionalDiff >= thresholdPercentage:
            self.log.warning("Absolute fractional difference between afwMatch.VARIANCECLIP and Cov[0,0] "
                             "is more than %f%%: %f", thresholdPercentage, fractionalDiff)

        return mu, varDiff, covDiffAstier
