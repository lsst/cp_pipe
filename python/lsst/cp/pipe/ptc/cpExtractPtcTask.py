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
                                arrangeFlatsByExpFlux, sigmaClipCorrection,
                                CovFastFourierTransform)

import lsst.pipe.base.connectionTypes as cT

from lsst.ip.isr import PhotonTransferCurveDataset
from lsst.ip.isr import IsrTask
from lsst.geom import Box2I, Point2I

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
    taskMetadata = cT.Input(
        name="isr_metadata",
        doc="Input task metadata to extract statistics from.",
        storageClass="TaskMetadata",
        dimensions=("instrument", "exposure", "detector"),
        multiple=True,
    )
    outputCovariances = cT.Output(
        name="ptcCovariances",
        doc="Extracted flat (co)variances.",
        storageClass="PhotonTransferCurveDataset",
        dimensions=("instrument", "exposure", "detector"),
        isCalibration=True,
        multiple=True,
    )


class PhotonTransferCurveExtractConfig(pipeBase.PipelineTaskConfig,
                                       pipelineConnections=PhotonTransferCurveExtractConnections):
    """Configuration for the measurement of covariances from flats.
    """
    matchExposuresType = pexConfig.ChoiceField(
        dtype=str,
        doc="Match input exposures by time, flux, or expId",
        default='TIME',
        allowed={
            "TIME": "Match exposures by exposure time.",
            "FLUX": "Match exposures by target flux. Use header keyword"
                " in matchExposuresByFluxKeyword to find the flux.",
            "EXPID": "Match exposures by exposure ID."
        }
    )
    matchExposuresByFluxKeyword = pexConfig.Field(
        dtype=str,
        doc="Header keyword for flux if matchExposuresType is FLUX.",
        default='CCOBFLUX',
    )
    maximumRangeCovariancesAstier = pexConfig.Field(
        dtype=int,
        doc="Maximum range of covariances as in Astier+19",
        default=8,
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
    doGain = pexConfig.Field(
        dtype=bool,
        doc="Calculate a gain per input flat pair.",
        default=True,
    )
    gainCorrectionType = pexConfig.ChoiceField(
        dtype=str,
        doc="Correction type for the gain.",
        default='FULL',
        allowed={
            'NONE': 'No correction.',
            'SIMPLE': 'First order correction.',
            'FULL': 'Second order correction.'
        }
    )
    numSubregionsX = pexConfig.Field(
        dtype=int,
        doc="Number of subregions in each amp in the X direction.",
        default=1,
    )
    numSubregionsY = pexConfig.Field(
        dtype=int,
        doc="Number of subregions in each amp in the Y direction.",
        default=1,
    )


class PhotonTransferCurveExtractTask(pipeBase.PipelineTask):
    """Task to measure covariances from flat fields.

    This task receives as input a list of flat-field images
    (flats), and sorts these flats in pairs taken at the
    same time (the task will raise if there is one one flat
    at a given exposure time, and it will discard extra flats if
    there are more than two per exposure time). This task measures
    the  mean, variance, and covariances from a region (e.g.,
    an amplifier) of the difference image of the two flats with
    the same exposure time (alternatively, all input images could have
    the same exposure time but their flux changed).

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
    input flats in pairs. However, it is required at this moment
    that the number of input dimensions matches
    bijectively the number of output dimensions. Therefore, a number
    of "dummy" PTC datasets are inserted in the output list.  This
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

        # Dictionary, keyed by expTime (or expFlux or expId), with tuples
        # containing flat exposures and their IDs.
        matchType = self.config.matchExposuresType
        if matchType == 'TIME':
            inputs['inputExp'] = arrangeFlatsByExpTime(inputs['inputExp'], inputs['inputDims'])
        elif matchType == 'FLUX':
            inputs['inputExp'] = arrangeFlatsByExpFlux(inputs['inputExp'], inputs['inputDims'],
                                                       self.config.matchExposuresByFluxKeyword)
        else:
            inputs['inputExp'] = arrangeFlatsByExpId(inputs['inputExp'], inputs['inputDims'])

        outputs = self.run(**inputs)
        butlerQC.put(outputs, outputRefs)

    def run(self, inputExp, inputDims, taskMetadata):
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
        taskMetadata : `list` [`lsst.pipe.base.TaskMetadata`]
            List of exposures metadata from ISR.

        Returns
        -------
        results : `lsst.pipe.base.Struct`
            The resulting Struct contains:

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

        # We have the option to split each amp into a number of sub-regions
        # These parameters determine how many subregions we split the amp into.
        nSubX = self.config.numSubregionsX
        nSubY = self.config.numSubregionsY        

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

        readNoiseDict = {ampName: 0.0 for ampName in ampNames}
        for ampName in ampNames:
            # Initialize amps of `dummyPtcDatset`.
            dummyPtcDataset.setAmpValues(ampName)
            # Overscan readnoise from post-ISR exposure metadata.
            # It will be used to estimate the gain from a pair of flats.
            readNoiseDict[ampName] = self.getReadNoiseFromMetadata(taskMetadata, ampName)

        # Output list with PTC datasets.
        partialPtcDatasetList = []
        # The number of output references needs to match that of input
        # references: initialize outputlist with dummy PTC datasets.
        for i in range(len(inputDims) * nSubX * nSubY):
            partialPtcDatasetList.append(dummyPtcDataset)

        if self.config.numEdgeSuspect > 0:
            isrTask = IsrTask()
            self.log.info("Masking %d pixels from the edges of all exposures as SUSPECT.",
                          self.config.numEdgeSuspect)

        # Depending on the value of config.matchExposuresType
        # 'expTime' can stand for exposure time, flux, or ID.
        for expTime in inputExp:
            exposures = inputExp[expTime]
            if len(exposures) == 1:
                self.log.warning("Only one exposure found at %s %f. Dropping exposure %d.",
                                 self.config.matchExposuresType, expTime, exposures[0][1])
                continue
            else:
                # Only use the first two exposures at expTime. Each
                # element is a tuple (exposure, expId)
                expRef1, expId1 = exposures[0]
                expRef2, expId2 = exposures[1]
                # use get() to obtain `lsst.afw.image.Exposure`
                exp1, exp2 = expRef1.get(), expRef2.get()

                if len(exposures) > 2:
                    self.log.warning("Already found 2 exposures at %s %f. Ignoring exposures: %s",
                                     self.config.matchExposuresType, expTime,
                                     ", ".join(str(i[1]) for i in exposures[2:]))
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
                if self.config.detectorMeasurementRegion == 'AMP':
                    region = amp.getBBox()
                elif self.config.detectorMeasurementRegion == 'FULL':
                    region = None

                # Now split the region into subregions (if nSubX * nSubY > 1)    
                xStep = int(region.width / nSubX)
                yStep = int(region.height / nSubY)
                for iSub in range(nSubX):
                    xmin = region.minX + iSub * xStep
                    if iSub < (nSubX - 1):
                        xmax = region.minX + (iSub + 1) * xStep
                    else:
                        xmax = region.maxX
                    for jSub in range(nSubY):
                        ymin = region.minY + jSub * yStep
                        if jSub < (nSubY - 1):
                            ymax = region.minY + (jSub + 1) * yStep
                        else:
                            ymax = region.maxY
                        subRegion = Box2I(minimum=Point2I(xmin,ymin), maximum=Point2I(xmax,ymax))

                        # Get masked image regions, masking planes, statistic control
                        # objects, and clipped means. Calculate once to reuse in
                        # `measureMeanVarCov` and `getGainFromFlatPair`.
                        im1Area, im2Area, imStatsCtrl, mu1, mu2 = self.getImageAreasMasksStats(exp1, exp2,
                                                                                               region=subRegion)

                        # `measureMeanVarCov` is the function that measures
                        # the variance and covariances from a region of
                        # the difference image of two flats at the same
                        # exposure time.  The variable `covAstier` that is
                        # returned is of the form:
                        # [(i, j, var (cov[0,0]), cov, npix) for (i,j) in
                        # {maxLag, maxLag}^2].
                        muDiff, varDiff, covAstier = self.measureMeanVarCov(im1Area, im2Area, imStatsCtrl, mu1, mu2)
                        # Estimate the gain from the flat pair
                        if self.config.doGain:
                            gain = self.getGainFromFlatPair(im1Area, im2Area, imStatsCtrl, mu1, mu2,
                                                            correctionType=self.config.gainCorrectionType,
                                                            readNoise=readNoiseDict[ampName])
                        else:
                            gain = np.nan

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
                            self.log.warning("NaN mean or var, or None cov in amp %s in exposure pair %d, %d of "
                                             "detector %d.", ampName, expId1, expId2, detNum)
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
                            # Turn the tuples with the measured information
                            # into covariance arrays.
                            # covrow: (i, j, var (cov[0,0]), cov, npix)
                            tupleRows = [(muDiff, varDiff) + covRow + (ampNumber, expTime,
                                                                       ampName) for covRow in covAstier]
                            tempStructArray = np.array(tupleRows, dtype=tags)
                            covArray, vcov, _ = self.makeCovArray(tempStructArray,
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
                                                       covSqrtWeights=covSqrtWeights, gain=gain,
                                                       noise=readNoiseDict[ampName])
                        # Use location of exp1 to save PTC dataset from (exp1, exp2) pair.
                        # Below, np.where(expId1 == np.array(inputDims)) returns a tuple
                        # with a single-element array, so [0][0]
                        # is necessary to extract the required index.
                        #datasetIndex = np.where(expId1 == np.array(inputDims))[0][0]
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
                        partialPtcDataset.updateMetadata(setDate=True, detector=detector)
                        #partialPtcDatasetList[datasetIndex] = partialPtcDataset
                        partialPtcDatasetList.append(partialPtcDataset)                        

                if nAmpsNan == len(ampNames):
                    msg = f"NaN mean in all amps of exposure pair {expId1}, {expId2} of detector {detNum}."
                    self.log.warning(msg)
        return pipeBase.Struct(
            outputCovariances=partialPtcDatasetList,
        )

    def makeCovArray(self, inputTuple, maxRangeFromTuple):
        """Make covariances array from tuple.

        Parameters
        ----------
        inputTuple : `numpy.ndarray`
            Structured array with rows with at least
            (mu, afwVar, cov, var, i, j, npix), where:
            mu : `float`
                0.5*(m1 + m2), where mu1 is the mean value of flat1
                and mu2 is the mean value of flat2.
            afwVar : `float`
                Variance of difference flat, calculated with afw.
            cov : `float`
                Covariance value at lag(i, j)
            var : `float`
                Variance(covariance value at lag(0, 0))
            i : `int`
                Lag in dimension "x".
            j : `int`
                Lag in dimension "y".
            npix : `int`
                Number of pixels used for covariance calculation.
        maxRangeFromTuple : `int`
            Maximum range to select from tuple.

        Returns
        -------
        cov : `numpy.array`
            Covariance arrays, indexed by mean signal mu.
        vCov : `numpy.array`
            Variance of the [co]variance arrays, indexed by mean signal mu.
        muVals : `numpy.array`
            List of mean signal values.
        """
        if maxRangeFromTuple is not None:
            cut = (inputTuple['i'] < maxRangeFromTuple) & (inputTuple['j'] < maxRangeFromTuple)
            cutTuple = inputTuple[cut]
        else:
            cutTuple = inputTuple
        # increasing mu order, so that we can group measurements with the
        # same mu
        muTemp = cutTuple['mu']
        ind = np.argsort(muTemp)

        cutTuple = cutTuple[ind]
        # should group measurements on the same image pairs(same average)
        mu = cutTuple['mu']
        xx = np.hstack(([mu[0]], mu))
        delta = xx[1:] - xx[:-1]
        steps, = np.where(delta > 0)
        ind = np.zeros_like(mu, dtype=int)
        ind[steps] = 1
        ind = np.cumsum(ind)  # this acts as an image pair index.
        # now fill the 3-d cov array(and variance)
        muVals = np.array(np.unique(mu))
        i = cutTuple['i'].astype(int)
        j = cutTuple['j'].astype(int)
        c = 0.5*cutTuple['cov']
        n = cutTuple['npix']
        v = 0.5*cutTuple['var']
        # book and fill
        cov = np.ndarray((len(muVals), np.max(i)+1, np.max(j)+1))
        var = np.zeros_like(cov)
        cov[ind, i, j] = c
        var[ind, i, j] = v**2/n
        var[:, 0, 0] *= 2  # var(v) = 2*v**2/N

        return cov, var, muVals

    def measureMeanVarCov(self, im1Area, im2Area, imStatsCtrl, mu1, mu2):
        """Calculate the mean of each of two exposures and the variance
        and covariance of their difference. The variance is calculated
        via afwMath, and the covariance via the methods in Astier+19
        (appendix A). In theory, var = covariance[0,0]. This should
        be validated, and in the future, we may decide to just keep
        one (covariance).

        Parameters
        ----------
        im1Area : `lsst.afw.image.maskedImage.MaskedImageF`
            Masked image from exposure 1.
        im2Area : `lsst.afw.image.maskedImage.MaskedImageF`
            Masked image from exposure 2.
        imStatsCtrl : `lsst.afw.math.StatisticsControl`
            Statistics control object.
        mu1: `float`
            Clipped mean of im1Area (ADU).
        mu2: `float`
            Clipped mean of im2Area (ADU).

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

        # Variance calculation via afwMath
        varDiff = 0.5*(afwMath.makeStatistics(diffIm, afwMath.VARIANCECLIP, imStatsCtrl).getValue())

        # Covariances calculations
        # Get the pixels that were not clipped
        varClip = afwMath.makeStatistics(diffIm, afwMath.VARIANCECLIP, imStatsCtrl).getValue()
        meanClip = afwMath.makeStatistics(diffIm, afwMath.MEANCLIP, imStatsCtrl).getValue()
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

        # Calculate covariances via FFT.
        shapeDiff = np.array(diffIm.image.array.shape)
        # Calculate the sizes of FFT dimensions.
        s = shapeDiff + maxRangeCov
        tempSize = np.array(np.log(s)/np.log(2.)).astype(int)
        fftSize = np.array(2**(tempSize+1)).astype(int)
        fftShape = (fftSize[0], fftSize[1])
        c = CovFastFourierTransform(diffIm.image.array, w, fftShape, maxRangeCov)
        # np.sum(w) is the same as npix[0][0] returned in covDiffAstier
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

    def getImageAreasMasksStats(self, exposure1, exposure2, region=None):
        """Get image areas in a region as well as masks and statistic objects.

        Parameters
        ----------
        exposure1 : `lsst.afw.image.exposure.ExposureF`
            First exposure of flat field pair.
        exposure2 : `lsst.afw.image.exposure.ExposureF`
            Second exposure of flat field pair.
        region : `lsst.geom.Box2I`, optional
            Region of each exposure where to perform the calculations
            (e.g, an amplifier).

        Returns
        -------
        im1Area : `lsst.afw.image.maskedImage.MaskedImageF`
            Masked image from exposure 1.
        im2Area : `lsst.afw.image.maskedImage.MaskedImageF`
            Masked image from exposure 2.
        imStatsCtrl : `lsst.afw.math.StatisticsControl`
            Statistics control object.
        mu1: `float`
            Clipped mean of im1Area (ADU).
        mu2: `float`
            Clipped mean of im2Area (ADU).
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

        # Get mask planes and construct statistics control object from one
        # of the exposures
        imMaskVal = exposure1.getMask().getPlaneBitMask(self.config.maskNameList)
        imStatsCtrl = afwMath.StatisticsControl(self.config.nSigmaClipPtc,
                                                self.config.nIterSigmaClipPtc,
                                                imMaskVal)
        imStatsCtrl.setNanSafe(True)
        imStatsCtrl.setAndMask(imMaskVal)

        mu1 = afwMath.makeStatistics(im1Area, afwMath.MEANCLIP, imStatsCtrl).getValue()
        mu2 = afwMath.makeStatistics(im2Area, afwMath.MEANCLIP, imStatsCtrl).getValue()

        return (im1Area, im2Area, imStatsCtrl, mu1, mu2)

    def getGainFromFlatPair(self, im1Area, im2Area, imStatsCtrl, mu1, mu2,
                            correctionType='NONE', readNoise=None):
        """Estimate the gain from a single pair of flats.

        The basic premise is 1/g = <(I1 - I2)^2/(I1 + I2)> = 1/const,
        where I1 and I2 correspond to flats 1 and 2, respectively.
        Corrections for the variable QE and the read-noise are then
        made following the derivation in Robert Lupton's forthcoming
        book, which gets

        1/g = <(I1 - I2)^2/(I1 + I2)> - 1/mu(sigma^2 - 1/2g^2).

        This is a quadratic equation, whose solutions are given by:

        g = mu +/- sqrt(2*sigma^2 - 2*const*mu + mu^2)/(2*const*mu*2
            - 2*sigma^2)

        where 'mu' is the average signal level and 'sigma' is the
        amplifier's readnoise. The positive solution will be used.
        The way the correction is applied depends on the value
        supplied for correctionType.

        correctionType is one of ['NONE', 'SIMPLE' or 'FULL']
            'NONE' : uses the 1/g = <(I1 - I2)^2/(I1 + I2)> formula.
            'SIMPLE' : uses the gain from the 'NONE' method for the
                       1/2g^2 term.
            'FULL'   : solves the full equation for g, discarding the
                       non-physical solution to the resulting quadratic.

        Parameters
        ----------
        im1Area : `lsst.afw.image.maskedImage.MaskedImageF`
            Masked image from exposure 1.
        im2Area : `lsst.afw.image.maskedImage.MaskedImageF`
            Masked image from exposure 2.
        imStatsCtrl : `lsst.afw.math.StatisticsControl`
            Statistics control object.
        mu1: `float`
            Clipped mean of im1Area (ADU).
        mu2: `float`
            Clipped mean of im2Area (ADU).
        correctionType : `str`, optional
            The correction applied, one of ['NONE', 'SIMPLE', 'FULL']
        readNoise : `float`, optional
            Amplifier readout noise (ADU).

        Returns
        -------
        gain : `float`
            Gain, in e/ADU.

        Raises
        ------
        RuntimeError
            Raise if `correctionType` is not one of 'NONE',
            'SIMPLE', or 'FULL'.
        """
        if correctionType not in ['NONE', 'SIMPLE', 'FULL']:
            raise RuntimeError("Unknown correction type: %s" % correctionType)

        if correctionType != 'NONE' and not np.isfinite(readNoise):
            self.log.warning("'correctionType' in 'getGainFromFlatPair' is %s, "
                             "but 'readNoise' is NaN. Setting 'correctionType' "
                             "to 'NONE', so a gain value will be estimated without "
                             "corrections." % correctionType)
            correctionType = 'NONE'

        mu = 0.5*(mu1 + mu2)

        # ratioIm = (I1 - I2)^2 / (I1 + I2)
        temp = im2Area.clone()
        ratioIm = im1Area.clone()
        ratioIm -= temp
        ratioIm *= ratioIm

        # Sum of pairs
        sumIm = im1Area.clone()
        sumIm += temp

        ratioIm /= sumIm

        const = afwMath.makeStatistics(ratioIm, afwMath.MEAN, imStatsCtrl).getValue()
        gain = 1. / const

        if correctionType == 'SIMPLE':
            gain = 1/(const - (1/mu)*(readNoise**2 - (1/2*gain**2)))
        elif correctionType == 'FULL':
            root = np.sqrt(mu**2 - 2*mu*const + 2*readNoise**2)
            denom = (2*const*mu - 2*readNoise**2)
            positiveSolution = (root + mu)/denom
            gain = positiveSolution

        return gain

    def getReadNoiseFromMetadata(self, taskMetadata, ampName):
        """Gets readout noise for an amp from ISR metadata.

        Parameters
        ----------
        taskMetadata : `list` [`lsst.pipe.base.TaskMetadata`]
                    List of exposures metadata from ISR.
        ampName : `str`
            Amplifier name.

        Returns
        -------
        readNoise : `float`
            Median of the overscan readnoise in the
            post-ISR metadata of the input exposures (ADU).
            Returns 'None' if the median could not be calculated.
        """
        # Empirical readout noise [ADU] measured from an
        # overscan-subtracted overscan during ISR.
        expectedKey = f"RESIDUAL STDEV {ampName}"

        readNoises = []
        for expMetadata in taskMetadata:
            if 'isr' in expMetadata:
                overscanNoise = expMetadata['isr'][expectedKey]
            else:
                continue
            readNoises.append(overscanNoise)

        if len(readNoises):
            readNoise = np.median(np.array(readNoises))
        else:
            self.log.warning("Median readout noise from ISR metadata for amp %s "
                             "could not be calculated." % ampName)
            readNoise = np.nan

        return readNoise
