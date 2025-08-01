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
from collections import defaultdict
import logging
import numpy as np
from lmfit.models import GaussianModel
import scipy.stats
import warnings

import lsst.afw.math as afwMath
import lsst.afw.cameraGeom
import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase
from lsst.geom import (Box2I, Point2I, Extent2I)
from lsst.cp.pipe.utils import (arrangeFlatsByExpTime, arrangeFlatsByExpId,
                                arrangeFlatsByExpFlux, sigmaClipCorrection,
                                CovFastFourierTransform, getReadNoise)
from lsst.cp.pipe.utilsEfd import CpEfdClient

import lsst.pipe.base.connectionTypes as cT

from lsst.ip.isr import PhotonTransferCurveDataset
from lsst.ip.isr import IsrTask

__all__ = [
    'PhotonTransferCurveExtractConfig',
    'PhotonTransferCurveExtractTask',
    'PhotonTransferCurveExtractPairConfig',
    'PhotonTransferCurveExtractPairTask',
]


class PhotonTransferCurveExtractConnectionsBase(pipeBase.PipelineTaskConnections,
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
    inputPhotodiodeData = cT.Input(
        name="photodiode",
        doc="Photodiode readings data.",
        storageClass="IsrCalib",
        dimensions=("instrument", "exposure"),
        multiple=True,
        deferLoad=True,
    )

    def __init__(self, *, config=None):
        if not config.doExtractPhotodiodeData:
            del self.inputPhotodiodeData


class PhotonTransferCurveExtractConnections(
    PhotonTransferCurveExtractConnectionsBase,
    dimensions=("instrument", "detector"),
):
    outputCovariances = cT.Output(
        name="ptcCovariances",
        doc="Extracted flat (co)variances.",
        storageClass="PhotonTransferCurveDataset",
        dimensions=("instrument", "exposure", "detector"),
        isCalibration=True,
        multiple=True,
    )


class PhotonTransferCurveExtractPairConnections(
    PhotonTransferCurveExtractConnectionsBase,
    dimensions=("instrument", "detector", "exposure"),
):
    outputCovariance = cT.Output(
        name="ptcCovariance",
        doc="Extracted flat (co)variance.",
        storageClass="PhotonTransferCurveDataset",
        dimensions=("instrument", "exposure", "detector"),
        isCalibration=True,
    )

    def adjust_all_quanta(self, adjuster):
        _LOG = logging.getLogger(__name__)

        def _removeExposure(exposure, pop=False, msg=None):
            """Remove an exposure from the quanta.

            Parameters
            ----------
            exposure : `int`
                Exposure key to quantumIdDict.
            pop : `bool`, optional
                Pop the exposure from the dict? Otherwise change
                to an empty dict.
            msg : `str`, optional
                Warning string to log with exposure number.
            """
            if msg is not None:
                _LOG.warning("Exposure %d %s; dropping.", exposure, msg)

            # Remove all quanta associated with this exposure.
            for key, element in quantumIdDict[exposure].items():
                adjuster.remove_quantum(element)
            if pop:
                # Pop it completely from the dict.
                quantumIdDict.pop(exposure)
            else:
                # Set to empty dict.
                quantumIdDict[exposure] = {}

        def _addInputs(sourceExposure, targetExposure, remove=False):
            """Add inputs to a quantum.

            Parameters
            ----------
            sourceExposure : `int`
                Take the quanta from this source exposure.
            targetExposure : `int`
                And add them to the quanta for this target exposure.
            remove : `bool`, optional
                Remove the sourceExposure from the quantum dict?
            """
            for detector in quantumIdDict[sourceExposure].keys():
                inputs = adjuster.get_inputs(quantumIdDict[sourceExposure][detector])
                if detector in quantumIdDict[targetExposure]:
                    adjuster.add_input(
                        quantumIdDict[targetExposure][detector],
                        "inputExp",
                        inputs["inputExp"][0],
                    )
                    if self.config.doExtractPhotodiodeData:
                        adjuster.add_input(
                            quantumIdDict[targetExposure][detector],
                            "inputPhotodiodeData",
                            inputs["inputPhotodiodeData"][0],
                        )
                else:
                    _LOG.warning("Exposure %d does not include detector %d", targetExposure, detector)
                if remove:
                    adjuster.remove_quantum(quantumIdDict[sourceExposure][detector])

        # Build a dict keyed by exposure.
        # Each entry is a dict of {detector: quantumId}
        # And everything will be sorted by exposure and detector.
        quantumIdDict = defaultdict(dict)
        for quantumId in sorted(adjuster.iter_data_ids(), key=lambda d: (d["exposure"], d["detector"])):
            exposure = quantumId["exposure"]
            quantumIdDict[exposure][quantumId["detector"]] = quantumId

        # Ensure we are only using "flat" image types.
        for exposure in list(quantumIdDict.keys()):
            # We only need to check the first detector in each exposure list.
            firstDet = next(iter(quantumIdDict[exposure].keys()))
            obsType = adjuster.expand_quantum_data_id(
                quantumIdDict[exposure][firstDet],
            ).exposure.observation_type
            if obsType != "flat":
                _removeExposure(exposure, pop=True, msg="is not of 'flat' type")

        if self.config.matchExposuresType == "TIME":
            # Extract the exposure times.

            exposureTimes = []
            for exposure in quantumIdDict.keys():
                firstDet = list(quantumIdDict[exposure].keys())[0]
                exposureTimes.append(
                    adjuster.expand_quantum_data_id(quantumIdDict[exposure][firstDet]).exposure.exposure_time,
                )

            previousExposure = -1
            previousExposureTime = -1.0
            nInPair = 1
            for i, exposure in enumerate(quantumIdDict.keys()):
                if np.isclose(exposureTimes[i], previousExposureTime, atol=5e-3):
                    # This is a match.
                    if nInPair == 2:
                        # We already have a pair!
                        _removeExposure(exposure, msg="already has two exptime matches")
                    else:
                        # Add the inputs from the current exposure quantum
                        # to the previous one, and remove the current one
                        # from the set to run.
                        _addInputs(exposure, previousExposure, remove=True)
                        nInPair += 1
                else:
                    nInPair = 1
                    previousExposureTime = exposureTimes[i]
                    previousExposure = exposure

        elif self.config.matchExposuresType == "EXPID":
            # Pair by ExpId sequence.
            exposures = list(quantumIdDict.keys())
            for i in range(len(exposures)):
                if (i % 2) == 1:
                    # This is the second of a pair.
                    _addInputs(exposures[i], exposures[i - 1], remove=True)

        elif self.config.matchExposuresType == "FLUX":
            # When matching by flux keyword, we do not yet have access
            # to the header information to uniquely pair the exposures.
            # We do know that exposure pairs will be back-to-back, so
            # we set up triplets and do the final sorting in runQuantum.

            # For each triplet, we take the index of the sorted exposures,
            # and set it up with the previous and next exposure.
            # In runQuantum we will then look at the headers and match.
            # If all 3 have the same keyword, we are confused and we warn
            # and raise NoWorkFound.
            # If all 3 have different keywords, we do not have a pair and
            # we warn and raise NoWorkFound.
            # If there is a matched pair, if the first in the pair has the
            # same exposure number as the quantum id, it is "primary" and
            # we continue.
            # If there is a matched pair, if the second in the pair has the
            # same exposure number as the quantum id, it is "secondary" and
            # we raise NoWorkFound.

            # For example, the first column is the quantum id and the next
            # is the triplet (or double at the boundaries).
            #  0: 0, 1  - this has a match (0 == 1), 0 == 0 do work
            #  1: 0, 1, 2  - this has a match (0 == 1), 0 != 1 -> no work
            #  2: 1, 2, 3  - this has a match (2 == 3), 2 == 2 -> do work
            #  3: 2, 3, 4  - this has a match (2 == 3), 2 != 3 -> no work
            #  4: 3, 4, 5  - this has a match (4 == 5), 4 == 4 -> do work
            #  5: 4, 5  - this has a match (4 == 5), 4 != 5 -> no work

            exposures = list(quantumIdDict.keys())
            for i in range(len(exposures)):
                if (i - 1) >= 0:
                    _addInputs(exposures[i - 1], exposures[i], remove=False)
                if (i + 1) < len(exposures):
                    _addInputs(exposures[i + 1], exposures[i], remove=False)


class PhotonTransferCurveExtractConfigBase(
    pipeBase.PipelineTaskConfig,
    pipelineConnections=PhotonTransferCurveExtractConnectionsBase
):
    """Configuration for the measurement of covariances from flats.
    """
    matchExposuresType = pexConfig.ChoiceField(
        dtype=str,
        doc="Match input exposures by time, flux, or expId",
        default='TIME',
        allowed={
            "TIME": "Match exposures by reported exposure time. Entries "
                    "that are within 5e-3 seconds.",
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
    doKsHistMeasurement = pexConfig.Field(
        dtype=bool,
        doc="Do the Ks test of gaussianity? If False, kspValue will be filled "
            "with all 1.0s.",
        default=False,
    )
    ksHistNBins = pexConfig.Field(
        dtype=int,
        doc="Number of bins for the KS test histogram.",
        default=100,
    )
    ksHistLimitMultiplier = pexConfig.Field(
        dtype=float,
        doc="Number of sigma (as predicted from the mean value) to compute KS test histogram.",
        default=8.0,
    )
    ksHistMinDataValues = pexConfig.Field(
        dtype=int,
        doc="Minimum number of good data values to compute KS test histogram.",
        default=100,
    )
    auxiliaryHeaderKeys = pexConfig.ListField(
        dtype=str,
        doc="Auxiliary header keys to store with the PTC dataset.",
        default=[],
    )
    doExtractPhotodiodeData = pexConfig.Field(
        dtype=bool,
        doc="Extract photodiode data?",
        default=False,
    )
    useEfdPhotodiodeData = pexConfig.Field(
        dtype=bool,
        doc="Extract photodiode values from EFD?",
        default=False,
    )
    efdPhotodiodeSeries = pexConfig.Field(
        dtype=str,
        doc="EFD series to use to get photodiode values.",
        default="lsst.sal.Electrometer.logevent_logMessage",
    )
    efdSalIndex = pexConfig.Field(
        dtype=int,
        doc="EFD SAL Index to select electrometer. This is 101 for mainTel, 201 for auxTel.",
        default=201,
    )

    photodiodeIntegrationMethod = pexConfig.ChoiceField(
        dtype=str,
        doc="Integration method for photodiode monitoring data.",
        default="TRIMMED_SUM",
        allowed={
            "DIRECT_SUM": ("Use numpy's trapezoid integrator on all photodiode "
                           "readout entries"),
            "TRIMMED_SUM": ("Use numpy's trapezoid integrator, clipping the "
                            "leading and trailing entries, which are "
                            "nominally at zero baseline level."),
            "CHARGE_SUM": ("Treat the current values as integrated charge "
                           "over the sampling interval and simply sum "
                           "the values, after subtracting a baseline level."),
            "MEAN": {"Take the average of the photodiode measurements and "
                     "multiply by the exposure time."},
        },
    )
    photodiodeCurrentScale = pexConfig.Field(
        dtype=float,
        doc="Scale factor to apply to photodiode current values for the "
            "``CHARGE_SUM``, ``TRIMMED_SUM``, and ``MEAN`` integration methods.",
        default=-1.0,
    )

    doVignetteFunctionRegionSelection = pexConfig.Field(
        dtype=bool,
        doc="Use vignette function to select PTC region?",
        default=False,
    )
    vignetteFunctionPolynomialCoeffs = pexConfig.ListField(
        dtype=float,
        doc="Polynomial terms for radial vignetting function. This is used with "
            "vig = np.polyval(coeffs, radius), where radius is in meters. This must be "
            "set if doVignetteFunctionRegionSelection is True.",
        default=None,
        optional=True,
    )
    vignetteFunctionRegionSelectionMinimumPixels = pexConfig.Field(
        dtype=int,
        doc="Minumum number of pixels to select for vignette function region selection.",
        default=250_000,
    )
    vignetteFunctionRegionSelectionPercent = pexConfig.Field(
        dtype=float,
        doc="Vignetting function variation over the focal plane that is determined to be "
            "satisfactory. If this does not yield vignetteFunctionRegionSelectionMinimumPixels "
            "then this will be increased until vignetteFunctionRegionSelectionMinimumPixels is "
            "reached.",
        default=2.0,
    )

    def validate(self):
        super().validate()
        if self.doExtractPhotodiodeData and self.useEfdPhotodiodeData:
            # These get information from different places, so let's
            # disallow both being true.
            raise ValueError("doExtractPhotodiodeData and useEfdPhotodiodeData are mutually exclusive.")

        if self.doVignetteFunctionRegionSelection and self.vignetteFunctionPolynomialCoeffs is None:
            raise ValueError(
                "vignetteFunctionPolynomialTerms must be set if "
                "doVignetteFunctionRegionSelection is True.",
            )


class PhotonTransferCurveExtractConfig(
    PhotonTransferCurveExtractConfigBase,
    pipelineConnections=PhotonTransferCurveExtractConnections,
):
    pass


class PhotonTransferCurveExtractPairConfig(
    PhotonTransferCurveExtractConfigBase,
    pipelineConnections=PhotonTransferCurveExtractPairConnections,
):
    pass


class PhotonTransferCurveExtractTaskBase(pipeBase.PipelineTask):
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

    ConfigClass = PhotonTransferCurveExtractConfigBase
    _DefaultName = 'cpPtcExtractBase'

    def _extractPhotoChargeDict(self, inputPhotodiodeData, inputExp):
        """Extract the photodiode charge.

        Parameters
        ----------
        inputPhotoDiodeData : `dict` [`str`,
            `lsst.pipe.base.connections.DeferredDatasetRef`]
            Photodiode readings data (optional).
        inputExp : `dict` [`float`, `list`
                          [`~lsst.pipe.base.connections.DeferredDatasetRef`]]
            Dictionary that groups references to flat-field exposures.

        Returns
        -------
        photoChargeDict : `dict` [`int`, `float`]
            Dict of photocharge values, keyed by expId. May be empty.
        """
        # Extract the photodiode data if requested.
        photoChargeDict = {}
        if self.config.doExtractPhotodiodeData:
            # Compute the photodiode integrals once, at the start.
            for pdHandle, expHandle in zip(inputPhotodiodeData, inputExp):
                expId = pdHandle.dataId['exposure']
                pdCalib = pdHandle.get()
                pdCalib.integrationMethod = self.config.photodiodeIntegrationMethod
                pdCalib.currentScale = self.config.photodiodeCurrentScale
                visitInfo = expHandle.get(component="visitInfo")
                exposureTime = visitInfo.getExposureTime()
                photoChargeDict[expId] = pdCalib.integrate(exposureTime=exposureTime)
        elif self.config.useEfdPhotodiodeData:
            client = CpEfdClient()
            obsDates = {}
            obsMin = None
            obsMax = None
            for _, expList in inputExp.items():
                for ref, refId in expList:
                    ts = ref.dataId.exposure.timespan
                    dt = ts.end - ts.begin
                    obsDates[refId] = {
                        'mid': ts.begin + dt / 2.0,
                        'dt': dt,
                        'begin': ts.begin,
                        'end': ts.end,
                        'exptime': ref.dataId.exposure.exposure_time,
                    }
                    if not obsMin or ts.begin < obsMin:
                        obsMin = ts.begin - dt
                    if not obsMax or ts.end > obsMax:
                        obsMax = ts.end + dt

            pdData = client.getEfdElectrometerData(dataSeries=self.config.efdPhotodiodeSeries,
                                                   dateMin=obsMin.isot,
                                                   dateMax=obsMax.isot)
            for expId, expDate in obsDates.items():
                # This returns matchDate, intensity, and optional
                # endDate.  The matchDate is the most recent date
                # prior to that being searched.
                results = client.parseElectrometerStatus(pdData,
                                                         expDate['end'],
                                                         index=self.config.efdSalIndex)
                # The monitor diode charge is negative and a time average,
                # so we need to flip the sign and multiply by the exposure
                # time.
                photoChargeDict[expId] = -1.0 * results[1] * expDate['exptime']

            # The data can be large, so:
            del pdData
            del client

        return photoChargeDict

    def _extractOneFlatPair(
        self,
        expIds,
        exps,
        photoChargeDict,
        expTime,
    ):
        """Extract a single flat pair.

        Parameters
        ----------
        expIds : `tuple` [`int`]
            Tuple of two exposure IDs for the flat pair.
        exps : `tuple` [`lsst.afw.image.Exposure`]
            Tuple of two exposures for the flat pair.
        photoChargeDict : `dict` [`int`, `float`]
            Dict of photocharge values, keyed by expId. May be empty.
        expTime : `float`
            Exposure time.

        Returns
        -------
        partialPtcDataset : `lsst.ip.isr.PhotonTransferCurveDataset`
            PTC dataset for this flat pair.
        nAmpsNan : `int`
            Number of amps that have no measurements.
        """
        expId1, expId2 = expIds
        exp1, exp2 = exps

        expMjd1 = exp1.metadata['MJD']
        expMjd2 = exp2.metadata['MJD']
        inputExpPairMjdStart = min([expMjd1, expMjd2])

        self.log.info("Extracting PTC data from flat pair %d, %d", expId1, expId2)

        # These are the column names for `tupleRows` below.
        tags = [('mu', '<f8'), ('afwVar', '<f8'), ('i', '<i8'), ('j', '<i8'), ('var', '<f8'),
                ('cov', '<f8'), ('npix', '<i8'), ('ext', '<i8'), ('expTime', '<f8'), ('ampName', '<U3')]

        detector = exp1.getDetector()
        detNum = detector.getId()
        amps = detector.getAmplifiers()
        ampNames = [amp.getName() for amp in amps]

        # Mask pixels at the edge of the detector or of each amp
        if self.config.numEdgeSuspect > 0:
            self.isrTask.maskEdges(exp1, numEdgePixels=self.config.numEdgeSuspect,
                                   maskPlane="SUSPECT", level=self.config.edgeMaskLevel)
            self.isrTask.maskEdges(exp2, numEdgePixels=self.config.numEdgeSuspect,
                                   maskPlane="SUSPECT", level=self.config.edgeMaskLevel)

        # Mask "vignetted" subregions.
        if self.config.doVignetteFunctionRegionSelection:
            self._maskVignetteFunctionRegion([exp1, exp2])

        # Extract any metadata keys from the headers.
        auxDict = {}
        metadata = exp1.getMetadata()
        for key in self.config.auxiliaryHeaderKeys:
            if key not in metadata:
                self.log.warning(
                    "Requested auxiliary keyword %s not found in exposure metadata for %d",
                    key,
                    expId1,
                )
                value = np.nan
            else:
                value = metadata[key]

            auxDict[key] = value

        # Pull key visitInfo data into the auxDict.
        visitInfo = exp1.info.getVisitInfo()
        auxDict["observationType"] = visitInfo.getObservationType()
        auxDict["observationReason"] = visitInfo.getObservationReason()
        auxDict["scienceProgram"] = visitInfo.getScienceProgram()

        nAmpsNan = 0
        partialPtcDataset = PhotonTransferCurveDataset(
            ampNames, 'PARTIAL',
            covMatrixSide=self.config.maximumRangeCovariancesAstier)

        # These should be identical, but we can use them both.
        maskValue1 = exp1.mask.getPlaneBitMask(self.config.maskNameList)
        maskValue2 = exp2.mask.getPlaneBitMask(self.config.maskNameList)

        # Get the following statistics for each amp
        for ampNumber, amp in enumerate(detector):
            ampName = amp.getName()
            if self.config.detectorMeasurementRegion == 'AMP':
                region = amp.getBBox()
            elif self.config.detectorMeasurementRegion == 'FULL':
                region = None

            # Get masked image regions, masking planes, statistic control
            # objects, and clipped means. Calculate once to reuse in
            # `measureMeanVarCov` and `getGainFromFlatPair`.
            im1Area, im2Area, imStatsCtrl, mu1, mu2 = self.getImageAreasMasksStats(exp1, exp2,
                                                                                   region=region)

            nPixelCovariance1 = ((im1Area.mask.array & maskValue1) == 0).sum()
            nPixelCovariance2 = ((im2Area.mask.array & maskValue2) == 0).sum()

            if nPixelCovariance1 != nPixelCovariance2:
                raise RuntimeError("Inconsistent selections in pair of exposures.")

            readNoise1 = getReadNoise(exp1, ampName)
            readNoise2 = getReadNoise(exp2, ampName)
            if not np.isfinite(readNoise1) and not np.isfinite(readNoise2):
                # We allow the mean read noise to be nan if both read noise
                # values are nan, so suppress this warning (e.g. bad amps)
                meanReadNoise = np.nan
            else:
                meanReadNoise = np.nanmean([readNoise1, readNoise2])

            overscanMedianLevel1 = exp1.metadata[f'LSST ISR OVERSCAN SERIAL MEDIAN {ampName}']
            overscanMedianLevel2 = exp2.metadata[f'LSST ISR OVERSCAN SERIAL MEDIAN {ampName}']
            if not np.isfinite(overscanMedianLevel1) and \
               not np.isfinite(overscanMedianLevel2):
                # We allow the mean overscan median level to be nan if both
                # statistics are nan, so suppress this warning (e.g. bad amps)
                overscanMedianLevel = np.nan
            else:
                overscanMedianLevel = np.nanmean([overscanMedianLevel1, overscanMedianLevel2])

            # We demand that both mu1 and mu2 be finite and greater than 0.
            if not np.isfinite(mu1) or not np.isfinite(mu2) \
               or ((np.nan_to_num(mu1) + np.nan_to_num(mu2)/2.) <= 0.0):
                self.log.warning(
                    "Illegal mean value(s) detected for amp %s on exposure pair %d/%d",
                    ampName,
                    expId1,
                    expId2,
                )
                partialPtcDataset.setAmpValuesPartialDataset(
                    ampName,
                    inputExpIdPair=(expId1, expId2),
                    rawExpTime=expTime,
                    expIdMask=False,
                    nPixelCovariance=nPixelCovariance1,
                )
                continue

            # `measureMeanVarCov` is the function that measures
            # the variance and covariances from a region of
            # the difference image of two flats at the same
            # exposure time.  The variable `covAstier` that is
            # returned is of the form:
            # [(i, j, var (cov[0,0]), cov, npix) for (i,j) in
            # {maxLag, maxLag}^2].
            muDiff, varDiff, covAstier, rowMeanVariance = self.measureMeanVarCov(im1Area,
                                                                                 im2Area,
                                                                                 imStatsCtrl,
                                                                                 mu1,
                                                                                 mu2)
            # Estimate the gain from the flat pair
            if self.config.doGain:
                gain = self.getGainFromFlatPair(im1Area, im2Area, imStatsCtrl, mu1, mu2,
                                                correctionType=self.config.gainCorrectionType,
                                                readNoise=meanReadNoise)
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
            if (np.isnan(muDiff) or np.isnan(varDiff) or (covAstier is None)
               or (rowMeanVariance is np.nan)):
                self.log.warning("NaN mean, var or rowmeanVariance, or None cov in amp %s "
                                 "in exposure pair %d, %d of "
                                 "detector %d.", ampName, expId1, expId2, detNum)
                nAmpsNan += 1
                expIdMask = False
                covArray = np.full((1, self.config.maximumRangeCovariancesAstier,
                                    self.config.maximumRangeCovariancesAstier), np.nan)
                covSqrtWeights = np.full_like(covArray, np.nan)

            if covAstier is not None:
                # Turn the tuples with the measured information
                # into covariance arrays.
                # covrow: (i, j, var (cov[0,0]), cov, npix)
                tupleRows = [(muDiff, varDiff) + covRow + (ampNumber, expTime,
                                                           ampName) for covRow in covAstier]
                tempStructArray = np.array(tupleRows, dtype=tags)

                covArray, vcov, _ = self.makeCovArray(tempStructArray,
                                                      self.config.maximumRangeCovariancesAstier)

                # The returned covArray should only have 1 entry;
                # raise if this is not the case.
                if covArray.shape[0] != 1:
                    raise RuntimeError("Serious programming error in covArray shape.")

                covSqrtWeights = np.nan_to_num(1./np.sqrt(vcov))

            # Correct covArray for sigma clipping:
            # 1) Apply varFactor twice for the whole covariance matrix
            covArray *= varFactor**2
            # 2) But, only once for the variance element of the
            # matrix, covArray[0, 0, 0] (so divide one factor out).
            # (the first 0 is because this is a 3D array for insertion into
            # the combined dataset).
            covArray[0, 0, 0] /= varFactor

            if expIdMask and self.config.doKsHistMeasurement:
                # Run the Gaussian histogram only if this is a legal
                # amplifier and configured to do so.
                histVar, histChi2Dof, kspValue = self.computeGaussianHistogramParameters(
                    im1Area,
                    im2Area,
                    imStatsCtrl,
                    mu1,
                    mu2,
                )
            else:
                histVar = np.nan
                histChi2Dof = np.nan
                if self.config.doKsHistMeasurement:
                    kspValue = 0.0
                else:
                    # When this is turned off, we will always allow it to
                    # pass downstream.
                    kspValue = 1.0

            if self.config.doExtractPhotodiodeData or self.config.useEfdPhotodiodeData:
                nExps = 0
                photoCharge = 0.0
                for expId in [expId1, expId2]:
                    if expId in photoChargeDict:
                        photoCharge += photoChargeDict[expId]
                        nExps += 1
                if nExps > 0:
                    photoCharge /= nExps
                else:
                    photoCharge = np.nan
            else:
                photoCharge = np.nan

            # Get amp offsets if available.
            ampOffsetKey = f"LSST ISR AMPOFFSET PEDESTAL {ampName}"
            ampOffset1 = exp1.metadata.get(ampOffsetKey, np.nan)
            ampOffset2 = exp2.metadata.get(ampOffsetKey, np.nan)
            ampOffset = (ampOffset1 + ampOffset2) / 2.0

            partialPtcDataset.setAmpValuesPartialDataset(
                ampName,
                inputExpIdPair=(expId1, expId2),
                inputExpPairMjdStart=inputExpPairMjdStart,
                rawExpTime=expTime,
                rawMean=muDiff,
                rawVar=varDiff,
                photoCharge=photoCharge,
                ampOffset=ampOffset,
                expIdMask=expIdMask,
                nPixelCovariance=nPixelCovariance1,
                covariance=covArray[0, :, :],
                covSqrtWeights=covSqrtWeights[0, :, :],
                gain=gain,
                noise=meanReadNoise,
                overscanMedianLevel=overscanMedianLevel,
                histVar=histVar,
                histChi2Dof=histChi2Dof,
                kspValue=kspValue,
                rowMeanVariance=rowMeanVariance,
            )

        partialPtcDataset.setAuxValuesPartialDataset(auxDict)

        return partialPtcDataset, nAmpsNan

    def _maskVignetteFunctionRegion(self, exposures, maskPlane="SUSPECT"):
        """Mask regions with strong vignetting.

        Parameters
        ----------
        exposures : `list` [`lsst.afw.image.Exposure`]
            Exposure to mask.
        maskPlane : `str`, optional
            Mask plane to set.
        """
        detector = exposures[0].getDetector()
        coeffs = np.asarray(self.config.vignetteFunctionPolynomialCoeffs)

        transform = detector.getTransform(
            fromSys=lsst.afw.cameraGeom.PIXELS,
            toSys=lsst.afw.cameraGeom.FOCAL_PLANE,
        )
        nx = detector.getBBox().getWidth()
        ny = detector.getBBox().getHeight()

        x = np.repeat(np.arange(nx, dtype=np.float64), ny)
        y = np.tile(np.arange(ny, dtype=np.float64), nx)
        xy = np.vstack((x, y))
        x2, y2 = np.vsplit(transform.getMapping().applyForward(xy), 2)
        # Convert from mm to meters for the coefficients.
        fpRad = np.sqrt((x2.ravel()/1000.)**2. + (y2.ravel()/1000.)**2.)

        vigImage = lsst.afw.image.ImageF(detector.getBBox())
        vigImage.array[:, :] = np.polyval(coeffs, fpRad).reshape(nx, ny).T

        bitmask = exposures[0].mask.getPlaneBitMask(maskPlane)

        for amp in detector:
            vigSubImage = vigImage[amp.getBBox()].array.copy()

            nSelected = 0
            subMax = vigSubImage.max()
            sel = self.config.vignetteFunctionRegionSelectionPercent
            while (nSelected < self.config.vignetteFunctionRegionSelectionMinimumPixels) and sel < 100.0:
                threshold = subMax * (1.0 - sel / 100.0)
                high = (vigSubImage > threshold)
                nSelected = high.sum()

                # Raise by a percent at a time.
                sel += 1.0

            if sel > (self.config.vignetteFunctionRegionSelectionPercent + 1.0):
                self.log.warning(
                    "Amplifier %s_%s required %.1f%% threshold to reach %d pixel threshold.",
                    detector.getName(),
                    amp.getName(),
                    sel - 1.0,
                    self.config.vignetteFunctionRegionSelectionMinimumPixels,
                )

            for exposure in exposures:
                exposure[amp.getBBox()].mask.array[~high] |= bitmask

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
        mu : `float`
            0.5*(mu1 + mu2), where mu1, and mu2 are the clipped means
            of the regions in both exposures. If either mu1 or m2 are
            NaN's, the returned value is NaN.
        varDiff : `float`
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
        rowMeanVariance : `float`
            Variance of the means of each row in the difference image.
            Taken from `github.com/lsst-camera-dh/eo_pipe`.

            If either mu1 or m2 are NaN's, the returned value is NaN.
        """
        if np.isnan(mu1) or np.isnan(mu2):
            self.log.warning("Mean of amp in image 1 or 2 is NaN: %f, %f.", mu1, mu2)
            return np.nan, np.nan, None, np.nan
        mu = 0.5*(mu1 + mu2)

        # Take difference of pairs
        # symmetric formula: diff = (mu2*im1-mu1*im2)/(0.5*(mu1+mu2))
        temp = im2Area.clone()
        temp *= mu1
        diffIm = im1Area.clone()
        diffIm *= mu2
        diffIm -= temp
        diffIm /= mu

        if self.config.binSize > 1:
            diffIm = afwMath.binImage(diffIm, self.config.binSize)

        # Calculate the variance (in ADU^2) of the means of rows for diffIm.
        # Taken from eo_pipe
        region = diffIm.getBBox()
        rowMeans = []
        for row in range(region.minY, region.maxY):
            regionRow = Box2I(Point2I(region.minX, row),
                              Extent2I(region.width, 1))
            rowMeans.append(afwMath.makeStatistics(diffIm[regionRow],
                                                   afwMath.MEAN,
                                                   imStatsCtrl).getValue())
            rowMeanVariance = afwMath.makeStatistics(
                np.array(rowMeans), afwMath.VARIANCECLIP,
                imStatsCtrl).getValue()

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
        wDiff = np.where(diffIm.mask.array & imStatsCtrl.getAndMask() == 0, 1, 0)
        # Combine the two sets of pixels ('1': use; '0': don't use)
        # into a final weight matrix to be used in the covariance
        # calculations below.
        w = unmasked*wDiff

        if np.sum(w) < self.config.minNumberGoodPixelsForCovariance/(self.config.binSize**2):
            self.log.warning("Number of good points for covariance calculation (%s) is less "
                             "(than threshold %s)", np.sum(w),
                             self.config.minNumberGoodPixelsForCovariance/(self.config.binSize**2))
            return np.nan, np.nan, None, np.nan

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
        try:
            covDiffAstier = c.reportCovFastFourierTransform(maxRangeCov)
        except ValueError:
            # This is raised if there are not enough pixels.
            self.log.warning("Not enough pixels covering the requested covariance range in x/y (%d)",
                             self.config.maximumRangeCovariancesAstier)
            return np.nan, np.nan, None, np.nan

        # Compare Cov[0,0] and afwMath.VARIANCECLIP covDiffAstier[0]
        # is the Cov[0,0] element, [3] is the variance, and there's a
        # factor of 0.5 difference with afwMath.VARIANCECLIP.
        thresholdPercentage = self.config.thresholdDiffAfwVarVsCov00
        fractionalDiff = 100*np.fabs(1 - varDiff/(covDiffAstier[0][3]*0.5))
        if fractionalDiff >= thresholdPercentage:
            self.log.warning("Absolute fractional difference between afwMatch.VARIANCECLIP and Cov[0,0] "
                             "is more than %f%%: %f", thresholdPercentage, fractionalDiff)

        return mu, varDiff, covDiffAstier, rowMeanVariance

    def getImageAreasMasksStats(self, exposure1, exposure2, region=None):
        """Get image areas in a region as well as masks and statistic objects.

        Parameters
        ----------
        exposure1 : `lsst.afw.image.ExposureF`
            First exposure of flat field pair.
        exposure2 : `lsst.afw.image.ExposureF`
            Second exposure of flat field pair.
        region : `lsst.geom.Box2I`, optional
            Region of each exposure where to perform the calculations
            (e.g, an amplifier).

        Returns
        -------
        im1Area : `lsst.afw.image.MaskedImageF`
            Masked image from exposure 1.
        im2Area : `lsst.afw.image.MaskedImageF`
            Masked image from exposure 2.
        imStatsCtrl : `lsst.afw.math.StatisticsControl`
            Statistics control object.
        mu1 : `float`
            Clipped mean of im1Area (ADU).
        mu2 : `float`
            Clipped mean of im2Area (ADU).
        """
        if region is not None:
            im1Area = exposure1.maskedImage[region]
            im2Area = exposure2.maskedImage[region]
        else:
            im1Area = exposure1.maskedImage
            im2Area = exposure2.maskedImage

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

    def computeGaussianHistogramParameters(self, im1Area, im2Area, imStatsCtrl, mu1, mu2):
        """Compute KS test for a Gaussian model fit to a histogram of the
        difference image.

        Parameters
        ----------
        im1Area : `lsst.afw.image.MaskedImageF`
            Masked image from exposure 1.
        im2Area : `lsst.afw.image.MaskedImageF`
            Masked image from exposure 2.
        imStatsCtrl : `lsst.afw.math.StatisticsControl`
            Statistics control object.
        mu1 : `float`
            Clipped mean of im1Area (ADU).
        mu2 : `float`
            Clipped mean of im2Area (ADU).

        Returns
        -------
        varFit : `float`
            Variance from the Gaussian fit.
        chi2Dof : `float`
            Chi-squared per degree of freedom of Gaussian fit.
        kspValue : `float`
            The KS test p-value for the Gaussian fit.

        Notes
        -----
        The algorithm here was originally developed by Aaron Roodman.
        Tests on the full focal plane of LSSTCam during testing has shown
        that a KS test p-value cut of 0.01 is a good discriminant for
        well-behaved flat pairs (p>0.01) and poorly behaved non-Gaussian
        flat pairs (p<0.01).
        """
        diffExp = im1Area.clone()
        diffExp -= im2Area

        sel = (((diffExp.mask.array & imStatsCtrl.getAndMask()) == 0)
               & np.isfinite(diffExp.mask.array))
        diffArr = diffExp.image.array[sel]

        numOk = len(diffArr)

        if numOk >= self.config.ksHistMinDataValues and np.isfinite(mu1) and np.isfinite(mu2):
            # Create a histogram symmetric around zero, with a bin size
            # determined from the expected variance given by the average of
            # the input signal levels.
            lim = self.config.ksHistLimitMultiplier * np.sqrt((mu1 + mu2)/2.)
            yVals, binEdges = np.histogram(diffArr, bins=self.config.ksHistNBins, range=[-lim, lim])

            # Fit the histogram with a Gaussian model.
            model = GaussianModel()
            yVals = yVals.astype(np.float64)
            xVals = ((binEdges[0: -1] + binEdges[1:])/2.).astype(np.float64)
            errVals = np.sqrt(yVals)
            errVals[(errVals == 0.0)] = 1.0
            pars = model.guess(yVals, x=xVals)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                # The least-squares fitter sometimes spouts (spurious) warnings
                # when the model is very bad. Swallow these warnings now and
                # let the KS test check the model below.
                out = model.fit(
                    yVals,
                    pars,
                    x=xVals,
                    weights=1./errVals,
                    calc_covar=True,
                    method="least_squares",
                )

            # Calculate chi2.
            chiArr = out.residual
            nDof = len(yVals) - 3
            chi2Dof = np.sum(chiArr**2.)/nDof
            sigmaFit = out.params["sigma"].value

            # Calculate KS test p-value for the fit.
            ksResult = scipy.stats.ks_1samp(
                diffArr,
                scipy.stats.norm.cdf,
                (out.params["center"].value, sigmaFit),
            )

            kspValue = ksResult.pvalue
            if kspValue < 1e-15:
                kspValue = 0.0

            varFit = sigmaFit**2.

        else:
            varFit = np.nan
            chi2Dof = np.nan
            kspValue = 0.0

        return varFit, chi2Dof, kspValue


class PhotonTransferCurveExtractTask(PhotonTransferCurveExtractTaskBase):
    ConfigClass = PhotonTransferCurveExtractConfig
    _DefaultName = 'cpPtcExtract'

    def runQuantum(self, butlerQC, inputRefs, outputRefs):
        """Ensure that the input and output dimensions are passed along.

        Parameters
        ----------
        butlerQC : `~lsst.daf.butler.QuantumContext`
            Butler to operate on.
        inputRefs : `~lsst.pipe.base.InputQuantizedConnection`
            Input data refs to load.
        ouptutRefs : `~lsst.pipe.base.OutputQuantizedConnection`
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
            inputs['inputExp'] = arrangeFlatsByExpTime(inputs['inputExp'], inputs['inputDims'], log=self.log)
        elif matchType == 'FLUX':
            inputs['inputExp'] = arrangeFlatsByExpFlux(
                inputs['inputExp'],
                inputs['inputDims'],
                self.config.matchExposuresByFluxKeyword,
                log=self.log,
            )
        else:
            inputs['inputExp'] = arrangeFlatsByExpId(inputs['inputExp'], inputs['inputDims'])

        outputs = self.run(**inputs)
        outputs = self._guaranteeOutputs(inputs['inputDims'], outputs, outputRefs)
        butlerQC.put(outputs, outputRefs)

    def _guaranteeOutputs(self, inputDims, outputs, outputRefs):
        """Ensure that all outputRefs have a matching output, and if they do
        not, fill the output with dummy PTC datasets.

        Parameters
        ----------
        inputDims : `dict` [`str`, `int`]
            Input exposure dimensions.
        outputs : `lsst.pipe.base.Struct`
            Outputs from the ``run`` method.  Contains the entry:

            ``outputCovariances``
                Output PTC datasets (`list` [`lsst.ip.isr.IsrCalib`])
        outputRefs : `~lsst.pipe.base.OutputQuantizedConnection`
            Container with all of the outputs expected to be generated.

        Returns
        -------
        outputs : `lsst.pipe.base.Struct`
            Dummy dataset padded version of the input ``outputs`` with
            the same entries.
        """
        newCovariances = []
        for ref in outputRefs.outputCovariances:
            outputExpId = ref.dataId['exposure']
            if outputExpId in inputDims:
                entry = inputDims.index(outputExpId)
                newCovariances.append(outputs.outputCovariances[entry])
            else:
                newPtc = PhotonTransferCurveDataset(['no amp'], 'DUMMY', covMatrixSide=1)
                newPtc.setAmpValuesPartialDataset('no amp')
                newCovariances.append(newPtc)
        return pipeBase.Struct(outputCovariances=newCovariances)

    def run(self, inputExp, inputDims, inputPhotodiodeData=None):
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
        taskMetadata : `list` [`lsst.pipe.base.TaskMetadata`], optional
            List of exposures metadata from ISR.  This is not used,
            and will be completely removed on DM-45802.
        inputPhotodiodeData : `dict` [`str`, `lsst.ip.isr.PhotodiodeCalib`]
            Photodiode readings data (optional).

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

        # Create a dummy ptcDataset. Dummy datasets will be
        # used to ensure that the number of output and input
        # dimensions match.
        dummyPtcDataset = PhotonTransferCurveDataset(
            ampNames, 'DUMMY',
            covMatrixSide=self.config.maximumRangeCovariancesAstier)
        for ampName in ampNames:
            dummyPtcDataset.setAmpValuesPartialDataset(ampName)

        photoChargeDict = self._extractPhotoChargeDict(inputPhotodiodeData, inputExp)

        # Output list with PTC datasets.
        partialPtcDatasetList = []
        # The number of output references needs to match that of input
        # references: initialize outputlist with dummy PTC datasets.
        for i in range(len(inputDims)):
            partialPtcDatasetList.append(dummyPtcDataset)

        if self.config.numEdgeSuspect > 0:
            self.isrTask = IsrTask()
            self.log.info("Masking %d pixels from the edges of all %ss as SUSPECT.",
                          self.config.numEdgeSuspect, self.config.edgeMaskLevel)

        # Depending on the value of config.matchExposuresType
        # 'expTime' can stand for exposure time, flux, or ID.
        for expTime in inputExp:
            exposures = inputExp[expTime]
            if not np.isfinite(expTime):
                self.log.warning("Illegal/missing %s found (%s). Dropping exposure %d",
                                 self.config.matchExposuresType, expTime, exposures[0][1])
                continue
            elif len(exposures) == 1:
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

            partialPtcDataset, nAmpsNan = self._extractOneFlatPair(
                [expId1, expId2],
                [exp1, exp2],
                photoChargeDict,
                expTime,
            )

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
            partialPtcDataset.updateMetadataFromExposures([exp1, exp2])
            partialPtcDataset.updateMetadata(setDate=True, detector=detector)
            partialPtcDatasetList[datasetIndex] = partialPtcDataset

            if nAmpsNan == len(ampNames):
                msg = f"NaN mean in all amps of exposure pair {expId1}, {expId2} of detector {detNum}."
                self.log.warning(msg)

        return pipeBase.Struct(
            outputCovariances=partialPtcDatasetList,
        )


class PhotonTransferCurveExtractPairTask(PhotonTransferCurveExtractTaskBase):
    ConfigClass = PhotonTransferCurveExtractPairConfig
    _DefaultName = "cpPtcExtractPair"

    def runQuantum(self, butlerQC, inputRefs, outputRefs):
        inputs = butlerQC.get(inputRefs)

        quantumExposure = butlerQC.quantum.dataId["exposure"]

        if len(inputRefs.inputExp) == 1:
            self.log.warning("Exposure %d does not have a pair; skipping.", quantumExposure)
            raise pipeBase.NoWorkFound(f"Exposure {quantumExposure} does not have a pair.")

        if self.config.matchExposuresType == "FLUX":
            # Get the flux keyword value for each of the inputs.
            if len(inputs["inputExp"]) > 3:
                raise RuntimeError("Error in matching inputs to PhotonTransferCurveExtractPairTask")

            fluxValues = []
            inputExposures = []
            for inputExpRef in inputs["inputExp"]:
                md = inputExpRef.get(component="metadata")
                # Note that nan will never match anything.
                fluxValue = md.get(self.config.matchExposuresByFluxKeyword, np.nan)
                if np.isnan(fluxValue):
                    self.log.warning(
                        "Exposure %d is missing flux keyword %s",
                        inputExpRef.dataId["exposure"],
                        self.config.matchExposuresByFluxKeyword,
                    )
                fluxValues.append(fluxValue)
                inputExposures.append(inputExpRef.dataId["exposure"])

            if len(fluxValues) == 2:
                # The simple case that we are at a boundary.
                if fluxValues[0] != fluxValues[1]:
                    self.log.warning(
                        "Exposure %d may not have a match via %s",
                        quantumExposure,
                        self.config.matchExposuresByFluxKeyword,
                    )
                    raise pipeBase.NoWorkFound(
                        f"Exposure {quantumExposure} does not have a match."
                    )
                elif quantumExposure != inputs["inputExp"][0].dataId["exposure"]:
                    # Not primary; that's okay, we can skip.
                    raise pipeBase.NoWorkFound(
                        f"Exposure {quantumExposure} is not the primary in a pair.",
                    )
            else:
                # We have a set of three.
                if np.all(np.asarray(fluxValues) == fluxValues[0]):
                    self.log.warning(
                        "More than two exposures have matching %s to exposure %d",
                        self.config.matchExposuresByFluxKeyword,
                        quantumExposure,
                    )
                    raise pipeBase.NoWorkFound(
                        f"Exposure {quantumExposure} matches too many "
                        f"{self.config.matchExposuresByFluxKeyword}",
                    )
                if fluxValues[0] == fluxValues[1] and inputExposures[0] == quantumExposure:
                    # Remove the last input.
                    inputs["inputExp"].pop(2)
                    if self.config.doExtractPhotodiodeData:
                        inputs["inputPhotodiodeData"].pop(2)
                elif fluxValues[1] == fluxValues[2] and inputExposures[1] == quantumExposure:
                    # Remove the first input
                    inputs["inputExp"].pop(0)
                    if self.config.doExtractPhotodiodeData:
                        inputs["inputPhotodiodeData"].pop(0)
                elif fluxValues[0] != fluxValues[1] and fluxValues[1] != fluxValues[2]:
                    self.log.warning(
                        "Exposure %d may not have a match via %s",
                        quantumExposure,
                        self.config.matchExposuresByFluxKeyword,
                    )
                    raise pipeBase.NoWorkFound(
                        f"Exposure {quantumExposure} does not have a match."
                    )
                else:
                    # Not primary; that's okay, we can skip.
                    raise pipeBase.NoWorkFound(
                        f"Exposure {quantumExposure} is not the primary in a pair.",
                    )

        inputs["inputDims"] = [expRef.dataId["exposure"] for expRef in inputs["inputExp"]]

        outputs = self.run(**inputs)
        butlerQC.put(outputs, outputRefs)

    def run(self, *, inputExp, inputDims, inputPhotodiodeData=None):
        """Measure covariances from a single flat pair.

        Parameters
        ----------
        inputExp : `list` [`lsst.pipe.base.connections.DeferredDatasetRef`]
            List of 2 references to the input flat exposures.
        inputDims : `list` [`int`]
            List of 2 exposure numbers for the input flat exposures.
        inputPhotodiodeData : `list`
                [`lsst.pipe.base.connections.DeferredDatasetRef`], optional
            List of 2 references to input photodiode data.

        Returns
        -------
        results : `lsst.pipe.base.Struct`
            The resulting Struct contains:

            ``outputCovariance``
                The single-pair PTC measurement
                `lsst.ip.isr.PhotonTransferCurveDataset`
        """
        partialPtcDataset = None

        expId1, expId2 = inputDims
        exp1 = inputExp[0].get()
        exp2 = inputExp[1].get()

        detector = exp1.getDetector()
        detNum = detector.getId()
        amps = detector.getAmplifiers()
        ampNames = [amp.getName() for amp in amps]

        # Set the exposure time to the real exposure time.
        expTime = exp1.info.getVisitInfo().exposureTime

        photoChargeDict = self._extractPhotoChargeDict(inputPhotodiodeData, inputExp)

        if self.config.numEdgeSuspect > 0:
            self.isrTask = IsrTask()
            self.log.info("Masking %d pixels from the edges of all %ss as SUSPECT.",
                          self.config.numEdgeSuspect, self.config.edgeMaskLevel)

        partialPtcDataset, nAmpsNan = self._extractOneFlatPair(
            [expId1, expId2],
            [exp1, exp2],
            photoChargeDict,
            expTime,
        )

        partialPtcDataset.updateMetadataFromExposures([exp1, exp2])
        partialPtcDataset.updateMetadata(setDate=True, detector=detector)

        if nAmpsNan == len(ampNames):
            self.log.warning(
                "NaN mean in all amps of exposure pair %d, %d of detector %d.",
                expId1,
                expId2,
                detNum,
            )

        return pipeBase.Struct(
            outputCovariance=partialPtcDataset,
        )
