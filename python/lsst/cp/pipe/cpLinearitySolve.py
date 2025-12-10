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

__all__ = [
    "LinearitySolveTask",
    "LinearitySolveConfig",
    "LinearityDoubleSplineSolveTask",
    "LinearityDoubleSplineSolveConfig",
]

import copy
import logging
import numpy as np
import esutil
from scipy.stats import median_abs_deviation
from scipy.interpolate import Akima1DInterpolator

import lsst.afw.image as afwImage
import lsst.pipe.base as pipeBase
import lsst.pipe.base.connectionTypes as cT
import lsst.pex.config as pexConfig

from lsstDebug import getDebugFrame
from lsst.ip.isr import (Linearizer, IsrProvenance)

from .utils import (funcPolynomial, irlsFit, AstierSplineLinearityFitter,
                    extractCalibDate)


def ptcLookup(datasetType, registry, quantumDataId, collections):
    """Butler lookup function to allow PTC to be found.

    Parameters
    ----------
    datasetType : `lsst.daf.butler.DatasetType`
        Dataset type to look up.
    registry : `lsst.daf.butler.Registry`
        Registry for the data repository being searched.
    quantumDataId : `lsst.daf.butler.DataCoordinate`
        Data ID for the quantum of the task this dataset will be passed to.
        This must include an "instrument" key, and should also include any
        keys that are present in ``datasetType.dimensions``.  If it has an
        ``exposure`` or ``visit`` key, that's a sign that this function is
        not actually needed, as those come with the temporal information that
        would allow a real validity-range lookup.
    collections : `lsst.daf.butler.registry.CollectionSearch`
        Collections passed by the user when generating a QuantumGraph.  Ignored
        by this function (see notes below).

    Returns
    -------
    refs : `list` [ `DatasetRef` ]
        A zero- or single-element list containing the matching
        dataset, if one was found.

    Raises
    ------
    RuntimeError
        Raised if more than one PTC reference is found.
    """
    refs = list(registry.queryDatasets(datasetType, dataId=quantumDataId, collections=collections,
                                       findFirst=False))
    if len(refs) >= 2:
        RuntimeError("Too many PTC connections found. Incorrect collections supplied?")

    return refs


class LinearitySolveConnections(pipeBase.PipelineTaskConnections,
                                dimensions=("instrument", "detector")):
    dummy = cT.Input(
        name="raw",
        doc="Dummy exposure.",
        storageClass='Exposure',
        dimensions=("instrument", "exposure", "detector"),
        multiple=True,
        deferLoad=True,
    )

    camera = cT.PrerequisiteInput(
        name="camera",
        doc="Camera Geometry definition.",
        storageClass="Camera",
        dimensions=("instrument", ),
        isCalibration=True,
    )

    inputPtc = cT.PrerequisiteInput(
        name="ptc",
        doc="Input PTC dataset.",
        storageClass="PhotonTransferCurveDataset",
        dimensions=("instrument", "detector"),
        isCalibration=True,
        lookupFunction=ptcLookup,
    )

    inputLinearizerPtc = cT.Input(
        name="linearizerPtc",
        doc="Input linearizer PTC dataset.",
        storageClass="PhotonTransferCurveDataset",
        dimensions=("instrument", "detector"),
        isCalibration=True,
    )

    inputPhotodiodeCorrection = cT.Input(
        name="pdCorrection",
        doc="Input photodiode correction.",
        storageClass="IsrCalib",
        dimensions=("instrument", ),
        isCalibration=True,
    )

    inputNormalization = cT.Input(
        name="cpLinearizerPtcNormalization",
        doc="Focal-plane normalization table.",
        storageClass="ArrowAstropy",
        dimensions=("instrument",),
        isCalibration=True,
    )

    outputLinearizer = cT.Output(
        name="linearity",
        doc="Output linearity measurements.",
        storageClass="Linearizer",
        dimensions=("instrument", "detector"),
        isCalibration=True,
    )

    def __init__(self, *, config=None):
        super().__init__(config=config)

        if not config.applyPhotodiodeCorrection:
            del self.inputPhotodiodeCorrection

        if config.useLinearizerPtc:
            del self.inputPtc
        else:
            del self.inputLinearizerPtc

        if not config.useFocalPlaneNormalization:
            del self.inputNormalization


class LinearitySolveConfig(pipeBase.PipelineTaskConfig,
                           pipelineConnections=LinearitySolveConnections):
    """Configuration for solving the linearity from PTC dataset.
    """
    linearityType = pexConfig.ChoiceField(
        dtype=str,
        doc="Type of linearizer to construct.",
        default="Squared",
        allowed={
            "LookupTable": "Create a lookup table solution.",
            "Polynomial": "Create an arbitrary polynomial solution.",
            "Squared": "Create a single order squared solution.",
            "Spline": "Create a spline based solution.",
            "None": "Create a dummy solution.",
        }
    )
    polynomialOrder = pexConfig.RangeField(
        dtype=int,
        doc="Degree of polynomial to fit.  Must be at least 2.",
        default=3,
        min=2,
    )
    splineKnots = pexConfig.Field(
        dtype=int,
        doc="Number of spline knots to use in fit.",
        default=10,
    )

    trimmedState = pexConfig.Field(
        dtype=bool,
        doc="Will this linearizer be used on trimmed data?",
        default=True,
    )

    maxLookupTableAdu = pexConfig.Field(
        dtype=int,
        doc="Maximum DN value for a LookupTable linearizer.",
        default=2**18,
    )
    maxLinearAdu = pexConfig.Field(
        dtype=float,
        doc="Maximum adu value to use to estimate linear term; not used with spline fits.",
        default=20000.0,
    )
    minLinearAdu = pexConfig.Field(
        dtype=float,
        doc="Minimum adu value to use to estimate linear term.",
        default=30.0,
    )
    nSigmaClipLinear = pexConfig.Field(
        dtype=float,
        doc="Maximum deviation from linear solution for Poissonian noise.",
        default=5.0,
    )
    ignorePtcMask = pexConfig.Field(
        dtype=bool,
        doc="Ignore the expIdMask set by the PTC solver?",
        default=False,
        deprecated="This field is no longer used. Will be removed after v28.",
    )
    maxFracLinearityDeviation = pexConfig.Field(
        dtype=float,
        doc="Maximum fraction deviation from raw linearity to compute "
            "linearityTurnoff and linearityMaxSignal.",
        # TODO: DM-46811 investigate if this can be raised to 0.05.
        default=0.01,
    )
    minSignalFitLinearityTurnoff = pexConfig.Field(
        dtype=float,
        doc="Minimum signal to compute raw linearity slope for linearityTurnoff.",
        default=1000.0,
    )
    usePhotodiode = pexConfig.Field(
        dtype=bool,
        doc="Use the photodiode info instead of the raw expTimes?",
        default=False,
    )
    applyPhotodiodeCorrection = pexConfig.Field(
        dtype=bool,
        doc="Calculate and apply a correction to the photodiode readings?",
        default=False,
    )
    minPhotodiodeCurrent = pexConfig.Field(
        dtype=float,
        doc="Minimum value to trust photodiode signals.",
        default=0.0,
    )
    doAutoGrouping = pexConfig.Field(
        dtype=bool,
        doc="Do automatic group detection? Cannot be True if splineGroupingColumn is also set. "
            "The automatic group detection will use the ratio of signal to exposure time (if "
            "autoGroupingUseExptime is True) or photodiode (if False) to determine which "
            "flat pairs were taken with different illumination settings.",
        default=False,
    )
    autoGroupingUseExptime = pexConfig.Field(
        dtype=bool,
        doc="Use exposure time to determine automatic grouping. Used if doAutoGrouping=True.",
        default=True,
    )
    autoGroupingThreshold = pexConfig.Field(
        dtype=float,
        doc="Minimum relative jump from sorted conversion values to determine a group.",
        default=0.1,
    )
    autoGroupingMaxSignalFraction = pexConfig.Field(
        dtype=float,
        doc="Only do auto-grouping when the signal is this fraction of the maximum signal. "
            "All exposures with signal higher than this threshold will be put into the "
            "largest signal group. This config is needed if the input PTC goes beyond "
            "the linearity turnoff.",
        default=0.9,
    )
    splineGroupingColumn = pexConfig.Field(
        dtype=str,
        doc="Column to use for grouping together points for Spline mode, to allow "
            "for different proportionality constants. If None, then grouping will "
            "only be done if doAutoGrouping is True.",
        default=None,
        optional=True,
    )
    splineGroupingMinPoints = pexConfig.Field(
        dtype=int,
        doc="Minimum number of linearity points to allow grouping together points "
            "for Spline mode with splineGroupingColumn. This configuration is here "
            "to prevent misuse of the Spline code to avoid over-fitting.",
        default=100,
    )
    splineFitMinIter = pexConfig.Field(
        dtype=int,
        doc="Minimum number of iterations for spline fit.",
        default=3,
    )
    splineFitMaxIter = pexConfig.Field(
        dtype=int,
        doc="Maximum number of iterations for spline fit.",
        default=20,
    )
    splineFitMaxRejectionPerIteration = pexConfig.Field(
        dtype=int,
        doc="Maximum number of rejections per iteration for spline fit.",
        default=5,
    )
    doSplineFitOffset = pexConfig.Field(
        dtype=bool,
        doc="Fit a scattered light offset in the spline fit.",
        default=True,
    )
    doSplineFitWeights = pexConfig.Field(
        dtype=bool,
        doc="Fit linearity weight parameters in the spline fit.",
        default=False,
    )
    splineFitWeightParsStart = pexConfig.ListField(
        dtype=float,
        doc="Starting parameters for weight fit, if doSplineFitWeights=True. "
            "Parameters are such that sigma = sqrt(par[0]**2. + par[1]**2./mu)."
            "If doSplineFitWeights=False then these are used as-is; otherwise "
            "they are used as the initial values for fitting these parameters.",
        length=2,
        default=[1.0, 0.0],
    )
    doSplineFitTemperature = pexConfig.Field(
        dtype=bool,
        doc="Fit temperature coefficient in spline fit?",
        default=False,
    )
    splineFitTemperatureColumn = pexConfig.Field(
        dtype=str,
        doc="Name of the temperature column to use when fitting temperature "
            "coefficients in spline fit; this must not be None if "
            "doSplineFitTemperature is True.",
        default=None,
        optional=True,
    )
    doSplineFitTemporal = pexConfig.Field(
        dtype=bool,
        doc="Fit a linear temporal parameter coefficient in spline fit?",
        default=False,
    )
    useLinearizerPtc = pexConfig.Field(
        dtype=bool,
        doc="Use a linearizer ptc in a single pipeline?",
        default=False,
    )
    useFocalPlaneNormalization = pexConfig.Field(
        dtype=bool,
        doc="Use focal-plane normalization in addition to/instead of photodiode? "
            "(Only used with spline fitting).",
        default=False,
    )

    def validate(self):
        super().validate()

        if self.doSplineFitTemperature and self.splineFitTemperatureColumn is None:
            raise ValueError("Must set splineFitTemperatureColumn if doSplineFitTemperature is True.")

        if self.doAutoGrouping and self.splineGroupingColumn is not None:
            raise ValueError("Must not set doAutoGrouping and also splineGroupingColumn")
        if self.doAutoGrouping:
            if not self.autoGroupingUseExptime and not self.usePhotodiode:
                raise ValueError("If doAutoGrouping is True and autoGroupingUseExptime is False, then "
                                 "usePhotodiode must be True.")


class LinearitySolveTask(pipeBase.PipelineTask):
    """Fit the linearity from the PTC dataset.
    """

    ConfigClass = LinearitySolveConfig
    _DefaultName = 'cpLinearitySolve'

    def runQuantum(self, butlerQC, inputRefs, outputRefs):
        """Ensure that the input and output dimensions are passed along.

        Parameters
        ----------
        butlerQC : `lsst.daf.butler.QuantumContext`
            Butler to operate on.
        inputRefs : `lsst.pipe.base.InputQuantizedConnection`
            Input data refs to load.
        ouptutRefs : `lsst.pipe.base.OutputQuantizedConnection`
            Output data refs to persist.
        """
        inputs = butlerQC.get(inputRefs)

        # Use the dimensions to set calib/provenance information.

        if self.config.useLinearizerPtc:
            inputs["inputDims"] = dict(inputRefs.inputLinearizerPtc.dataId.required)
            inputs["inputPtc"] = inputs["inputLinearizerPtc"]
            del inputs["inputLinearizerPtc"]
        else:
            inputs["inputDims"] = dict(inputRefs.inputPtc.dataId.required)

        # Add calibration provenance info to header.
        kwargs = dict()
        reference = getattr(inputRefs, "inputPtc", None)

        if reference is not None and hasattr(reference, "run"):
            runKey = "PTC_RUN"
            runValue = reference.run
            idKey = "PTC_UUID"
            idValue = str(reference.id)
            dateKey = "PTC_DATE"
            calib = inputs.get("inputPtc", None)
            dateValue = extractCalibDate(calib)

            kwargs[runKey] = runValue
            kwargs[idKey] = idValue
            kwargs[dateKey] = dateValue

            self.log.info("Using " + str(reference.run))

        outputs = self.run(**inputs)
        outputs.outputLinearizer.updateMetadata(setDate=False, **kwargs)

        butlerQC.put(outputs, outputRefs)

    def run(self, inputPtc, dummy, camera, inputDims,
            inputPhotodiodeCorrection=None, inputNormalization=None):
        """Fit non-linearity to PTC data, returning the correct Linearizer
        object.

        Parameters
        ----------
        inputPtc : `lsst.ip.isr.PtcDataset`
            Pre-measured PTC dataset.
        dummy : `lsst.afw.image.Exposure`
            The exposure used to select the appropriate PTC dataset.
            In almost all circumstances, one of the input exposures
            used to generate the PTC dataset is the best option.
        camera : `lsst.afw.cameraGeom.Camera`
            Camera geometry.
        inputDims : `lsst.daf.butler.DataCoordinate` or `dict`
            DataIds to use to populate the output calibration.
        inputPhotodiodeCorrection :
            `lsst.ip.isr.PhotodiodeCorrection`, optional
            Pre-measured photodiode correction used in the case when
            applyPhotodiodeCorrection=True.
        inputNormalization : `astropy.table.Table`, optional
            Focal plane normalization table to use if
            useFocalPlaneNormalization is True.

        Returns
        -------
        results : `lsst.pipe.base.Struct`
            The results struct containing:

            ``outputLinearizer``
                Final linearizer calibration (`lsst.ip.isr.Linearizer`).
            ``outputProvenance``
                Provenance data for the new calibration
                (`lsst.ip.isr.IsrProvenance`).

        Notes
        -----
        This task currently fits only polynomial-defined corrections,
        where the correction coefficients are defined such that:
        :math:`corrImage = uncorrImage + \\sum_i c_i uncorrImage^(2 + i)`
        These :math:`c_i` are defined in terms of the direct polynomial fit:
        :math:`meanVector ~ P(x=timeVector) = \\sum_j k_j x^j`
        such that :math:`c_(j-2) = -k_j/(k_1^j)` in units of DN^(1-j) (c.f.,
        Eq. 37 of 2003.05978). The `config.polynomialOrder` or
        `config.splineKnots` define the maximum order of :math:`x^j` to fit.
        As :math:`k_0` and :math:`k_1` are degenerate with bias level and gain,
        they are not included in the non-linearity correction.
        """
        if len(dummy) == 0:
            self.log.warning("No dummy exposure found.")

        detector = camera[inputDims['detector']]
        if self.config.linearityType == 'LookupTable':
            table = np.zeros((len(detector), self.config.maxLookupTableAdu), dtype=np.float32)
            tableIndex = 0
        else:
            table = None
            tableIndex = None  # This will fail if we increment it.

        # Initialize the linearizer.
        linearizer = Linearizer(detector=detector, table=table, log=self.log)
        linearizer.updateMetadataFromExposures([inputPtc])
        if self.config.usePhotodiode and self.config.applyPhotodiodeCorrection:
            abscissaCorrections = inputPhotodiodeCorrection.abscissaCorrections

        groupingValues = _determineInputGroups(
            inputPtc,
            self.config.doAutoGrouping,
            self.config.autoGroupingUseExptime,
            self.config.autoGroupingMaxSignalFraction,
            self.config.autoGroupingThreshold,
            self.config.splineGroupingColumn,
            self.config.minPhotodiodeCurrent,
        )

        if self.config.linearityType == 'Spline':
            if self.config.doSplineFitTemperature:
                if self.config.splineFitTemperatureColumn not in inputPtc.auxValues:
                    raise ValueError("Config requests fitting temperature coefficient for "
                                     f"{self.config.splineFitTemperatureColumn} but this column "
                                     "is not available in inputPtc.auxValues.")
                temperatureValues = inputPtc.auxValues[self.config.splineFitTemperatureColumn]
            else:
                temperatureValues = None

            # We set this to have a value to fill the bad amps.
            fitOrder = self.config.splineKnots
        else:
            fitOrder = self.config.polynomialOrder

        for i, amp in enumerate(detector):
            ampName = amp.getName()

            # Save the input gains
            linearizer.inputGain[ampName] = inputPtc.gain[ampName]

            if ampName in inputPtc.badAmps:
                linearizer = self.fillBadAmp(linearizer, fitOrder, inputPtc, amp)
                self.log.warning("Amp %s in detector %s has no usable PTC information. Skipping!",
                                 ampName, detector.getName())
                continue

            # Check for too few points.
            if self.config.linearityType == "Spline" \
               and self.config.splineGroupingColumn is not None \
               and len(inputPtc.inputExpIdPairs[ampName]) < self.config.splineGroupingMinPoints:
                raise RuntimeError(
                    "The input PTC has too few points to reliably run with PD grouping. "
                    "The recommended course of action is to set splineGroupingColumn to None. "
                    "If you really know what you are doing, you may reduce "
                    "config.splineGroupingMinPoints.")

            # We start with all finite values.
            mask = np.isfinite(inputPtc.rawMeans[ampName])

            if self.config.linearityType == "Spline" and temperatureValues is not None:
                mask &= np.isfinite(temperatureValues)

            if self.config.usePhotodiode:
                modExpTimes = inputPtc.photoCharges[ampName].copy()
                # Make sure any exposure pairs that do not have photodiode data
                # are masked.
                mask[~np.isfinite(modExpTimes)] = False

                # Make sure any photodiode measurements below the configured
                # minimum are masked.
                mask[modExpTimes < self.config.minPhotodiodeCurrent] = False

                # Get the photodiode correction.
                if self.config.applyPhotodiodeCorrection:
                    for j, pair in enumerate(inputPtc.inputExpIdPairs[ampName]):
                        try:
                            correction = abscissaCorrections[str(pair)]
                        except KeyError:
                            correction = 0.0
                        modExpTimes[j] += correction

                inputAbscissa = modExpTimes
            else:
                inputAbscissa = inputPtc.rawExpTimes[ampName].copy()

            # Normalize if configured.
            inputNorm = np.ones_like(inputAbscissa)
            if self.config.useFocalPlaneNormalization:
                exposures = np.asarray(inputPtc.inputExpIdPairs[ampName])[:, 0]
                a, b = esutil.numpy_util.match(exposures, inputNormalization["exposure"])
                inputNorm[a] = inputNormalization["normalization"][b]
                inputAbscissa *= inputNorm

            # Compute linearityTurnoff and linearitySignalMax.
            turnoffMask = inputPtc.expIdMask[ampName].copy()
            turnoffMask &= mask

            turnoffIndex, turnoff, maxSignal, _ = _computeTurnoffAndMax(
                inputAbscissa,
                inputPtc.rawMeans[ampName],
                turnoffMask,
                groupingValues,
                ampName=ampName,
                minSignalFitLinearityTurnoff=self.config.minSignalFitLinearityTurnoff,
                maxFracLinearityDeviation=self.config.maxFracLinearityDeviation,
                log=self.log,
            )

            if np.isnan(turnoff):
                # This is a bad amp, with no linearizer.
                linearizer = self.fillBadAmp(linearizer, fitOrder, inputPtc, amp)
                self.log.warning("Amp %s in detector %s has no usable linearizer information. Skipping!",
                                 ampName, detector.getName())
                continue

            linearizer.linearityTurnoff[ampName] = turnoff
            linearizer.linearityMaxSignal[ampName] = maxSignal

            inputOrdinate = inputPtc.rawMeans[ampName].copy()

            linearizer.inputAbscissa[ampName] = inputAbscissa.copy()
            linearizer.inputOrdinate[ampName] = inputOrdinate.copy()
            linearizer.inputGroupingIndex[ampName] = groupingValues.copy()
            linearizer.inputNormalization[ampName] = inputNorm.copy()

            if self.config.linearityType != 'Spline':
                mask &= (inputOrdinate < self.config.maxLinearAdu)
            else:
                # For spline fits, cut above the turnoff.
                self.log.info("Using linearityTurnoff of %.4f adu for amplifier %s", turnoff, ampName)
                extraMask = np.ones(len(inputOrdinate), dtype=bool)
                extraMask[turnoffIndex + 1:] = False
                mask &= extraMask

            mask &= (inputOrdinate > self.config.minLinearAdu)

            # Initial value for the mask.
            linearizer.inputMask[ampName] = mask.copy()

            if mask.sum() < 2:
                linearizer = self.fillBadAmp(linearizer, fitOrder, inputPtc, amp)
                self.log.warning("Amp %s in detector %s has not enough points for fit. Skipping!",
                                 ampName, detector.getName())
                continue

            if self.config.linearityType != 'Spline':
                linearFit, linearFitErr, chiSq, weights = irlsFit([0.0, 100.0], inputAbscissa[mask],
                                                                  inputOrdinate[mask], funcPolynomial)

                # Convert this proxy-to-flux fit into an expected linear flux
                linearOrdinate = linearFit[0] + linearFit[1] * inputAbscissa
                # Exclude low end outliers.
                # This is compared to the original values.
                threshold = self.config.nSigmaClipLinear * np.sqrt(abs(inputOrdinate))

                mask[np.abs(inputOrdinate - linearOrdinate) >= threshold] = False

                linearizer.inputMask[ampName] = mask.copy()

                if mask.sum() < 2:
                    linearizer = self.fillBadAmp(linearizer, fitOrder, inputPtc, amp)
                    self.log.warning("Amp %s in detector %s has not enough points in linear ordinate. "
                                     "Skipping!", ampName, detector.getName())
                    continue

                self.debugFit('linearFit', inputAbscissa, inputOrdinate, linearOrdinate, mask, ampName)

            # Do fits
            if self.config.linearityType in ['Polynomial', 'Squared', 'LookupTable']:
                polyFit = np.zeros(fitOrder + 1)
                polyFit[1] = 1.0
                polyFit, polyFitErr, chiSq, weights = irlsFit(polyFit, linearOrdinate[mask],
                                                              inputOrdinate[mask], funcPolynomial)

                # Truncate the polynomial fit to the squared term.
                k1 = polyFit[1]
                linearityCoeffs = np.array(
                    [-coeff/(k1**order) for order, coeff in enumerate(polyFit)]
                )[2:]
                significant = np.where(np.abs(linearityCoeffs) > 1e-10)
                self.log.info("Significant polynomial fits: %s", significant)

                modelOrdinate = funcPolynomial(polyFit, linearOrdinate)

                self.debugFit(
                    'polyFit',
                    inputAbscissa[mask],
                    inputOrdinate[mask],
                    modelOrdinate[mask],
                    None,
                    ampName,
                )

                if self.config.linearityType == 'Squared':
                    # The first term is the squared term.
                    linearityCoeffs = linearityCoeffs[0: 1]
                elif self.config.linearityType == 'LookupTable':
                    # Use linear part to get time at which signal is
                    # maxAduForLookupTableLinearizer DN
                    tMax = (self.config.maxLookupTableAdu - polyFit[0])/polyFit[1]
                    timeRange = np.linspace(0, tMax, self.config.maxLookupTableAdu)
                    signalIdeal = polyFit[0] + polyFit[1]*timeRange
                    signalUncorrected = funcPolynomial(polyFit, timeRange)
                    lookupTableRow = signalIdeal - signalUncorrected  # LinearizerLookupTable has correction

                    linearizer.tableData[tableIndex, :] = lookupTableRow
                    linearityCoeffs = np.array([tableIndex, 0])
                    tableIndex += 1
            elif self.config.linearityType in ['Spline']:
                # This is a spline fit with photodiode data based on a model
                # from Pierre Astier.
                # This model fits a spline with (optional) nuisance parameters
                # to allow for different linearity coefficients with different
                # photodiode settings.  The minimization is a least-squares
                # fit with the residual of
                # Sum[(S(mu_i) + mu_i - O)/(k_j * D_i) - 1]**2, where S(mu_i)
                # is an Akima Spline function of mu_i, the observed flat-pair
                # mean; D_j is the photo-diode measurement corresponding to
                # that flat-pair; and k_j is a constant of proportionality
                # which is over index j as it is allowed to
                # be different based on different photodiode settings (e.g.
                # CCOBCURR); and O is a constant offset to allow for light
                # leaks (and is only fit if doSplineFitOffset=True). In
                # addition, if config.doSplineFitTemperature is True then
                # the fit will adjust mu such that
                # mu = mu_input*(1 + alpha*(T - T_ref))
                # and T_ref is taken as the median temperature of the run.
                # Finally, if config.doSplineFitTemporal is True then the
                # fit will further adjust mu such that
                # mu = mu_input*(1 + beta*(mjd - mjd_ref))
                # and mjd_ref is taken as the median mjd of the run.
                # Note that this fit is only valid if the input data
                # was taken with a randomly shuffled order of exposure
                # levels.

                # The fit has additional constraints to ensure that the spline
                # goes through the (0, 0) point, as well as a normalization
                # condition so that the average of the spline over the full
                # range is 0. The normalization ensures that the spline only
                # fits deviations from linearity, rather than the linear
                # function itself which is degenerate with the gain.

                # We want to make sure the top node is above the top value
                # to avoid edge issues with the top point.
                nodes = np.linspace(0.0, np.max(inputOrdinate[mask]) + 1.0, self.config.splineKnots)

                if temperatureValues is not None:
                    temperatureValuesScaled = temperatureValues - np.median(temperatureValues[mask])
                else:
                    temperatureValuesScaled = None

                if self.config.doSplineFitTemporal:
                    inputMjdScaled = inputPtc.inputExpPairMjdStartList[ampName].copy()
                    inputMjdScaled -= np.nanmedian(inputMjdScaled)
                else:
                    inputMjdScaled = None

                fitter = AstierSplineLinearityFitter(
                    nodes,
                    groupingValues,
                    inputAbscissa,
                    inputOrdinate,
                    mask=mask,
                    log=self.log,
                    fit_offset=self.config.doSplineFitOffset,
                    fit_weights=self.config.doSplineFitWeights,
                    weight_pars_start=self.config.splineFitWeightParsStart,
                    fit_temperature=self.config.doSplineFitTemperature,
                    temperature_scaled=temperatureValuesScaled,
                    max_signal_nearly_linear=inputPtc.ptcTurnoff[ampName],
                    fit_temporal=self.config.doSplineFitTemporal,
                    mjd_scaled=inputMjdScaled,
                )
                p0 = fitter.estimate_p0()
                pars = fitter.fit(
                    p0,
                    min_iter=self.config.splineFitMinIter,
                    max_iter=self.config.splineFitMaxIter,
                    max_rejection_per_iteration=self.config.splineFitMaxRejectionPerIteration,
                    n_sigma_clip=self.config.nSigmaClipLinear,
                )

                # Confirm that the first parameter is 0, and set it to
                # exactly zero.
                if not np.isclose(pars[0], 0):
                    raise RuntimeError("Programmer error! First spline parameter must "
                                       "be consistent with zero.")
                pars[0] = 0.0

                linearityChisq = fitter.compute_chisq_dof(pars)

                linearityCoeffs = np.concatenate([nodes, pars[fitter.par_indices["values"]]])
                linearFit = np.array([0.0, np.mean(pars[fitter.par_indices["groups"]])])

                # We must modify the inputOrdinate according to the
                # nuisance terms in the linearity fit for the residual
                # computation code to work properly.
                # The true mu (inputOrdinate) is given by
                #  mu = mu_in * (1 + alpha*t_scale) * (1 + beta*mjd_scale)
                if self.config.doSplineFitTemperature:
                    inputOrdinate *= (1.0
                                      + pars[fitter.par_indices["temperature_coeff"]]*temperatureValuesScaled)
                if self.config.doSplineFitTemporal:
                    inputOrdinate *= (1.0
                                      + pars[fitter.par_indices["temporal_coeff"]]*inputMjdScaled)
                # We have to adjust the abscissa for the different groups.
                # This is because we need a corrected abscissa to get a
                # reasonable linear fit to look for residuals, particularly in
                # the case of significantly different signal-vs-photodiode or
                # signal-vs-exptime scalings for different groups. This then
                # becomes a multiplication by the relative scaling of the
                # different groups.
                for j, group_index in enumerate(fitter.group_indices):
                    inputAbscissa[group_index] *= (pars[fitter.par_indices["groups"][j]] / linearFit[1])
                # And remove the offset term.
                if self.config.doSplineFitOffset:
                    inputOrdinate -= pars[fitter.par_indices["offset"]]

                linearOrdinate = linearFit[1] * inputOrdinate
                # For the spline fit, reuse the "polyFit -> fitParams"
                # field to record the linear coefficients for the groups.
                # We additionally append the offset and weight_pars;
                # however these will be zero-length arrays if these were
                # not configured to be fit.
                polyFit = np.concatenate((
                    pars[fitter.par_indices["groups"]],
                    pars[fitter.par_indices["offset"]],
                    pars[fitter.par_indices["weight_pars"]],
                    pars[fitter.par_indices["temperature_coeff"]],
                    pars[fitter.par_indices["temporal_coeff"]],
                ))
                polyFitErr = np.zeros_like(polyFit)
                chiSq = linearityChisq

                # Update mask based on what the fitter rejected.
                mask = fitter.mask

                linearizer.inputMask[ampName] = mask.copy()
            else:
                polyFit = np.zeros(1)
                polyFitErr = np.zeros(1)
                chiSq = np.nan
                linearityCoeffs = np.zeros(1)

            linearizer.linearityType[ampName] = self.config.linearityType
            linearizer.linearityCoeffs[ampName] = linearityCoeffs
            if self.config.trimmedState:
                linearizer.linearityBBox[ampName] = amp.getBBox()
            else:
                linearizer.linearityBBox[ampName] = amp.getRawBBox()
            linearizer.fitParams[ampName] = polyFit
            linearizer.fitParamsErr[ampName] = polyFitErr
            linearizer.fitChiSq[ampName] = chiSq
            linearizer.linearFit[ampName] = linearFit

            image = afwImage.ImageF(len(inputOrdinate), 1)
            image.array[:, :] = inputOrdinate
            linearizeFunction = linearizer.getLinearityTypeByName(linearizer.linearityType[ampName])
            linearizeFunction()(
                image,
                **{'coeffs': linearizer.linearityCoeffs[ampName],
                   'table': linearizer.tableData,
                   'log': linearizer.log}
            )
            linearizeModel = image.array[0, :]

            # The residuals that we record are the final residuals compared to
            # a linear model, after everything has been (properly?) linearized.
            if mask.sum() < 2:
                self.log.warning("Amp %s in detector %s has not enough points in linear ordinate "
                                 "for residuals. Skipping!", ampName, detector.getName())
                residuals = np.full_like(linearizeModel, np.nan)
                residualsUnmasked = residuals.copy()
            else:
                postLinearFit, _, _, _ = irlsFit(
                    linearFit,
                    inputAbscissa[mask],
                    linearizeModel[mask],
                    funcPolynomial,
                )
                # When computing residuals, we only care about the slope of
                # the postLinearFit and not the intercept. The intercept
                # itself depends on a possibly unknown zero in the abscissa
                # (often photodiode) which may have an arbitrary value.
                residuals = linearizeModel - (postLinearFit[1] * inputAbscissa)
                residualsUnmasked = residuals.copy()
                # We set masked residuals to nan.
                residuals[~mask] = np.nan

            linearizer.fitResidualsUnmasked[ampName] = residualsUnmasked
            linearizer.fitResiduals[ampName] = residuals
            linearizer.fitResidualsModel[ampName] = linearizeModel.copy()

            finite = np.isfinite(residuals)
            if finite.sum() == 0:
                sigmad = np.nan
            else:
                sigmad = median_abs_deviation(residuals[finite]/inputOrdinate[finite], scale="normal")
            linearizer.fitResidualsSigmaMad[ampName] = sigmad

            self.debugFit(
                'solution',
                inputOrdinate[mask],
                linearOrdinate[mask],
                linearizeModel[mask],
                None,
                ampName,
            )

        self.fixupBadAmps(linearizer)

        linearizer.hasLinearity = True
        linearizer.validate()
        linearizer.updateMetadata(camera=camera, detector=detector, filterName='NONE')
        linearizer.updateMetadata(setDate=True, setCalibId=True)
        linearizer.updateMetadataFromExposures([inputPtc])
        provenance = IsrProvenance(calibType='linearizer')

        return pipeBase.Struct(
            outputLinearizer=linearizer,
            outputProvenance=provenance,
        )

    def fillBadAmp(self, linearizer, fitOrder, inputPtc, amp):
        # Need to fill linearizer with empty values
        # if the amp is non-functional
        ampName = amp.getName()
        nEntries = 1
        pEntries = 1
        if self.config.linearityType in ['Polynomial']:
            # We discard the first 2 entries in the polynomial.
            nEntries = fitOrder + 1 - 2
            pEntries = fitOrder + 1 - 2
        elif self.config.linearityType in ['Spline']:
            nEntries = fitOrder * 2
        elif self.config.linearityType in ['Squared', 'None']:
            nEntries = 1
            pEntries = fitOrder + 1
        elif self.config.linearityType in ['LookupTable']:
            nEntries = 2
            pEntries = fitOrder + 1

        nPair = len(inputPtc.inputExpIdPairs[ampName])

        linearizer.linearityType[ampName] = "None"
        linearizer.linearityCoeffs[ampName] = np.zeros(nEntries)
        if self.config.trimmedState:
            linearizer.linearityBBox[ampName] = amp.getBBox()
        else:
            linearizer.linearityBBox[ampName] = amp.getRawBBox()
        linearizer.fitParams[ampName] = np.zeros(pEntries)
        linearizer.fitParamsErr[ampName] = np.zeros(pEntries)
        linearizer.fitChiSq[ampName] = np.nan
        linearizer.fitResiduals[ampName] = np.zeros(nPair)
        linearizer.fitResidualsSigmaMad[ampName] = np.nan
        linearizer.fitResidualsUnmasked[ampName] = np.zeros(nPair)
        linearizer.fitResidualsModel[ampName] = np.zeros(nPair)
        linearizer.linearFit[ampName] = np.zeros(2)
        linearizer.linearityTurnoff[ampName] = np.nan
        linearizer.linearityMaxSignal[ampName] = np.nan
        linearizer.inputMask[ampName] = np.zeros(nPair, dtype=np.bool_)
        linearizer.inputAbscissa[ampName] = np.zeros(nPair)
        linearizer.inputOrdinate[ampName] = np.zeros(nPair)
        linearizer.inputGroupingIndex[ampName] = np.zeros(nPair, dtype=np.int64)
        linearizer.inputNormalization[ampName] = np.ones(nPair)

        return linearizer

    def fixupBadAmps(self, linearizer):
        """Fix nan padding in bad amplifiers.

        Parameters
        ----------
        linearizer : `lsst.ip.isr.Linearizer`
        """
        fitParamsMaxLen = 0
        for ampName in linearizer.ampNames:
            if (length := len(linearizer.fitParams[ampName])) > fitParamsMaxLen:
                fitParamsMaxLen = length

        for ampName in linearizer.ampNames:
            if linearizer.linearityType[ampName] == "None":
                # Bad amplifier.
                linearizer.fitParams[ampName] = np.zeros(fitParamsMaxLen)
                linearizer.fitParamsErr[ampName] = np.zeros(fitParamsMaxLen)
            elif len(linearizer.fitParams[ampName]) != fitParamsMaxLen:
                raise RuntimeError("Linearity has mismatched fitParams; check code/data.")

    def debugFit(self, stepname, xVector, yVector, yModel, mask, ampName):
        """Debug method for linearity fitting.

        Parameters
        ----------
        stepname : `str`
            A label to use to check if we care to debug at a given
            line of code.
        xVector : `numpy.array`, (N,)
            The values to use as the independent variable in the
            linearity fit.
        yVector : `numpy.array`, (N,)
            The values to use as the dependent variable in the
            linearity fit.
        yModel : `numpy.array`, (N,)
            The values to use as the linearized result.
        mask : `numpy.array` [`bool`], (N,) , optional
            A mask to indicate which entries of ``xVector`` and
            ``yVector`` to keep.
        ampName : `str`
            Amplifier name to lookup linearity correction values.
        """
        frame = getDebugFrame(self._display, stepname)
        if frame:
            import matplotlib.pyplot as plt
            fig, axs = plt.subplots(2)

            if mask is None:
                mask = np.ones_like(xVector, dtype=bool)

            fig.suptitle(f"{stepname} {ampName} {self.config.linearityType}")
            if stepname == 'linearFit':
                axs[0].set_xlabel("Input Abscissa (time or mondiode)")
                axs[0].set_ylabel("Input Ordinate (flux)")
                axs[1].set_xlabel("Linear Ordinate (linear flux)")
                axs[1].set_ylabel("Flux Difference: (input - linear)")
            elif stepname in ('polyFit', 'splineFit'):
                axs[0].set_xlabel("Linear Abscissa (linear flux)")
                axs[0].set_ylabel("Input Ordinate (flux)")
                axs[1].set_xlabel("Linear Ordinate (linear flux)")
                axs[1].set_ylabel("Flux Difference: (input - full model fit)")
            elif stepname == 'solution':
                axs[0].set_xlabel("Input Abscissa (time or mondiode)")
                axs[0].set_ylabel("Linear Ordinate (linear flux)")
                axs[1].set_xlabel("Model flux (linear flux)")
                axs[1].set_ylabel("Flux Difference: (linear - model)")

            axs[0].set_yscale('log')
            axs[0].set_xscale('log')
            axs[0].scatter(xVector, yVector)
            axs[0].scatter(xVector[~mask], yVector[~mask], c='red', marker='x')
            axs[1].set_xscale('log')

            axs[1].scatter(yModel, yVector[mask] - yModel)
            fig.tight_layout()
            fig.show()

            prompt = "Press Enter or c to continue [chpx]..."
            while True:
                ans = input(prompt).lower()
                if ans in ("", " ", "c",):
                    break
                elif ans in ("p", ):
                    import pdb
                    pdb.set_trace()
                elif ans in ("h", ):
                    print("[h]elp [c]ontinue [p]db")
                elif ans in ('x', ):
                    exit()
            plt.close()


class LinearityDoubleSplineSolveConnections(
    pipeBase.PipelineTaskConnections,
    dimensions=("instrument", "detector"),
):
    dummy = cT.Input(
        name="raw",
        doc="Dummy exposure.",
        storageClass='Exposure',
        dimensions=("instrument", "exposure", "detector"),
        multiple=True,
        deferLoad=True,
    )
    camera = cT.PrerequisiteInput(
        name="camera",
        doc="Camera Geometry definition.",
        storageClass="Camera",
        dimensions=("instrument", ),
        isCalibration=True,
    )
    inputLinearizerPtc = cT.Input(
        name="linearizerPtc",
        doc="Input linearizer PTC dataset.",
        storageClass="PhotonTransferCurveDataset",
        dimensions=("instrument", "detector"),
        isCalibration=True,
    )
    inputNormalization = cT.Input(
        name="cpLinearizerPtcNormalization",
        doc="Focal-plane normalization table.",
        storageClass="ArrowAstropy",
        dimensions=("instrument",),
        isCalibration=True,
    )
    inputBinnedImagesHandles = cT.Input(
        name="cpPtcPairBinned",
        doc="Tabulated binned exposure pairs.",
        storageClass="ArrowAstropy",
        dimensions=("instrument", "exposure", "detector"),
        multiple=True,
        deferLoad=True,
    )
    outputLinearizer = cT.Output(
        name="linearity",
        doc="Output linearity measurements.",
        storageClass="Linearizer",
        dimensions=("instrument", "detector"),
        isCalibration=True,
    )

    def __init__(self, *, config=None):
        super().__init__(config=config)

        if not config.useFocalPlaneNormalization:
            del self.inputNormalization


class LinearityDoubleSplineSolveConfig(
    pipeBase.PipelineTaskConfig,
    pipelineConnections=LinearityDoubleSplineSolveConnections,
):
    maxFracLinearityDeviation = pexConfig.Field(
        dtype=float,
        doc="Maximum fraction deviation from raw linearity to compute "
            "linearityTurnoff and linearityMaxSignal.",
        # TODO: DM-46811 investigate if this can be raised to 0.05.
        default=0.01,
    )
    minSignalFitLinearityTurnoff = pexConfig.Field(
        dtype=float,
        doc="Minimum signal to compute raw linearity slope for linearityTurnoff.",
        default=1000.0,
    )
    maxLinearityTurnoffRelativeToPtcTurnoff = pexConfig.Field(
        dtype=float,
        doc="Maximum fractional allowed linearity turnoff relative to the PTC turnoff. Used "
            "to keep extra-high odd values from contaminating the fit.",
        default=1.3,
    )
    maxNoiseReference = pexConfig.Field(
        dtype=float,
        doc="Maximum read noise (e-) in the PTC for an amp to be considered as a reference.",
        default=12.0,
    )
    usePhotodiode = pexConfig.Field(
        dtype=bool,
        doc="Use the photodiode info instead of the raw expTimes?",
        default=False,
    )
    minPhotodiodeCurrent = pexConfig.Field(
        dtype=float,
        doc="Minimum value to trust photodiode signals.",
        default=0.0,
    )
    doAutoGrouping = pexConfig.Field(
        dtype=bool,
        doc="Do automatic group detection? Cannot be True if splineGroupingColumn is also set. "
            "The automatic group detection will use the ratio of signal to exposure time (if "
            "autoGroupingUseExptime is True) or photodiode (if False) to determine which "
            "flat pairs were taken with different illumination settings.",
        default=False,
    )
    autoGroupingUseExptime = pexConfig.Field(
        dtype=bool,
        doc="Use exposure time to determine automatic grouping. Used if doAutoGrouping=True.",
        default=True,
    )
    autoGroupingThreshold = pexConfig.Field(
        dtype=float,
        doc="Minimum relative jump from sorted conversion values to determine a group.",
        default=0.1,
    )
    autoGroupingMaxSignalFraction = pexConfig.Field(
        dtype=float,
        doc="Only do auto-grouping when the signal is this fraction of the maximum signal. "
            "All exposures with signal higher than this threshold will be put into the "
            "largest signal group. This config is needed if the input PTC goes beyond "
            "the linearity turnoff.",
        default=0.9,
    )
    groupingColumn = pexConfig.Field(
        dtype=str,
        doc="Column to use for grouping together points, to allow "
            "for different proportionality constants. If None, then grouping will "
            "only be done if doAutoGrouping is True.",
        default=None,
        optional=True,
    )
    absoluteSplineMinimumSignalNode = pexConfig.Field(
        dtype=float,
        doc="Smallest node (above 0) for absolute spline (adu).",
        default=0.0,
    )
    absoluteSplineLowThreshold = pexConfig.Field(
        dtype=float,
        doc="Threshold for the low-level linearity nodes for absolute spline (adu). "
            "If this is below ``absoluteSplineMinimumSignalNode`` then the low "
            "level checks will be skipped.",
        default=0.0,
    )
    absoluteSplineLowNodeSize = pexConfig.Field(
        dtype=float,
        doc="Minimum size for low-level linearity nodes for absolute spline (adu).",
        default=2000.0,
    )
    absoluteSplineNodeSize = pexConfig.Field(
        dtype=float,
        doc="Minimum size for linearity nodes for absolute spline above absoluteSplineLowThreshold e(adu); "
            "note that there will always be a node at the reference PTC turnoff.",
        default=3000.0,
    )
    absoluteSplineFitMinIter = pexConfig.Field(
        dtype=int,
        doc="Minimum number of iterations for absolute spline fit.",
        default=3,
    )
    absoluteSplineFitMaxIter = pexConfig.Field(
        dtype=int,
        doc="Maximum number of iterations for absolute spline fit.",
        default=20,
    )
    absoluteSplineFitMaxRejectionPerIteration = pexConfig.Field(
        dtype=int,
        doc="Maximum number of rejections per iteration for absolute spline fit.",
        default=5,
    )
    absoluteNSigmaClipLinear = pexConfig.Field(
        dtype=float,
        doc="Sigma-clipping for absolute spline solution.",
        default=5.0,
    )
    doAbsoluteSplineFitOffset = pexConfig.Field(
        dtype=bool,
        doc="Fit a scattered light offset in the spline fit.",
        default=True,
    )
    doAbsoluteSplineFitWeights = pexConfig.Field(
        dtype=bool,
        doc="Fit linearity weight parameters in the spline fit.",
        default=False,
    )
    absoluteSplineFitWeightParsStart = pexConfig.ListField(
        dtype=float,
        doc="Starting parameters for weight fit, if doSplineFitWeights=True. "
            "Parameters are such that sigma = sqrt(par[0]**2. + par[1]**2./mu)."
            "If doSplineFitWeights=False then these are used as-is; otherwise "
            "they are used as the initial values for fitting these parameters.",
        length=2,
        default=[1.0, 0.0],
    )
    doAbsoluteSplineFitTemperature = pexConfig.Field(
        dtype=bool,
        doc="Fit temperature coefficient in spline fit?",
        default=False,
    )
    absoluteSplineFitTemperatureColumn = pexConfig.Field(
        dtype=str,
        doc="Name of the temperature column to use when fitting temperature "
            "coefficients in spline fit; this must not be None if "
            "doSplineFitTemperature is True.",
        default=None,
        optional=True,
    )
    doAbsoluteSplineFitTemporal = pexConfig.Field(
        dtype=bool,
        doc="Fit a linear temporal parameter coefficient in spline fit?",
        default=False,
    )
    useFocalPlaneNormalization = pexConfig.Field(
        dtype=bool,
        doc="Use focal-plane normalization in addition to/instead of photodiode? "
            "(Only used with for absolute spline fitting).",
        default=False,
    )
    relativeSplineReferenceCounts = pexConfig.Field(
        dtype=float,
        doc="Number of target counts (adu) to select a reference image for "
            "relative spline solution.",
        default=10000.0,
    )
    relativeSplineMinimumSignalNode = pexConfig.Field(
        dtype=float,
        doc="Smallest node (above 0) for relative spline (adu).",
        default=100.0,
    )
    relativeSplineLowThreshold = pexConfig.Field(
        dtype=float,
        doc="Threshold for the low-level linearity nodes for relative spline (adu)."
            "If this is below ``relativeSplineMinimumSignalNode`` then the low "
            "level checks will be skipped.",
        default=5000.0,
    )
    relativeSplineLowNodeSize = pexConfig.Field(
        dtype=float,
        doc="Minimum size for low-level linearity nodes for relative spline (adu).",
        default=750.0,
    )
    relativeSplineMidNodeSize = pexConfig.Field(
        dtype=float,
        doc="Minimum size for mid-level linearity nodes for relative spline (adu); "
            "this applies to counts between relativeSplineLowThreshold and the "
            "PTC turnoff.",
        default=5000.0,
    )
    relativeSplineHighNodeSize = pexConfig.Field(
        dtype=float,
        doc="Minimum size for high-level linearity nodes for relative spline (adu); "
            "this applies to counts between the PTC and linearity turnoffs.",
        default=2000.0,
    )
    relativeSplineFitMinIter = pexConfig.Field(
        dtype=int,
        doc="Minimum number of iterations for relative spline fit.",
        default=3,
    )
    relativeSplineFitMaxIter = pexConfig.Field(
        dtype=int,
        doc="Maximum number of iterations for relative spline fit.",
        default=20,
    )
    relativeSplineFitMaxRejectionPerIteration = pexConfig.Field(
        dtype=int,
        doc="Maximum number of rejections per iteration for relative spline fit.",
        default=5,
    )
    relativeNSigmaClipLinear = pexConfig.Field(
        dtype=float,
        doc="Sigma-clipping for relative spline solution.",
        default=5.0,
    )

    def validate(self):
        super().validate()

        if self.doAbsoluteSplineFitTemperature and self.absoluteSplineFitTemperatureColumn is None:
            raise ValueError(
                "Must set absoluteSplineFitTemperatureColumn if doAbsoluteSplineFitTemperature is True.",
            )

        if self.doAutoGrouping and self.groupingColumn is not None:
            raise ValueError("Must not set doAutoGrouping and also groupingColumn")
        if self.doAutoGrouping:
            if not self.autoGroupingUseExptime and not self.usePhotodiode:
                raise ValueError("If doAutoGrouping is True and autoGroupingUseExptime is False, then "
                                 "usePhotodiode must be True.")


class LinearityDoubleSplineSolveTask(pipeBase.PipelineTask):
    ConfigClass = LinearityDoubleSplineSolveConfig
    _DefaultName = "cpLinearityDoubleSplineSolve"

    def runQuantum(self, butlerQC, inputRefs, outputRefs):
        # docstring inherited
        inputs = butlerQC.get(inputRefs)

        if self.config.useFocalPlaneNormalization:
            inputNormalization = inputs["inputNormalization"]
        else:
            inputNormalization = None

        # Add calibration provenance info to header.
        kwargs = dict()
        reference = getattr(inputRefs, "inputLinearizerPtc", None)

        if reference is not None and hasattr(reference, "run"):
            runKey = "PTC_RUN"
            runValue = reference.run
            idKey = "PTC_UUID"
            idValue = str(reference.id)
            dateKey = "PTC_DATE"
            calib = inputs.get("inputPtc", None)
            dateValue = extractCalibDate(calib)

            kwargs[runKey] = runValue
            kwargs[idKey] = idValue
            kwargs[dateKey] = dateValue

            self.log.info("Using " + str(reference.run))

        outputs = self.run(
            inputPtc=inputs["inputLinearizerPtc"],
            camera=inputs["camera"],
            inputBinnedImagesHandles=inputs["inputBinnedImagesHandles"],
            inputNormalization=inputNormalization,
        )
        outputs.outputLinearizer.updateMetadata(setDate=False, **kwargs)

        butlerQC.put(outputs, outputRefs)

    def run(
        self,
        *,
        inputPtc,
        camera,
        inputBinnedImagesHandles,
        inputNormalization,
    ):
        """Fit the double-spline relative/absolute linearity correction.

        Parameters
        ----------
        inputPtc : `lsst.ip.isr.PtcDataset`
            Pre-measured PTC dataset.
        camera : `lsst.afw.cameraGeom.Camera`
            Camera geometry.
        inputBinnedImagesHandles : `list` [`DeferredDatasetHandle`]
            Handles for input binned image pairs.
        inputNormalization : `astropy.table.Table`, optional
            Focal plane normalization table to use if
            useFocalPlaneNormalization is True.

        Returns
        -------
        results : `lsst.pipe.base.Struct`
            The results struct containing:

            ``outputLinearizer``
                Final linearizer calibration (`lsst.ip.isr.Linearizer`).
            ``outputProvenance``
                Provenance data for the new calibration
                (`lsst.ip.isr.IsrProvenance`).
        """
        detector = camera[inputPtc.metadata["DETECTOR"]]

        binnedImagesHandleDict = {
            handle.dataId["exposure"]: handle for handle in inputBinnedImagesHandles
        }

        linearizer = Linearizer(detector=detector, log=self.log)
        linearizer.updateMetadataFromExposures([inputPtc])

        groupingValues = _determineInputGroups(
            inputPtc,
            self.config.doAutoGrouping,
            self.config.autoGroupingUseExptime,
            self.config.autoGroupingMaxSignalFraction,
            self.config.autoGroupingThreshold,
            self.config.groupingColumn,
            self.config.minPhotodiodeCurrent,
        )

        if self.config.doAbsoluteSplineFitTemperature:
            if self.config.absoluteSplineFitTemperatureColumn not in inputPtc.auxValues:
                raise ValueError(
                    "Config requests fitting temperature coefficient for "
                    f"{self.config.splineFitTemperatureColumn} but this column "
                    "is not available in inputPtc.auxValues.",
                )
                temperatureValues = inputPtc.auxValues[self.config.splineFitTemperatureColumn]
        else:
            temperatureValues = None

        # Fill the linearizer with empty values.
        firstAmp = None
        for ampName in inputPtc.ampNames:
            if ampName not in inputPtc.badAmps:
                firstAmp = ampName
                break
        if firstAmp is None:
            raise pipeBase.NoWorkFound("No valid amps in input PTC.")
        nExp = len(inputPtc.inputExpIdPairs[firstAmp]) * 2
        nAmp = len(inputPtc.ampNames)

        for amp in detector:
            ampName = amp.getName()

            linearizer.inputGain[ampName] = inputPtc.gain[ampName]
            linearizer.linearityType[ampName] = "None"
            linearizer.linearityCoeffs[ampName] = np.zeros(1)
            # This is not used; kept for compatibility.
            linearizer.linearityBBox[ampName] = amp.getBBox()
            linearizer.fitParams[ampName] = np.zeros(1)
            linearizer.fitParamsErr[ampName] = np.zeros(1)
            linearizer.fitChiSq[ampName] = np.nan
            linearizer.fitResiduals[ampName] = np.zeros(nExp)
            linearizer.fitResidualsSigmaMad[ampName] = np.nan
            linearizer.fitResidualsUnmasked[ampName] = np.zeros(nExp)
            linearizer.fitResidualsModel[ampName] = np.zeros(nExp)
            linearizer.linearFit[ampName] = np.zeros(2)
            linearizer.linearityTurnoff[ampName] = np.nan
            linearizer.linearityMaxSignal[ampName] = np.nan
            linearizer.inputMask[ampName] = np.zeros(nExp, dtype=np.bool_)
            linearizer.inputAbscissa[ampName] = np.zeros(nExp)
            linearizer.inputOrdinate[ampName] = np.zeros(nExp)
            linearizer.inputGroupingIndex[ampName] = np.zeros(nExp, dtype=np.int64)
            linearizer.inputNormalization[ampName] = np.ones(nExp)

        linearizer.absoluteReferenceAmplifier = ""

        # Extract values in common, and per-amp.
        data = np.zeros(
            nExp,
            dtype=[
                ("exp_id", "i8"),
                ("exptime", "f8"),
                ("photocharge", "f8"),
                ("mjd", "f8"),
                ("raw_mean", ("f8", nAmp)),
                ("abscissa", "f8"),
                ("grouping", "i4"),
                # The following are computed in the relative scaling
                # measurements.
                ("ref_counts", "f8"),
                ("gain_ratio", ("f8", nAmp)),
            ],
        )

        data["exp_id"] = np.asarray(inputPtc.inputExpIdPairs[firstAmp]).ravel()
        data["exptime"] = np.repeat(inputPtc.rawExpTimes[firstAmp], 2)
        data["mjd"] = np.repeat(inputPtc.inputExpPairMjdStartList[firstAmp], 2)
        data["photocharge"] = np.repeat(inputPtc.photoCharges[firstAmp], 2)
        data["photocharge"][::2] -= inputPtc.photoChargeDeltas[firstAmp] / 2.
        data["photocharge"][1::2] += inputPtc.photoChargeDeltas[firstAmp] / 2.
        data["grouping"] = np.repeat(groupingValues, 2)

        for i, amp in enumerate(detector):
            ampName = amp.getName()

            data["raw_mean"][:, i] = np.repeat(inputPtc.rawMeans[ampName], 2)
            data["raw_mean"][::2, i] -= inputPtc.rawDeltas[ampName] / 2.
            data["raw_mean"][1::2, i] += inputPtc.rawDeltas[ampName] / 2.

        if self.config.usePhotodiode:
            data["abscissa"][:] = data["photocharge"]

            data["abscissa"][data["photocharge"] < self.config.minPhotodiodeCurrent] = np.nan
        else:
            data["abscissa"][:] = data["exptime"]

        # Normalize if configured.
        inputNorm = np.ones(nExp, dtype=np.float64)
        if self.config.useFocalPlaneNormalization:
            a, b = esutil.numpy_util.match(data["exp_id"], inputNormalization["exposure"])
            inputNorm[a] = inputNormalization["normalization"][b]
            data["abscissa"] *= inputNorm

        postTurnoffMasks = {}

        # Compute linearity turnoff for each amp.
        for i, amp in enumerate(detector):
            ampName = amp.getName()

            if ampName in inputPtc.badAmps:
                self.log.warning(
                    "Amp %s in detector %s has no usable PTC information. Skipping!",
                    ampName,
                    detector.getName(),
                )
                continue

            mask = np.isfinite(data["raw_mean"][:, i])

            turnoffMask = np.repeat(inputPtc.expIdMask[ampName], 2)
            turnoffMask &= mask

            _, turnoff, maxSignal, goodPoints = _computeTurnoffAndMax(
                data["abscissa"],
                data["raw_mean"][:, i],
                turnoffMask,
                data["grouping"],
                ampName=ampName,
                minSignalFitLinearityTurnoff=self.config.minSignalFitLinearityTurnoff,
                maxFracLinearityDeviation=self.config.maxFracLinearityDeviation,
                log=self.log,
                maxTurnoff=inputPtc.ptcTurnoff[ampName] * self.config.maxLinearityTurnoffRelativeToPtcTurnoff,
            )

            # Use the goodPoints as an initial estimate of the mask
            # above the turnoff. But we only want to maintain the
            # "high end" outliers.
            postTurnoffMask = goodPoints
            postTurnoffMask[data["raw_mean"][:, i] < np.median(data["raw_mean"][goodPoints, i])] = True
            postTurnoffMasks[ampName] = postTurnoffMask

            if np.isnan(turnoff):
                # This is a bad amp, with no linearizer.
                self.log.warning(
                    "Amp %s in detector %s has no usable linearizer information. Skipping!",
                    ampName,
                    detector.getName(),
                )
                continue

            linearizer.linearityTurnoff[ampName] = turnoff
            linearizer.linearityMaxSignal[ampName] = maxSignal

            self.log.info("Amplifier %s has a linearity turnoff of %.2f adu.", ampName, turnoff)

        # Choose the reference amplifier as the one with the largest
        # turnoff. This ensures that the absolute fit covers the full
        # range. We additionally confirm that the ptc turnoff is
        # finite for this amplifier.
        turnoffArray = np.asarray([linearizer.linearityTurnoff[ampName] for ampName in inputPtc.ampNames])
        # This is a possibly redundant check to make sure that a bad amp is
        # not chosen as a reference amp. We also check that a high noise
        # amp is not chosen as the reference amp.
        for i, ampName in enumerate(inputPtc.ampNames):
            if ampName in inputPtc.badAmps \
               or not np.isfinite(inputPtc.ptcTurnoff[ampName]) \
               or inputPtc.noise[ampName] > self.config.maxNoiseReference:
                turnoffArray[i] = np.nan

        if np.all(~np.isfinite(turnoffArray)):
            # Return the default blank linearizer.
            linearizer.hasLinearity = True
            linearizer.validate()
            linearizer.updateMetadata(camera=camera, detector=detector, filterName='NONE')
            linearizer.updateMetadata(setDate=True, setCalibId=True)
            linearizer.updateMetadataFromExposures([inputPtc])
            provenance = IsrProvenance(calibType='linearizer')

            return pipeBase.Struct(
                outputLinearizer=linearizer,
                outputProvenance=provenance,
            )

        refAmpIndex = np.argmax(np.nan_to_num(turnoffArray))
        refAmpName = inputPtc.ampNames[refAmpIndex]
        linearizer.absoluteReferenceAmplifier = refAmpName

        # Choose a reference image.
        refExpIndex = np.argmin(
            np.abs(
                np.nan_to_num(data["raw_mean"][:, refAmpIndex]) - self.config.relativeSplineReferenceCounts
            )
        )
        refExpId = data["exp_id"][refExpIndex]

        self.log.info(
            "Using exposure %d (%.2f adu in amp %s) as reference.",
            refExpId,
            data["raw_mean"][refExpIndex, refAmpIndex],
            refAmpName,
        )

        # We need to know if the reference exposure is the first or second
        # in the pair, because the binned are pairs.
        offset = refExpIndex % 2

        refBinned = binnedImagesHandleDict[data["exp_id"][refExpIndex - offset]].get()
        refBinned = copy.copy(refBinned)
        if offset == 0:
            refBinned["value"] = refBinned["value1"]
        else:
            refBinned["value"] = refBinned["value2"]

        # Scale reference according to reference amplifier.
        refScaling = np.median(refBinned["value"][refBinned["amp_index"] == refAmpIndex])
        refBinned["value"] /= refScaling

        # Get the invidual amp scalings.
        # These are the relative gains for the reference image.
        ampScalings = np.asarray(
            [
                np.median(refBinned["value"][refBinned["amp_index"] == ampIndex])
                for ampIndex in range(nAmp)
            ],
        )

        # Compute the gain ratios for every exposure.
        # The binned images are stored as pairs.
        self.log.info("Computing gain ratios for %d exposures.", len(data))
        for i in range(len(data)):
            expId = data["exp_id"][i]
            if (i % 2) == 0:
                binned = binnedImagesHandleDict[expId].get()
                binned["value"] = binned["value1"]
            else:
                binned["value"] = binned["value2"]

            binned["value"] /= refBinned["value"]

            gainRatios = np.asarray(
                [
                    np.median(binned["value"][binned["amp_index"] == ampIndex])
                    for ampIndex in range(nAmp)
                ]
            )
            ref_counts = gainRatios[refAmpIndex]
            gainRatios /= ref_counts

            data["ref_counts"][i] = ref_counts
            data["gain_ratio"][i, :] = gainRatios

        # We need to know which group has the largest size.
        groupAmplitudes = np.zeros(len(np.unique(data["grouping"])))
        for g in range(len(groupAmplitudes)):
            use = (data["grouping"] == g)
            groupAmplitudes[g] = np.nanmax(data["ref_counts"][use]) - np.nanmin(data["ref_counts"][use])
        maxAmplitudeGroup = np.argmax(groupAmplitudes)

        self.log.info("Illumination group %d has the largest signal amplitude.", maxAmplitudeGroup)

        # Compute relative linearization first.
        maxRelNodes = 0

        for i, amp in enumerate(detector):
            if i == refAmpIndex:
                continue

            ampName = amp.getName()

            ptcTurnoff = inputPtc.ptcTurnoff[ampName]
            linearityTurnoff = linearizer.linearityTurnoff[ampName]

            if not np.isfinite(ptcTurnoff) or not np.isfinite(linearityTurnoff):
                # This is a bad amp; skip it.
                continue

            if ptcTurnoff < self.config.relativeSplineLowThreshold:
                lowThreshold = 0.0
            else:
                lowThreshold = self.config.relativeSplineLowThreshold

            relAbscissa = data["ref_counts"] * ampScalings[i]
            relOrdinate = data["ref_counts"] * data["gain_ratio"][:, i] * ampScalings[i]

            # The mask here must exclude everything beyond the turnoff.
            # Note that we need to do this before we use the actual
            # turnoff to compute the nodes to avoid nodes going past the
            # data domain.
            relMask = (
                np.isfinite(relAbscissa)
                & np.isfinite(relOrdinate)
                & (relOrdinate < linearityTurnoff)
            )

            # Make sure that the linearity turnoff used here does not
            # go beyond the max value of the relOrdinate
            relTurnoff = min(linearityTurnoff, np.max(relOrdinate[relMask]))

            relNodes = _noderator(
                lowThreshold,
                ptcTurnoff,
                relTurnoff,
                self.config.relativeSplineMinimumSignalNode,
                self.config.relativeSplineLowNodeSize,
                self.config.relativeSplineMidNodeSize,
                self.config.relativeSplineHighNodeSize,
            )

            self.log.info(
                "Relative linearity for amplifier %s using %d nodes from %.2f to %.2f counts.",
                ampName,
                len(relNodes),
                relNodes[0],
                relNodes[-1],
            )

            # Update the number of relative nodes to concatenation.
            if len(relNodes) > maxRelNodes:
                maxRelNodes = len(relNodes)

            linearizer.inputMask[ampName] = relMask.copy()
            linearizer.inputAbscissa[ampName] = relAbscissa.copy()
            linearizer.inputOrdinate[ampName] = relOrdinate.copy()
            linearizer.inputGroupingIndex[ampName] = data["grouping"].copy()
            linearizer.inputNormalization[ampName] = np.ones_like(relAbscissa)

            fitter = AstierSplineLinearityFitter(
                relNodes,
                data["grouping"],
                relAbscissa,
                relOrdinate,
                mask=relMask,
                fit_offset=False,
                fit_weights=False,
                fit_temperature=False,
                max_signal_nearly_linear=ptcTurnoff,
                fit_temporal=False,
                # Put a cap on the maximum correction in absolute value.
                max_frac_correction=np.inf,
                max_correction=10_000.0,
            )
            p0 = fitter.estimate_p0()
            pars = fitter.fit(
                p0,
                min_iter=self.config.relativeSplineFitMinIter,
                max_iter=self.config.relativeSplineFitMaxIter,
                max_rejection_per_iteration=self.config.relativeSplineFitMaxRejectionPerIteration,
                n_sigma_clip=self.config.relativeNSigmaClipLinear,
            )

            # Confirm that the first parameter is 0, and set it to
            # exactly zero.
            relValues = pars[fitter.par_indices["values"]]
            if not np.isclose(relValues[0], 0):
                raise RuntimeError("Programmer error! First spline parameter must "
                                   "be consistent with zero.")
            relValues[0] = 0.0

            if np.any(np.abs(pars[fitter.par_indices["values"]]) > 10_000.0):
                self.log.warning("Unconstrained nodes escaped containment; clipping.")
                lo = (pars[fitter.par_indices["values"]] < -10_000.0)
                if np.sum(lo) > 0:
                    pars[fitter.par_indices["values"][lo]] = -10_000.0
                hi = (pars[fitter.par_indices["values"]] > 10_000.0)
                if np.sum(hi) > 0:
                    pars[fitter.par_indices["values"][hi]] = 10_000.0

            # We adjust the node values according to the slope of the
            # group with the largest amplitude.  This removes a degeneracy
            # in the normalization and ensures that the overall linearized
            # correction is as close to the reference as possible.
            relValues -= (1.0 - pars[fitter.par_indices["groups"][maxAmplitudeGroup]]) * relNodes

            relChisq = fitter.compute_chisq_dof(pars)

            # Our reference fit is always 1.0 slope.
            relLinearFit = np.array([0.0, 1.0])

            # Adjust the abscissa for different groups for residuals.
            for j, groupIndex in enumerate(fitter.group_indices):
                relAbscissa[groupIndex] *= (pars[fitter.par_indices["groups"][j]] / relLinearFit[1])

            relMask = fitter.mask

            # Record values in the linearizer.
            linearizer.linearityType[ampName] = "DoubleSpline"
            # Note that we have a placeholder for the number of nodes in
            # the absolute spline.
            linearizer.linearityCoeffs[ampName] = np.concatenate([[len(relNodes), 0], relNodes, relValues])
            linearizer.fitChiSq[ampName] = relChisq
            linearizer.linearFit[ampName] = relLinearFit

            # Compute residuals.
            spl = Akima1DInterpolator(relNodes, relValues, method="akima")
            relOffset = spl(np.clip(relOrdinate, relNodes[0], relNodes[-1]))
            relModel = relOrdinate - relOffset

            if relMask.sum() < 2:
                self.log.warning("Amp %s in detector %s has not enough points in linear ordinate "
                                 "for residuals. Skipping!", ampName, detector.getName())
                relResiduals = np.full_like(relModel, np.nan)
                relResidualsUnmasked = relResiduals.copy()
            else:
                postLinearFit, _, _, _ = irlsFit(
                    relLinearFit,
                    relAbscissa[relMask],
                    relModel[relMask],
                    funcPolynomial,
                )
                # When computing residuals, we only care about the slope of
                # the postLinearFit and not the intercept. The intercept
                # itself depends on a possibly unknown zero in the abscissa
                # (often photodiode) which may have an arbitrary value.
                relResiduals = relModel - (postLinearFit[1] * relAbscissa)
                relResidualsUnmasked = relResiduals.copy()
                # We set masked residuals to nan.
                relResiduals[~relMask] = np.nan

            linearizer.fitResidualsUnmasked[ampName] = relResidualsUnmasked
            linearizer.fitResiduals[ampName] = relResiduals
            linearizer.fitResidualsModel[ampName] = relModel.copy()

            finite = np.isfinite(relResiduals)
            if finite.sum() == 0:
                sigmad = np.nan
            else:
                sigmad = median_abs_deviation(relResiduals[finite]/relOrdinate[finite], scale="normal")
            linearizer.fitResidualsSigmaMad[ampName] = sigmad

        # Now compute absolute linearization.

        if temperatureValues is not None:
            temperatureValuesScaled = temperatureValues - np.median(temperatureValues)
        else:
            temperatureValuesScaled = None

        if self.config.doAbsoluteSplineFitTemporal:
            inputMjdScaled = data["mjd"].copy()
            inputMjdScaled -= np.nanmedian(inputMjdScaled)
        else:
            inputMjdScaled = None

        absAbscissa = data["abscissa"].copy()
        absOrdinate = data["ref_counts"].copy()

        # These are guaranteed to be finite (as checked previously).
        absPtcTurnoff = inputPtc.ptcTurnoff[refAmpName]
        absLinearityTurnoff = linearizer.linearityTurnoff[refAmpName]

        if absPtcTurnoff < self.config.absoluteSplineLowThreshold:
            lowThreshold = 0.0
        else:
            lowThreshold = self.config.absoluteSplineLowThreshold

        # The mask here must exclude everything beyond the turnoff.
        # Note that we need to do this before we use the actual
        # turnoff to compute the nodes to avoid nodes going past the
        # data domain.
        absMask = postTurnoffMasks[refAmpName] & np.isfinite(absAbscissa) & np.isfinite(absOrdinate)

        absLinearityTurnoff = min(absLinearityTurnoff, np.max(absOrdinate[absMask]))

        absNodes = _noderator(
            lowThreshold,
            absPtcTurnoff,
            absLinearityTurnoff,
            self.config.absoluteSplineMinimumSignalNode,
            self.config.absoluteSplineLowNodeSize,
            # The medium and high are matched for absolute spline.
            self.config.absoluteSplineNodeSize,
            self.config.absoluteSplineNodeSize,
        )

        self.log.info("Absolute linearity for using %d nodes.", len(absNodes))

        # We store the absolute residuals with the reference amplifier.
        linearizer.linearityType[refAmpName] = "DoubleSpline"
        linearizer.inputMask[refAmpName] = absMask.copy()
        linearizer.inputAbscissa[refAmpName] = absAbscissa.copy()
        linearizer.inputOrdinate[refAmpName] = absOrdinate.copy()
        linearizer.inputGroupingIndex[refAmpName] = data["grouping"].copy()
        linearizer.inputNormalization[refAmpName] = inputNorm.copy()

        fitter = AstierSplineLinearityFitter(
            absNodes,
            data["grouping"].copy(),
            absAbscissa,
            absOrdinate,
            mask=absMask,
            log=self.log,
            fit_offset=self.config.doAbsoluteSplineFitOffset,
            fit_weights=self.config.doAbsoluteSplineFitWeights,
            weight_pars_start=self.config.absoluteSplineFitWeightParsStart,
            fit_temperature=self.config.doAbsoluteSplineFitTemperature,
            temperature_scaled=temperatureValuesScaled,
            max_signal_nearly_linear=absPtcTurnoff,
            fit_temporal=self.config.doAbsoluteSplineFitTemporal,
            mjd_scaled=inputMjdScaled,
        )
        p0 = fitter.estimate_p0()
        pars = fitter.fit(
            p0,
            min_iter=self.config.absoluteSplineFitMinIter,
            max_iter=self.config.absoluteSplineFitMaxIter,
            max_rejection_per_iteration=self.config.absoluteSplineFitMaxRejectionPerIteration,
            n_sigma_clip=self.config.absoluteNSigmaClipLinear,
        )

        # Confirm that the first parameter is 0, and set it to
        # exactly zero.
        absValues = pars[fitter.par_indices["values"]]
        if not np.isclose(absValues[0], 0):
            raise RuntimeError("Programmer error! First spline parameter must "
                               "be consistent with zero.")
        absValues[0] = 0.0

        # We need a place to store this.
        linearizer.fitChiSq[refAmpName] = fitter.compute_chisq_dof(pars)

        absLinearFit = np.array([0.0, np.mean(pars[fitter.par_indices["groups"]])])

        # We must modify the inputOrdinate according to the
        # nuisance terms in the linearity fit for the residual
        # computation code to work properly.
        # The true mu (inputOrdinate) is given by
        #  mu = mu_in * (1 + alpha*t_scale) * (1 + beta*mjd_scale)
        if self.config.doAbsoluteSplineFitTemperature:
            absOrdinate *= (1.0
                            + pars[fitter.par_indices["temperature_coeff"]]*temperatureValuesScaled)
        if self.config.doAbsoluteSplineFitTemporal:
            absOrdinate *= (1.0
                            + pars[fitter.par_indices["temporal_coeff"]]*inputMjdScaled)

        # Adjust the abscissa for different groups for residuals.
        for j, groupIndex in enumerate(fitter.group_indices):
            absAbscissa[groupIndex] *= (pars[fitter.par_indices["groups"][j]] / absLinearFit[1])
        # And remove the offset term.
        if self.config.doAbsoluteSplineFitOffset:
            absOrdinate -= pars[fitter.par_indices["offset"]]

        absMask = fitter.mask

        # Compute residuals.
        spl = Akima1DInterpolator(absNodes, absValues, method="akima")
        absOffset = spl(np.clip(absOrdinate, absNodes[0], absNodes[-1]))
        absModel = absOrdinate - absOffset

        if absMask.sum() < 2:
            self.log.warning("Detector %s has not enough points in linear ordinate "
                             "for residuals. Skipping!", detector.getName())
            # We have to KICK OUT HERE something is VERY wrong.
            absResiduals = np.full_like(absModel, np.nan)
            absResidualsUnmasked = relResiduals.copy()
        else:
            postLinearFit, _, _, _ = irlsFit(
                absLinearFit,
                absAbscissa[absMask],
                absModel[absMask],
                funcPolynomial,
            )
            # When computing residuals, we only care about the slope of
            # the postLinearFit and not the intercept. The intercept
            # itself depends on a possibly unknown zero in the abscissa
            # (often photodiode) which may have an arbitrary value.
            absResiduals = absModel - (postLinearFit[1] * absAbscissa)
            absResidualsUnmasked = absResiduals.copy()
            # We set masked residuals to nan.
            absResiduals[~absMask] = np.nan

        linearizer.fitResidualsUnmasked[refAmpName] = absResidualsUnmasked
        linearizer.fitResiduals[refAmpName] = absResiduals
        linearizer.fitResidualsModel[refAmpName] = absModel.copy()

        finite = np.isfinite(absResiduals)
        if finite.sum() == 0:
            sigmad = np.nan
        else:
            sigmad = median_abs_deviation(absResiduals[finite]/absOrdinate[finite], scale="normal")
        linearizer.fitResidualsSigmaMad[refAmpName] = sigmad

        # Record the absolute nodes and values in each individual amplifier,
        # along with extra padding for alignment.
        nAbsNodes = len(absNodes)
        for i, amp in enumerate(detector):
            ampName = amp.getName()

            coeffs = np.zeros(2 * nAbsNodes + 2 * maxRelNodes + 2)
            if ampName == refAmpName:
                # The reference amplifier only has the absolute spline.
                coeffs[1] = nAbsNodes
                coeffs[2: 2 + 2 * nAbsNodes] = np.concatenate([absNodes, absValues])
            else:
                nRelNodes = int(linearizer.linearityCoeffs[ampName][0])

                coeffs = np.zeros(2 * nAbsNodes + 2 * maxRelNodes + 2)
                coeffs[0] = nRelNodes
                coeffs[1] = nAbsNodes
                relStart = 2
                relEnd = relStart + 2 * nRelNodes
                coeffs[relStart: relEnd] = linearizer.linearityCoeffs[ampName][relStart: relEnd]
                absStart = relEnd
                absEnd = absStart + 2 * nAbsNodes
                coeffs[absStart: absEnd] = np.concatenate([absNodes, absValues])

            linearizer.linearityCoeffs[ampName] = coeffs

        linearizer.hasLinearity = True
        linearizer.validate()
        linearizer.updateMetadata(camera=camera, detector=detector, filterName='NONE')
        linearizer.updateMetadata(setDate=True, setCalibId=True)
        linearizer.updateMetadataFromExposures([inputPtc])
        provenance = IsrProvenance(calibType='linearizer')

        return pipeBase.Struct(
            outputLinearizer=linearizer,
            outputProvenance=provenance,
        )


def _determineInputGroups(
    ptc,
    doAutoGrouping,
    autoGroupingUseExptime,
    autoGroupingMaxSignalFraction,
    autoGroupingThreshold,
    groupingColumn,
    minPhotodiodeCurrent,
):
    """Determine input groups for linearity fit.

    If ``splineGroupingColumn`` is set, then grouping will be done
    based on this. Otherwise, if ``doAutoGrouping`` is False, then
    no grouping will be done. Finally, grouping will be done by measuring
    the ratio of signal to exposure time (if
    ``autoGroupingUseExptime`` is set; recommended) or photocharge.
    These are then clustered with a simple algorithm to split into groups.
    If the data was taking by varying exposure time at different
    illumination levels, this grouping is very robust as the clusters are
    very well separated.

    Parameters
    ----------
    ptc : `lsst.ip.isr.PhotonTransferCurveDataset`
        Input PTC to do grouping.
    doAutoGrouping : `bool`
        Do automatic grouping of pairs?
    autoGroupingUseExptime : `bool`
        Use exposure time for automatic grouping of pairs?
    autoGroupingMaxSignalFraction : `float`
        All exposures with signal higher than this threshold will
        be put into the largest signal group.
    autoGroupingThreshold : `float`
        Minimum relative jump from sorted values to determine a group.
    minPhotodiodeCurrent : `float`
        Minimum photodiode current if auto-grouping is used and
        autoGroupingUseExptime is False.
    splineGroupingColumn : `str` or `None`
        Column to be used for spline grouping (if doAutoGrouping is False).

    Returns
    -------
    groupingValues : `np.ndarray`
        Array of values that are unique for a given group.
    """
    nPair = np.asarray(ptc.inputExpIdPairs[ptc.ampNames[0]]).shape[0]
    groupingValues = np.zeros(nPair, dtype=np.int64)

    if not doAutoGrouping:
        if groupingColumn is not None:
            if groupingColumn not in ptc.auxValues:
                raise ValueError(f"Config requests grouping by {groupingColumn}, "
                                 "but this column is not available in ptc.auxValues.")

            uGroupValues = np.unique(ptc.auxValues[groupingColumn])
            for i, uGroupValue in enumerate(uGroupValues):
                groupingValues[ptc.auxValues[groupingColumn] == uGroupValue] = i
    else:
        means = np.zeros((nPair, len(ptc.ampNames)))
        exptimes = np.zeros_like(means)
        for i, ampName in enumerate(ptc.ampNames):
            means[:, i] = ptc.rawMeans[ampName] * ptc.gain[ampName]
            exptimes[:, i] = ptc.rawExpTimes[ampName]
        detMeans = np.nanmean(means, axis=1)
        detExptimes = np.nanmean(exptimes, axis=1)

        if autoGroupingUseExptime:
            abscissa = detExptimes
        else:
            abscissa = ptc.photoCharges[ptc.ampNames[0]].copy()
            # Set illegal photocharges to NaN.
            abscissa[abscissa < minPhotodiodeCurrent] = np.nan

        ratio = detMeans / abscissa
        ratio /= np.nanmedian(ratio)

        # Adjust those that are above threshold so they fall into the
        # largest group.
        above = (detMeans > autoGroupingMaxSignalFraction*np.nanmax(detMeans))
        maxIndex = np.argmax(detMeans[~above])
        ratio[above] = ratio[maxIndex]

        # The clustering of ratios into groups is performed with a simple
        # algorithm based on sorting and looking for the largest gaps.
        # See https://stackoverflow.com/a/18385795
        st = np.argsort(ratio)
        stratio = ratio[st]
        delta = stratio[1:] - stratio[0: -1]

        transitions, = np.where(delta > autoGroupingThreshold)
        # If there are no transitions then everything ends up in group 0.
        if len(transitions) > 0:
            ratioCuts = stratio[transitions] + autoGroupingThreshold/2.

            for i in range(len(transitions) + 1):
                if i == 0:
                    inGroup, = np.where(ratio < ratioCuts[i])
                elif i == len(transitions):
                    inGroup, = np.where(ratio > ratioCuts[i - 1])
                else:
                    inGroup, = np.where((ratio > ratioCuts[i - 1]) & (ratio < ratioCuts[i]))
                groupingValues[inGroup] = i

        # Ensure out-of-range photoCharges/exptimes are in their own group.
        # These are masked later on.
        groupingValues[~np.isfinite(abscissa)] = -1
        # And put the high ones in the max group.
        groupingValues[above] = groupingValues[maxIndex]

    return groupingValues


def _computeTurnoffAndMax(
    abscissa,
    ordinate,
    initialMask,
    groupingValues,
    ampName="UNKNOWN",
    minSignalFitLinearityTurnoff=1000.0,
    maxFracLinearityDeviation=0.01,
    log=None,
    maxTurnoff=np.inf,
):
    """Compute the turnoff and max signal.

    Parameters
    ----------
    abscissa : `np.ndarray`
        Input x values, either photoCharges or exposure times.
        These should be cleaned of any non-finite values.
    ordinate : `np.ndarray`
        Input y values, the raw mean values for the amp.
        These should be cleaned of any non-finite values.
    initialMask : `np.ndarray`
        Mask to use for initial fit (usually from PTC).
    groupingValues : `np.ndarray`
        Array of values that are used to group different fits.
    ampName : `str`, optional
        Amplifier name (used for logging).
    minSignalFitLinearityTurnoff : `float`, optional
        Minimum signal to cmpute raw linearity slope for linearityTurnoff.
    maxFracLinearityDeviation : `float`, optional
        Maximum fraction deviation from raw linearity to compute turnoff.
    log : `logging.Logger`, optional
        Log object.
    maxTurnoff : `float`, optional
        Maximum turnoff allowed (will be set above PTC turnoff).

    Returns
    -------
    turnoffIndex : `int`
        Fit turnoff index (keyed to raw input).
    turnoff : `float`
        Fit turnoff value.
    maxSignal : `float`
        Fit maximum signal value.
    goodPoints : `np.ndarray`
        Mask of good points used in turnoff fit.
    """
    if log is None:
        log = logging.getLogger(__name__)

    # Follow eo_pipe:
    # https://github.com/lsst-camera-dh/eo_pipe/blob/6afa546569f622b8d604921e248200481c445730/python/lsst/eo/pipe/linearityPlotsTask.py#L50
    # Replacing flux with abscissa, Ne with ordinate.

    # Fit a line with the y-intercept fixed to zero, using the
    # signal counts Ne as the variance in the chi-square, i.e.,
    # chi2 = sum( (ordinate - aa*abscissa)**2/ordinate )
    # Minimizing chi2 wrt aa, gives
    # aa = sum(abscissa) / sum(abscissa**2/ordinate)

    fitMask = initialMask.copy()
    fitMask[ordinate < minSignalFitLinearityTurnoff] = False
    fitMask[ordinate > maxTurnoff] = False
    fitMask[~np.isfinite(abscissa) | ~np.isfinite(ordinate)] = False
    goodPoints = fitMask.copy()

    gValues = np.unique(groupingValues)
    groupIndicesList = []
    for gValue in gValues:
        use, = np.where(groupingValues == gValue)
        groupIndicesList.append(use)

    found = False
    firstIteration = True
    while (fitMask.sum() >= 4) and not found:
        residuals = np.zeros_like(ordinate)

        abscissaMasked = abscissa.copy()
        abscissaMasked[~fitMask] = np.nan
        ordinateMasked = ordinate.copy()
        ordinateMasked[~fitMask] = np.nan

        for i, groupIndices in enumerate(groupIndicesList):
            num = np.nansum(abscissaMasked[groupIndices])
            denom = np.nansum(abscissaMasked[groupIndices]**2./ordinateMasked[groupIndices])

            if num == 0.0 or denom == 0.0:
                if firstIteration:
                    log.info(
                        "All points for %s were masked in linearity turnoff for group %d (first iteration).",
                        ampName,
                        i,
                    )
                    # We can try to recover this.
                    nTry = min(10, len(groupIndices))
                    num = np.nansum(abscissa[groupIndices][0: nTry])
                    denom = np.nansum(abscissa[groupIndices][0: nTry]**2./ordinate[groupIndices][0: nTry])
                    aa = num / denom
                else:
                    log.warning("All points masked in linearity turnoff for group %d.", i)
                    aa = np.nan
            else:
                aa = num / denom

            residuals[groupIndices] = (ordinate[groupIndices] - aa*abscissa[groupIndices]) / \
                ordinate[groupIndices]

        # Use the residuals to compute the turnoff.
        # Only subtract off the median from the previously estimated fitMask.
        residuals -= np.nanmedian(residuals[fitMask])

        goodPoints = (np.abs(residuals) < maxFracLinearityDeviation) & (ordinate < maxTurnoff)

        if goodPoints.sum() > 4:
            # This was an adequate fit.
            found = True
            turnoff = np.max(ordinate[goodPoints])
            turnoffIndex = np.where(np.isclose(ordinate, turnoff))[0][0]
        else:
            # This was a bad fit; remove the largest outlier.
            badIndex = np.argmax(np.abs(residuals)[fitMask])
            fitIndices, = np.nonzero(fitMask)
            fitMask[fitIndices[badIndex]] = False

        firstIteration = False

    if not found:
        # Could not find any reasonable value.
        log.warning(
            "Could not find a reasonable initial linear fit to compute linearity turnoff for "
            "amplifier %s; may need finer sampling of input data?",
            ampName,
        )
        if np.all(~fitMask):
            return -1, np.nan, np.nan, goodPoints

        turnoff = np.max(ordinate[fitMask])
        turnoffIndex = np.where(np.isclose(ordinate, turnoff))[0][0]

        residuals = np.zeros(len(ordinate))

    # Fit the maximum signal.
    if turnoffIndex == (len(residuals) - 1):
        # This is the last point; we can't do a fit.
        # This is not a warning because we do not actually need this
        # value in practice.
        log.info(
            "No linearity turnoff detected for amplifier %s; try to increase the signal range.",
            ampName,
        )
        maxSignal = ordinate[turnoffIndex]
    else:
        maxSignalInitial = np.nanmax(ordinate)

        highFluxPoints = (np.nan_to_num(ordinate)
                          > (1.0 - maxFracLinearityDeviation)*maxSignalInitial)
        maxSignal = np.median(ordinate[highFluxPoints])

    return turnoffIndex, turnoff, maxSignal, goodPoints


def _noderator(turnoff0, turnoff1, turnoff2, minNode, lowNodeSize, midNodeSize, highNodeSize):
    """The "noderator" node-finder.

    Parameters
    ----------
    turnoff0 : `float`
        Zeroth turnoff value (e.g. expectation of low-level
        non-linearity threshold) (adu).
    turnoff1 : `float`
        First turnoff value (e.g. ptc turnoff) (adu).
    turnoff2 : `float`
        Second turnoff value (e.g. linearity turnoff) (adu).
    minNode : `float`
        Location to place the first node after 0.0 (if this is <= 0.0
        it will be ignored) (adu).
    lowNodeSize : `float`
        Minimum node size in the low-level non-linearity regime
        (below turnoff0) (adu).
    midNodeSize : `float`
        Minimum node size in the mid-level non-linearity regime
        (between turnoff0 and turnoff1) (adu).
    highNodeSize : `float`
        Minimum node size in the high-level non-linearity regime
        (between turnoff1 and turnoff2) (adu).

    Returns
    -------
    nodes : `np.ndarray`
        Array of node values (adu).
    """
    if turnoff0 > minNode:
        # At least 2 nodes (edges) in the low signal regime.
        nNodesLow = np.clip(int(np.ceil((turnoff0 - minNode) / lowNodeSize)), 2, None)
        midStart = turnoff0
    else:
        nNodesLow = 0
        midStart = 0.0
    # At least 5 nodes (akima minimum) in the mid signal regime.
    nNodesMid = np.clip(int(np.ceil((turnoff1 - midStart) / midNodeSize)), 5, None)
    if turnoff2 > turnoff1:
        # At least 2 nodes (edges) in the high signal regime.
        nNodesHigh = np.clip(int(np.ceil((turnoff2 - turnoff1) / highNodeSize)), 3, None)
    else:
        nNodesHigh = 0
    nodesLow = np.linspace(minNode, turnoff0, nNodesLow)
    nodesMid = np.linspace(midStart, turnoff1, nNodesMid)
    nodesHigh = np.linspace(turnoff1, turnoff2, nNodesHigh)

    # Make sure we do not duplicate nodes when concatenating.
    nodeList = []
    if nNodesLow > 1:
        nodeList.append([0.0])
        nodeList.append(nodesLow[:-1])
    if nNodesMid > 1:
        nodeList.append(nodesMid)
    if nNodesHigh > 1:
        nodeList.append(nodesHigh[1:])
    return np.concatenate(nodeList)
