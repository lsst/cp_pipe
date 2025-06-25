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

__all__ = ["LinearitySolveTask", "LinearitySolveConfig"]

import numpy as np
from scipy.stats import median_abs_deviation

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

    outputLinearizer = cT.Output(
        name="linearity",
        doc="Output linearity measurements.",
        storageClass="Linearizer",
        dimensions=("instrument", "detector"),
        isCalibration=True,
    )

    def __init__(self, *, config=None):
        if not config.applyPhotodiodeCorrection:
            del self.inputPhotodiodeCorrection

        if config.useLinearizerPtc:
            del self.inputPtc
        else:
            del self.inputLinearizerPtc


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
        doc="Do automatic group detection? Cannot be True if splineGroupingColumn is also set.",
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
            "largest signal group.",
        default=0.9,
    )
    splineGroupingColumn = pexConfig.Field(
        dtype=str,
        doc="Column to use for grouping together points for Spline mode, to allow "
            "for different proportionality constants. If not set, no grouping "
            "will be done.",
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
    useLinearizerPtc = pexConfig.Field(
        dtype=bool,
        doc="Use a linearizer ptc in a single pipeline?",
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
            inputPhotodiodeCorrection=None):
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
        inputPhotodiodeCorrection : `lsst.ip.isr.PhotodiodeCorrection`
            Pre-measured photodiode correction used in the case when
            applyPhotodiodeCorrection=True.
        camera : `lsst.afw.cameraGeom.Camera`
            Camera geometry.
        inputDims : `lsst.daf.butler.DataCoordinate` or `dict`
            DataIds to use to populate the output calibration.

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

        groupingValues = self._determineInputGroups(inputPtc)

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

            # Compute linearityTurnoff and linearitySignalMax.
            turnoffMask = inputPtc.expIdMask[ampName].copy()
            turnoffMask &= mask
            turnoffIndex, turnoff, maxSignal = self._computeTurnoffAndMax(
                inputAbscissa,
                inputPtc.rawMeans[ampName],
                turnoffMask,
                groupingValues,
                ampName=ampName,
            )
            linearizer.linearityTurnoff[ampName] = turnoff
            linearizer.linearityMaxSignal[ampName] = maxSignal

            inputOrdinate = inputPtc.rawMeans[ampName].copy()

            if self.config.linearityType != 'Spline':
                mask &= (inputOrdinate < self.config.maxLinearAdu)
            else:
                # For spline fits, cut above the turnoff.
                self.log.info("Using linearityTurnoff of %.4f adu for amplifier %s", turnoff, ampName)
                extraMask = np.ones(len(inputOrdinate), dtype=bool)
                extraMask[turnoffIndex + 1:] = False
                mask &= extraMask

            mask &= (inputOrdinate > self.config.minLinearAdu)

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
                #  mu = mu_in * (1 + alpha*t_scale)
                if self.config.doSplineFitTemperature:
                    inputOrdinate *= (1.0
                                      + pars[fitter.par_indices["temperature_coeff"]]*temperatureValuesScaled)
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
                ))
                polyFitErr = np.zeros_like(polyFit)
                chiSq = linearityChisq

                # Update mask based on what the fitter rejected.
                mask = fitter.mask
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
                # We set masked residuals to nan.
                residuals[~mask] = np.nan

            linearizer.fitResiduals[ampName] = residuals

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

        linearizer.linearityType[ampName] = "None"
        linearizer.linearityCoeffs[ampName] = np.zeros(nEntries)
        if self.config.trimmedState:
            linearizer.linearityBBox[ampName] = amp.getBBox()
        else:
            linearizer.linearityBBox[ampName] = amp.getRawBBox()
        linearizer.fitParams[ampName] = np.zeros(pEntries)
        linearizer.fitParamsErr[ampName] = np.zeros(pEntries)
        linearizer.fitChiSq[ampName] = np.nan
        linearizer.fitResiduals[ampName] = np.zeros(len(inputPtc.expIdMask[ampName]))
        linearizer.fitResidualsSigmaMad[ampName] = np.nan
        linearizer.linearFit[ampName] = np.zeros(2)
        linearizer.linearityTurnoff[ampName] = np.nan
        linearizer.linearityMaxSignal[ampName] = np.nan
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

    def _determineInputGroups(self, ptc):
        """Determine input groups for linearity fit.

        Parameters
        ----------
        ptc : `lsst.ip.isr.PhotonTransferCurveDataset`

        Returns
        -------
        groupingValues : `np.ndarray`
            Array of values that are unique for a given group.
        """
        nPair = np.asarray(ptc.inputExpIdPairs[ptc.ampNames[0]]).shape[0]
        groupingValues = np.zeros(nPair, dtype=np.int64)

        if not self.config.doAutoGrouping:
            if self.config.splineGroupingColumn is not None:
                if self.config.splineGroupingColumn not in ptc.auxValues:
                    raise ValueError(f"Config requests grouping by {self.config.splineGroupingColumn}, "
                                     "but this column is not available in ptc.auxValues.")

                uGroupValues = np.unique(ptc.auxValues[self.config.splineGroupingColumn])
                for i, uGroupValue in enumerate(uGroupValues):
                    groupingValues[ptc.auxValues[self.config.splineGroupingColumn] == uGroupValue] = i
        else:
            means = np.zeros((nPair, len(ptc.ampNames)))
            exptimes = np.zeros_like(means)
            for i, ampName in enumerate(ptc.ampNames):
                means[:, i] = ptc.rawMeans[ampName] * ptc.gain[ampName]
                exptimes[:, i] = ptc.rawExpTimes[ampName]
            detMeans = np.nanmean(means, axis=1)
            detExptimes = np.nanmean(exptimes, axis=1)

            if self.config.autoGroupingUseExptime:
                abscissa = detExptimes
            else:
                abscissa = ptc.photoCharges[ptc.ampNames[0]].copy()
                # Set illegal photocharges to NaN.
                abscissa[abscissa < self.config.minPhotodiodeCurrent] = np.nan

            ratio = detMeans / abscissa
            ratio /= np.nanmedian(ratio)

            # Adjust those that are above threshold so they fall into the
            # largest group.
            above = (detMeans > self.config.autoGroupingMaxSignalFraction*np.nanmax(detMeans))
            maxIndex = np.argmax(detMeans[~above])
            ratio[above] = ratio[maxIndex]

            st = np.argsort(ratio)
            stratio = ratio[st]
            delta = stratio[1:] - stratio[0: -1]

            transitions, = np.where(delta > self.config.autoGroupingThreshold)
            if len(transitions) > 0:
                ratioCuts = stratio[transitions] + self.config.autoGroupingThreshold/2.

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

    def _computeTurnoffAndMax(self, abscissa, ordinate, initialMask, groupingValues, ampName="UNKNOWN"):
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

        Returns
        -------
        turnoffIndex : `int`
            Fit turnoff index (keyed to raw input).
        turnoff : `float`
            Fit turnoff value.
        maxSignal : `float`
            Fit maximum signal value.
        """
        # Follow eo_pipe:
        # https://github.com/lsst-camera-dh/eo_pipe/blob/6afa546569f622b8d604921e248200481c445730/python/lsst/eo/pipe/linearityPlotsTask.py#L50
        # Replacing flux with abscissa, Ne with ordinate.

        # Fit a line with the y-intercept fixed to zero, using the
        # signal counts Ne as the variance in the chi-square, i.e.,
        # chi2 = sum( (ordinate - aa*abscissa)**2/ordinate )
        # Minimizing chi2 wrt aa, gives
        # aa = sum(abscissa) / sum(abscissa**2/ordinate)

        fitMask = initialMask.copy()
        fitMask[ordinate < self.config.minSignalFitLinearityTurnoff] = False

        gValues = np.unique(groupingValues)
        groupIndicesList = []
        for gValue in gValues:
            use, = np.where(groupingValues == gValue)
            groupIndicesList.append(use)

        found = False
        while (fitMask.sum() >= 4) and not found:
            residuals = np.zeros_like(ordinate)

            abscissaMasked = abscissa.copy()
            abscissaMasked[~fitMask] = np.nan
            ordinateMasked = ordinate.copy()
            ordinateMasked[~fitMask] = np.nan

            for groupIndices in groupIndicesList:
                num = np.nansum(abscissaMasked[groupIndices])
                denom = np.nansum(abscissaMasked[groupIndices]**2./ordinateMasked[groupIndices])
                aa = num / denom

                residuals[groupIndices] = (ordinate[groupIndices] - aa*abscissa[groupIndices]) / \
                    ordinate[groupIndices]

            # Use the residuals to compute the turnoff.
            residuals -= np.nanmedian(residuals)

            goodPoints = np.abs(residuals) < self.config.maxFracLinearityDeviation

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

        if not found:
            # Could not find any reasonable value.
            self.log.warning(
                "Could not find a reasonable initial linear fit to compute linearity turnoff for "
                "amplifier %s; may need finer sampling of input data?",
                ampName,
            )
            turnoff = np.max(ordinate[fitMask])
            turnoffIndex = np.where(np.isclose(ordinate, turnoff))[0][0]

            residuals = np.zeros(len(ordinate))

        # Fit the maximum signal.
        if turnoffIndex == (len(residuals) - 1):
            # This is the last point; we can't do a fit.
            self.log.warning(
                "No linearity turnoff detected for amplifier %s; try to increase the signal range.",
                ampName,
            )
            maxSignal = ordinate[turnoffIndex]
        else:
            maxSignalInitial = np.nanmax(ordinate)

            highFluxPoints = (np.nan_to_num(ordinate)
                              > (1.0 - self.config.maxFracLinearityDeviation)*maxSignalInitial)
            maxSignal = np.median(ordinate[highFluxPoints])

        return turnoffIndex, turnoff, maxSignal

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
