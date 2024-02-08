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
    maxLookupTableAdu = pexConfig.Field(
        dtype=int,
        doc="Maximum DN value for a LookupTable linearizer.",
        default=2**18,
    )
    maxLinearAdu = pexConfig.Field(
        dtype=float,
        doc="Maximum DN value to use to estimate linear term.",
        default=20000.0,
    )
    minLinearAdu = pexConfig.Field(
        dtype=float,
        doc="Minimum DN value to use to estimate linear term.",
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
        inputs['inputDims'] = dict(inputRefs.inputPtc.dataId.required)

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

        if self.config.linearityType == 'Spline':
            if self.config.splineGroupingColumn is not None:
                if self.config.splineGroupingColumn not in inputPtc.auxValues:
                    raise ValueError(f"Config requests grouping by {self.config.splineGroupingColumn}, "
                                     "but this column is not available in inputPtc.auxValues.")
                groupingValue = inputPtc.auxValues[self.config.splineGroupingColumn]
            else:
                groupingValue = np.ones(len(inputPtc.rawMeans[inputPtc.ampNames[0]]), dtype=int)
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

            if (len(inputPtc.expIdMask[ampName]) == 0) or self.config.ignorePtcMask:
                self.log.warning("Mask not found for %s in detector %s in fit. Using all points.",
                                 ampName, detector.getName())
                mask = np.ones(len(inputPtc.expIdMask[ampName]), dtype=bool)
            else:
                mask = inputPtc.expIdMask[ampName].copy()

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

            inputOrdinate = inputPtc.rawMeans[ampName].copy()

            mask &= (inputOrdinate < self.config.maxLinearAdu)
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
                # Sum[(S(mu_i) + mu_i)/(k_j * D_i) - 1]**2, where S(mu_i) is
                # an Akima Spline function of mu_i, the observed flat-pair
                # mean; D_j is the photo-diode measurement corresponding to
                # that flat-pair; and k_j is a constant of proportionality
                # which is over index j as it is allowed to
                # be different based on different photodiode settings (e.g.
                # CCOBCURR).

                # The fit has additional constraints to ensure that the spline
                # goes through the (0, 0) point, as well as a normalization
                # condition so that the average of the spline over the full
                # range is 0. The normalization ensures that the spline only
                # fits deviations from linearity, rather than the linear
                # function itself which is degenerate with the gain.

                nodes = np.linspace(0.0, np.max(inputOrdinate[mask]), self.config.splineKnots)

                fitter = AstierSplineLinearityFitter(
                    nodes,
                    groupingValue,
                    inputAbscissa,
                    inputOrdinate,
                    mask=mask,
                    log=self.log,
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

                linearityCoeffs = np.concatenate([nodes, pars[0: len(nodes)]])
                linearFit = np.array([0.0, np.mean(pars[len(nodes):])])

                # We modify the inputAbscissa according to the linearity fits
                # here, for proper residual computation.
                for j, group_index in enumerate(fitter.group_indices):
                    inputOrdinate[group_index] /= (pars[len(nodes) + j] / linearFit[1])

                linearOrdinate = linearFit[1] * inputOrdinate
                # For the spline fit, reuse the "polyFit -> fitParams"
                # field to record the linear coefficients for the groups.
                polyFit = pars[len(nodes):]
                polyFitErr = np.zeros_like(polyFit)
                chiSq = np.nan

                # Update mask based on what the fitter rejected.
                mask = fitter.mask
            else:
                polyFit = np.zeros(1)
                polyFitErr = np.zeros(1)
                chiSq = np.nan
                linearityCoeffs = np.zeros(1)

            linearizer.linearityType[ampName] = self.config.linearityType
            linearizer.linearityCoeffs[ampName] = linearityCoeffs
            linearizer.linearityBBox[ampName] = amp.getBBox()
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
                    [0.0, 100.0],
                    inputAbscissa[mask],
                    linearizeModel[mask],
                    funcPolynomial,
                )
                residuals = linearizeModel - (postLinearFit[0] + postLinearFit[1] * inputAbscissa)
                # We set masked residuals to nan.
                residuals[~mask] = np.nan

            linearizer.fitResiduals[ampName] = residuals

            self.debugFit(
                'solution',
                inputOrdinate[mask],
                linearOrdinate[mask],
                linearizeModel[mask],
                None,
                ampName,
            )

        linearizer.hasLinearity = True
        linearizer.validate()
        linearizer.updateMetadata(camera=camera, detector=detector, filterName='NONE')
        linearizer.updateMetadata(setDate=True, setCalibId=True)
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
            nEntries = fitOrder + 1
            pEntries = fitOrder + 1
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
        linearizer.linearityBBox[ampName] = amp.getBBox()
        linearizer.fitParams[ampName] = np.zeros(pEntries)
        linearizer.fitParamsErr[ampName] = np.zeros(pEntries)
        linearizer.fitChiSq[ampName] = np.nan
        linearizer.fitResiduals[ampName] = np.zeros(len(inputPtc.expIdMask[ampName]))
        linearizer.linearFit[ampName] = np.zeros(2)
        return linearizer

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
