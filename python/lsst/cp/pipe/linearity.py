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
import lsst.afw.math as afwMath
import lsst.pipe.base as pipeBase
import lsst.pipe.base.connectionTypes as cT
import lsst.pex.config as pexConfig

from lsstDebug import getDebugFrame
from lsst.ip.isr import (Linearizer, IsrProvenance)

from .utils import (funcPolynomial, irlsFit)
from ._lookupStaticCalibration import lookupStaticCalibration


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
        lookupFunction=lookupStaticCalibration,
    )

    inputPtc = cT.Input(
        name="ptc",
        doc="Input PTC dataset.",
        storageClass="PhotonTransferCurveDataset",
        dimensions=("instrument", "detector"),
        isCalibration=True,
    )

    inputPhotodiodeData = cT.PrerequisiteInput(
        name="photodiode",
        doc="Photodiode readings data.",
        storageClass="IsrCalib",
        dimensions=("instrument", "exposure"),
        multiple=True,
        deferLoad=True,
        minimum=0,
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
        super().__init__(config=config)

        if config.applyPhotodiodeCorrection is not True:
            self.inputs.discard("inputPhotodiodeCorrection")

        if config.usePhotodiode is not True:
            self.inputs.discard("inputPhotodiodeData")


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
    polynomialOrder = pexConfig.Field(
        dtype=int,
        doc="Degree of polynomial to fit.",
        default=3,
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
    photodiodeIntegrationMethod = pexConfig.ChoiceField(
        dtype=str,
        doc="Integration method for photodiode monitoring data.",
        default="DIRECT_SUM",
        allowed={
            "DIRECT_SUM": ("Use numpy's trapz integrator on all photodiode "
                           "readout entries"),
            "TRIMMED_SUM": ("Use numpy's trapz integrator, clipping the "
                            "leading and trailing entries, which are "
                            "nominally at zero baseline level."),
        }
    )
    applyPhotodiodeCorrection = pexConfig.Field(
        dtype=bool,
        doc="Calculate and apply a correction to the photodiode readings?",
        default=False,
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
        butlerQC : `lsst.daf.butler.butlerQuantumContext.ButlerQuantumContext`
            Butler to operate on.
        inputRefs : `lsst.pipe.base.connections.InputQuantizedConnection`
            Input data refs to load.
        ouptutRefs : `lsst.pipe.base.connections.OutputQuantizedConnection`
            Output data refs to persist.
        """
        inputs = butlerQC.get(inputRefs)

        # Use the dimensions to set calib/provenance information.
        inputs['inputDims'] = inputRefs.inputPtc.dataId.byName()

        outputs = self.run(**inputs)
        butlerQC.put(outputs, outputRefs)

    def run(self, inputPtc, dummy, camera, inputDims, inputPhotodiodeData=None,
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
        inputPhotodiodeData : `dict` [`str`, `lsst.ip.isr.PhotodiodeCalib`]
            Photodiode readings data.
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

        if self.config.linearityType == 'Spline':
            fitOrder = self.config.splineKnots
        else:
            fitOrder = self.config.polynomialOrder

        # Initialize the linearizer.
        linearizer = Linearizer(detector=detector, table=table, log=self.log)
        linearizer.updateMetadataFromExposures([inputPtc])
        if self.config.usePhotodiode:
            # Compute the photodiode integrals once, outside the loop
            # over amps.
            monDiodeCharge = {}
            for handle in inputPhotodiodeData:
                expId = handle.dataId['exposure']
                pd_calib = handle.get()
                pd_calib.integrationMethod = self.config.photodiodeIntegrationMethod
                monDiodeCharge[expId] = pd_calib.integrate()[0]
            if self.config.applyPhotodiodeCorrection:
                abscissaCorrections = inputPhotodiodeCorrection.abscissaCorrections

        for i, amp in enumerate(detector):
            ampName = amp.getName()
            if ampName in inputPtc.badAmps:
                linearizer = self.fillBadAmp(linearizer, fitOrder, inputPtc, amp)
                self.log.warning("Amp %s in detector %s has no usable PTC information. Skipping!",
                                 ampName, detector.getName())
                continue

            if (len(inputPtc.expIdMask[ampName]) == 0) or self.config.ignorePtcMask:
                self.log.warning("Mask not found for %s in detector %s in fit. Using all points.",
                                 ampName, detector.getName())
                mask = np.repeat(True, len(inputPtc.expIdMask[ampName]))
            else:
                mask = np.array(inputPtc.expIdMask[ampName], dtype=bool)

            if self.config.usePhotodiode:
                modExpTimes = []
                for i, pair in enumerate(inputPtc.inputExpIdPairs[ampName]):
                    pair = pair[0]
                    modExpTime = 0.0
                    nExps = 0
                    for j in range(2):
                        expId = pair[j]
                        if expId in monDiodeCharge:
                            modExpTime += monDiodeCharge[expId]
                            nExps += 1
                    if nExps > 0:
                        modExpTime = modExpTime / nExps
                    else:
                        mask[i] = False

                    # Get the photodiode correction
                    if self.config.applyPhotodiodeCorrection:
                        try:
                            correction = abscissaCorrections[str(pair)]
                        except KeyError:
                            correction = 0.0
                    else:
                        correction = 0.0
                    modExpTimes.append(modExpTime + correction)
                inputAbscissa = np.array(modExpTimes)[mask]
            else:
                inputAbscissa = np.array(inputPtc.rawExpTimes[ampName])[mask]

            inputOrdinate = np.array(inputPtc.rawMeans[ampName])[mask]
            # Determine proxy-to-linear-flux transformation
            fluxMask = inputOrdinate < self.config.maxLinearAdu
            lowMask = inputOrdinate > self.config.minLinearAdu
            fluxMask = fluxMask & lowMask
            linearAbscissa = inputAbscissa[fluxMask]
            linearOrdinate = inputOrdinate[fluxMask]
            if len(linearAbscissa) < 2:
                linearizer = self.fillBadAmp(linearizer, fitOrder, inputPtc, amp)
                self.log.warning("Amp %s in detector %s has not enough points for linear fit. Skipping!",
                                 ampName, detector.getName())
                continue

            linearFit, linearFitErr, chiSq, weights = irlsFit([0.0, 100.0], linearAbscissa,
                                                              linearOrdinate, funcPolynomial)
            # Convert this proxy-to-flux fit into an expected linear flux
            linearOrdinate = linearFit[0] + linearFit[1] * inputAbscissa
            # Exclude low end outliers
            threshold = self.config.nSigmaClipLinear * np.sqrt(abs(linearOrdinate))
            fluxMask = np.abs(inputOrdinate - linearOrdinate) < threshold
            linearOrdinate = linearOrdinate[fluxMask]
            fitOrdinate = inputOrdinate[fluxMask]
            fitAbscissa = inputAbscissa[fluxMask]
            if len(linearOrdinate) < 2:
                linearizer = self.fillBadAmp(linearizer, fitOrder, inputPtc, amp)
                self.log.warning("Amp %s in detector %s has not enough points in linear ordinate. Skipping!",
                                 ampName, detector.getName())
                continue

            self.debugFit('linearFit', inputAbscissa, inputOrdinate, linearOrdinate, fluxMask, ampName)
            # Do fits
            if self.config.linearityType in ['Polynomial', 'Squared', 'LookupTable']:
                polyFit = np.zeros(fitOrder + 1)
                polyFit[1] = 1.0
                polyFit, polyFitErr, chiSq, weights = irlsFit(polyFit, linearOrdinate,
                                                              fitOrdinate, funcPolynomial)

                # Truncate the polynomial fit
                k1 = polyFit[1]
                linearityFit = [-coeff/(k1**order) for order, coeff in enumerate(polyFit)]
                significant = np.where(np.abs(linearityFit) > 1e-10, True, False)
                self.log.info("Significant polynomial fits: %s", significant)

                modelOrdinate = funcPolynomial(polyFit, fitAbscissa)

                self.debugFit('polyFit', linearAbscissa, fitOrdinate, modelOrdinate, None, ampName)

                if self.config.linearityType == 'Squared':
                    linearityFit = [linearityFit[2]]
                elif self.config.linearityType == 'LookupTable':
                    # Use linear part to get time at which signal is
                    # maxAduForLookupTableLinearizer DN
                    tMax = (self.config.maxLookupTableAdu - polyFit[0])/polyFit[1]
                    timeRange = np.linspace(0, tMax, self.config.maxLookupTableAdu)
                    signalIdeal = polyFit[0] + polyFit[1]*timeRange
                    signalUncorrected = funcPolynomial(polyFit, timeRange)
                    lookupTableRow = signalIdeal - signalUncorrected  # LinearizerLookupTable has correction

                    linearizer.tableData[tableIndex, :] = lookupTableRow
                    linearityFit = [tableIndex, 0]
                    tableIndex += 1
            elif self.config.linearityType in ['Spline']:
                # See discussion in `lsst.ip.isr.linearize.py` before
                # modifying.
                numPerBin, binEdges = np.histogram(linearOrdinate, bins=fitOrder)
                with np.errstate(invalid="ignore"):
                    # Algorithm note: With the counts of points per
                    # bin above, the next histogram calculates the
                    # values to put in each bin by weighting each
                    # point by the correction value.
                    values = np.histogram(linearOrdinate, bins=fitOrder,
                                          weights=(inputOrdinate[fluxMask] - linearOrdinate))[0]/numPerBin

                    # After this is done, the binCenters are
                    # calculated by weighting by the value we're
                    # binning over.  This ensures that widely
                    # spaced/poorly sampled data aren't assigned to
                    # the midpoint of the bin (as could be done using
                    # the binEdges above), but to the weighted mean of
                    # the inputs.  Note that both histograms are
                    # scaled by the count per bin to normalize what
                    # the histogram returns (a sum of the points
                    # inside) into an average.
                    binCenters = np.histogram(linearOrdinate, bins=fitOrder,
                                              weights=linearOrdinate)[0]/numPerBin
                    values = values[numPerBin > 0]
                    binCenters = binCenters[numPerBin > 0]

                self.debugFit('splineFit', binCenters, np.abs(values), values, None, ampName)
                interp = afwMath.makeInterpolate(binCenters.tolist(), values.tolist(),
                                                 afwMath.stringToInterpStyle("AKIMA_SPLINE"))
                modelOrdinate = linearOrdinate + interp.interpolate(linearOrdinate)
                self.debugFit('splineFit', linearOrdinate, fitOrdinate, modelOrdinate, None, ampName)

                # If we exclude a lot of points, we may end up with
                # less than fitOrder points.  Pad out the low-flux end
                # to ensure equal lengths.
                if len(binCenters) != fitOrder:
                    padN = fitOrder - len(binCenters)
                    binCenters = np.pad(binCenters, (padN, 0), 'linear_ramp',
                                        end_values=(binCenters.min() - 1.0, ))
                    # This stores the correction, which is zero at low values.
                    values = np.pad(values, (padN, 0))

                # Pack the spline into a single array.
                linearityFit = np.concatenate((binCenters.tolist(), values.tolist())).tolist()
                polyFit = [0.0]
                polyFitErr = [0.0]
                chiSq = np.nan
            else:
                polyFit = [0.0]
                polyFitErr = [0.0]
                chiSq = np.nan
                linearityFit = [0.0]

            linearizer.linearityType[ampName] = self.config.linearityType
            linearizer.linearityCoeffs[ampName] = np.array(linearityFit)
            linearizer.linearityBBox[ampName] = amp.getBBox()
            linearizer.fitParams[ampName] = np.array(polyFit)
            linearizer.fitParamsErr[ampName] = np.array(polyFitErr)
            linearizer.fitChiSq[ampName] = chiSq
            linearizer.linearFit[ampName] = linearFit
            residuals = fitOrdinate - modelOrdinate

            # The residuals only include flux values which are
            # not masked out. To be able to access this later and
            # associate it with the PTC flux values, we need to
            # fill out the residuals with NaNs where the flux
            # value is masked.

            # First convert mask to a composite of the two masks:
            mask[mask] = fluxMask
            fullResiduals = np.full(len(mask), np.nan)
            fullResiduals[mask] = residuals
            linearizer.fitResiduals[ampName] = fullResiduals
            image = afwImage.ImageF(len(inputOrdinate), 1)
            image.getArray()[:, :] = inputOrdinate
            linearizeFunction = linearizer.getLinearityTypeByName(linearizer.linearityType[ampName])
            linearizeFunction()(image,
                                **{'coeffs': linearizer.linearityCoeffs[ampName],
                                   'table': linearizer.tableData,
                                   'log': linearizer.log})
            linearizeModel = image.getArray()[0, :]

            self.debugFit('solution', inputOrdinate[fluxMask], linearOrdinate,
                          linearizeModel[fluxMask], None, ampName)

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
