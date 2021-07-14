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

import lsst.afw.image as afwImage
import lsst.afw.math as afwMath
import lsst.pipe.base as pipeBase
import lsst.pipe.base.connectionTypes as cT
import lsst.pex.config as pexConfig

from lsstDebug import getDebugFrame
from lsst.ip.isr import (Linearizer, IsrProvenance)

from .utils import (funcPolynomial, irlsFit)
from ._lookupStaticCalibration import lookupStaticCalibration

__all__ = ["LinearitySolveTask", "LinearitySolveConfig", "MeasureLinearityTask"]


class LinearitySolveConnections(pipeBase.PipelineTaskConnections,
                                dimensions=("instrument", "exposure", "detector")):
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
    inputPtc = cT.PrerequisiteInput(
        name="ptc",
        doc="Input PTC dataset.",
        storageClass="PhotonTransferCurveDataset",
        dimensions=("instrument", "detector"),
        isCalibration=True,
    )

    outputLinearizer = cT.Output(
        name="linearity",
        doc="Output linearity measurements.",
        storageClass="Linearizer",
        dimensions=("instrument", "detector"),
        isCalibration=True,
    )


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
        default=2000.0,
    )
    nSigmaClipLinear = pexConfig.Field(
        dtype=float,
        doc="Maximum deviation from linear solution for Poissonian noise.",
        default=5.0,
    )
    ignorePtcMask = pexConfig.Field(
        dtype=bool,
        doc="Ignore the values masked by PTC?",
        default=False,
    )


class LinearitySolveTask(pipeBase.PipelineTask, pipeBase.CmdLineTask):
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

    def run(self, inputPtc, dummy, camera, inputDims):
        """Fit non-linearity to PTC data, returning the correct Linearizer
        object.

        Parameters
        ----------
        inputPtc : `lsst.cp.pipe.PtcDataset`
            Pre-measured PTC dataset.
        camera : `lsst.afw.cameraGeom.Camera`
            Camera geometry.
        inputDims : `lsst.daf.butler.DataCoordinate` or `dict`
            DataIds to use to populate the output calibration.

        Returns
        -------
        results : `lsst.pipe.base.Struct`
            The results struct containing:

            ``outputLinearizer`` : `lsst.ip.isr.Linearizer`
                Final linearizer calibration.
            ``outputProvenance`` : `lsst.ip.isr.IsrProvenance`
                Provenance data for the new calibration.

        Notes
        -----
        This task currently fits only polynomial-defined corrections,
        where the correction coefficients are defined such that:
            corrImage = uncorrImage + sum_i c_i uncorrImage^(2 + i)
        These `c_i` are defined in terms of the direct polynomial fit:
            meanVector ~ P(x=timeVector) = sum_j k_j x^j
        such that c_(j-2) = -k_j/(k_1^j) in units of DN^(1-j) (c.f.,
        Eq. 37 of 2003.05978). The `config.polynomialOrder` or
        `config.splineKnots` define the maximum order of x^j to fit.
        As k_0 and k_1 are degenerate with bias level and gain, they
        are not included in the non-linearity correction.
        """
        if len(dummy) == 0:
            self.log.warn("No dummy exposure found.")

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

        for i, amp in enumerate(detector):
            ampName = amp.getName()
            if ampName in inputPtc.badAmps:
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
                self.log.warn("Amp %s has no usable PTC information.  Skipping!", ampName)
                continue

            if (len(inputPtc.expIdMask[ampName]) == 0) or self.config.ignorePtcMask:
                self.log.warn(f"Mask not found for {ampName} in non-linearity fit. Using all points.")
                mask = np.repeat(True, len(inputPtc.expIdMask[ampName]))
            else:
                mask = np.array(inputPtc.expIdMask[ampName], dtype=bool)

            inputAbscissa = np.array(inputPtc.rawExpTimes[ampName])[mask]
            inputOrdinate = np.array(inputPtc.rawMeans[ampName])[mask]

            # Determine proxy-to-linear-flux transformation
            fluxMask = inputOrdinate < self.config.maxLinearAdu
            lowMask = inputOrdinate > self.config.minLinearAdu
            fluxMask = fluxMask & lowMask
            linearAbscissa = inputAbscissa[fluxMask]
            linearOrdinate = inputOrdinate[fluxMask]

            linearFit, linearFitErr, chiSq, weights = irlsFit([0.0, 100.0], linearAbscissa,
                                                              linearOrdinate, funcPolynomial)
            # Convert this proxy-to-flux fit into an expected linear flux
            linearOrdinate = linearFit[0] + linearFit[1] * inputAbscissa

            # Exclude low end outliers
            threshold = self.config.nSigmaClipLinear * np.sqrt(linearOrdinate)
            fluxMask = np.abs(inputOrdinate - linearOrdinate) < threshold
            linearOrdinate = linearOrdinate[fluxMask]
            fitOrdinate = inputOrdinate[fluxMask]
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
                self.log.info(f"Significant polynomial fits: {significant}")

                modelOrdinate = funcPolynomial(polyFit, linearAbscissa)
                self.debugFit('polyFit', linearAbscissa, fitOrdinate, modelOrdinate, None, ampName)

                if self.config.linearityType == 'Squared':
                    linearityFit = [linearityFit[2]]
                elif self.config.linearityType == 'LookupTable':
                    # Use linear part to get time at wich signal is maxAduForLookupTableLinearizer DN
                    tMax = (self.config.maxLookupTableAdu - polyFit[0])/polyFit[1]
                    timeRange = np.linspace(0, tMax, self.config.maxLookupTableAdu)
                    signalIdeal = polyFit[0] + polyFit[1]*timeRange
                    signalUncorrected = funcPolynomial(polyFit, timeRange)
                    lookupTableRow = signalIdeal - signalUncorrected  # LinearizerLookupTable has correction

                    linearizer.tableData[tableIndex, :] = lookupTableRow
                    linearityFit = [tableIndex, 0]
                    tableIndex += 1
            elif self.config.linearityType in ['Spline']:
                # See discussion in `lsst.ip.isr.linearize.py` before modifying.
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

    def debugFit(self, stepname, xVector, yVector, yModel, mask, ampName):
        """Debug method for linearity fitting.

        Parameters
        ----------
        stepname : `str`
            A label to use to check if we care to debug at a given
            line of code.
        xVector : `numpy.array`
            The values to use as the independent variable in the
            linearity fit.
        yVector : `numpy.array`
            The values to use as the dependent variable in the
            linearity fit.
        yModel : `numpy.array`
            The values to use as the linearized result.
        mask : `numpy.array` [ `bool` ], optional
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


class MeasureLinearityConfig(pexConfig.Config):
    solver = pexConfig.ConfigurableField(
        target=LinearitySolveTask,
        doc="Task to convert PTC data to linearity solutions.",
    )


class MeasureLinearityTask(pipeBase.CmdLineTask):
    """Stand alone Gen2 linearity measurement.

    This class wraps the Gen3 linearity task to allow it to be run as
    a Gen2 CmdLineTask.
    """
    ConfigClass = MeasureLinearityConfig
    _DefaultName = "measureLinearity"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.makeSubtask("solver")

    def runDataRef(self, dataRef):
        """Run new linearity code for gen2.

        Parameters
        ----------
        dataRef : `lsst.daf.persistence.ButlerDataRef`
            Input dataref for the photon transfer curve data.

        Returns
        -------
        results : `lsst.pipe.base.Struct`
            The results struct containing:

            ``outputLinearizer`` : `lsst.ip.isr.Linearizer`
                Final linearizer calibration.
            ``outputProvenance`` : `lsst.ip.isr.IsrProvenance`
                Provenance data for the new calibration.
        """
        ptc = dataRef.get('photonTransferCurveDataset')
        camera = dataRef.get('camera')
        inputDims = dataRef.dataId  # This is the closest gen2 has.
        linearityResults = self.solver.run(ptc, camera=camera, inputDims=inputDims)

        inputDims['calibDate'] = linearityResults.outputLinearizer.getMetadata().get('CALIBDATE')
        butler = dataRef.getButler()
        butler.put(linearityResults.outputLinearizer, "linearizer", inputDims)
        return linearityResults
