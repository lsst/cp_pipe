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

import lsst.pipe.base as pipeBase
import lsst.pipe.base.connectionTypes as cT
import lsst.pex.config as pexConfig

from lsstDebug import getDebugFrame
from lsst.ip.isr import (Linearizer, IsrProvenance)

from .utils import (fitLeastSq, funcPolynomial)


__all__ = ["LinearitySolveTask", "LinearitySolveConfig"]


class LinearitySolveConnections(pipeBase.PipelineTaskConnections,
                                dimensions=("instrument", "detector")):

    inputPtc = cT.Input(
        name="inputPtc",
        doc="Input PTC dataset.",
        storageClass="StructuredDataDict",
        dimensions=("instrument", "detector"),
        multiple=False,
    )
    camera = cT.Input(
        name="camera",
        doc="Camera Geometry definition.",
        storageClass="Camera",
        dimensions=("instrument", ),
    )
    outputLinearizer = cT.Output(
        name="linearity",
        doc="Output linearity measurements.",
        storageClass="Linearizer",
        dimensions=("instrument", "detector"),
    )


class LinearitySolveConfig(pipeBase.PipelineTaskConfig,
                           pipelineConnections=LinearitySolveConnections):
    """Configuration for solving the linearity from PTC dataset.
    """
    linearityType = pexConfig.ChoiceField(
        dtype=str,
        doc="Type of linearizer to construct.",
        default="Polynomial",
        allowed={
            "LookupTable": "Create a lookup table solution.",
            "Polynomial": "Create an arbitrary polynomial solution.",
            "Squared": "Create a single order squared solution.",
            "None": "Create a dummy solution.",
        }
    )
    polynomialOrder = pexConfig.Field(
        dtype=int,
        doc="Degree of polynomial to fit.",
        default=3,
    )
    maxLookupTableAdu = pexConfig.Field(
        dtype=int,
        doc="Maximum DN value for a LookupTable linearizer.",
        default=2**18,
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
        inputs['inputDims'] = [exp.dataId.byName() for exp in inputRefs.inputPtc]

        outputs = self.run(**inputs)
        butlerQC.put(outputs, outputRefs)

    def run(self, inputPtc, camera=None, inputDims=None):
        """Fit non-linearity to PTC data, returning the correct Linearizer
        object.

        Parameters
        ----------
        inputPtc : `lsst.cp.pipe.PtcDataset`
            Pre-measured PTC dataset.
        camera : `lsst.afw.cameraGeom.Camera`, optional
            Camera geometry.
        inputDims : `lsst.daf.butler.DataCoordinate` or `dict`, optional
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
        Eq. 37 of 2003.05978). The `config.polynomialOrder` defines
        the maximum order of x^j to fit.  As k_0 and k_1 are
        degenerate with bias level and gain, they are not included in
        the non-linearity correction.

        """
        if camera:
            detector = camera[inputDims['detector']]

        if self.config.linearityType == 'LookupTable':
            table = np.zeros((len(detector), self.config.maxLookupTableAdu), dtype=np.float32)
            tableIndex = 0
        else:
            table = None
            tableIndex = None  # This will fail if we increment it.

        # Initialize the linearizer.
        linearizer = Linearizer(detector=detector, table=table, log=self.log)

        for i, amp in enumerate(detector):
            ampName = amp.getName()
            if (len(inputPtc.visitMask[ampName]) == 0):
                self.log.warn(f"Mask not found for {ampName} in non-linearity fit. Using all points.")
                mask = np.repeat(True, len(inputPtc.rawExpTimes[ampName]))
            else:
                mask = inputPtc.visitMask[ampName]

            timeVector = np.array(inputPtc.rawExpTimes[ampName])[mask]
            meanVector = np.array(inputPtc.rawMeans[ampName])[mask]

            if self.config.linearityType in ['Polynomial', 'Squared', 'LookupTable']:
                polyFit = np.zeros(self.config.polynomialOrder + 1)
                polyFit[1] = 1.0
                polyFit, polyFitErr, chiSq = fitLeastSq(polyFit, timeVector, meanVector, funcPolynomial)

                # Truncate the polynomial fit
                k1 = polyFit[1]
                linearityFit = [-coeff/(k1**order) for order, coeff in enumerate(polyFit)]
                significant = np.where(np.abs(linearityFit) > 1e-10, True, False)
                self.log.info(f"Significant polynomial fits: {significant}")

                if self.config.linearityType == 'Squared':
                    linearityFit = [linearityFit[2]]
                elif self.config.linearityType == 'LookupTable':
                    # Use linear part to get time at wich signal is maxAduForLookupTableLinearizer DN
                    tMax = (self.config.maxLookupTableAdu - polyFit[0])/polyFit[1]
                    timeRange = np.linspace(0, tMax, self.config.maxLookupTableAdu)
                    signalIdeal = polyFit[0] + polyFit[1]*timeRange
                    signalUncorrected = funcPolynomial(polyFit, timeRange)
                    lookupTableRow = signalIdeal - signalUncorrected  # LinearizerLookupTable has corrections

                    linearizer.tableData[tableIndex, :] = lookupTableRow
                    linearityFit = [tableIndex, 0]
                    tableIndex += 1
            else:
                polyFit = [0.0]
                polyFitErr = [0.0]
                chiSq = np.nan
                linearityFit = [0.0]

            linearizer.linearityType[ampName] = 'self.config.linearityType'
            linearizer.linearityCoeffs[ampName] = linearityFit
            linearizer.linearityBBox[ampName] = amp.getBBox()
            linearizer.fitParams[ampName] = polyFit
            linearizer.fitParamsErr[ampName] = polyFitErr
            linearizer.fitChiSq[ampName] = chiSq
            self.debugFit('solution', timeVector, meanVector, linearizer, ampName)

        linearizer.validate()
        linearizer.updateMetadata(setDate=True)
        provenance = IsrProvenance(calibType='linearizer')

        return pipeBase.Struct(
            outputLinearizer=linearizer,
            outputProvenance=provenance,
        )

    def debugFit(self, stepname, timeVector, meanVector, linearizer, ampName):
        """Debug method for linearity fitting.

        Parameters
        ----------
        stepname : `str`
            A label to use to check if we care to debug at a given
            line of code.
        timeVector : `numpy.array`
            The values to use as the independent variable in the
            linearity fit.
        meanVector : `numpy.array`
            The values to use as the dependent variable in the
            linearity fit.
        linearizer : `lsst.ip.isr.Linearizer`
            The linearity correction to compare.
        ampName : `str`
            Amplifier name to lookup linearity correction values.
        """
        frame = getDebugFrame(self._display, stepname)
        if frame:
            import matplotlib.pyplot as plot
            figure = plot.figure(1)
            figure.clear()

            axes = figure.add_axes((min(timeVector), min(meanVector),
                                    max(timeVector), max(meanVector)))
            axes.plot(timeVector, meanVector, 'k+')

            axes.plot(timeVector,
                      np.polynomial.polynomial.polyval(linearizer.fitParams[ampName],
                                                       timeVector), 'r')
            plot.xlabel("Exposure Time")
            plot.ylabel("Mean Flux")
            plot.title(f"Linearity {ampName} {linearizer.linearityType[ampName]}"
                       f" chi={linearizer.fitChiSq[ampName]}")
            figure.show()

            prompt = "Press enter to continue: "
            while True:
                ans = input(prompt).lower()
                if ans in ("", "c",):
                    break
            plot.close()
