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
"""Calculation of brighter-fatter effect correlations and kernels."""

__all__ = ['BrighterFatterKernelSolveTask',
           'BrighterFatterKernelSolveConfig']

import numpy as np

import lsst.afw.math as afwMath
import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase
import lsst.pipe.base.connectionTypes as cT

from lsst.ip.isr import (BrighterFatterKernel)
from .utils import (funcPolynomial, irlsFit)
from ._lookupStaticCalibration import lookupStaticCalibration


class BrighterFatterKernelSolveConnections(pipeBase.PipelineTaskConnections,
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
        doc="Camera associated with this data.",
        storageClass="Camera",
        dimensions=("instrument", ),
        isCalibration=True,
        lookupFunction=lookupStaticCalibration,
    )
    inputPtc = cT.PrerequisiteInput(
        name="ptc",
        doc="Photon transfer curve dataset.",
        storageClass="PhotonTransferCurveDataset",
        dimensions=("instrument", "detector"),
        isCalibration=True,
    )

    outputBFK = cT.Output(
        name="brighterFatterKernel",
        doc="Output measured brighter-fatter kernel.",
        storageClass="BrighterFatterKernel",
        dimensions=("instrument", "detector"),
        isCalibration=True,
    )


class BrighterFatterKernelSolveConfig(pipeBase.PipelineTaskConfig,
                                      pipelineConnections=BrighterFatterKernelSolveConnections):
    level = pexConfig.ChoiceField(
        doc="The level at which to calculate the brighter-fatter kernels",
        dtype=str,
        default="AMP",
        allowed={
            "AMP": "Every amplifier treated separately",
            "DETECTOR": "One kernel per detector",
        }
    )
    ignoreAmpsForAveraging = pexConfig.ListField(
        dtype=str,
        doc="List of amp names to ignore when averaging the amplifier kernels into the detector"
        " kernel. Only relevant for level = DETECTOR",
        default=[]
    )
    xcorrCheckRejectLevel = pexConfig.Field(
        dtype=float,
        doc="Rejection level for the sum of the input cross-correlations. Arrays which "
        "sum to greater than this are discarded before the clipped mean is calculated.",
        default=2.0
    )
    nSigmaClip = pexConfig.Field(
        dtype=float,
        doc="Number of sigma to clip when calculating means for the cross-correlation",
        default=5
    )
    forceZeroSum = pexConfig.Field(
        dtype=bool,
        doc="Force the correlation matrix to have zero sum by adjusting the (0,0) value?",
        default=False,
    )
    useAmatrix = pexConfig.Field(
        dtype=bool,
        doc="Use the PTC 'a' matrix (Astier et al. 2019 equation 20) "
        "instead of the average of measured covariances?",
        default=False,
    )

    maxIterSuccessiveOverRelaxation = pexConfig.Field(
        dtype=int,
        doc="The maximum number of iterations allowed for the successive over-relaxation method",
        default=10000
    )
    eLevelSuccessiveOverRelaxation = pexConfig.Field(
        dtype=float,
        doc="The target residual error for the successive over-relaxation method",
        default=5.0e-14
    )

    correlationQuadraticFit = pexConfig.Field(
        dtype=bool,
        doc="Use a quadratic fit to find the correlations instead of simple averaging?",
        default=False,
    )
    correlationModelRadius = pexConfig.Field(
        dtype=int,
        doc="Build a model of the correlation coefficients for radii larger than this value in pixels?",
        default=100,
    )
    correlationModelSlope = pexConfig.Field(
        dtype=float,
        doc="Slope of the correlation model for radii larger than correlationModelRadius",
        default=-1.35,
    )


class BrighterFatterKernelSolveTask(pipeBase.PipelineTask, pipeBase.CmdLineTask):
    """Measure appropriate Brighter-Fatter Kernel from the PTC dataset.
    """

    ConfigClass = BrighterFatterKernelSolveConfig
    _DefaultName = 'cpBfkMeasure'

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
        """Combine covariance information from PTC into brighter-fatter
        kernels.

        Parameters
        ----------
        inputPtc : `lsst.ip.isr.PhotonTransferCurveDataset`
            PTC data containing per-amplifier covariance measurements.
        dummy : `lsst.afw.image.Exposure`
            The exposure used to select the appropriate PTC dataset.
            In almost all circumstances, one of the input exposures
            used to generate the PTC dataset is the best option.
        camera : `lsst.afw.cameraGeom.Camera`
            Camera to use for camera geometry information.
        inputDims : `lsst.daf.butler.DataCoordinate` or `dict`
            DataIds to use to populate the output calibration.

        Returns
        -------
        results : `lsst.pipe.base.Struct`
            The resulst struct containing:

            ``outputBfk``
                Resulting Brighter-Fatter Kernel
                (`lsst.ip.isr.BrighterFatterKernel`).
        """
        if len(dummy) == 0:
            self.log.warning("No dummy exposure found.")

        detector = camera[inputDims['detector']]
        detName = detector.getName()

        if self.config.level == 'DETECTOR':
            detectorCorrList = list()
            detectorFluxes = list()

        bfk = BrighterFatterKernel(camera=camera, detectorId=detector.getId(), level=self.config.level)
        bfk.means = inputPtc.finalMeans  # ADU
        bfk.variances = inputPtc.finalVars  # ADU^2
        # Use the PTC covariances as the cross-correlations.  These
        # are scaled before the kernel is generated, which performs
        # the conversion.
        bfk.rawXcorrs = inputPtc.covariances  # ADU^2
        bfk.badAmps = inputPtc.badAmps
        bfk.shape = (inputPtc.covMatrixSide*2 + 1, inputPtc.covMatrixSide*2 + 1)
        bfk.gain = inputPtc.gain
        bfk.noise = inputPtc.noise
        bfk.meanXcorrs = dict()
        bfk.valid = dict()

        for amp in detector:
            ampName = amp.getName()
            gain = bfk.gain[ampName]

            # Using the inputPtc.expIdMask works if the covariance
            # array has the same length as the rawMeans/rawVars.  This
            # isn't the case, as it's the same size as the
            # finalMeans/finalVars.  However, these arrays (and the
            # covariance) are padded with NAN values to match the
            # longest amplifier vector.  We do not want to include
            # these NAN values, so we construct a mask for all non-NAN
            # values in finalMeans, and use that to filter finalVars
            # and the covariances.
            mask = np.isfinite(bfk.means[ampName])
            fluxes = np.array(bfk.means[ampName])[mask]
            variances = np.array(bfk.variances[ampName])[mask]
            xCorrList = np.array([np.array(xcorr) for xcorr in bfk.rawXcorrs[ampName]])[mask]

            if gain <= 0:
                # We've received very bad data.
                self.log.warning("Impossible gain recieved from PTC for %s: %f.  Skipping amplifier.",
                                 ampName, gain)
                bfk.meanXcorrs[ampName] = np.zeros(bfk.shape)
                bfk.ampKernels[ampName] = np.zeros(bfk.shape)
                bfk.valid[ampName] = False
                continue

            fluxes = np.array([flux*gain for flux in fluxes])  # Now in e^-
            variances = np.array([variance*gain*gain for variance in variances])  # Now in e^2-

            # This should duplicate Coulton et al. 2017 Equation 22-29
            # (arxiv:1711.06273)
            scaledCorrList = list()
            corrList = list()
            truncatedFluxes = list()
            for xcorrNum, (xcorr, flux, var) in enumerate(zip(xCorrList, fluxes, variances), 1):
                q = np.array(xcorr) * gain * gain  # xcorr now in e^-
                q *= 2.0  # Remove factor of 1/2 applied in PTC.
                self.log.info("Amp: %s %d/%d Flux: %f  Var: %f  Q(0,0): %g  Q(1,0): %g  Q(0,1): %g",
                              ampName, xcorrNum, len(xCorrList), flux, var, q[0][0], q[1][0], q[0][1])

                # Normalize by the flux, which removes the (0,0)
                # component attributable to Poisson noise.  This
                # contains the two "t I delta(x - x')" terms in
                # Coulton et al. 2017 equation 29
                q[0][0] -= 2.0*(flux)

                if q[0][0] > 0.0:
                    self.log.warning("Amp: %s %d skipped due to value of (variance-mean)=%f",
                                     ampName, xcorrNum, q[0][0])
                    # If we drop an element of ``scaledCorrList``
                    # (which is what this does), we need to ensure we
                    # drop the flux entry as well.
                    continue

                # This removes the "t (I_a^2 + I_b^2)" factor in
                # Coulton et al. 2017 equation 29.
                # The quadratic fit option needs the correlations unscaled
                q /= -2.0
                unscaled = self._tileArray(q)
                q /= flux**2
                scaled = self._tileArray(q)
                xcorrCheck = np.abs(np.sum(scaled))/np.sum(np.abs(scaled))
                if (xcorrCheck > self.config.xcorrCheckRejectLevel) or not (np.isfinite(xcorrCheck)):
                    self.log.warning("Amp: %s %d skipped due to value of triangle-inequality sum %f",
                                     ampName, xcorrNum, xcorrCheck)
                    continue

                scaledCorrList.append(scaled)
                corrList.append(unscaled)
                truncatedFluxes.append(flux)
                self.log.info("Amp: %s %d/%d  Final: %g  XcorrCheck: %f",
                              ampName, xcorrNum, len(xCorrList), q[0][0], xcorrCheck)

            fluxes = np.array(truncatedFluxes)

            if len(scaledCorrList) == 0:
                self.log.warning("Amp: %s All inputs rejected for amp!", ampName)
                bfk.meanXcorrs[ampName] = np.zeros(bfk.shape)
                bfk.ampKernels[ampName] = np.zeros(bfk.shape)
                bfk.valid[ampName] = False
                continue

            if self.config.useAmatrix:
                # Use the aMatrix, ignoring the meanXcorr generated above.
                preKernel = np.pad(self._tileArray(np.array(inputPtc.aMatrix[ampName])), ((1, 1)))
            elif self.config.correlationQuadraticFit:
                # Use a quadratic fit to the correlations as a
                # function of flux.
                preKernel = self.quadraticCorrelations(corrList, fluxes, f"Amp: {ampName}")
            else:
                # Use a simple average of the measured correlations.
                preKernel = self.averageCorrelations(scaledCorrList, f"Amp: {ampName}")

            center = int((bfk.shape[0] - 1) / 2)

            if self.config.forceZeroSum:
                totalSum = np.sum(preKernel)

                if self.config.correlationModelRadius < (preKernel.shape[0] - 1) / 2:
                    # Assume a correlation model of
                    # Corr(r) = -preFactor * r^(2 * slope)
                    preFactor = np.sqrt(preKernel[center, center + 1] * preKernel[center + 1, center])
                    slopeFactor = 2.0 * np.abs(self.config.correlationModelSlope)
                    totalSum += 2.0*np.pi*(preFactor / (slopeFactor*(center + 0.5))**slopeFactor)

                preKernel[center, center] -= totalSum
                self.log.info("%s Zero-Sum Scale: %g", ampName, totalSum)

            finalSum = np.sum(preKernel)
            bfk.meanXcorrs[ampName] = preKernel

            postKernel = self.successiveOverRelax(preKernel)
            bfk.ampKernels[ampName] = postKernel
            if self.config.level == 'DETECTOR':
                detectorCorrList.extend(scaledCorrList)
                detectorFluxes.extend(fluxes)
            bfk.valid[ampName] = True
            self.log.info("Amp: %s Sum: %g  Center Info Pre: %g  Post: %g",
                          ampName, finalSum, preKernel[center, center], postKernel[center, center])

        # Assemble a detector kernel?
        if self.config.level == 'DETECTOR':
            if self.config.correlationQuadraticFit:
                preKernel = self.quadraticCorrelations(detectorCorrList, detectorFluxes, f"Amp: {ampName}")
            else:
                preKernel = self.averageCorrelations(detectorCorrList, f"Det: {detName}")
            finalSum = np.sum(preKernel)
            center = int((bfk.shape[0] - 1) / 2)

            postKernel = self.successiveOverRelax(preKernel)
            bfk.detKernels[detName] = postKernel
            self.log.info("Det: %s Sum: %g  Center Info Pre: %g  Post: %g",
                          detName, finalSum, preKernel[center, center], postKernel[center, center])

        return pipeBase.Struct(
            outputBFK=bfk,
        )

    def averageCorrelations(self, xCorrList, name):
        """Average input correlations.

        Parameters
        ----------
        xCorrList : `list` [`numpy.array`]
            List of cross-correlations.  These are expected to be
            square arrays.
        name : `str`
            Name for log messages.

        Returns
        -------
        meanXcorr : `numpy.array`, (N, N)
            The averaged cross-correlation.
        """
        meanXcorr = np.zeros_like(xCorrList[0])
        xCorrList = np.transpose(xCorrList)
        sctrl = afwMath.StatisticsControl()
        sctrl.setNumSigmaClip(self.config.nSigmaClip)
        for i in range(np.shape(meanXcorr)[0]):
            for j in range(np.shape(meanXcorr)[1]):
                meanXcorr[i, j] = afwMath.makeStatistics(xCorrList[i, j],
                                                         afwMath.MEANCLIP, sctrl).getValue()

        # To match previous definitions, pad by one element.
        meanXcorr = np.pad(meanXcorr, ((1, 1)))

        return meanXcorr

    def quadraticCorrelations(self, xCorrList, fluxList, name):
        """Measure a quadratic correlation model.

        Parameters
        ----------
        xCorrList : `list` [`numpy.array`]
            List of cross-correlations.  These are expected to be
            square arrays.
        fluxList : `numpy.array`, (Nflux,)
            Associated list of fluxes.
        name : `str`
            Name for log messages.

        Returns
        -------
        meanXcorr : `numpy.array`, (N, N)
            The averaged cross-correlation.
        """
        meanXcorr = np.zeros_like(xCorrList[0])
        fluxList = np.square(fluxList)
        xCorrList = np.array(xCorrList)

        for i in range(np.shape(meanXcorr)[0]):
            for j in range(np.shape(meanXcorr)[1]):
                # Fit corrlation_i(x, y) = a0 + a1 * (flux_i)^2 The
                # i,j indices are inverted to apply the transposition,
                # as is done in the averaging case.
                linearFit, linearFitErr, chiSq, weights = irlsFit([0.0, 1e-4], fluxList,
                                                                  xCorrList[:, j, i], funcPolynomial,
                                                                  scaleResidual=False)
                meanXcorr[i, j] = linearFit[1]  # Discard the intercept.
                self.log.info("Quad fit meanXcorr[%d,%d] = %g", i, j, linearFit[1])

        # To match previous definitions, pad by one element.
        meanXcorr = np.pad(meanXcorr, ((1, 1)))

        return meanXcorr

    @staticmethod
    def _tileArray(in_array):
        """Given an input quarter-image, tile/mirror it and return full image.

        Given a square input of side-length n, of the form

        input = array([[1, 2, 3],
                       [4, 5, 6],
                       [7, 8, 9]])

        return an array of size 2n-1 as

        output = array([[ 9,  8,  7,  8,  9],
                        [ 6,  5,  4,  5,  6],
                        [ 3,  2,  1,  2,  3],
                        [ 6,  5,  4,  5,  6],
                        [ 9,  8,  7,  8,  9]])

        Parameters
        ----------
        input : `np.array`, (N, N)
            The square input quarter-array

        Returns
        -------
        output : `np.array`, (2*N + 1, 2*N + 1)
            The full, tiled array
        """
        assert(in_array.shape[0] == in_array.shape[1])
        length = in_array.shape[0] - 1
        output = np.zeros((2*length + 1, 2*length + 1))

        for i in range(length + 1):
            for j in range(length + 1):
                output[i + length, j + length] = in_array[i, j]
                output[-i + length, j + length] = in_array[i, j]
                output[i + length, -j + length] = in_array[i, j]
                output[-i + length, -j + length] = in_array[i, j]
        return output

    def successiveOverRelax(self, source, maxIter=None, eLevel=None):
        """An implementation of the successive over relaxation (SOR) method.

        A numerical method for solving a system of linear equations
        with faster convergence than the Gauss-Seidel method.

        Parameters
        ----------
        source : `numpy.ndarray`, (N, N)
            The input array.
        maxIter : `int`, optional
            Maximum number of iterations to attempt before aborting.
        eLevel : `float`, optional
            The target error level at which we deem convergence to have
            occurred.

        Returns
        -------
        output : `numpy.ndarray`, (N, N)
            The solution.
        """
        if not maxIter:
            maxIter = self.config.maxIterSuccessiveOverRelaxation
        if not eLevel:
            eLevel = self.config.eLevelSuccessiveOverRelaxation

        assert source.shape[0] == source.shape[1], "Input array must be square"
        # initialize, and set boundary conditions
        func = np.zeros([source.shape[0] + 2, source.shape[1] + 2])
        resid = np.zeros([source.shape[0] + 2, source.shape[1] + 2])
        rhoSpe = np.cos(np.pi/source.shape[0])  # Here a square grid is assumed

        # Calculate the initial error
        for i in range(1, func.shape[0] - 1):
            for j in range(1, func.shape[1] - 1):
                resid[i, j] = (func[i, j - 1] + func[i, j + 1] + func[i - 1, j]
                               + func[i + 1, j] - 4*func[i, j] - source[i - 1, j - 1])
        inError = np.sum(np.abs(resid))

        # Iterate until convergence
        # We perform two sweeps per cycle,
        # updating 'odd' and 'even' points separately
        nIter = 0
        omega = 1.0
        dx = 1.0
        while nIter < maxIter*2:
            outError = 0
            if nIter%2 == 0:
                for i in range(1, func.shape[0] - 1, 2):
                    for j in range(1, func.shape[1] - 1, 2):
                        resid[i, j] = float(func[i, j-1] + func[i, j + 1] + func[i - 1, j]
                                            + func[i + 1, j] - 4.0*func[i, j] - dx*dx*source[i - 1, j - 1])
                        func[i, j] += omega*resid[i, j]*.25
                for i in range(2, func.shape[0] - 1, 2):
                    for j in range(2, func.shape[1] - 1, 2):
                        resid[i, j] = float(func[i, j - 1] + func[i, j + 1] + func[i - 1, j]
                                            + func[i + 1, j] - 4.0*func[i, j] - dx*dx*source[i - 1, j - 1])
                        func[i, j] += omega*resid[i, j]*.25
            else:
                for i in range(1, func.shape[0] - 1, 2):
                    for j in range(2, func.shape[1] - 1, 2):
                        resid[i, j] = float(func[i, j - 1] + func[i, j + 1] + func[i - 1, j]
                                            + func[i + 1, j] - 4.0*func[i, j] - dx*dx*source[i - 1, j - 1])
                        func[i, j] += omega*resid[i, j]*.25
                for i in range(2, func.shape[0] - 1, 2):
                    for j in range(1, func.shape[1] - 1, 2):
                        resid[i, j] = float(func[i, j - 1] + func[i, j + 1] + func[i - 1, j]
                                            + func[i + 1, j] - 4.0*func[i, j] - dx*dx*source[i - 1, j - 1])
                        func[i, j] += omega*resid[i, j]*.25
            outError = np.sum(np.abs(resid))
            if outError < inError*eLevel:
                break
            if nIter == 0:
                omega = 1.0/(1 - rhoSpe*rhoSpe/2.0)
            else:
                omega = 1.0/(1 - rhoSpe*rhoSpe*omega/4.0)
            nIter += 1

        if nIter >= maxIter*2:
            self.log.warning("Failure: SuccessiveOverRelaxation did not converge in %s iterations."
                             "\noutError: %s, inError: %s,", nIter//2, outError, inError*eLevel)
        else:
            self.log.info("Success: SuccessiveOverRelaxation converged in %s iterations."
                          "\noutError: %s, inError: %s", nIter//2, outError, inError*eLevel)
        return func[1: -1, 1: -1]
