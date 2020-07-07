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
        doc="Use the PTC 'a' matrix instead of the average of measured covariances?",
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

    # These are unused.  Are they worth implementing?
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
        """Combine covariance information from PTC into brighter-fatter kernels.

        Parameters
        ----------
        inputPtc : `lsst.ip.isr.PhotonTransferCurveDataset`
            PTC data containing per-amplifier covariance measurements.
        dummy : `lsst.afw.image.Exposure
            The exposure used to select the appropriate PTC dataset.
        camera : `lsst.afw.cameraGeom.Camera`
            Camera to use for camera geometry information.
        inputDims : `lsst.daf.butler.DataCoordinate` or `dict`
            DataIds to use to populate the output calibration.

        Returns
        -------
        results : `lsst.pipe.base.Struct`
            The resulst struct containing:

            ``outputBfk`` : `lsst.ip.isr.BrighterFatterKernel`
                Resulting Brighter-Fatter Kernel.
        """
        if len(dummy) == 0:
            self.log.warn("No dummy exposure found.")

        detector = camera[inputDims['detector']]
        detName = detector.getName()

        if self.config.level == 'DETECTOR':
            detectorCorrList = list()

        bfk = BrighterFatterKernel(camera=camera, detectorId=detector.getId(), level=self.config.level)
        bfk.means = inputPtc.finalMeans  # ADU
        bfk.rawMeans = inputPtc.rawMeans  # ADU

        bfk.variances = inputPtc.finalVars  # ADU^2
        bfk.rawXcorrs = inputPtc.covariances  # ADU^2

        bfk.gain = inputPtc.gain
        bfk.noise = inputPtc.noise
        bfk.meanXCorrs = dict()

        for amp in detector:
            ampName = amp.getName()
            mask = inputPtc.expIdMask[ampName]

            gain = bfk.gain[ampName]
            fluxes = np.array(bfk.means[ampName])[mask]
            variances = np.array(bfk.variances[ampName])[mask]
            xCorrList = [np.array(xcorr) for xcorr in bfk.rawXcorrs[ampName]]
            xCorrList = np.array(xCorrList)[mask]

            fluxes = np.array([flux*gain for flux in fluxes])  # Now in e^-
            variances = np.array([variance*gain*gain for variance in variances])  # Now in e^2-

            # This should duplicate the else block in generateKernel@L1358,
            # which in turn is based on Coulton et al Equation 22.
            scaledCorrList = list()
            for xcorrNum, (xcorr, flux, var) in enumerate(zip(xCorrList, fluxes, variances), 1):
                q = np.array(xcorr) * gain * gain  # xcorr now in e^-
                q *= 2.0  # Remove factor of 1/2 applied in PTC.
                self.log.info("Amp: %s %d/%d Flux: %f  Var: %f  Q(0,0): %g  Q(1,0): %g  Q(0,1): %g",
                              ampName, xcorrNum, len(xCorrList), flux, var, q[0][0], q[1][0], q[0][1])

                # Normalize by the flux, which removes the (0,0)
                # component attributable to Poisson noise.
                q[0][0] -= 2.0*(flux)

                if q[0][0] > 0.0:
                    self.log.warn("Amp: %s %d skipped due to value of (variance-mean)=%f",
                                  ampName, xcorrNum, q[0][0])
                    continue

                q /= -2.0*(flux**2)
                scaled = self._tileArray(q)

                xcorrCheck = np.abs(np.sum(scaled))/np.sum(np.abs(scaled))
                if xcorrCheck > self.config.xcorrCheckRejectLevel:
                    self.log.warn("Amp: %s %d skipped due to value of triangle-inequality sum %f",
                                  ampName, xcorrNum, xcorrCheck)
                    continue

                scaledCorrList.append(scaled)
                self.log.info("Amp: %s %d/%d  Final: %g  XcorrCheck: %f",
                              ampName, xcorrNum, len(xCorrList), q[0][0], xcorrCheck)

            if len(scaledCorrList) == 0:
                self.log.warn("Amp: %s All inputs rejected for amp!", ampName)
                bfk.ampKernels[ampName] = np.zeros_like(np.pad(scaled, ((1, 1))))
                continue

            if self.config.level == 'DETECTOR':
                detectorCorrList.extend(scaledCorrList)

            if self.config.useAmatrix:
                # This is mildly wasteful
                preKernel = np.pad(self._tileArray(np.array(inputPtc.aMatrix[ampName])), ((1, 1)))
            else:
                preKernel = self.averageCorrelations(scaledCorrList, f"Amp: {ampName}")

            finalSum = np.sum(preKernel)
            center = int((preKernel.shape[0] - 1) / 2)
            bfk.meanXCorrs[ampName] = preKernel

            postKernel = self.successiveOverRelax(preKernel)
            bfk.ampKernels[ampName] = postKernel
            self.log.info("Amp: %s Sum: %g  Center Info Pre: %g  Post: %g",
                          ampName, finalSum, preKernel[center, center], postKernel[center, center])

        # Assemble a detector kernel?
        if self.config.level == 'DETECTOR':
            preKernel = self.averageCorrelations(detectorCorrList, f"Det: {detName}")
            finalSum = np.sum(preKernel)
            center = int((preKernel.shape[0] - 1) / 2)

            postKernel = self.successiveOverRelax(preKernel)
            bfk.detKernels[detName] = postKernel
            self.log.info("Det: %s Sum: %g  Center Info Pre: %g  Post: %g",
                          detName, finalSum, preKernel[center, center], postKernel[center, center])

        bfk.shape = postKernel.shape

        return pipeBase.Struct(
            outputBFK=bfk,
        )

    def averageCorrelations(self, xCorrList, name):
        """Average input correlations.

        Parameters
        ----------
        xCorrList : `list` [`numpy.array`]
            List of cross-correlations.
        name : `str`
            Name for log messages.

        Returns
        -------
        meanXcorr : `numpy.array`
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
        center = int((meanXcorr.shape[0] - 1) / 2)
        if self.config.forceZeroSum or True:
            totalSum = np.sum(meanXcorr)
            meanXcorr[center, center] -= totalSum
            self.log.info("%s Zero-Sum Scale: %g", name, totalSum)

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

        Parameters:
        -----------
        input : `np.array`
            The square input quarter-array

        Returns:
        --------
        output : `np.array`
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

        Parameters:
        -----------
        source : `numpy.ndarray`
            The input array.
        maxIter : `int`, optional
            Maximum number of iterations to attempt before aborting.
        eLevel : `float`, optional
            The target error level at which we deem convergence to have
            occurred.

        Returns:
        --------
        output : `numpy.ndarray`
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
            self.log.warn("Failure: SuccessiveOverRelaxation did not converge in %s iterations."
                          "\noutError: %s, inError: %s," % (nIter//2, outError, inError*eLevel))
        else:
            self.log.info("Success: SuccessiveOverRelaxation converged in %s iterations."
                          "\noutError: %s, inError: %s", nIter//2, outError, inError*eLevel)
        return func[1: -1, 1: -1]


# class MakeBrighterFatterKernelTask(pipeBase.CmdLineTask):
#     """Brighter-fatter effect correction-kernel calculation task.

#     A command line task for calculating the brighter-fatter correction
#     kernel from pairs of flat-field images (with the same exposure length).

#     The following operations are performed:

#     - The configurable isr task is called, which unpersists and assembles the
#       raw images, and performs the selected instrument signature removal tasks.
#       For the purpose of brighter-fatter coefficient calculation is it
#       essential that certain components of isr are *not* performed, and
#       recommended that certain others are. The task checks the selected isr
#       configuration before it is run, and if forbidden components have been
#       selected task will raise, and if recommended ones have not been selected,
#       warnings are logged.

#     - The gain of the each amplifier in the detector is calculated using
#       the photon transfer curve (PTC) method and used to correct the images
#       so that all calculations are done in units of electrons, and so that the
#       level across amplifier boundaries is continuous.
#       Outliers in the PTC are iteratively rejected
#       before fitting, with the nSigma rejection level set by
#       config.nSigmaClipRegression. Individual pixels are ignored in the input
#       images the image based on config.nSigmaClipGainCalc.

#     - Each image is then cross-correlated with the one it's paired with
#       (with the pairing defined by the --visit-pairs command line argument),
#       which is done either the whole-image to whole-image,
#       or amplifier-by-amplifier, depending on config.level.

#     - Once the cross-correlations have been calculated for each visit pair,
#       these are used to generate the correction kernel.
#       The maximum lag used, in pixels, and hence the size of the half-size
#       of the kernel generated, is given by config.maxLag,
#       i.e. a value of 10 will result in a kernel of size 2n-1 = 19x19 pixels.
#       Outlier values in these cross-correlations are rejected by using a
#       pixel-wise sigma-clipped thresholding to each cross-correlation in
#       the visit-pairs-length stack of cross-correlations.
#       The number of sigma clipped to is set by config.nSigmaClipKernelGen.

#     @pipeBase.timeMethod
#     def runDataRef(self, dataRef, visitPairs):
#         """Run the brighter-fatter measurement task.

#         For a dataRef (which is each detector here),
#         and given a list of visit pairs, calculate the
#         brighter-fatter kernel for the detector.

#         Parameters
#         ----------
#         dataRef : `list` of `lsst.daf.persistence.ButlerDataRef`
#             dataRef for the detector for the visits to be fit.
#         visitPairs : `iterable` of `tuple` of `int`
#             Pairs of visit numbers to be processed together
#         """
#         np.random.seed(0)  # used in the PTC fit bootstrap

#         # setup necessary objects
#         # NB: don't use dataRef.get('raw_detector')
#         # this currently doesn't work for composites because of the way
#         # composite objects (i.e. LSST images) are handled/constructed
#         # these need to be retrieved from the camera and dereferenced
#         # rather than accessed directly
#         detNum = dataRef.dataId[self.config.ccdKey]
#         detector = dataRef.get('camera')[dataRef.dataId[self.config.ccdKey]]
#         amps = detector.getAmplifiers()
#         ampNames = [amp.getName() for amp in amps]

#         if self.config.level == 'DETECTOR':
#             kernels = {detNum: []}
#             means = {detNum: []}
#             xcorrs = {detNum: []}
#             meanXcorrs = {detNum: []}
#         elif self.config.level == 'AMP':
#             kernels = {key: [] for key in ampNames}
#             means = {key: [] for key in ampNames}
#             xcorrs = {key: [] for key in ampNames}
#             meanXcorrs = {key: [] for key in ampNames}
#         else:
#             raise RuntimeError("Unsupported level: {}".format(self.config.level))

#         # we must be able to get the gains one way or the other, so check early
#         if not self.config.doCalcGains:
#             deleteMe = None
#             try:
#                 deleteMe = dataRef.get('photonTransferCurveDataset')
#             except butlerExceptions.NoResults:
#                 try:
#                     deleteMe = dataRef.get('brighterFatterGain')
#                 except butlerExceptions.NoResults:
#                     pass
#             if not deleteMe:
#                 raise RuntimeError("doCalcGains == False and gains could not be got from butler") from None
#             else:
#                 del deleteMe

#         # if the level is DETECTOR we need to have the gains first so that each
#         # amp can be gain corrected in order to treat the detector as a single
#         # imaging area. However, if the level is AMP we can wait, calculate
#         # the correlations and correct for the gains afterwards
#         if self.config.level == 'DETECTOR':
#             if self.config.doCalcGains:
#                 self.log.info('Computing gains for detector %s' % detNum)
#                 gains, ptcData, nomGains = self.estimateGains(dataRef, visitPairs)
#                 # dataRef.put(gains, datasetType='brighterFatterGain')
#                 self.log.debug('Finished gain estimation for detector %s' % detNum)
#             else:
#                 gainsObj = dataRef.get('brighterFatterGain')
#                 gains = gainsObj.gains
#                 if not gains:
#                     raise RuntimeError('Failed to retrieved gains for detector %s' % detNum)
#                 self.log.info('Retrieved stored gain for detector %s' % detNum)
#             self.log.debug('Detector %s has gains %s' % (detNum, gains))
#         else:  # we fake the gains as 1 for now, and correct later
#             gains = {}
#             for ampName in ampNames:
#                 gains[ampName] = 1.0
#             # We'll use the ptc.py code to calculate the gains, so we set this up
#             ptcConfig = MeasurePhotonTransferCurveTaskConfig()
#             ptcConfig.isrForbiddenSteps = []
#             ptcConfig.doFitBootstrap = True
#             ptcConfig.ptcFitType = 'POLYNOMIAL'  # default Astier doesn't work for gain correction
#             ptcConfig.polynomialFitDegree = 3
#             ptcConfig.minMeanSignal = self.config.minMeanSignal
#             ptcConfig.maxMeanSignal = self.config.maxMeanSignal
#             ptcTask = MeasurePhotonTransferCurveTask(config=ptcConfig)
#             ptcDataset = PhotonTransferCurveDataset(ampNames)

#         # Loop over pairs of visits
#         # calculating the cross-correlations at the required level
#         for (v1, v2) in visitPairs:
#             dataRef.dataId['expId'] = v1
#             exp1 = self.isr.runDataRef(dataRef).exposure
#             dataRef.dataId['expId'] = v2
#             exp2 = self.isr.runDataRef(dataRef).exposure
#             del dataRef.dataId['expId']
#             checkExpLengthEqual(exp1, exp2, v1, v2, raiseWithMessage=True)

#             self.log.info('Preparing images for cross-correlation calculation for detector %s' % detNum)
#             # note the shape of these returns depends on level
#             _scaledMaskedIms1, _means1 = self._makeCroppedExposures(exp1, gains, self.config.level)
#             _scaledMaskedIms2, _means2 = self._makeCroppedExposures(exp2, gains, self.config.level)

#             # Compute the cross-correlation and means
#             # at the appropriate config.level:
#             # - "DETECTOR": one key, so compare the two visits to each other
#             # - "AMP": n_amp keys, comparing each amplifier of one visit
#             #          to the same amplifier in the visit its paired with
#             for det_object in _scaledMaskedIms1.keys():  # det_object is ampName or detName depending \CZW
# on level
#                 self.log.debug("Calculating correlations for %s" % det_object)
#                 _xcorr, _mean = self._crossCorrelate(_scaledMaskedIms1[det_object],
#                                                      _scaledMaskedIms2[det_object])
#                 xcorrs[det_object].append(_xcorr)
#                 means[det_object].append([_means1[det_object], _means2[det_object]])
#                 if self.config.level != 'DETECTOR':
#                     # Populate the ptcDataset for running fitting in the PTC task
#                     expTime = exp1.getInfo().getVisitInfo().getExposureTime()
#                     ptcDataset.rawExpTimes[det_object].append(expTime)
#                     ptcDataset.rawMeans[det_object].append((_means1[det_object] + _means2[det_object])
# / 2.0)
#                     ptcDataset.rawVars[det_object].append(_xcorr[0, 0] / 2.0)

#                 # TODO: DM-15305 improve debug functionality here.
#                 # This is position 1 for the removed code.

#         # Save the raw means and xcorrs so we can look at them before any modifications
#         rawMeans = copy.deepcopy(means)
#         rawXcorrs = copy.deepcopy(xcorrs)

#         # gains are always and only pre-applied for DETECTOR
#         # so for all other levels we now calculate them from the correlations
#         # and apply them
#         if self.config.level != 'DETECTOR':
#             if self.config.doCalcGains:  # Run the PTC task for calculating the gains, put results
#                 self.log.info('Calculating gains for detector %s using PTC task' % detNum)
#                 ptcDataset = ptcTask.fitPtc(ptcDataset, ptcConfig.ptcFitType)
#                 dataRef.put(ptcDataset, datasetType='photonTransferCurveDataset')
#                 self.log.debug('Finished gain estimation for detector %s' % detNum)
#             else:  # load results  - confirmed to work much earlier on, so can be relied upon here
#                 ptcDataset = dataRef.get('photonTransferCurveDataset')

#             self._applyGains(means, xcorrs, ptcDataset)

#             if self.config.doPlotPtcs:
#                 dirname = dataRef.getUri(datasetType='cpPipePlotRoot', write=True)
#                 if not os.path.exists(dirname):
#                     os.makedirs(dirname)
#                 detNum = dataRef.dataId[self.config.ccdKey]
#                 filename = f"PTC_det{detNum}.pdf"
#                 filenameFull = os.path.join(dirname, filename)
#                 with PdfPages(filenameFull) as pdfPages:
#                     ptcTask._plotPtc(ptcDataset, ptcConfig.ptcFitType, pdfPages)

#         # having calculated and applied the gains for all code-paths we can now
#         # generate the kernel(s)
#         self.log.info('Generating kernel(s) for %s' % detNum)
#         for det_object in xcorrs.keys():  # looping over either detectors or amps
#             if self.config.level == 'DETECTOR':
#                 objId = 'detector %s' % det_object
#             elif self.config.level == 'AMP':
#                 objId = 'detector %s AMP %s' % (detNum, det_object)

#             try:
#                 meanXcorr, kernel = self.generateKernel(xcorrs[det_object], means[det_object], objId)
#                 kernels[det_object] = kernel
#                 meanXcorrs[det_object] = meanXcorr
#             except RuntimeError:
#                 # bad amps will cause failures here which we want to ignore
#                 self.log.warn('RuntimeError during kernel generation for %s' % objId)
#                 continue

#         bfKernel = BrighterFatterKernel(self.config.level)
#         bfKernel.means = means
#         bfKernel.rawMeans = rawMeans
#         bfKernel.rawXcorrs = rawXcorrs
#         bfKernel.xCorrs = xcorrs
#         bfKernel.meanXcorrs = meanXcorrs
#         bfKernel.originalLevel = self.config.level
#         try:
#             bfKernel.gain = ptcDataset.gain
#             bfKernel.gainErr = ptcDataset.gainErr
#             bfKernel.noise = ptcDataset.noise
#             bfKernel.noiseErr = ptcDataset.noiseErr
#         except NameError:  # we don't have a ptcDataset to store results from
#             pass

#         bfKernel.gains = gains
#         if self.config.level == 'AMP':
#             bfKernel.ampKernels = kernels
#             ex = self.config.ignoreAmpsForAveraging
#             bfKernel.makeDetectorKernelFromAmpwiseKernels(detector.getName(),
#                                                           ampsToExclude=ex)
#         elif self.config.level == 'DETECTOR':
#             bfKernel.detectorKernel = kernels
#         else:
#             raise RuntimeError('Invalid level for kernel calculation; this should not be possible.')

#         dataRef.put(bfKernel)

#         self.log.info('Finished generating kernel(s) for %s' % detNum)
#         return pipeBase.Struct(exitStatus=0)

#     def _iterativeRegression(self, x, y, fixThroughOrigin=False, nSigmaClip=None, maxIter=None):
#         """Use linear regression to fit a line, iteratively removing outliers.

#         Useful when you have a sufficiently large numbers of points on your PTC.
#         This function iterates until either there are no outliers of
#         config.nSigmaClip magnitude, or until the specified maximum number
#         of iterations has been performed.

#         Parameters:
#         -----------
#         x : `numpy.array`
#             The independent variable. Must be a numpy array, not a list.
#         y : `numpy.array`
#             The dependent variable. Must be a numpy array, not a list.
#         fixThroughOrigin : `bool`, optional
#             Whether to fix the PTC through the origin or allow an y-intercept.
#         nSigmaClip : `float`, optional
#             The number of sigma to clip to.
#             Taken from the task config if not specified.
#         maxIter : `int`, optional
#             The maximum number of iterations allowed.
#             Taken from the task config if not specified.

#         Returns:
#         --------
#         slope : `float`
#             The slope of the line of best fit
#         intercept : `float`
#             The y-intercept of the line of best fit
#         """
#         if not maxIter:
#             maxIter = self.config.maxIterRegression
#         if not nSigmaClip:
#             nSigmaClip = self.config.nSigmaClipRegression

#         nIter = 0
#         sctrl = afwMath.StatisticsControl()
#         sctrl.setNumSigmaClip(nSigmaClip)

#         if fixThroughOrigin:
#             while nIter < maxIter:
#                 nIter += 1
#                 self.log.debug("Origin fixed, iteration # %s using %s elements:" % (nIter, np.shape(x)[0]))
#                 TEST = x[:, np.newaxis]
#                 slope, _, _, _ = np.linalg.lstsq(TEST, y)
#                 slope = slope[0]
#                 res = (y - slope * x) / x
#                 resMean = afwMath.makeStatistics(res, afwMath.MEANCLIP, sctrl).getValue()
#                 resStd = np.sqrt(afwMath.makeStatistics(res, afwMath.VARIANCECLIP, sctrl).getValue())
#                 index = np.where((res > (resMean + nSigmaClip*resStd)) |
#                                  (res < (resMean - nSigmaClip*resStd)))
#                 self.log.debug("%.3f %.3f %.3f %.3f" % (resMean, resStd, np.max(res), nSigmaClip))
#                 if np.shape(np.where(index))[1] == 0 or (nIter >= maxIter):  # run out of points or iters
#                     break
#                 x = np.delete(x, index)
#                 y = np.delete(y, index)

#             return slope, 0

#         while nIter < maxIter:
#             nIter += 1
#             self.log.debug("Iteration # %s using %s elements:" % (nIter, np.shape(x)[0]))
#             xx = np.vstack([x, np.ones(len(x))]).T
#             ret, _, _, _ = np.linalg.lstsq(xx, y)
#             slope, intercept = ret
#             res = y - slope*x - intercept
#             resMean = afwMath.makeStatistics(res, afwMath.MEANCLIP, sctrl).getValue()
#             resStd = np.sqrt(afwMath.makeStatistics(res, afwMath.VARIANCECLIP, sctrl).getValue())
#             index = np.where((res > (resMean + nSigmaClip * resStd)) | (res < \CZW
# resMean - nSigmaClip * resStd))
#             self.log.debug("%.3f %.3f %.3f %.3f" % (resMean, resStd, np.max(res), nSigmaClip))
#             if np.shape(np.where(index))[1] == 0 or (nIter >= maxIter):  # run out of points, or iterations
#                 break
#             x = np.delete(x, index)
#             y = np.delete(y, index)

#         return slope, intercept

#     def generateKernel(self, corrs, means, objId, rejectLevel=None):
#         """Generate the full kernel from a list of cross-correlations and means.

#         Taking a list of quarter-image, gain-corrected cross-correlations,
#         do a pixel-wise sigma-clipped mean of each,
#         and tile into the full-sized kernel image.

#         Each corr in corrs is one quarter of the full cross-correlation,
#         and has been gain-corrected. Each mean in means is a tuple of the means
#         of the two individual images, corresponding to that corr.

#         Parameters:
#         -----------
#         corrs : `list` of `numpy.ndarray`, (Ny, Nx)
#             A list of the quarter-image cross-correlations
#         means : `list` of `tuples` of `floats`
#             The means of the input images for each corr in corrs
#         rejectLevel : `float`, optional
#             This is essentially is a sanity check parameter.
#             If this condition is violated there is something unexpected
#             going on in the image, and it is discarded from the stack before
#             the clipped-mean is calculated.
#             If not provided then config.xcorrCheckRejectLevel is used

#         Returns:
#         --------
#         kernel : `numpy.ndarray`, (Ny, Nx)
#             The output kernel
#         """
#         self.log.info('Calculating kernel for %s'%objId)

#         if not rejectLevel:
#             rejectLevel = self.config.xcorrCheckRejectLevel

#         if self.config.correlationQuadraticFit:
#             xcorrList = []
#             fluxList = []

#             for corrNum, ((mean1, mean2), corr) in enumerate(zip(means, corrs)):
#                 msg = 'For item %s, initial corr[0,0] = %g, corr[1,0] = %g'%(corrNum, \CZW
# corr[0, 0], corr[1, 0])
#                 self.log.info(msg)
#                 if self.config.level == 'DETECTOR':
#                     #  This is now done in _applyGains() but only if level is not DETECTOR
#                     corr[0, 0] -= (mean1 + mean2)
#                 fullCorr = self._tileArray(corr)

#                 # Craig Lage says he doesn't understand the negative sign, but it needs to be there
#                 xcorrList.append(-fullCorr / 2.0)
#                 flux = (mean1 + mean2) / 2.0
#                 fluxList.append(flux * flux)
#                 # We're using the linear fit algorithm to find a quadratic fit,
#                 # so we square the x-axis.
#                 # The step below does not need to be done, but is included
#                 # so that correlations can be compared
#                 # directly to existing code.  We might want to take it out.
#                 corr /= -1.0*(mean1**2 + mean2**2)

#             if not xcorrList:
#                 raise RuntimeError("Cannot generate kernel because all inputs were discarded. "
#                                    "Either the data is bad, or config.xcorrCheckRejectLevel is too low")

#             # This method fits a quadratic vs flux and keeps only the quadratic term.
#             meanXcorr = np.zeros_like(fullCorr)
#             xcorrList = np.asarray(xcorrList)

#             for i in range(np.shape(meanXcorr)[0]):
#                 for j in range(np.shape(meanXcorr)[1]):
#                     # Note the i,j inversion.  This serves the same function as the transpose step in
#                     # the base code.  I don't understand why it is there, but I put it in to be consistent.
#                     slopeRaw, interceptRaw, rVal, pVal, stdErr = stats.linregress(np.asarray(fluxList),
#                                                                                   xcorrList[:, j, i])
#                     try:
#                         slope, intercept = self._iterativeRegression(np.asarray(fluxList),
#                                                                      xcorrList[:, j, i],
#                                                                      fixThroughOrigin=True)
#                         msg = "(%s,%s):Slope of raw fit: %s, intercept: %s p value: %s" % (i, j, slopeRaw,
#                                                                                            interceptRaw,
# pVal)
#                         self.log.debug(msg)
#                         self.log.debug("(%s,%s):Slope of fixed fit: %s" % (i, j, slope))

#                         meanXcorr[i, j] = slope
#                     except ValueError:
#                         meanXcorr[i, j] = slopeRaw

#                     msg = f"i={i}, j={j}, slope = {slope:.6g}, slopeRaw = {slopeRaw:.6g}"
#                     self.log.debug(msg)
#             self.log.info('Quad Fit meanXcorr[0,0] = %g, meanXcorr[1,0] = %g'%(meanXcorr[8, 8],
#                                                                                meanXcorr[9, 8]))

#         else:
#             # Try to average over a set of possible inputs.
#             # This generates a simple function of the kernel that
#             # should be constant across the images, and averages that.
#             xcorrList = []
#             sctrl = afwMath.StatisticsControl()
#             sctrl.setNumSigmaClip(self.config.nSigmaClipKernelGen)

#             for corrNum, ((mean1, mean2), corr) in enumerate(zip(means, corrs)):
#                 corr[0, 0] -= (mean1 + mean2)
#                 if corr[0, 0] > 0:
#                     self.log.warn('Skipped item %s due to unexpected value of (variance-mean)' % corrNum)
#                     continue
#                 corr /= -1.0*(mean1**2 + mean2**2)

#                 fullCorr = self._tileArray(corr)

#                 xcorrCheck = np.abs(np.sum(fullCorr))/np.sum(np.abs(fullCorr))
#                 if xcorrCheck > rejectLevel:
#                     self.log.warn("Sum of the xcorr is unexpectedly high. " \CZW
# "Investigate item num %s for %s. \n"
#                                   "value = %s" % (corrNum, objId, xcorrCheck))
#                     continue
#                 xcorrList.append(fullCorr)

#             if not xcorrList:
#                 raise RuntimeError("Cannot generate kernel because all inputs were discarded. "
#                                    "Either the data is bad, or config.xcorrCheckRejectLevel is too low")

#             # stack the individual xcorrs and apply a per-pixel clipped-mean
#             meanXcorr = np.zeros_like(fullCorr)
#             xcorrList = np.transpose(xcorrList)
#             for i in range(np.shape(meanXcorr)[0]):
#                 for j in range(np.shape(meanXcorr)[1]):
#                     meanXcorr[i, j] = afwMath.makeStatistics(xcorrList[i, j],
#                                                              afwMath.MEANCLIP, sctrl).getValue()

#         if self.config.correlationModelRadius < (meanXcorr.shape[0] - 1) / 2:
#             sumToInfinity = self._buildCorrelationModel(meanXcorr, self.config.correlationModelRadius,
#                                                         self.config.correlationModelSlope)
#             self.log.info("SumToInfinity = %s" % sumToInfinity)
#         else:
#             sumToInfinity = 0.0
#         if self.config.forceZeroSum:
#             self.log.info("Forcing sum of correlation matrix to zero")
#             meanXcorr = self._forceZeroSum(meanXcorr, sumToInfinity)

#         return meanXcorr, self.successiveOverRelax(meanXcorr)

#     @staticmethod
#     def _buildCorrelationModel(array, replacementRadius, slope):
#         """Given an array of correlations, build a model
#         for correlations beyond replacementRadius pixels from the center
#         and replace the measured values with the model.

#         Parameters:
#         -----------
#         input : `np.array`
#             The square input array, assumed square and with
#             shape (2n+1) x (2n+1)

#         Returns:
#         --------
#         output : `np.array`
#             The same array, with the outer values
#             replaced with a smoothed model.
#         """
#         assert(array.shape[0] == array.shape[1])
#         assert(array.shape[0] % 2 == 1)
#         assert(replacementRadius > 1)
#         center = int((array.shape[0] - 1) / 2)
#         # First we check if either the [0,1] or [1,0] correlation is positive.
#         # If so, the data is seriously messed up. This has happened in some bad amplifiers.
#         # In this case, we just return the input array unchanged.
#         if (array[center, center + 1] >= 0.0) or (array[center + 1, center] >= 0.0):
#             return 0.0

#         intercept = (np.log10(-array[center, center + 1]) + np.log10(-array[center + 1, center])) / 2.0
#         preFactor = 10**intercept
#         slopeFactor = 2.0*abs(slope) - 2.0
#         sumToInfinity = 2.0*np.pi*preFactor / (slopeFactor*(float(center)+0.5)**slopeFactor)
#         # sum_to_ininity is the integral of the model beyond what is measured.
#         # It is used to adjust C00 in the case of forcing zero sum

#         # Now replace the pixels beyond replacementRadius with the model values
#         for i in range(array.shape[0]):
#             for j in range(array.shape[1]):
#                 r2 = float((i-center)*(i-center) + (j-center)*(j-center))
#                 if abs(i-center) < replacementRadius and abs(j-center) < replacementRadius:
#                     continue
#                 else:
#                     newCvalue = -preFactor * r2**slope
#                     array[i, j] = newCvalue
#         return sumToInfinity
