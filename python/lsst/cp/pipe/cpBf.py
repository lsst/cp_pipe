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

__all__ = ['ElectrostaticBrighterFatterSolveTask',
           'ElectrostaticBrighterFatterSolveConfig']

import numpy as np

import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase
import lsst.pipe.base.connectionTypes as cT
from scipy.optimize import leastsq

from .utils import (funcPolynomial, irlsFit, extractCalibDate)
from .cpLinearitySolve import ptcLookup
from lsst.ip.isr import symmetrize
from lsst.ip.isr import (
    ElectrostaticBrighterFatterCalibration,
    ElectrostaticFit,
    BoundaryShifts,
)


class ElectrostaticBrighterFatterSolveConnections(pipeBase.PipelineTaskConnections,
                                                  dimensions=("instrument", "detector")):
    camera = cT.PrerequisiteInput(
        name="camera",
        doc="Camera associated with this data.",
        storageClass="Camera",
        dimensions=("instrument", ),
        isCalibration=True,
    )
    inputPtc = cT.PrerequisiteInput(
        name="ptc",
        doc="Photon transfer curve dataset.",
        storageClass="PhotonTransferCurveDataset",
        dimensions=("instrument", "detector"),
        isCalibration=True,
        lookupFunction=ptcLookup,
    )
    inputBfkPtc = cT.Input(
        name="bfkPtc",
        doc="Input BFK PTC dataset.",
        storageClass="PhotonTransferCurveDataset",
        dimensions=("instrument", "detector"),
        isCalibration=True,
    )

    outputBF = cT.Output(
        name="bf",
        doc="Output measured brighter-fatter electrostatic model.",
        storageClass="ElectrostaticBrighterFatterCalibration",
        dimensions=("instrument", "detector"),
        isCalibration=True,
    )

    def __init__(self, *, config=None):
        if config.useBfkPtc:
            del self.inputPtc
            del self.dummy
        else:
            del self.inputBfkPtc


class ElectrostaticBrighterFatterSolveConfig(pipeBase.PipelineTaskConfig,
                                             pipelineConnections=BrighterFatterKernelSolveConnections):

    maxFitRange = pexConfig.Field(
        dtype=float,
        doc="Maximum pixel range to compute the electrostatic fit.",
        default=8,
    )
    doCheckValidity = pexConfig.Field(
        dtype=bool,
        doc="Check the AMP kernels for basic validity criteria? "
            "Will set bfk.valid for each amp.",
        default=True,
    )
    fixThicknessTo = pexConfig.Field(
        dtype=float,
        doc="If set, fix thickness to this value in the fit (microns). Use None to leave free.",
        default=None,
    )
    fixPixSizeTo = pexConfig.Field(
        dtype=float,
        doc="If set, fix pixel size to this value in the fit (microns). Use None to leave free.",
        default=None,
    )
    fixAlphaTo = pexConfig.Field(
        dtype=float,
        doc="If set, fix alpha (see https://arxiv.org/abs/2301.03274, Equation 16) to this value in the fit. Use None to leave free.",
        default=None,
    )
    fixBetaTo = pexConfig.Field(
        dtype=float,
        doc="If set, fix beta (see https://arxiv.org/abs/2301.03274, Equation 16) to this value in the fit. Use None to leave free.",
        default=None,
    )


class ElectrostaticBrighterFatterSolveTask(pipeBase.PipelineTask):
    """Measure appropriate Brighter-Fatter Kernel from the PTC dataset.
    """

    ConfigClass = ElectrostaticBrighterFatterSolveConfig
    _DefaultName = 'cpElectrostaticBfMeasure'

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

        if self.config.useBfkPtc:
            inputs["inputDims"] = dict(inputRefs.inputBfkPtc.dataId.required)
            inputs["inputPtc"] = inputs["inputBfkPtc"]
            del inputs["inputBfkPtc"]
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
        outputs.outputBFK.updateMetadata(setDate=False, **kwargs)

        butlerQC.put(outputs, outputRefs)

    def run(self, inputPtc, camera, inputDims):
        """Combine covariance information from PTC into brighter-fatter
        kernels.

        Parameters
        ----------
        inputPtc : `lsst.ip.isr.PhotonTransferCurveDataset`
            PTC data containing per-amplifier covariance measurements.
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
        detector = camera[inputDims['detector']]
        detName = detector.getName()


        if not inputPtc.ptcFitType.startswith("FULLCOVARIANCE"):
            raise ValueError("ptcFitType must be FULLCOVARIANCE* to solve for electrostatic solution.")
        if maxFitRange > inputPtc.covMatrixSide:
            raise ValueError("Cannot compute the electrostatic solution if int(inputPtc.covMatrixSide) < maxFitRange.")

        # Get flux sample dictionary
        fluxSampleDict = {ampName: 0.0 for ampName in inputPtc.ampNames}
        for ampName in inputPtc.ampNames:
            if 'ALL_AMPS' in self.config.covModelFluxSample:
                fluxSampleDict[ampName] = self.config.covModelFluxSample['ALL_AMPS']
            elif ampName in self.config.covModelFluxSample:
                fluxSampleDict[ampName] = self.config.covModelFluxSample[ampName]

        bf = ElectrostaticBrighterFatterCalibration(
            camera=camera,
            detectorId=detector.getId(),
        )
        bf.rawMeans = inputPtc.rawMeans  # ADU
        bf.rawVariances = inputPtc.rawVars  # ADU^2
        bf.expIdMask = inputPtc.expIdMask
        bf.aMatrix = inputPtc.aMatrix
        bf.aMatrixErr = inputPtc.ptcFitParsError

        # Use the PTC covariances as the cross-correlations.
        # The input covariances are in (x, y) index
        # ordering, as is the aMatrix.
        bf.rawXcorrs = inputPtc.covariances  # ADU^2
        bf.badAmps = inputPtc.badAmps
        maxFitRange = self.config.maxFitRange
        bf.shape = (maxFitRange, maxFitRange)
        bf.gain = inputPtc.gain
        bf.noise = inputPtc.noise
        bf.valid = dict()
        bf.updateMetadataFromExposures([inputPtc])

        for amp in detector:
            ampName = amp.getName()
            gain = inp.gain[ampName]
            aMatrix = inputPtc.aMatrix[ampName]
            mask = inputPtc.expIdMask[ampName]

            aMatrixSigma = inputPtc.ptcFitParsError

            if np.isfinite(aMatrixSigma) or None:
                # If the general uncertainty is one, the measurements uncertainties
                # along the axes are sqrt(2), and sqrt(8) in (0,0) (because the
                # slope of C00 is fitted)
                #
                # So, this sets variances at (1,2,8) for the three groups.
                # then the number of replicas (when going to 4 quadrants)
                # are (4,2,1) for the 3 same groups.
                #
                # The effective variances are then in the ratios (1/4,1,8)
                # or (1,4,32)
                aMatrixSigma = np.ones_like(aMatrix)
                aMatrixSigma[0,:] = 2
                aMatrixSigma[:,0] = 2
                aMatrixSigma[0,0] = np.sqrt(32)
                self.log.warning("No aMatrix sigma found for amp %s in input PTC, setting unceratinties "
                                 "to default value for electrostatic fit.", ampName)

            if gain <= 0:
                # We've received very bad data.
                self.log.warning("Impossible gain recieved from PTC for %s: %f. Skipping bad amplifier.",
                                 ampName, gain)
                bfk.meanXcorrs[ampName] = np.zeros(bfk.shape)
                bfk.ampKernels[ampName] = np.zeros(bfk.shape)
                bfk.rawXcorrs[ampName] = np.zeros((len(mask), inputPtc.covMatrixSide, inputPtc.covMatrixSide))
                bfk.valid[ampName] = False
                continue

            # Compute the electrostatic fit
            fit = electro_fit(meas_a, sig_meas_a,
                        input_range=options.input_range,
                        output_range=options.output_range)

            # Set initial parameters
            fit.set_params({
                'thickness': 100.,
                'pixsize' : 10,
                'z_q' : 1,
                'zsh': 2.,
                'zsv' : 3.,
                'a': 1.,
                'b': 3.,
                'alpha': 1.,
                'beta': 0,
            })

            # Fix parameters if config values are set
            if self.config.fixThicknessTo is not None:
                fit.params['thickness'] = self.config.fixThickness
                fit.params.fix('thickness')
            if self.config.fixPixSizeTo is not None:
                fit.params['pixsize'] = self.config.fixPixsize
                fit.params.fix('pixsize')
            if self.config.fixBetaTo is not None:
                fit.params['beta'] = self.config.fixBeta
                fit.params.fix('beta')

            params = fit.get_params()
            print("Starting parameters: ", params)

            fitted_params, cov_params, ls_stuff, mesg, ierr = leastsq(fit.wres, params, full_output=True, maxfev=200000)
            if ierr not in [1,2,3,4] :
                raise RuntimeError(mesg)
            else :
                fitted_params = params
            fit.params.free = fitted_params

            del fitted_params # to make sure the sign flip just below cannot be reversed.
            # a and b can go negative, but the calculations only
            # depend on the abs value
            fit.params['a'] = np.abs(fit.params['a'].full[0])
            fit.params['b'] = np.abs(fit.params['b'].full[0])
            self.log.info('%s %s a,b (microns): %f %f' % (detName, ampName, round(fit.params['a'].full[0], 2), round(fit.params['b'].full[0], 2)))
            self.log.info('%s %s Chi2: %g' % (detName, ampName, fit.getChi2()))
            self.log.info('%s %s Model: \n' % (detName, ampName, fit.model()))
            self.log.info('%s %s Data: \n' % (detName, ampName, fit.get_a()))
            self.log.info('%s %s Params:' % (detName, ampName, fit.params))
            self.log.info('%s %s Sum: data %g model %g' % (detName, ampName, (symmetrize(fit.get_a()).sum(), symmetrize(fit.model()).sum())))

            # fit.write_results_np('avalues%s.npy'%tag)
            # fit.write_results_txt('bfshifts%s.list'%tag,'avalues%s.npy'%tag)
            # fit.write_params('elec_params%s.pkl'%tag)
            # pickle.dump(cov_params, open('cov_params%s.pkl'%tag,'wb'))

            # Save the fit
            fitParamNames = np.array([x for x in fit.params._struct])
            bf.fitParamNames = fitParamNames
            bf.freeFitParams = paramNames[np.array(fit.params._free)]
            bf.fitParams[ampName] = fit.params
            bf.chi2[ampName] = fit.getChi2()
            bf.aMatrixData[ampName] = fit.get_a()
            bf.aMatrixModel[ampName] = fit.model()
            bf.aMatrixDataSum[ampName] = symmetrize(fit.get_a()).sum()
            bf.aMatrixModelSum[ampName] = symmetrize(fit.model()).sum()

            if self.config.normalizeElectrostaticModel:
                m, o = fit.normalize_model(fit.raw_model())
                self.log.info("Normalization (multiplicative factor, offset) for amp %s: (%f,%f)", ampName, m, o)

            # Compute boundary shifts for a single electron
            pixelDistortions = fit.computePixelDistortions(conversion_weights=None) # Todo: add conversion depth probability distribution for each band/wavelength
            bf.pixelDistortions[ampName] = pixelDistortions

            # Check for validity
            if self.config.doCheckValidity:
                # Todo:
                bf.valid[ampName] = True
            else:
                # The kernel at this point will be valid
                bf.valid[ampName] = True

        # Assemble a detector kernel?
        # if self.config.level == 'DETECTOR':
        #     if self.config.correlationQuadraticFit:
        #         preKernel = self.quadraticCorrelations(detectorCorrList, detectorFluxes, f"Amp: {ampName}")
        #     else:
        #         preKernel = self.averageCorrelations(detectorCorrList, f"Det: {detName}")
        #     finalSum = np.sum(preKernel)
        #     center = int((bfk.shape[0] - 1) / 2)

        #     postKernel = self.successiveOverRelax(preKernel)
        #     bfk.detKernels[detName] = postKernel
        #     self.log.info("Det: %s Sum: %g  Center Info Pre: %g  Post: %g",
        #                   detName, finalSum, preKernel[center, center], postKernel[center, center])

        return pipeBase.Struct(
            outputBF=bf,
        )

