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
"""Computation of electrostatic solution of brighter-fatter effect impact
on pixel distortions"""

__all__ = ['ElectrostaticBrighterFatterSolveTask',
           'ElectrostaticBrighterFatterSolveConfig']

import numpy as np
import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase
import lsst.pipe.base.connectionTypes as cT

from .utils import (
    extractCalibDate,
    ElectrostaticFit,
)
from .cpLinearitySolve import ptcLookup
from lsst.ip.isr.isrFunctions import symmetrize
from lsst.ip.isr import ElectrostaticBrighterFatterDistortionMatrix
from lmfit import Parameters, report_fit


class ElectrostaticBrighterFatterSolveConnections(pipeBase.PipelineTaskConnections,
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
        doc="Camera associated with this data.",
        storageClass="Camera",
        dimensions=("instrument", ),
        isCalibration=True,
    )
    inputPtc = cT.PrerequisiteInput(
        name="ptc",
        doc="Photon transfer curve dataset.",
        storageClass="IsrCalib",
        dimensions=("instrument", "detector"),
        isCalibration=True,
        lookupFunction=ptcLookup,
    )
    inputBfPtc = cT.Input(
        name="bfPtc",
        doc="Input BF PTC dataset.",
        storageClass="IsrCalib",
        dimensions=("instrument", "detector"),
        isCalibration=True,
    )

    output = cT.Output(
        name="electroBfDistortionMatrix",
        doc="Output measured brighter-fatter electrostatic model.",
        storageClass="IsrCalib",
        dimensions=("instrument", "detector"),
        isCalibration=True,
    )

    def __init__(self, *, config=None):
        if config.useBfPtc:
            del self.inputPtc
            del self.dummy
        else:
            del self.inputBfPtc


class ElectrostaticBrighterFatterSolveConfig(pipeBase.PipelineTaskConfig,
                                             pipelineConnections=ElectrostaticBrighterFatterSolveConnections):

    useBfPtc = pexConfig.Field(
        dtype=bool,
        doc="Use a BF ptc in a single pipeline?",
        default=False,
    )
    fitRange = pexConfig.Field(
        dtype=int,
        doc="Maximum pixel range to compute the electrostatic fit.",
        default=8,
    )
    fitMethod = pexConfig.Field(
        dtype=str,
        doc="Minimization technique to fit the electrostatic solution. "
            "Should be one of the available fitting methods in "
            "`lmfit.minimizer.Minimizer.minimize`. For list of all possible "
            "methods see the documentation.",
        default="leastsq",
    )
    doNormalizeElectrostaticModel = pexConfig.Field(
        dtype=bool,
        doc="Do you want apply a final normalization to the modeled "
            "aMatrix?",
        default=False,
    )
    doFitNormalizationOffset = pexConfig.Field(
        dtype=bool,
        doc="Do you want to fit an offset to the a matrix? This caused "
            "by long range correlations in the data. Only used if "
            "doNormalizeElectrostaticModel.",
        default=True,
    )
    nImageChargePairs = pexConfig.Field(
        dtype=int,
        doc="Number of image charge pairs to use when computing "
            "Gauss's law. The larger number, the better, and an "
            "odd number is preferred.",
        default=11,
    )
    initialParametersDict = pexConfig.DictField(
        keytype=str,
        itemtype=float,
        doc="Initial fit parameters, should contain `thickness`, "
            "`pixelsize`, `zq`, zsh`, `zsv`, `a`, `b`, `alpha`, "
            " and `beta`. See the class docstring for descriptions "
            " and units of each parameter.",
        default={
            'thickness': 100.0,
            'pixelsize': 10.0,
            'zq': 1.0,
            'zsh': 2.0,
            'zsv': 3.0,
            'a': 2.0,
            'b': 2.0,
            'alpha': 1.0,
            'beta': 0.0,
        },
    )
    parametersToVary = pexConfig.DictField(
        keytype=str,
        itemtype=bool,
        doc="Dictionary of parameters and booleans which will configure "
            "if the parameter is allowed to vary in the fit, should contain "
            "`thickness`,`pixelsize`, `zq`, zsh`, `zsv`, `a`, `b`, `alpha`, "
            "and `beta`. If False, the parameter will be fixed to the initial "
            "value set in initialParameterDict. See the class docstring for "
            "descriptions and units of each parameter.",
        default={
            'thickness': False,
            'pixelsize': False,
            'zq': True,
            'zsh': True,
            'zsv': True,
            'a': True,
            'b': True,
            'alpha': True,
            'beta': True,
        },
    )


class ElectrostaticBrighterFatterSolveTask(pipeBase.PipelineTask):
    """Find the complete electrostatic solution to the given PTC.
    """

    ConfigClass = ElectrostaticBrighterFatterSolveConfig
    _DefaultName = 'cpElectrostaticBfSolve'

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

        # Use the dimensions to set
        # electroBfDistortionMatrix/provenance
        # information.
        if self.config.useBfPtc:
            inputs["inputDims"] = dict(inputRefs.inputBfPtc.dataId.required)
            inputs["inputPtc"] = inputs["inputBfPtc"]
            inputs["dummy"] = []
            del inputs["inputBfPtc"]
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
        outputs.output.updateMetadata(setDate=False, **kwargs)

        butlerQC.put(outputs, outputRefs)

    def run(self, inputPtc, dummy, camera, inputDims):
        """Fit the PTC A MATRIX into a vectorized a matrix form
        based on a complete electrostatic solution.

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

            ``output``
                Resulting Brighter-Fatter electrostatic solution
                (`lsst.ip.isr.ElectrostaticBrighterFatterDistortionMatrix`).
        """
        detector = camera[inputDims['detector']]

        inputRange = int(inputPtc.covMatrixSideFullCovFit)
        fitRange = int(self.config.fitRange)

        if not inputPtc.ptcFitType.startswith("FULLCOVARIANCE"):
            raise ValueError(
                "ptcFitType must be FULLCOVARIANCE* to solve for electrostatic solution."
            )
        if fitRange > inputRange:
            raise ValueError(
                "Cannot compute the electrostatic solution if "
                "int(inputPtc.covMatrixSide) < fitRange."
            )

        # Initialize the output calibration
        electroBfDistortionMatrix = ElectrostaticBrighterFatterDistortionMatrix(
            camera=camera,
            detectorId=detector.getId(),
            inputRange=inputRange,
            fitRange=fitRange,
        )

        # Inherit data + metadata
        electroBfDistortionMatrix.updateMetadataFromExposures([inputPtc])

        badAmps = inputPtc.badAmps
        electroBfDistortionMatrix.badAmps = badAmps
        electroBfDistortionMatrix.gain = inputPtc.gain

        aMatrixDict = inputPtc.aMatrix
        aMatrixList = [m for _, m in aMatrixDict.items() if _ not in badAmps]

        nGoodAmps = len(detector.getAmplifiers()) - len(badAmps)
        if nGoodAmps == 0:
            self.log.warning("The entire detector is bad and cannot generate a "
                             "detector solution.")
            return pipeBase.Struct(
                output=electroBfDistortionMatrix,
            )
        elif nGoodAmps < 2:
            # If the general uncertainty is one, the measurement
            # uncertainties along the axes are sqrt(2), and sqrt(8)
            # in (0,0) (because the slope of C00 is fitted).
            #
            # This sets variances at (1, 2, 8) for the three groups.
            # Then the number of replicas (when going to 4 quadrants)
            # are (4, 2, 1) for the same three groups.
            #
            # The effective variances are then in the ratios (1/4, 1, 8)
            # or (1, 4, 32).
            self.log.warning("Not enough good amplifiers in this detector "
                             "to confidently solve. Setting aMatrixSigma "
                             "to default.")
            aMatrix = np.mean(aMatrixList, axis=0)
            aMatrixSigma = np.ones_like(aMatrix, dtype=np.float64)
            aMatrixSigma[0, :] = 2.0
            aMatrixSigma[:, 0] = 2.0
            aMatrixSigma[0, 0] = np.sqrt(32)

        else:
            aMatrix = np.mean(aMatrixList, axis=0)
            aMatrixSigma = np.std(aMatrixList, axis=0)

        # Ensure we have numpy arrays in 64-bit float precision
        aMatrix = np.asarray(aMatrix, dtype=np.float64)
        aMatrixSigma = np.asarray(aMatrixSigma, dtype=np.float64)

        # Set initial parameters using config
        thickness = np.float64(self.config.initialParametersDict['thickness'])
        pixelsize = np.float64(self.config.initialParametersDict['pixelsize'])
        zq = np.float64(self.config.initialParametersDict['zq'])
        zsh = np.float64(self.config.initialParametersDict['zsh'])
        zsv = np.float64(self.config.initialParametersDict['zsv'])
        a = np.float64(self.config.initialParametersDict['a'])
        b = np.float64(self.config.initialParametersDict['b'])
        alpha = np.float64(self.config.initialParametersDict['alpha'])
        beta = np.float64(self.config.initialParametersDict['beta'])

        initialParams = Parameters()
        initialParams.add(
            "thickness",
            value=thickness,
            min=0,
            max=1.25*thickness,
            vary=self.config.parametersToVary["thickness"],
        )
        initialParams.add(
            "pixelsize",
            value=pixelsize,
            min=0.5*np.abs(pixelsize),
            max=1.5*np.abs(pixelsize),
            vary=self.config.parametersToVary["pixelsize"],
        )
        initialParams.add(
            "zq",
            value=zq,
            vary=self.config.parametersToVary["zq"],
            min=0.0,
            max=0.5*thickness,
        )
        # These nuisance parameters ensure that
        # (zsh > zq) & (zsv > zq)
        initialParams.add(
            "zsh_minus_zq",
            value=zsh - zq,
            vary=self.config.parametersToVary["zsh"],
            min=1.0e-12,
            max=0.1*thickness,
        )
        initialParams.add(
            "zsh",
            vary=self.config.parametersToVary["zsh"],
            min=0.0,
            max=0.5*thickness,
            expr="zq + zsh_minus_zq" if self.config.parametersToVary["zsh"] else f"{zsh}",
        )
        initialParams.add(
            "zsv_minus_zq",
            value=zsv - zq,
            vary=self.config.parametersToVary["zsv"],
            min=1.0e-12,
            max=0.1*thickness,
        )
        initialParams.add(
            "zsv",
            vary=self.config.parametersToVary["zsv"],
            min=0.0,
            max=0.5*thickness,
            expr="zq + zsv_minus_zq" if self.config.parametersToVary["zsv"] else f"{zsv}",
        )
        initialParams.add(
            "a",
            value=a,
            vary=self.config.parametersToVary["a"],
            min=1.0e-5,
            max=0.35*pixelsize,
        )
        initialParams.add(
            "b",
            value=b,
            vary=self.config.parametersToVary["b"],
            min=1.0e-5,
            max=0.35*pixelsize,
        )
        initialParams.add(
            "alpha",
            value=alpha,
            vary=self.config.parametersToVary["alpha"],
            min=-10.0,
            max=10.0,
        )
        initialParams.add(
            "beta",
            value=beta,
            vary=self.config.parametersToVary["beta"],
            min=-10.0,
            max=10.0,
        )

        # Compute the electrostatic fit
        electrostaticFit = ElectrostaticFit(
            initialParams=initialParams,
            fitMethod=self.config.fitMethod,
            aMatrix=aMatrix,
            aMatrixSigma=aMatrixSigma,
            fitRange=fitRange,
            doFitNormalizationOffset=self.config.doFitNormalizationOffset,
            nImageChargePairs=self.config.nImageChargePairs,
        )

        # Do the fit
        result = electrostaticFit.fit()

        # Check if fit was successful
        if not result.success:
            raise RuntimeError(f"Fit was not successful: {result.message}")

        # Save the fit
        finalParams = result.params
        finalParamsDict = finalParams.valuesdict()

        # No longer need these nusiance variables
        if 'zsh_minus_zq' in finalParamsDict:
            del finalParamsDict['zsh_minus_zq']
        if 'zsv_minus_zq' in finalParamsDict:
            del finalParamsDict['zsv_minus_zq']

        fitParamNames = list(finalParamsDict.keys())
        freeFitParamNames = result.var_names
        electroBfDistortionMatrix.fitParamNames = fitParamNames
        electroBfDistortionMatrix.freeFitParamNames = freeFitParamNames
        electroBfDistortionMatrix.fitParams = finalParamsDict
        fitParamErrors = dict()
        for fitParamName in fitParamNames:
            if fitParamName in freeFitParamNames:
                fitParamErrors[fitParamName] = finalParams[fitParamName].stderr
            else:
                fitParamErrors[fitParamName] = 0.0
        electroBfDistortionMatrix.fitParamErrors = fitParamErrors

        electroBfDistortionMatrix.fitChi2 = result.chisqr
        electroBfDistortionMatrix.fitReducedChi2 = result.redchi
        electroBfDistortionMatrix.fitParamCovMatrix = result.covar

        # Compute the final model
        aMatrixModel = electrostaticFit.model(result.params)

        # Optional:
        # Perform the final model normalization
        modelNormalization = [1.0, 0.0]
        if self.config.doNormalizeElectrostaticModel:
            m, o = electrostaticFit.normalizeModel(aMatrixModel)
            modelNormalization = [m, o]
            aMatrixModel = m*aMatrixModel + o
            self.log.info(
                "Normalization (factor, offset) for amp %s: (%.3f, %.3f)", m, o
            )

        # Save the original data and the final model.
        electroBfDistortionMatrix.aMatrix = aMatrix
        electroBfDistortionMatrix.aMatrixSigma = aMatrixSigma
        electroBfDistortionMatrix.aMatrixModel = aMatrixModel
        electroBfDistortionMatrix.aMatrixSum = symmetrize(aMatrix).sum()
        electroBfDistortionMatrix.aMatrixModelSum = symmetrize(aMatrixModel).sum()
        electroBfDistortionMatrix.modelNormalization = modelNormalization

        # Fit result information
        self.log.info(report_fit(result))

        # TODO: add conversion depth probability distribution
        # for each band/wavelength. Compute boundary shifts
        # for a single electron
        pd = electrostaticFit.computePixelDistortions(conversionWeights=None)

        aN, aS, aE, aW = (pd.aN, pd.aS, pd.aE, pd.aW)
        ath = pd.ath
        athMinusBeta = pd.athMinusBeta
        fitMask = np.zeros_like(aN, dtype=bool)
        fitMask[:fitRange, :fitRange] = True

        electroBfDistortionMatrix.fitMask = fitMask
        electroBfDistortionMatrix.ath = ath
        electroBfDistortionMatrix.athMinusBeta = athMinusBeta
        electroBfDistortionMatrix.aN = aN
        electroBfDistortionMatrix.aS = aS
        electroBfDistortionMatrix.aE = aE
        electroBfDistortionMatrix.aW = aW

        # Optional: Check for validity
        # if self.config.doCheckValidity:
        #     # Todo:
        #     pass

        return pipeBase.Struct(
            output=electroBfDistortionMatrix,
        )
