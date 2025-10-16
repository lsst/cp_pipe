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

from .utils import (
    extractCalibDate,
    ElectrostaticFit,
)
from .cpLinearitySolve import ptcLookup
from lsst.ip.isr.isrFunctions import symmetrize
from lsst.ip.isr import ElectrostaticBrighterFatter
from lmfit import Parameters


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

    output = cT.Output(
        name="ebf",
        doc="Output measured brighter-fatter electrostatic model.",
        storageClass="ElectrostaticBrighterFatter",
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
                                             pipelineConnections=ElectrostaticBrighterFatterSolveConnections):

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
            "methods see the documentation. Default: 'leastsq'.",
        default="leastsq",
    )
    doNormalizeElectrostaticModel = pexConfig.Field(
        dtype=bool,
        doc="Do you want apply a final normalization to the modeled "
            "aMatrix? Default: False.",
        default=False,
    )
    doFitOffset = pexConfig.Field(
        dtype=bool,
        doc="Do you want to fit an offset to the a matrix? This caused "
            "by long range correlations in the data. Default: False.",
        default=False,
    )
    nImageChargePairs = pexConfig.Field(
        dtype=int,
        doc="Number of image charge pairs to use when computing "
            "Gauss's law. The larger number, the better, and an "
            "odd number is preferred. Default: 11.",
        default=11,
    )
    doCheckValidity = pexConfig.Field(
        dtype=bool,
        doc="Check the AMP kernels for basic validity criteria? "
            "Will set electrostaticBfCalib.valid for each amp.",
        default=True,
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
            "`zq`, zsh`, `zsv`, `a`, `b`, `alpha`,  and `beta`. If False, "
            "the parameter will be fixed to the initial value set in "
            "initialParameterDict. `thickness` and `pixelsize` are always "
            "fixed.See the class docstring for descriptions and units of each "
            "parameter.",
        default={
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

        # Use the dimensions to set
        # electrostaticBfCalib/provenance
        # information.
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
        outputs.output.updateMetadata(setDate=False, **kwargs)

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

            ``output``
                Resulting Brighter-Fatter Kernel
                (`lsst.ip.isr.BrighterFatterKernel`).
        """
        detector = camera[inputDims['detector']]
        detName = detector.getName()

        inputRange = inputPtc.covMatrixSide
        fitRange = self.config.fitRange

        if not inputPtc.ptcFitType.startswith("FULLCOVARIANCE"):
            raise ValueError(
            "ptcFitType must be FULLCOVARIANCE* to solve for electrostatic solution."
            )
        if self.config.fitRange > inputPtc.covMatrixSide:
            raise ValueError(
            "Cannot compute the electrostatic solution if "
            "int(inputPtc.covMatrixSide) < fitRange."
            )

        # Initialize the output calibration
        electrostaticBfCalib = ElectrostaticBrighterFatter(
            camera=camera,
            detectorId=detector.getId(),
            inputRange=inputRange,
            fitRange=fitRange,
        )

        badAmps = inputPtc.badAmps
        electrostaticBfCalib.badAmps = badAmps
        electrostaticBfCalib.gain = inputPtc.gain

        aMatrixDict = inputPtc.aMatrix
        aMatrixSigmaDict = {amp: np.nan for amp in aMatrixDict.keys()}  # inputPtc.aMatrixError
        aMatrixList = [m for _, m in aMatrixDict.items() if _ not in badAmps]


        nGoodAmps = len(detector.getAmplifiers()) - len(badAmps)
        if nGoodAmps == 0:
            self.log.warning("The entire detector is bad and cannot generate a "
                                "detector solution.")
            return pipeBase.Struct(
                outputBF=electrostaticBfCalib,
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
            aMatrixSigma = np.ones_like(aMatrix)
            aMatrixSigma[0, :] = 2
            aMatrixSigma[:, 0] = 2
            aMatrixSigma[0, 0] = np.sqrt(32)

        else:
            aMatrix = np.mean(aMatrixList, axis=0)
            aMatrixSigma = np.std(aMatrixList, axis=0)


        electrostaticBfCalib.updateMetadataFromExposures([inputPtc])

        # Set initial parameters using config
        thickness = self.config.initialParametersDict['thickness']
        pixelsize = self.config.initialParametersDict['pixelsize']
        zq = self.config.initialParametersDict['zq']
        zsh = self.config.initialParametersDict['zsh']
        zsv = self.config.initialParametersDict['zsv']
        a = self.config.initialParametersDict['a']
        b = self.config.initialParametersDict['b']
        alpha = self.config.initialParametersDict['alpha']
        beta = self.config.initialParametersDict['beta']

        initialParams = Parameters()
        initialParams.add(
            "thickness",
            value=thickness,
            vary=False,
        )
        initialParams.add(
            "pixelsize",
            value=pixelsize,
            vary=False,
        )
        initialParams.add(
            "zq",
            value=zq,
            vary=self.config.parametersToVary["zq"],
            min=0,
            max=0.25*thickness,
        )
        initialParams.add(
            "zsh",
            value=zsh,
            vary=self.config.parametersToVary["zsh"],
            min=0,
            max=0.25*thickness,
        )
        initialParams.add(
            "zsv",
            value=zsv,
            vary=self.config.parametersToVary["zsv"],
            min=0,
            max=0.25*thickness,
        )
        initialParams.add(
            "a",
            value=a,
            vary=self.config.parametersToVary["a"],
            min=0,
            max=pixelsize,
        )
        initialParams.add(
            "b",
            value=b,
            vary=self.config.parametersToVary["b"],
            min=0,
            max=pixelsize,
        )
        initialParams.add(
            "alpha",
            value=alpha,
            vary=self.config.parametersToVary["alpha"],
            min=-10,
            max=10,
        )
        initialParams.add(
            "beta",
            value=beta,
            vary=self.config.parametersToVary["beta"],
            min=-10,
            max=10,
        )

        # Compute the electrostatic fit
        electrostaticFit = ElectrostaticFit(
            initialParams=initialParams,
            fitMethod=self.config.fitMethod,
            aMatrix=aMatrix,
            aMatrixSigma=aMatrixSigma,
            fitRange=fitRange,
            doFitOffset=self.config.doFitOffset,
            nImageChargePairs=self.config.nImageChargePairs,
        )

        # Do the fit
        result = electrostaticFit.fit()

        # Check if fit was successful
        if not result.success:
            raise RuntimeError(f"Fit was not successful: {result.message}")

        # Save the fit
        finalParams = result.params
        electrostaticBfCalib.fitParamNames = np.array(
            [name for name, p in finalParams.items() if p.vary]
        )
        electrostaticBfCalib.freeFitParamNames = np.array(
            [name for name, p in finalParams.items() if p.vary]
        )
        electrostaticBfCalib.fitParams = finalParams.valuesdict()
        electrostaticBfCalib.fitParamErrors = {
            name: p.stderr for name, p in finalParams.items()
        }
        electrostaticBfCalib.fitChi2 = result.chisqr
        electrostaticBfCalib.fitReducedChi2 = result.redchi
        electrostaticBfCalib.fitParamCovMatrix = result.covar

        # Compute the final model
        aMatrixModel = electrostaticFit.model(result.params)

        modelNormalization = [1, 0]
        if self.config.doNormalizeElectrostaticModel:
            m, o = electrostaticFit.normalizeModel(aMatrixModel)
            modelNormalization = [m, o]
            aMatrixModel = m*aMatrixModel + o
            self.log.info(
                "Normalization (factor, offset) for amp %s: (%.3f, %.3f)", m, o
            )

        # Save the original data and the final model.
        electrostaticBfCalib.aMatrix = aMatrix
        electrostaticBfCalib.aMatrixSigma = aMatrixSigma
        electrostaticBfCalib.aMatrixModel = aMatrixModel
        electrostaticBfCalib.aMatrixSum = symmetrize(aMatrix).sum()
        electrostaticBfCalib.aMatrixModelSum = symmetrize(aMatrixModel).sum()
        electrostaticBfCalib.modelNormalization = modelNormalization

        # Fit result information
        self.log.info(
            '%s a,b (microns): %f %f',
            detName,
            round(finalParams['a'].value, 3),
            round(finalParams['b'].value, 3)
        )
        self.log.info(
            '%s Reduced Chi2: %g',
            detName,
            finalParams
        )
        self.log.info(
            finalParams.pretty_print(
                columns=['value', 'min', 'max', 'stderr', 'vary']
            )
        )
        self.log.info(
            '%s Sum: data %g model %g',
            detName,
            symmetrize(aMatrix).sum(),
            symmetrize(aMatrixModel).sum()
        )

        # TODO: add conversion depth probability distribution
        # for each band/wavelength. Compute boundary shifts
        # for a single electron
        pd = electrostaticFit.computePixelDistortions(conversionWeights=None)

        aN, aS, aE, aW = (pd.aN, pd.aS, pd.aE, pd.aW)
        ath = pd.ath
        athMinusBeta = pd.athMinusBeta
        fitMask = np.zeros_like(aN, dtype=bool)
        fitMask[:fitRange, :fitRange] = True

        electrostaticBfCalib.fitMask = fitMask
        electrostaticBfCalib.ath = ath
        electrostaticBfCalib.athMinusBeta = athMinusBeta
        electrostaticBfCalib.aN = aN
        electrostaticBfCalib.aS = aS
        electrostaticBfCalib.aE = aE
        electrostaticBfCalib.aW = aW

        # Check for validity
        if self.config.doCheckValidity:
            # Todo:
            pass

        return pipeBase.Struct(
            outputBF=electrostaticBfCalib,
        )
