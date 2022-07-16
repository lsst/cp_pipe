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
from collections import Counter

import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase
from lsst.cp.pipe.utils import (fitLeastSq, fitBootstrap, funcPolynomial, funcAstier, symmetrize)

from scipy.signal import fftconvolve
from scipy.optimize import least_squares
from itertools import groupby
from operator import itemgetter

import lsst.pipe.base.connectionTypes as cT

from lsst.ip.isr import PhotonTransferCurveDataset

from lsst.cp.pipe._lookupStaticCalibration import lookupStaticCalibration

import copy


__all__ = ['PhotonTransferCurveSolveConfig', 'PhotonTransferCurveSolveTask']


class PhotonTransferCurveSolveConnections(pipeBase.PipelineTaskConnections,
                                          dimensions=("instrument", "detector")):
    inputCovariances = cT.Input(
        name="ptcCovariances",
        doc="Tuple with measured covariances from flats.",
        storageClass="PhotonTransferCurveDataset",
        dimensions=("instrument", "exposure", "detector"),
        multiple=True,
    )
    camera = cT.PrerequisiteInput(
        name="camera",
        doc="Camera the input data comes from.",
        storageClass="Camera",
        dimensions=("instrument",),
        isCalibration=True,
        lookupFunction=lookupStaticCalibration,
    )
    outputPtcDataset = cT.Output(
        name="ptcDatsetProposal",
        doc="Output proposed ptc dataset.",
        storageClass="PhotonTransferCurveDataset",
        dimensions=("instrument", "detector"),
        multiple=False,
        isCalibration=True,
    )


class PhotonTransferCurveSolveConfig(pipeBase.PipelineTaskConfig,
                                     pipelineConnections=PhotonTransferCurveSolveConnections):
    """Configuration for fitting measured covariances.
    """

    ptcFitType = pexConfig.ChoiceField(
        dtype=str,
        doc="Fit PTC to Eq. 16, Eq. 20 in Astier+19, or to a polynomial.",
        default="POLYNOMIAL",
        allowed={
            "POLYNOMIAL": "n-degree polynomial (use 'polynomialFitDegree' to set 'n').",
            "EXPAPPROXIMATION": "Approximation in Astier+19 (Eq. 16).",
            "FULLCOVARIANCE": "Full covariances model in Astier+19 (Eq. 20)"
        }
    )
    maximumRangeCovariancesAstier = pexConfig.Field(
        dtype=int,
        doc="Maximum range of covariances as in Astier+19",
        default=8,
    )
    sigmaClipFullFitCovariancesAstier = pexConfig.Field(
        dtype=float,
        doc="sigma clip for full model fit for FULLCOVARIANCE ptcFitType ",
        default=5.0,
    )
    maxIterFullFitCovariancesAstier = pexConfig.Field(
        dtype=int,
        doc="Maximum number of iterations in full model fit for FULLCOVARIANCE ptcFitType",
        default=3,
    )
    polynomialFitDegree = pexConfig.Field(
        dtype=int,
        doc="Degree of polynomial to fit the PTC, when 'ptcFitType'=POLYNOMIAL.",
        default=3,
    )
    sigmaCutPtcOutliers = pexConfig.Field(
        dtype=float,
        doc="Sigma cut for outlier rejection in PTC.",
        default=5.0,
    )
    maxIterationsPtcOutliers = pexConfig.RangeField(
        dtype=int,
        doc="Maximum number of iterations for outlier rejection in PTC.",
        default=2,
        min=0
    )
    minVarPivotSearch = pexConfig.Field(
        dtype=float,
        doc="The code looks for a pivot signal point after which the variance starts decreasing at high-flux"
            " to exclude then from the PTC model fit. However, sometimes at low fluxes, the variance"
            " decreases slightly. Set this variable for the variance value, in ADU^2, after which the pivot "
            " should be sought.",
        default=10000,
    )
    consecutivePointsVarDecreases = pexConfig.RangeField(
        dtype=int,
        doc="Required number of consecutive points/fluxes in the PTC where the variance "
            "decreases in order to find a first estimate of the PTC turn-off. ",
        default=2,
        min=2
    )
    doFitBootstrap = pexConfig.Field(
        dtype=bool,
        doc="Use bootstrap for the PTC fit parameters and errors?.",
        default=False,
    )


class PhotonTransferCurveSolveTask(pipeBase.PipelineTask,
                                   pipeBase.CmdLineTask):
    """Task to fit the PTC from flat covariances.

    The first task of the PTC measurement pipeline,
    ``PhotonTransferCurveMeasureTask`` (and assumed to have been run
    before this task), produced a list of
    `~lsst.ip.isr.PhotonTransferCurveDataset` objects. Each dataset
    contains the mean signal and covariances of the
    difference image of the flat-field images taken at
    the same exposure time. The list also contains dummy
    datasets (with no measurements), whose purpose is to have
    the input and output dimensions of ``PhotonTransferCurveMeasureTask``
    match.

    This task, ``PhotonTransferCurveSolveTask``, assembles the list
    of individual PTC datasets produced
    by ``PhotonTransferCurveMeasureTask`` into one single final PTC
    dataset, discarding the dummy datset as appropiate.
    The task fits the measured (co)variances to one of three models:
    a polynomial model of a given order,  or the models described
    in equations 16 and 20 of Astier+19. These options are referred
    to as ``POLYNOMIAL``, ``EXPAPPROXIMATION``, and ``FULLCOVARIANCE``
    in the configuration options of the task, respectively).
    Parameters of interest such as the gain and noise are derived
    from the fits. The ``FULLCOVARIANCE`` model is fitted to the
    full covariance data (as oppossed to the other two models, which
    are fit to the variance vs mean measurements only).

    Astier+19: "The Shape of the Photon Transfer Curve
    of CCD sensors", arXiv:1905.08677
    """

    ConfigClass = PhotonTransferCurveSolveConfig
    _DefaultName = 'cpPhotonTransferCurveSolve'

    def runQuantum(self, butlerQC, inputRefs, outputRefs):
        """Ensure that the input and output dimensions are passed along.

        Parameters
        ----------
        butlerQC : `~lsst.daf.butler.butlerQuantumContext.ButlerQuantumContext`
            Butler to operate on.
        inputRefs : `~lsst.pipe.base.connections.InputQuantizedConnection`
            Input data refs to load.
        ouptutRefs : `~lsst.pipe.base.connections.OutputQuantizedConnection`
            Output data refs to persist.
        """
        inputs = butlerQC.get(inputRefs)
        detId = inputRefs.inputCovariances[0].dataId['detector']
        outputs = self.run(inputCovariances=inputs['inputCovariances'], camera=inputs['camera'], detId=detId)
        butlerQC.put(outputs, outputRefs)

    def run(self, inputCovariances, camera=None, inputExpList=None, detId=0):
        """Fit measured covariances to different models.

        Parameters
        ----------
        inputCovariances : `list` [`lsst.ip.isr.PhotonTransferCurveDataset`]
            List of lsst.ip.isr.PhotonTransferCurveDataset datasets.
        camera : `lsst.afw.cameraGeom.Camera`, optional
            Input camera.
        inputExpList : `list` [`~lsst.afw.image.ExposureF`], optional
            List of exposures.
        detId : `int`
            Detector ID to locate the detector in teh camera and
            populate the `lsst.ip.isr.PhotonTransferCurveDataset`
            metadata.
        Returns
        -------
        results : `lsst.pipe.base.Struct`
            The resultins structure contains:
            ``outputPtcDatset``
                Final PTC dataset, containing information such as the
                means, variances, and exposure times
                (`lsst.ip.isr.PhotonTransferCurveDataset`).
        """
        # Assemble individual PTC datasets into a single PTC dataset.
        ampNames = np.unique(inputCovariances[0].ampNames)
        datasetPtc = PhotonTransferCurveDataset(ampNames=ampNames,
                                                ptcFitType=self.config.ptcFitType,
                                                covMatrixSide=self.config.maximumRangeCovariancesAstier)
        for partialPtcDataset in inputCovariances:
            # Ignore dummy datasets
            if partialPtcDataset.ptcFitType == 'DUMMY':
                continue
            for ampName in ampNames:
                datasetPtc.inputExpIdPairs[ampName].append(partialPtcDataset.inputExpIdPairs[ampName])
                if type(partialPtcDataset.rawExpTimes[ampName]) is list:
                    datasetPtc.rawExpTimes[ampName].append(partialPtcDataset.rawExpTimes[ampName][0])
                else:
                    datasetPtc.rawExpTimes[ampName].append(partialPtcDataset.rawExpTimes[ampName])
                if type(partialPtcDataset.rawMeans[ampName]) is list:
                    datasetPtc.rawMeans[ampName].append(partialPtcDataset.rawMeans[ampName][0])
                else:
                    datasetPtc.rawMeans[ampName].append(partialPtcDataset.rawMeans[ampName])
                if type(partialPtcDataset.rawVars[ampName]) is list:
                    datasetPtc.rawVars[ampName].append(partialPtcDataset.rawVars[ampName][0])
                else:
                    datasetPtc.rawVars[ampName].append(partialPtcDataset.rawVars[ampName])
                if type(partialPtcDataset.expIdMask[ampName]) is list:
                    datasetPtc.expIdMask[ampName].append(partialPtcDataset.expIdMask[ampName][0])
                else:
                    datasetPtc.expIdMask[ampName].append(partialPtcDataset.expIdMask[ampName])
                datasetPtc.covariances[ampName].append(np.array(partialPtcDataset.covariances[ampName][0]))
                datasetPtc.covariancesSqrtWeights[ampName].append(
                    np.array(partialPtcDataset.covariancesSqrtWeights[ampName][0]))
        # Sort arrays that are filled so far in the final dataset by
        # rawMeans index
        for ampName in ampNames:
            index = np.argsort(np.ravel(np.array(datasetPtc.rawMeans[ampName])))
            datasetPtc.inputExpIdPairs[ampName] = np.array(datasetPtc.inputExpIdPairs[ampName])[index]
            datasetPtc.rawExpTimes[ampName] = np.array(datasetPtc.rawExpTimes[ampName])[index]
            datasetPtc.rawMeans[ampName] = np.array(datasetPtc.rawMeans[ampName])[index]
            datasetPtc.rawVars[ampName] = np.array(datasetPtc.rawVars[ampName])[index]
            datasetPtc.expIdMask[ampName] = np.array(datasetPtc.expIdMask[ampName])[index]
            datasetPtc.covariances[ampName] = np.array(datasetPtc.covariances[ampName])[index]
            datasetPtc.covariancesSqrtWeights[ampName] = np.array(
                datasetPtc.covariancesSqrtWeights[ampName])[index]
        if self.config.ptcFitType == "FULLCOVARIANCE":
            # Fit the measured covariances vs mean signal to
            # the Astier+19 full model (Eq. 20). Before that
            # do a preliminary fit to the variance (C_00) vs mean
            # signal (mu) curve using the EXPAPPROXIMATION model
            # (Eq. 16 in Astier+19) in order to
            # get the flat pairs that are masked. The
            # points at these fluxes will also be masked when
            # calculating the other elements of the covariance
            # matrix, C_ij, i!=j).

            # Preliminary fit, usign a temp dataset to get the mask
            tempDatasetPtc = copy.copy(datasetPtc)
            tempDatasetPtc.ptcFitType = "EXPAPPROXIMATION"
            tempDatasetPtc = self.fitMeasurementsToModel(tempDatasetPtc)

            # "FULLCOVARIANCE", using the mask obtained from the
            # previous fit.
            for ampName in datasetPtc.ampNames:
                datasetPtc.expIdMask[ampName] = tempDatasetPtc.expIdMask[ampName]
            datasetPtc.fitType = "FULLCOVARIANCE"
            datasetPtc = self.fitMeasurementsToModel(datasetPtc)
        # The other options are: self.config.ptcFitType in
        # ("EXPAPPROXIMATION", "POLYNOMIAL")
        else:
            # Fit the PTC to a polynomial or to Astier+19 exponential
            # approximation (Eq. 16).  Fill up
            # PhotonTransferCurveDataset object.
            datasetPtc = self.fitMeasurementsToModel(datasetPtc)

        if camera:
            detector = camera[detId]
        else:
            detector = None
        datasetPtc.updateMetadata(setDate=True, camera=camera, detector=detector)

        return pipeBase.Struct(
            outputPtcDataset=datasetPtc,
        )

    def fitMeasurementsToModel(self, dataset):
        """Fit the measured covariances vs mean signal to a
        polynomial or one of the models in Astier+19
        (Eq. 16 or Eq.20).

        Parameters
        ----------
        dataset : `lsst.ip.isr.ptcDataset.PhotonTransferCurveDataset`
            The dataset containing information such as the means,
            (co)variances, and exposure times.

        Returns
        -------
        dataset : `lsst.ip.isr.ptcDataset.PhotonTransferCurveDataset`
            This is the same dataset as the input parameter, however,
            it has been modified to include information such as the
            fit vectors and the fit parameters. See the class
            `PhotonTransferCurveDatase`.
        """
        fitType = dataset.ptcFitType
        if fitType in ["FULLCOVARIANCE", ]:
            # This model uses the full covariance matrix in the fit.
            # The PTC is technically defined as variance vs signal,
            # with variance = Cov_00
            dataset = self.fitDataFullCovariance(dataset)
        elif fitType in ["POLYNOMIAL", "EXPAPPROXIMATION"]:
            # The PTC is technically defined as variance vs signal
            dataset = self.fitPtc(dataset)
        else:
            raise RuntimeError(
                f"Fitting option {fitType} not one of "
                "'POLYNOMIAL', 'EXPAPPROXIMATION', or 'FULLCOVARIANCE'"
            )

        return dataset

    def fitDataFullCovariance(self, dataset):
        """Fit measured flat covariances to the full model in
        Astier+19 (Eq. 20).

        Parameters
        ----------
        dataset : `lsst.ip.isr.ptcDataset.PhotonTransferCurveDataset`
            The dataset containing information such as the means,
            (co)variances, and exposure times.

        Returns
        -------
        dataset : `lsst.ip.isr.ptcDataset.PhotonTransferCurveDataset`
            This is the same dataset as the input parameter, however,
            it has been modified to include information such as the
            fit vectors and the fit parameters. See the class
            `PhotonTransferCurveDatase`.

        Notes
        -----
        The parameters of the full model for C_ij(mu) ("C_ij" and "mu"
        in ADU^2 and ADU, respectively) in Astier+19 (Eq. 20) are:
            "a" coefficients (r by r matrix), units: 1/e
            "b" coefficients (r by r matrix), units: 1/e
            noise matrix (r by r matrix), units: e^2
            gain, units: e/ADU
        "b" appears in Eq. 20 only through the "ab" combination, which
        is defined in this code as "c=ab".

        Total number of parameters: #entries(a) + #entries(c) + #entries(noise)
        + 1. This is equivalent to r^2 + r^2 + r^2 + 1, where "r" is the
        maximum lag considered for the covariances calculation, and the
        extra "1" is the gain. If "b" is 0, then "c" is 0, and len(pInit) will
        have r^2 fewer entries.
        """
        matrixSide = self.config.maximumRangeCovariancesAstier
        lenParams = matrixSide*matrixSide

        for ampName in dataset.ampNames:
            lenInputTimes = len(dataset.rawExpTimes[ampName])
            # Not used when ptcFitType is 'FULLCOVARIANCE'
            dataset.ptcFitPars[ampName] = [np.nan]
            dataset.ptcFitParsError[ampName] = [np.nan]
            dataset.ptcFitChiSq[ampName] = np.nan

            if ampName in dataset.badAmps:
                # Bad amp
                # Entries need to have proper dimensions so read/write
                # with astropy.Table works.
                nanMatrix = np.full((matrixSide, matrixSide), np.nan)
                listNanMatrix = np.full((lenInputTimes, matrixSide, matrixSide), np.nan)
                dataset.covariancesModel[ampName] = listNanMatrix
                dataset.covariancesSqrtWeights[ampName] = listNanMatrix
                dataset.aMatrix[ampName] = nanMatrix
                dataset.bMatrix[ampName] = nanMatrix
                dataset.covariancesModelNoB[ampName] = listNanMatrix
                dataset.aMatrixNoB[ampName] = nanMatrix

                dataset.expIdMask[ampName] = np.repeat(np.nan, lenInputTimes)
                dataset.gain[ampName] = np.nan
                dataset.gainErr[ampName] = np.nan
                dataset.noise[ampName] = np.nan
                dataset.noiseErr[ampName] = np.nan
                dataset.finalVars[ampName] = np.repeat(np.nan, lenInputTimes)
                dataset.finalModelVars[ampName] = np.repeat(np.nan, lenInputTimes)
                dataset.finalMeans[ampName] = np.repeat(np.nan, lenInputTimes)
                continue

            muAtAmp = dataset.rawMeans[ampName]
            maskAtAmp = dataset.expIdMask[ampName]
            if len(maskAtAmp) == 0:
                maskAtAmp = np.repeat(True, len(muAtAmp))

            muAtAmp = muAtAmp[maskAtAmp]
            covAtAmp = np.nan_to_num(dataset.covariances[ampName])[maskAtAmp]
            covSqrtWeightsAtAmp = np.nan_to_num(dataset.covariancesSqrtWeights[ampName])[maskAtAmp]

            # Initial fit, to approximate parameters, with c=0
            a0, c0, noise0, gain0 = self.initialFitFullCovariance(muAtAmp, covAtAmp, covSqrtWeightsAtAmp)

            # Fit full model (Eq. 20 of Astier+19) and same model with
            # b=0 (c=0 in this code)
            pInit = np.concatenate((a0.flatten(), c0.flatten(), noise0.flatten(), np.array(gain0)), axis=None)
            functionsDict = {'fullModel': self.funcFullCovarianceModel,
                             'fullModelNoB': self.funcFullCovarianceModelNoB}
            fitResults = {'fullModel': {'a': [], 'c': [], 'noise': [], 'gain': [], 'paramsErr': []},
                          'fullModelNoB': {'a': [], 'c': [], 'noise': [], 'gain': [], 'paramsErr': []}}
            for key in functionsDict:
                params, paramsErr, _ = fitLeastSq(pInit, muAtAmp,
                                                  covAtAmp.flatten(), functionsDict[key],
                                                  weightsY=covSqrtWeightsAtAmp.flatten())
                a = params[:lenParams].reshape((matrixSide, matrixSide))
                c = params[lenParams:2*lenParams].reshape((matrixSide, matrixSide))
                noise = params[2*lenParams:3*lenParams].reshape((matrixSide, matrixSide))
                gain = params[-1]

                fitResults[key]['a'] = a
                fitResults[key]['c'] = c
                fitResults[key]['noise'] = noise
                fitResults[key]['gain'] = gain
                fitResults[key]['paramsErr'] = paramsErr

            # Put the information in the PTC dataset

            # Not used when ptcFitType is 'FULLCOVARIANCE'
            dataset.ptcFitPars[ampName] = [np.nan]
            dataset.ptcFitParsError[ampName] = [np.nan]
            dataset.ptcFitChiSq[ampName] = np.nan

            # Save full covariances, covariances models, and their weights
            # dataset.expIdMask is already full
            dataset.covariances[ampName] = covAtAmp
            dataset.covariancesModel[ampName] = self.evalCovModel(muAtAmp,
                                                                  fitResults['fullModel']['a'],
                                                                  fitResults['fullModel']['c'],
                                                                  fitResults['fullModel']['noise'],
                                                                  fitResults['fullModel']['gain'])
            dataset.covariancesSqrtWeights[ampName] = covSqrtWeightsAtAmp
            dataset.aMatrix[ampName] = fitResults['fullModel']['a']
            dataset.bMatrix[ampName] = fitResults['fullModel']['c']/fitResults['fullModel']['a']
            dataset.covariancesModelNoB[ampName] = self.evalCovModel(muAtAmp,
                                                                     fitResults['fullModelNoB']['a'],
                                                                     fitResults['fullModelNoB']['c'],
                                                                     fitResults['fullModelNoB']['noise'],
                                                                     fitResults['fullModelNoB']['gain'],
                                                                     setBtoZero=True)
            dataset.aMatrixNoB[ampName] = fitResults['fullModelNoB']['a']
            dataset.gain[ampName] = fitResults['fullModel']['gain']
            dataset.gainErr[ampName] = fitResults['fullModel']['paramsErr'][-1]
            readoutNoise = fitResults['fullModel']['noise'][0][0]
            readoutNoiseSqrt = np.sqrt(np.fabs(readoutNoise))
            dataset.noise[ampName] = readoutNoise
            readoutNoiseSigma = fitResults['fullModel']['paramsErr'][2*lenParams]
            dataset.noiseErr[ampName] = 0.5*(readoutNoiseSigma/np.fabs(readoutNoise))*readoutNoiseSqrt
            dataset.finalVars[ampName] = covAtAmp[:, 0, 0]
            dataset.finalModelVars[ampName] = dataset.covariancesModel[ampName][:, 0, 0]
            dataset.finalMeans[ampName] = muAtAmp

        return dataset

    def initialFitFullCovariance(self, mu, cov, sqrtW):
        """ Performs a crude parabolic fit of the data in order to start
        the full fit close to the solution, setting b=0 (c=0) in Eq. 20
        of Astier+19.

        Parameters
        ----------
        mu : `numpy.array`, (N,)
            Signal `mu` (ADU)
        cov : `numpy.array`, (N, M, M)
            Covariance arrays of size `(M, M)` (with
            `M = config.maximumRangeCovariancesAstier`),
            indexed by mean signal `mu`.
        sqrtW : `numpy.array`, (N,)
            Covariance weights, defined as 1./sqrt(Variances)

        Returns
        -------
        a : `numpy.array`, (M, M)
            "a" parameter per flux in Eq. 20 of Astier+19.
        c : `numpy.array`, (M, M)
            "c"="ab" parameter per flux in Eq. 20 of Astier+19.
        noise : `numpy.array`, (M, M)
            "noise" parameter per flux in Eq. 20 of Astier+19.
        gain : `float`
            Amplifier gain (e/ADU)
        """
        matrixSide = self.config.maximumRangeCovariancesAstier

        # Initialize fit parameters
        a = np.zeros((matrixSide, matrixSide))
        c = np.zeros((matrixSide, matrixSide))
        noise = np.zeros((matrixSide, matrixSide))
        gain = 1.

        # iterate the fit to account for higher orders
        # the chi2 does not necessarily go down, so one could
        # stop when it increases
        oldChi2 = 1e30
        for _ in range(5):
            model = np.nan_to_num(self.evalCovModel(mu, a, c, noise, gain, setBtoZero=True))
            # loop on lags
            for i in range(matrixSide):
                for j in range(matrixSide):
                    # fit a parabola for a given lag
                    parsFit = np.polyfit(mu, cov[:, i, j] - model[:, i, j],
                                         2, w=sqrtW[:, i, j])
                    # model equation (Eq. 20) in Astier+19, with c=a*b=0:
                    a[i, j] += parsFit[0]
                    noise[i, j] += parsFit[2]
                    if(i + j == 0):
                        gain = 1./(1/gain+parsFit[1])
            weightedRes = (model - cov)*sqrtW
            chi2 = (weightedRes.flatten()**2).sum()
            if chi2 > oldChi2:
                break
            oldChi2 = chi2

        return a, c, noise, gain

    def funcFullCovarianceModel(self, params, x):
        """Model to fit covariances from flat fields; Equation 20 of
        Astier+19.

        Parameters
        ----------
        params : `list`
            Parameters of the model: aMatrix, CMatrix, noiseMatrix,
            gain (e/ADU).
        x : `numpy.array`, (N,)
            Signal `mu` (ADU)

        Returns
        -------
        y : `numpy.array`, (N,)
            Covariance matrix.
        """
        matrixSide = self.config.maximumRangeCovariancesAstier
        lenParams = matrixSide*matrixSide
        aMatrix = params[:lenParams].reshape((matrixSide, matrixSide))
        cMatrix = params[lenParams:2*lenParams].reshape((matrixSide, matrixSide))
        noiseMatrix = params[2*lenParams:3*lenParams].reshape((matrixSide, matrixSide))
        gain = params[-1]

        return self.evalCovModel(x, aMatrix, cMatrix, noiseMatrix, gain).flatten()

    def funcFullCovarianceModelNoB(self, params, x):
        """Model to fit covariances from flat fields; Equation 20 of
        Astier+19, with b=0 (equivalent to c=a*b=0 in this code).

        Parameters
        ----------
        params : `list`
            Parameters of the model: aMatrix, CMatrix, noiseMatrix,
            gain (e/ADU).
        x : `numpy.array`, (N,)
            Signal mu (ADU)

        Returns
        -------
        y : `numpy.array`, (N,)
            Covariance matrix.
        """
        matrixSide = self.config.maximumRangeCovariancesAstier
        lenParams = matrixSide*matrixSide
        aMatrix = params[:lenParams].reshape((matrixSide, matrixSide))
        cMatrix = params[lenParams:2*lenParams].reshape((matrixSide, matrixSide))
        noiseMatrix = params[2*lenParams:3*lenParams].reshape((matrixSide, matrixSide))
        gain = params[-1]

        return self.evalCovModel(x, aMatrix, cMatrix, noiseMatrix, gain, setBtoZero=True).flatten()

    def evalCovModel(self, mu, aMatrix, cMatrix, noiseMatrix, gain, setBtoZero=False):
        """Computes full covariances model (Eq. 20 of Astier+19).

        Parameters
        ----------
        mu : `numpy.array`, (N,)
            List of mean signals.
        aMatrix : `numpy.array`, (M, M)
            "a" parameter per flux in Eq. 20 of Astier+19.
        cMatrix : `numpy.array`, (M, M)
            "c"="ab" parameter per flux in Eq. 20 of Astier+19.
        noiseMatrix : `numpy.array`, (M, M)
            "noise" parameter per flux in Eq. 20 of Astier+19.
        gain : `float`
            Amplifier gain (e/ADU)
        setBtoZero=False : `bool`, optional
            Set "b" parameter in full model (see Astier+19) to zero.

        Returns
        -------
        covModel : `numpy.array`, (N, M, M)
            Covariances model.

        Notes
        -----
        By default, computes the covModel for the mu's stored(self.mu).
        Returns cov[Nmu, M, M]. The variance for the PTC is
        cov[:, 0, 0].  mu and cov are in ADUs and ADUs squared. To use
        electrons for both, the gain should be set to 1. This routine
        implements the model in Astier+19 (1905.08677).
        The parameters of the full model for C_ij(mu) ("C_ij" and "mu"
        in ADU^2 and ADU, respectively) in Astier+19 (Eq. 20) are:
        "a" coefficients (M by M matrix), units: 1/e
        "b" coefficients (M by M matrix), units: 1/e
        noise matrix (M by M matrix), units: e^2
        gain, units: e/ADU
        "b" appears in Eq. 20 only through the "ab" combination, which
        is defined in this code as "c=ab".
        """
        matrixSide = self.config.maximumRangeCovariancesAstier
        sa = (matrixSide, matrixSide)
        # pad a with zeros and symmetrize
        aEnlarged = np.zeros((int(sa[0]*1.5)+1, int(sa[1]*1.5)+1))
        aEnlarged[0:sa[0], 0:sa[1]] = aMatrix
        aSym = symmetrize(aEnlarged)
        # pad c with zeros and symmetrize
        cEnlarged = np.zeros((int(sa[0]*1.5)+1, int(sa[1]*1.5)+1))
        cEnlarged[0:sa[0], 0:sa[1]] = cMatrix
        cSym = symmetrize(cEnlarged)
        a2 = fftconvolve(aSym, aSym, mode='same')
        a3 = fftconvolve(a2, aSym, mode='same')
        ac = fftconvolve(aSym, cSym, mode='same')
        (xc, yc) = np.unravel_index(np.abs(aSym).argmax(), a2.shape)

        a1 = aMatrix[np.newaxis, :, :]
        a2 = a2[np.newaxis, xc:xc + matrixSide, yc:yc + matrixSide]
        a3 = a3[np.newaxis, xc:xc + matrixSide, yc:yc + matrixSide]
        ac = ac[np.newaxis, xc:xc + matrixSide, yc:yc + matrixSide]
        c1 = cMatrix[np.newaxis, ::]

        # assumes that mu is 1d
        bigMu = mu[:, np.newaxis, np.newaxis]*gain
        # c(=a*b in Astier+19) also has a contribution to the last
        # term, that is absent for now.
        if setBtoZero:
            c1 = np.zeros_like(c1)
            ac = np.zeros_like(ac)
        covModel = (bigMu/(gain*gain)*(a1*bigMu+2./3.*(bigMu*bigMu)*(a2 + c1)
                    + (1./3.*a3 + 5./6.*ac)*(bigMu*bigMu*bigMu)) + noiseMatrix[np.newaxis, :, :]/gain**2)
        # add the Poisson term, and the read out noise (variance)
        covModel[:, 0, 0] += mu/gain

        return covModel

    # EXPAPPROXIMATION and POLYNOMIAL fit methods
    @staticmethod
    def _initialParsForPolynomial(order):
        assert(order >= 2)
        pars = np.zeros(order, dtype=float)
        pars[0] = 10
        pars[1] = 1
        pars[2:] = 0.0001
        return pars

    @staticmethod
    def _boundsForPolynomial(initialPars, lowers=[], uppers=[]):
        if not len(lowers):
            lowers = [np.NINF for p in initialPars]
        if not len(uppers):
            uppers = [np.inf for p in initialPars]
        lowers[1] = 0  # no negative gains
        return (lowers, uppers)

    @staticmethod
    def _boundsForAstier(initialPars, lowers=[], uppers=[]):
        if not len(lowers):
            lowers = [np.NINF for p in initialPars]
        if not len(uppers):
            uppers = [np.inf for p in initialPars]
        return (lowers, uppers)

    @staticmethod
    def _getInitialGoodPoints(means, variances, minVarPivotSearch, consecutivePointsVarDecreases):
        """Return a boolean array to mask bad points.

        Parameters
        ----------
        means : `numpy.array`
            Input array with mean signal values.
        variances : `numpy.array`
            Input array with variances at each mean value.
        minVarPivotSearch : `float`
            The variance (in ADU^2), above which, the point
            of decreasing variance should be sought.
        consecutivePointsVarDecreases : `int`
            Required number of consecutive points/fluxes
            in the PTC where the variance
            decreases in order to find a first
            estimate of the PTC turn-off.

        Returns
        ------
        goodPoints : `numpy.array` [`bool`]
            Boolean array to select good (`True`) and bad (`False`)
            points.

        Notes
        -----
        Eliminate points beyond which the variance decreases.
        """
        goodPoints = np.ones_like(means, dtype=bool)
        # Variances are sorted and should monotonically increase
        pivotList = np.where(np.array(np.diff(variances)) < 0)[0]
        if len(pivotList) > 0:
            # For small values, sometimes the variance decreases slightly
            # Only look when var > self.config.minVarPivotSearch
            pivotList = [p for p in pivotList if variances[p] > minVarPivotSearch]
            # Require that the varince decreases during
            # consecutivePointsVarDecreases
            # consecutive points. This will give a first
            # estimate of the PTC turn-off, which
            # may be updated (reduced) further in the code.
            if len(pivotList) > 1:
                # enumerate(pivotList) creates tuples (index, value), for
                # each value in pivotList. The lambda function subtracts
                # each value from the index.
                # groupby groups elements by equal key value.
                for k, g in groupby(enumerate(pivotList), lambda x: x[0]-x[1]):
                    group = (map(itemgetter(1), g))
                    # Form groups of consecute values from pivotList
                    group = list(map(int, group))
                    # values in pivotList are indices where np.diff(variances)
                    # is negative, i.e., where the variance starts decreasing.
                    # Find the first group of consecutive numbers when
                    # variance decreases.
                    if len(group) >= consecutivePointsVarDecreases:
                        pivotIndex = np.min(group)
                        goodPoints[pivotIndex+1:] = False
                        break

        return goodPoints

    def _makeZeroSafe(self, array, substituteValue=1e-9):
        """"""
        array = np.array(array)
        nBad = Counter(np.ravel(array))[0]
        if nBad == 0:
            return array

        index, = np.where(array == 0)
        if len(index):
            msg = f"Found {nBad} zeros in array at elements {index}"
            self.log.warning(msg)

        array[index] = substituteValue

        return array

    def fitPtc(self, dataset):
        """Fit the photon transfer curve to a polynomial or to the
        Astier+19 approximation (Eq. 16).

        Fit the photon transfer curve with either a polynomial of
        the order specified in the task config, or using the
        exponential approximation in Astier+19 (Eq. 16).

        Sigma clipping is performed iteratively for the fit, as
        well as an initial clipping of data points that are more
        than `config.initialNonLinearityExclusionThreshold` away
        from lying on a straight line. This other step is necessary
        because the photon transfer curve turns over catastrophically
        at very high flux (because saturation
        drops the variance to ~0) and these far outliers cause the
        initial fit to fail, meaning the sigma cannot be calculated
        to perform the sigma-clipping.

        Parameters
        ----------
        dataset : `lsst.ip.isr.ptcDataset.PhotonTransferCurveDataset`
            The dataset containing the means, variances and
            exposure times.

        Returns
        -------
        dataset : `lsst.ip.isr.ptcDataset.PhotonTransferCurveDataset`
            This is the same dataset as the input parameter, however,
            it has been modified to include information such as the
            fit vectors and the fit parameters. See the class
            `PhotonTransferCurveDatase`.

        Raises
        ------
        RuntimeError:
            Raises if dataset.ptcFitType is None or empty.
        """
        if dataset.ptcFitType:
            ptcFitType = dataset.ptcFitType
        else:
            raise RuntimeError("ptcFitType is None of empty in PTC dataset.")
        matrixSide = self.config.maximumRangeCovariancesAstier
        nanMatrix = np.empty((matrixSide, matrixSide))
        nanMatrix[:] = np.nan

        for amp in dataset.ampNames:
            lenInputTimes = len(dataset.rawExpTimes[amp])
            listNanMatrix = np.empty((lenInputTimes, matrixSide, matrixSide))
            listNanMatrix[:] = np.nan

            dataset.covariancesModel[amp] = listNanMatrix
            dataset.aMatrix[amp] = nanMatrix
            dataset.bMatrix[amp] = nanMatrix
            dataset.covariancesModelNoB[amp] = listNanMatrix
            dataset.aMatrixNoB[amp] = nanMatrix

        def errFunc(p, x, y):
            return ptcFunc(p, x) - y

        sigmaCutPtcOutliers = self.config.sigmaCutPtcOutliers
        maxIterationsPtcOutliers = self.config.maxIterationsPtcOutliers

        for i, ampName in enumerate(dataset.ampNames):
            timeVecOriginal = np.ravel(np.array(dataset.rawExpTimes[ampName]))
            meanVecOriginal = np.ravel(np.array(dataset.rawMeans[ampName]))
            varVecOriginal = np.ravel(np.array(dataset.rawVars[ampName]))
            varVecOriginal = self._makeZeroSafe(varVecOriginal)

            # Discard points when the variance starts to decrease after two
            # consecutive signal levels
            goodPoints = self._getInitialGoodPoints(meanVecOriginal, varVecOriginal,
                                                    self.config.minVarPivotSearch,
                                                    self.config.consecutivePointsVarDecreases)

            # Check if all points are bad from the 'cpExtractPtcTask'
            initialExpIdMask = np.ravel(np.array(dataset.expIdMask[ampName]))

            if not (goodPoints.any() and initialExpIdMask.any()):
                msg = (f"SERIOUS: All points in goodPoints: {goodPoints} or "
                       f"in initialExpIdMask: {initialExpIdMask} are bad."
                       f"Setting {ampName} to BAD.")
                self.log.warning(msg)
                # Fill entries with NaNs
                self.fillBadAmp(dataset, ptcFitType, ampName)
                continue

            # Save the point where the variance starts decreasing as the
            # PTC turnoff point
            ptcTurnoff = meanVecOriginal[goodPoints][-1]
            dataset.ptcTurnoff[ampName] = ptcTurnoff

            mask = goodPoints

            if ptcFitType == 'EXPAPPROXIMATION':
                ptcFunc = funcAstier
                parsIniPtc = [-1e-9, 1.0, 10.]  # a00, gain, noise^2
                # lowers and uppers obtained from BOT data studies by
                # C. Lage (UC Davis, 11/2020).
                bounds = self._boundsForAstier(parsIniPtc, lowers=[-1e-4, 0.5, -2000],
                                               uppers=[1e-4, 2.5, 2000])
            if ptcFitType == 'POLYNOMIAL':
                ptcFunc = funcPolynomial
                parsIniPtc = self._initialParsForPolynomial(self.config.polynomialFitDegree + 1)
                bounds = self._boundsForPolynomial(parsIniPtc)

            # Before bootstrap fit, do an iterative fit to get rid of outliers.
            # This further process of outlier rejection be skipped
            # if self.config.maxIterationsPtcOutliers = 0.
            # We already did some initial outlier rejection above in
            # self._getInitialGoodPoints.
            count = 1
            newMask = np.ones_like(meanVecOriginal, dtype=bool)
            pars = parsIniPtc
            while count <= maxIterationsPtcOutliers:
                # Note that application of the mask actually shrinks the array
                # to size rather than setting elements to zero (as we want) so
                # always update mask itself and re-apply to the original data
                meanTempVec = meanVecOriginal[mask]
                varTempVec = varVecOriginal[mask]
                res = least_squares(errFunc, parsIniPtc, bounds=bounds, args=(meanTempVec, varTempVec))
                pars = res.x

                # change this to the original from the temp because
                # the masks are ANDed meaning once a point is masked
                # it's always masked, and the masks must always be the
                # same length for broadcasting
                sigResids = (varVecOriginal - ptcFunc(pars, meanVecOriginal))/np.sqrt(varVecOriginal)
                newMask = np.array([True if np.abs(r) < sigmaCutPtcOutliers else False for r in sigResids])
                mask = mask & newMask
                if not (mask.any() and newMask.any()):
                    msg = (f"SERIOUS: All points in either mask: {mask} or newMask: {newMask} are bad. "
                           f"Setting {ampName} to BAD.")
                    self.log.warning(msg)
                    # Fill entries with NaNs
                    self.fillBadAmp(dataset, ptcFitType, ampName)
                    break
                nDroppedTotal = Counter(mask)[False]
                self.log.debug("Iteration %d: discarded %d points in total for %s",
                               count, nDroppedTotal, ampName)
                count += 1
                # objects should never shrink
                assert (len(mask) == len(timeVecOriginal) == len(meanVecOriginal) == len(varVecOriginal))
            if not (mask.any() and newMask.any()):
                continue
            dataset.expIdMask[ampName] = np.array(dataset.expIdMask[ampName])
            # store the final mask
            if len(dataset.expIdMask[ampName]):
                dataset.expIdMask[ampName] &= mask  # bitwise_and if there is already a mask
            else:
                dataset.expIdMask[ampName] = mask
            parsIniPtc = pars
            meanVecFinal = meanVecOriginal[mask]
            varVecFinal = varVecOriginal[mask]

            if Counter(mask)[False] > 0:
                self.log.info("Number of points discarded in PTC of amplifier %s:"
                              " %d out of %d", ampName, Counter(mask)[False], len(meanVecOriginal))

            if (len(meanVecFinal) < len(parsIniPtc)):
                msg = (f"SERIOUS: Not enough data points ({len(meanVecFinal)}) compared to the number of "
                       f"parameters of the PTC model({len(parsIniPtc)}). Setting {ampName} to BAD.")
                self.log.warning(msg)
                # Fill entries with NaNs
                self.fillBadAmp(dataset, ptcFitType, ampName)
                continue
            # Fit the PTC
            if self.config.doFitBootstrap:
                parsFit, parsFitErr, reducedChiSqPtc = fitBootstrap(parsIniPtc, meanVecFinal,
                                                                    varVecFinal, ptcFunc,
                                                                    weightsY=1./np.sqrt(varVecFinal))
            else:
                parsFit, parsFitErr, reducedChiSqPtc = fitLeastSq(parsIniPtc, meanVecFinal,
                                                                  varVecFinal, ptcFunc,
                                                                  weightsY=1./np.sqrt(varVecFinal))
            dataset.ptcFitPars[ampName] = parsFit
            dataset.ptcFitParsError[ampName] = parsFitErr
            dataset.ptcFitChiSq[ampName] = reducedChiSqPtc
            # Masked variances (measured and modeled) and means. Need
            # to pad the array so astropy.Table does not crash (the
            # mask may vary per amp).
            padLength = len(dataset.rawExpTimes[ampName]) - len(varVecFinal)
            dataset.finalVars[ampName] = np.pad(varVecFinal, (0, padLength), 'constant',
                                                constant_values=np.nan)
            dataset.finalModelVars[ampName] = np.pad(ptcFunc(parsFit, meanVecFinal), (0, padLength),
                                                     'constant', constant_values=np.nan)
            dataset.finalMeans[ampName] = np.pad(meanVecFinal, (0, padLength), 'constant',
                                                 constant_values=np.nan)
            if ptcFitType == 'EXPAPPROXIMATION':
                ptcGain = parsFit[1]
                ptcGainErr = parsFitErr[1]
                ptcNoise = np.sqrt(np.fabs(parsFit[2]))
                ptcNoiseErr = 0.5*(parsFitErr[2]/np.fabs(parsFit[2]))*np.sqrt(np.fabs(parsFit[2]))
            if ptcFitType == 'POLYNOMIAL':
                ptcGain = 1./parsFit[1]
                ptcGainErr = np.fabs(1./parsFit[1])*(parsFitErr[1]/parsFit[1])
                ptcNoise = np.sqrt(np.fabs(parsFit[0]))*ptcGain
                ptcNoiseErr = (0.5*(parsFitErr[0]/np.fabs(parsFit[0]))*(np.sqrt(np.fabs(parsFit[0]))))*ptcGain
            dataset.gain[ampName] = ptcGain
            dataset.gainErr[ampName] = ptcGainErr
            dataset.noise[ampName] = ptcNoise
            dataset.noiseErr[ampName] = ptcNoiseErr

        if not len(dataset.ptcFitType) == 0:
            dataset.ptcFitType = ptcFitType
        if len(dataset.badAmps) == 0:
            dataset.badAmps = np.repeat(np.nan, len(list(dataset.rawExpTimes.values())[0]))

        return dataset

    def fillBadAmp(self, dataset, ptcFitType, ampName):
        """Fill the dataset with NaNs if there are not enough
        good points.

        Parameters
        ----------
        dataset : `lsst.ip.isr.ptcDataset.PhotonTransferCurveDataset`
            The dataset containing the means, variances and
            exposure times.
        ptcFitType : {'POLYNOMIAL', 'EXPAPPROXIMATION'}
            Fit a 'POLYNOMIAL' (degree: 'polynomialFitDegree') or
            'EXPAPPROXIMATION' (Eq. 16 of Astier+19) to the PTC.
        ampName : `str`
            Amplifier name.
        """
        dataset.badAmps.append(ampName)
        dataset.expIdMask[ampName] = np.repeat(False, len(dataset.rawExpTimes[ampName]))
        dataset.gain[ampName] = np.nan
        dataset.gainErr[ampName] = np.nan
        dataset.noise[ampName] = np.nan
        dataset.noiseErr[ampName] = np.nan
        dataset.ptcFitPars[ampName] = (np.repeat(np.nan, self.config.polynomialFitDegree + 1) if
                                       ptcFitType in ["POLYNOMIAL", ] else np.repeat(np.nan, 3))
        dataset.ptcFitParsError[ampName] = (np.repeat(np.nan, self.config.polynomialFitDegree + 1) if
                                            ptcFitType in ["POLYNOMIAL", ] else np.repeat(np.nan, 3))
        dataset.ptcFitChiSq[ampName] = np.nan
        dataset.ptcTurnoff[ampName] = np.nan
        dataset.finalVars[ampName] = np.repeat(np.nan, len(dataset.rawExpTimes[ampName]))
        dataset.finalModelVars[ampName] = np.repeat(np.nan, len(dataset.rawExpTimes[ampName]))
        dataset.finalMeans[ampName] = np.repeat(np.nan, len(dataset.rawExpTimes[ampName]))

        return
