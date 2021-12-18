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
from lsst.cp.pipe.utils import (fitLeastSq, fitBootstrap, funcPolynomial, funcAstier)

from scipy.optimize import least_squares

import lsst.pipe.base.connectionTypes as cT

from .astierCovPtcUtils import fitDataFullCovariance

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
    maxIterationsPtcOutliers = pexConfig.Field(
        dtype=int,
        doc="Maximum number of iterations for outlier rejection in PTC.",
        default=2,
    )
    minVarPivotSearch = pexConfig.Field(
        dtype=float,
        doc="The code looks for a pivot signal point after which the variance starts decreasing at high-flux"
            " to exclude then from the PTC model fit. However, sometimes at low fluxes, the variance"
            " decreases slightly. Set this variable for the variance value, in ADU^2, after which the pivot "
            " should be sought.",
        default=10000,
    )
    doFitBootstrap = pexConfig.Field(
        dtype=bool,
        doc="Use bootstrap for the PTC fit parameters and errors?.",
        default=False,
    )


class PhotonTransferCurveSolveTask(pipeBase.PipelineTask,
                                   pipeBase.CmdLineTask):
    """Task to fit the PTC from flat covariances.

    This task assembles the list of individual PTC datasets produced
    by ``PhotonTransferCurveSolveTask`` into one single final PTC
    dataset.  The task fits the measured (co)variances to a polynomial
    model or to the models described in equations 16 and 20 of
    Astier+19 (referred to as ``POLYNOMIAL``, ``EXPAPPROXIMATION``,
    and ``FULLCOVARIANCE`` in the configuration options of the task,
    respectively). Parameters of interest such as tghe gain and noise
    are derived from the fits.

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
        outputs = self.run(inputCovariances=inputs['inputCovariances'], camera=inputs['camera'])
        butlerQC.put(outputs, outputRefs)

    def run(self, inputCovariances, camera=None, inputExpList=None):
        """Fit measure covariances to different models.

        Parameters
        ----------
        inputCovariances : `list` [`lsst.ip.isr.PhotonTransferCurveDataset`]
            List of lsst.ip.isr.PhotonTransferCurveDataset datasets.

        camera : `lsst.afw.cameraGeom.Camera`, optional
            Input camera.

        inputExpList : `list` [`~lsst.afw.image.ExposureF`], optional
            List of exposures.

        Returns
        -------
        results : `lsst.pipe.base.Struct`
            The results struct containing:

            ``outputPtcDatset``
                Final PTC dataset, containing information such as the
                means, variances, and exposure times
                (`lsst.ip.isr.PhotonTransferCurveDataset`).
        """
        # Assemble partial PTC datasets into a single dataset.
        ampNames = np.unique(inputCovariances[0].ampNames)
        datasetPtc = PhotonTransferCurveDataset(ampNames, self.config.ptcFitType,
                                                self.config.maximumRangeCovariancesAstier)
        for partialPtcDataset in inputCovariances:
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
            # Calculate covariances and fit them, including the PTC,
            # to Astier+19 full model (Eq. 20) First, fit get the flat
            # pairs that are masked, fitting C_00 vs mu to the
            # EXPAPPROXIMATION model (Eq. 16 in Astier+19).  The
            # points at these fluxes will also be masked when
            # calculating the other covariances, C_ij)
            tempDatasetPtc = copy.copy(datasetPtc)
            tempDatasetPtc.ptcFitType = "EXPAPPROXIMATION"
            tempDatasetPtc = self.fitPtc(tempDatasetPtc)
            for ampName in datasetPtc.ampNames:
                datasetPtc.expIdMask[ampName] = tempDatasetPtc.expIdMask[ampName]
            datasetPtc.fitType = "FULLCOVARIANCE"
            datasetPtc = self.fitCovariancesAstier(datasetPtc)
        # The other options are: self.config.ptcFitType in
        # ("EXPAPPROXIMATION", "POLYNOMIAL")
        else:
            # Fit the PTC to a polynomial or to Astier+19 exponential
            # approximation (Eq. 16).  Fill up
            # PhotonTransferCurveDataset object.
            datasetPtc = self.fitPtc(datasetPtc)
        if inputExpList is not None:
            # It should be a list of exposures, to get the detector.
            detector = inputExpList[0].getDetector()
        else:
            detector = None
        datasetPtc.updateMetadata(setDate=True, camera=camera, detector=detector)

        return pipeBase.Struct(
            outputPtcDataset=datasetPtc,
        )

    def fitCovariancesAstier(self, dataset):
        """Fit measured flat covariances to full model in Astier+19.

        Parameters
        ----------
        dataset : `lsst.ip.isr.ptcDataset.PhotonTransferCurveDataset`
            The dataset containing information such as the means,
            (co)variances, and exposure times.

        Returns
        -------
        dataset : `lsst.ip.isr.ptcDataset.PhotonTransferCurveDataset`
            This is the same dataset as the input paramter, however,
            it has been modified to include information such as the
            fit vectors and the fit parameters. See the class
            `PhotonTransferCurveDatase`.
        """
        covFits, covFitsNoB = fitDataFullCovariance(dataset)
        dataset = self.getOutputPtcDataCovAstier(dataset, covFits, covFitsNoB)

        return dataset

    def getOutputPtcDataCovAstier(self, dataset, covFits, covFitsNoB):
        """Get output data for PhotonTransferCurveCovAstierDataset from CovFit
        objects.

        Parameters
        ----------
        dataset : `lsst.ip.isr.ptcDataset.PhotonTransferCurveDataset`
            The dataset containing information such as the means,
            variances and exposure times.
        covFits : `dict`
            Dictionary of CovFit objects, with amp names as keys.
        covFitsNoB : `dict`
             Dictionary of CovFit objects, with amp names as keys, and
             'b=0' in Eq. 20 of Astier+19.

        Returns
        -------
        dataset : `lsst.ip.isr.ptcDataset.PhotonTransferCurveDataset`
            This is the same dataset as the input paramter, however,
            it has been modified to include extra information such as
            the mask 1D array, gains, reoudout noise, measured signal,
            measured variance, modeled variance, a, and b coefficient
            matrices (see Astier+19) per amplifier.  See the class
            `PhotonTransferCurveDatase`.
        """
        assert(len(covFits) == len(covFitsNoB))

        for i, amp in enumerate(dataset.ampNames):
            lenInputTimes = len(dataset.rawExpTimes[amp])
            # Not used when ptcFitType is 'FULLCOVARIANCE'
            dataset.ptcFitPars[amp] = [np.nan]
            dataset.ptcFitParsError[amp] = [np.nan]
            dataset.ptcFitChiSq[amp] = np.nan
            if amp in covFits:
                fit = covFits[amp]
                fitNoB = covFitsNoB[amp]
                # Save full covariances, covariances models, and their weights
                # dataset.expIdMask is already full
                dataset.covariances[amp] = fit.cov
                dataset.covariancesModel[amp] = fit.evalCovModel()
                dataset.covariancesSqrtWeights[amp] = fit.sqrtW
                dataset.aMatrix[amp] = fit.getA()
                dataset.bMatrix[amp] = fit.getB()
                dataset.covariancesModelNoB[amp] = fitNoB.evalCovModel()
                dataset.aMatrixNoB[amp] = fitNoB.getA()

                (meanVecFinal, varVecFinal, varVecModel,
                    wc, varMask) = fit.getFitData(0, 0, divideByMu=False)
                gain = fit.getGain()

                dataset.gain[amp] = gain
                dataset.gainErr[amp] = fit.getGainErr()
                dataset.noise[amp] = np.sqrt(fit.getRon())
                dataset.noiseErr[amp] = fit.getRonErr()
                dataset.finalVars[amp] = varVecFinal
                dataset.finalModelVars[amp] = varVecModel
                dataset.finalMeans[amp] = meanVecFinal

            else:
                # Bad amp
                # Entries need to have proper dimensions so read/write
                # with astropy.Table works.
                matrixSide = self.config.maximumRangeCovariancesAstier
                nanMatrix = np.full((matrixSide, matrixSide), np.nan)
                listNanMatrix = np.full((lenInputTimes, matrixSide, matrixSide), np.nan)

                dataset.covariances[amp] = listNanMatrix
                dataset.covariancesModel[amp] = listNanMatrix
                dataset.covariancesSqrtWeights[amp] = listNanMatrix
                dataset.aMatrix[amp] = nanMatrix
                dataset.bMatrix[amp] = nanMatrix
                dataset.covariancesModelNoB[amp] = listNanMatrix
                dataset.aMatrixNoB[amp] = nanMatrix

                dataset.expIdMask[amp] = np.repeat(np.nan, lenInputTimes)
                dataset.gain[amp] = np.nan
                dataset.gainErr[amp] = np.nan
                dataset.noise[amp] = np.nan
                dataset.noiseErr[amp] = np.nan
                dataset.finalVars[amp] = np.repeat(np.nan, lenInputTimes)
                dataset.finalModelVars[amp] = np.repeat(np.nan, lenInputTimes)
                dataset.finalMeans[amp] = np.repeat(np.nan, lenInputTimes)

        return dataset

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
    def _getInitialGoodPoints(means, variances, minVarPivotSearch):
        """Return a boolean array to mask bad points.

        Parameters
        ----------
        means : `numpy.array`
            Input array with mean signal values.
        variances : `numpy.array`
            Input array with variances at each mean value.
        minVarPivotSearch : `float`
            Minimum variance point (in ADU^2) after which the pivot point
            wher the variance starts decreasing should be sought.

        Returns
        ------
        goodPoints : `numpy.array` [`bool`]
            Boolean array to select good (`True`) and bad (`False`)
            points.

        Notes
        -----
        Eliminate points beyond which the variance decreases
        """
        goodPoints = np.ones_like(means, dtype=bool)
        pivotList = np.where(np.array(np.diff(variances)) < 0)[0]
        if len(pivotList) > 0:
            # For small values, sometimes the variance decreases slightly
            # Only look when var > self.config.minVarPivotSearch
            pivotList = [p for p in pivotList if variances[p] > minVarPivotSearch]
            if len(pivotList) > 0:
                # Require that the decrease in variance happen for two
                # consecutive signal levels
                pivotIndex = np.min(np.where(np.diff(pivotList) == 1)[0])
                pivot = pivotList[pivotIndex]
                goodPoints[pivot+1:len(goodPoints)] = False

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
        """Fit the photon transfer curve to a polynomial or to Astier+19
        approximation.

        Fit the photon transfer curve with either a polynomial of the order
        specified in the task config, or using the exponential approximation
        in Astier+19 (Eq. 16).

        Sigma clipping is performed iteratively for the fit, as well as an
        initial clipping of data points that are more than
        config.initialNonLinearityExclusionThreshold away from lying on a
        straight line. This other step is necessary because the photon transfer
        curve turns over catastrophically at very high flux (because saturation
        drops the variance to ~0) and these far outliers cause the initial fit
        to fail, meaning the sigma cannot be calculated to perform the
        sigma-clipping.

        Parameters
        ----------
        dataset : `lsst.ip.isr.ptcDataset.PhotonTransferCurveDataset`
            The dataset containing the means, variances and exposure times.

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

            goodPoints = self._getInitialGoodPoints(meanVecOriginal, varVecOriginal,
                                                    self.config.minVarPivotSearch)
            if not (goodPoints.any()):
                msg = (f"SERIOUS: All points in goodPoints: {goodPoints} are bad."
                       f"Setting {ampName} to BAD.")
                self.log.warning(msg)
                # Fill entries with NaNs
                self.fillBadAmp(dataset, ptcFitType, ampName)
                continue

            mask = goodPoints

            if ptcFitType == 'EXPAPPROXIMATION':
                ptcFunc = funcAstier
                parsIniPtc = [-1e-9, 1.0, 10.]  # a00, gain, noisei^2
                # lowers and uppers obtained from BOT data studies by
                # C. Lage (UC Davis, 11/2020).
                bounds = self._boundsForAstier(parsIniPtc, lowers=[-1e-4, 0.5, -2000],
                                               uppers=[1e-4, 2.5, 2000])
            if ptcFitType == 'POLYNOMIAL':
                ptcFunc = funcPolynomial
                parsIniPtc = self._initialParsForPolynomial(self.config.polynomialFitDegree + 1)
                bounds = self._boundsForPolynomial(parsIniPtc)

            # Before bootstrap fit, do an iterative fit to get rid of outliers
            count = 1
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
        """Fill the dataset with NaNs if there are not enough good points.

        Parameters
        ----------
        dataset : `lsst.ip.isr.ptcDataset.PhotonTransferCurveDataset`
            The dataset containing the means, variances and exposure times.
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
        dataset.finalVars[ampName] = np.repeat(np.nan, len(dataset.rawExpTimes[ampName]))
        dataset.finalModelVars[ampName] = np.repeat(np.nan, len(dataset.rawExpTimes[ampName]))
        dataset.finalMeans[ampName] = np.repeat(np.nan, len(dataset.rawExpTimes[ampName]))

        return
