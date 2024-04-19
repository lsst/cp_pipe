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
import warnings

import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase
from lsst.cp.pipe.utils import (fitLeastSq, fitBootstrap, funcPolynomial,
                                funcAstier, symmetrize, Pol2D)

from scipy.signal import fftconvolve
from scipy.optimize import least_squares
from itertools import groupby
from operator import itemgetter

import lsst.pipe.base.connectionTypes as cT

from lsst.ip.isr import PhotonTransferCurveDataset

import copy


__all__ = ['PhotonTransferCurveSolveConfig', 'PhotonTransferCurveSolveTask']


class PhotonTransferCurveSolveConnections(pipeBase.PipelineTaskConnections,
                                          dimensions=("instrument", "detector")):
    inputCovariances = cT.Input(
        name="ptcCovariances",
        doc="Tuple with measured covariances from flats.",
        storageClass="PhotonTransferCurveDataset",
        dimensions=("instrument", "exposure", "detector"),
        isCalibration=True,
        multiple=True,
    )
    camera = cT.PrerequisiteInput(
        name="camera",
        doc="Camera the input data comes from.",
        storageClass="Camera",
        dimensions=("instrument",),
        isCalibration=True,
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
    minMeanSignal = pexConfig.DictField(
        keytype=str,
        itemtype=float,
        doc="Minimum values (inclusive) of mean signal (in ADU) per amp to use."
            " The same cut is applied to all amps if this parameter [`dict`] is passed as "
            " {'ALL_AMPS': value}",
        default={'ALL_AMPS': 0.0},
    )
    maxMeanSignal = pexConfig.DictField(
        keytype=str,
        itemtype=float,
        doc="Maximum values (inclusive) of mean signal (in ADU) below which to consider, per amp."
            " The same cut is applied to all amps if this dictionary is of the form"
            " {'ALL_AMPS': value}",
        default={'ALL_AMPS': 1e6},
    )
    maximumRangeCovariancesAstier = pexConfig.Field(
        dtype=int,
        doc="Maximum range of measured covariances as in Astier+19",
        default=8,
    )
    maximumRangeCovariancesAstierFullCovFit = pexConfig.Field(
        dtype=int,
        doc="Maximum range up to where to fit covariances as in Astier+19, "
            "for the FULLCOVARIANCE model."
            "This is different from  maximumRangeCovariancesAstier."
            "It should be less or equal than maximumRangeCovariancesAstier."
            "The number of parameters for this model is "
            "3*maximumRangeCovariancesAstierFullCovFit^2 + 1, so increase with care "
            "so that the fit is not too slow.",
        default=8,
    )
    doSubtractLongRangeCovariances = pexConfig.Field(
        dtype=bool,
        doc="Subtract long-range covariances before FULLCOVARIANCE fit, "
            "beyond startLongRangeCovariances?",
        default=False,
    )
    startLongRangeCovariances = pexConfig.Field(
        dtype=int,
        doc="If doSubtractLongRangeCovariances is True, subtract covariances "
            "beyond this range. It should be less than maximumRangeCovariancesAstier. ",
        default=4,
    )
    polyDegLongRangeCovariances = pexConfig.Field(
        dtype=int,
        doc="If doSubtractLongRangeCovariances is True, polynomial "
            "degree to fit data beyond startLongRangeCovariances.",
        default=1,
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
    doLegacyTurnoffSelection = pexConfig.Field(
        dtype=bool,
        doc="Use 'legacy' computation for PTC turnoff selection. If set "
            "to False, then the KS test p-value selection will be used instead.",
        default=False,
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
    maxSignalInitialPtcOutlierFit = pexConfig.Field(
        dtype=float,
        doc="Maximum signal considered for intial outlier fit. This should be below "
            "the PTC turnoff to ensure accurate outlier rejection. If "
            "scaleMaxSignalInitialPtcOutlierFit=True then the units are electrons; "
            "otherwise ADU.",
        default=50_000.,
    )
    maxDeltaInitialPtcOutlierFit = pexConfig.Field(
        dtype=float,
        doc="If there are any outliers in the initial fit that have mean greater than "
            "maxSignalInitialPtcOutlierFit, then no points that have this delta "
            "mean from the previous ``good`` point are allowed. If "
            "scaleMaxSignalInitialPtcOutlierFit=True then the units are electrons; "
            "otherwise ADU.",
        default=9_000.,
    )
    scaleMaxSignalInitialPtcOutlierFit = pexConfig.Field(
        dtype=bool,
        doc="Scale maxSignalInitialPtcOutlierFit and maxDeltaInitialPtcOutlierFit by "
            "approximate gain?  If yes then "
            "maxSignalInitialPtcOutlierFit and maxDeltaInitialPtcOutlierFit are assumed "
            "to have units of electrons, otherwise ADU.",
        default=True,
    )
    minVarPivotSearch = pexConfig.Field(
        dtype=float,
        doc="The code looks for a pivot signal point after which the variance starts decreasing at high-flux"
            " to exclude then from the PTC model fit. However, sometimes at low fluxes, the variance"
            " decreases slightly. Set this variable for the variance value, in ADU^2, after which the pivot "
            " should be sought. Only used if doLegacyTurnoffSelection is True.",
        default=10000,
    )
    consecutivePointsVarDecreases = pexConfig.RangeField(
        dtype=int,
        doc="Required number of consecutive points/fluxes in the PTC where the variance "
            "decreases in order to find a first estimate of the PTC turn-off. "
            "Only used if doLegacyTurnoffSelection is True.",
        default=2,
        min=2
    )
    ksTestMinPvalue = pexConfig.Field(
        dtype=float,
        doc="Minimum value of the Gaussian histogram KS test p-value to be used in PTC fit. "
            "Only used if doLegacyTurnoffSelection is False.",
        default=0.01,
    )
    doFitBootstrap = pexConfig.Field(
        dtype=bool,
        doc="Use bootstrap for the PTC fit parameters and errors?.",
        default=False,
    )
    binSize = pexConfig.Field(
        dtype=int,
        doc="Bin the image by this factor in both dimensions.",
        default=1,
    )

    def validate(self):
        super().validate()
        fitMatrixSide = self.maximumRangeCovariancesAstierFullCovFit
        measureMatrixSide = self.maximumRangeCovariancesAstier
        if self.ptcFitType == "FULLCOVARIANCE":
            if fitMatrixSide > measureMatrixSide:
                raise RuntimeError("Covariance fit size %s is larger than"
                                   "measurement size %s.",
                                   fitMatrixSide, measureMatrixSide)
            if self.doSubtractLongRangeCovariances:
                startLag = self.startLongRangeCovariances
                if measureMatrixSide < startLag:
                    raise RuntimeError("Covariance measure size %s is smaller than long"
                                       "-range covariance starting point %s.",
                                       measureMatrixSide, startLag)


class PhotonTransferCurveSolveTask(pipeBase.PipelineTask):
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
        butlerQC : `~lsst.daf.butler.QuantumContext`
            Butler to operate on.
        inputRefs : `~lsst.pipe.base.InputQuantizedConnection`
            Input data refs to load.
        ouptutRefs : `~lsst.pipe.base.OutputQuantizedConnection`
            Output data refs to persist.
        """
        inputs = butlerQC.get(inputRefs)
        detId = inputRefs.inputCovariances[0].dataId['detector']
        outputs = self.run(inputCovariances=inputs['inputCovariances'], camera=inputs['camera'], detId=detId)
        butlerQC.put(outputs, outputRefs)

    def run(self, inputCovariances, camera=None, detId=0):
        """Fit measured covariances to different models.

        Parameters
        ----------
        inputCovariances : `list` [`lsst.ip.isr.PhotonTransferCurveDataset`]
            List of lsst.ip.isr.PhotonTransferCurveDataset datasets.
        camera : `lsst.afw.cameraGeom.Camera`, optional
            Input camera.
        detId : `int`
            Detector ID to locate the detector in the camera and
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
        # Find the ampNames from a non-dummy ptc.
        ampNames = []
        for partialPtcDataset in inputCovariances:
            if partialPtcDataset.ptcFitType != 'DUMMY':
                ampNames = partialPtcDataset.ampNames
                break

        # Each amp may have a different min and max ADU signal
        # specified in the config.
        maxMeanSignalDict = {ampName: 1e6 for ampName in ampNames}
        minMeanSignalDict = {ampName: 0.0 for ampName in ampNames}
        for ampName in ampNames:
            if 'ALL_AMPS' in self.config.maxMeanSignal:
                maxMeanSignalDict[ampName] = self.config.maxMeanSignal['ALL_AMPS']
            elif ampName in self.config.maxMeanSignal:
                maxMeanSignalDict[ampName] = self.config.maxMeanSignal[ampName]

            if 'ALL_AMPS' in self.config.minMeanSignal:
                minMeanSignalDict[ampName] = self.config.minMeanSignal['ALL_AMPS']
            elif ampName in self.config.minMeanSignal:
                minMeanSignalDict[ampName] = self.config.minMeanSignal[ampName]

        # Assemble individual PTC datasets into a single PTC dataset.
        datasetPtc = PhotonTransferCurveDataset(
            ampNames=ampNames,
            ptcFitType=self.config.ptcFitType,
            covMatrixSide=self.config.maximumRangeCovariancesAstier,
            covMatrixSideFullCovFit=self.config.maximumRangeCovariancesAstierFullCovFit)

        for partialPtcDataset in inputCovariances:
            # Ignore dummy datasets
            if partialPtcDataset.ptcFitType == 'DUMMY':
                continue
            for ampName in ampNames:
                # The partial dataset consists of lists of values for each
                # quantity. In the case of the input exposure pairs, this is a
                # list of tuples. In all cases we only want the first
                # (and only) element of the list.
                datasetPtc.inputExpIdPairs[ampName].append(partialPtcDataset.inputExpIdPairs[ampName][0])
                datasetPtc.rawExpTimes[ampName] = np.append(datasetPtc.rawExpTimes[ampName],
                                                            partialPtcDataset.rawExpTimes[ampName][0])
                datasetPtc.rawMeans[ampName] = np.append(datasetPtc.rawMeans[ampName],
                                                         partialPtcDataset.rawMeans[ampName][0])
                datasetPtc.rawVars[ampName] = np.append(datasetPtc.rawVars[ampName],
                                                        partialPtcDataset.rawVars[ampName][0])
                datasetPtc.rowMeanVariance[ampName] = np.append(datasetPtc.rowMeanVariance[ampName],
                                                                partialPtcDataset.rowMeanVariance[ampName][0])
                datasetPtc.photoCharges[ampName] = np.append(datasetPtc.photoCharges[ampName],
                                                             partialPtcDataset.photoCharges[ampName][0])
                datasetPtc.histVars[ampName] = np.append(datasetPtc.histVars[ampName],
                                                         partialPtcDataset.histVars[ampName][0])
                datasetPtc.histChi2Dofs[ampName] = np.append(datasetPtc.histChi2Dofs[ampName],
                                                             partialPtcDataset.histChi2Dofs[ampName][0])
                datasetPtc.kspValues[ampName] = np.append(datasetPtc.kspValues[ampName],
                                                          partialPtcDataset.kspValues[ampName][0])
                datasetPtc.noiseList[ampName] = np.append(datasetPtc.noiseList[ampName],
                                                          partialPtcDataset.noise[ampName])
                datasetPtc.covariances[ampName] = np.append(
                    datasetPtc.covariances[ampName].ravel(),
                    partialPtcDataset.covariances[ampName].ravel()
                ).reshape(
                    (
                        len(datasetPtc.rawExpTimes[ampName]),
                        datasetPtc.covMatrixSide,
                        datasetPtc.covMatrixSide,
                    )
                )
                datasetPtc.covariancesSqrtWeights[ampName] = np.append(
                    datasetPtc.covariancesSqrtWeights[ampName].ravel(),
                    partialPtcDataset.covariancesSqrtWeights[ampName].ravel()
                ).reshape(
                    (
                        len(datasetPtc.rawExpTimes[ampName]),
                        datasetPtc.covMatrixSide,
                        datasetPtc.covMatrixSide,
                    )
                )

                # Apply min/max masking.
                rawMean = partialPtcDataset.rawMeans[ampName][0]
                rawVar = partialPtcDataset.rawVars[ampName][0]
                expIdMask = partialPtcDataset.expIdMask[ampName][0]
                if (rawMean <= minMeanSignalDict[ampName]) or (rawMean >= maxMeanSignalDict[ampName]) \
                   or not np.isfinite(rawMean) or not np.isfinite(rawVar):
                    expIdMask = False

                kspValue = partialPtcDataset.kspValues[ampName][0]
                if not self.config.doLegacyTurnoffSelection and \
                   kspValue < self.config.ksTestMinPvalue:
                    expIdMask = False

                datasetPtc.expIdMask[ampName] = np.append(datasetPtc.expIdMask[ampName], expIdMask)

            for key, value in partialPtcDataset.auxValues.items():
                if key in datasetPtc.auxValues:
                    datasetPtc.auxValues[key] = np.append(datasetPtc.auxValues[key], value)
                else:
                    datasetPtc.auxValues[key] = value

        # Sort arrays that are filled so far in the final dataset by
        # rawMeans index.
        # First compute the mean across all the amps to make sure that they are
        # all sorted the same way.
        detectorMeans = np.zeros(len(datasetPtc.inputExpIdPairs[ampNames[0]]))

        for i in range(len(detectorMeans)):
            arr = np.array([datasetPtc.rawMeans[ampName][i] for ampName in ampNames])
            good, = (np.isfinite(arr)).nonzero()
            if good.size == 0:
                detectorMeans[i] = np.nan
            else:
                detectorMeans[i] = np.mean(arr[good])

        index = np.argsort(detectorMeans)

        for ampName in ampNames:
            datasetPtc.inputExpIdPairs[ampName] = np.array(
                datasetPtc.inputExpIdPairs[ampName]
            )[index].tolist()
            datasetPtc.rawExpTimes[ampName] = datasetPtc.rawExpTimes[ampName][index]
            datasetPtc.rawMeans[ampName] = datasetPtc.rawMeans[ampName][index]
            datasetPtc.rawVars[ampName] = datasetPtc.rawVars[ampName][index]
            datasetPtc.rowMeanVariance[ampName] = datasetPtc.rowMeanVariance[ampName][index]
            datasetPtc.photoCharges[ampName] = datasetPtc.photoCharges[ampName][index]
            datasetPtc.histVars[ampName] = datasetPtc.histVars[ampName][index]
            datasetPtc.histChi2Dofs[ampName] = datasetPtc.histChi2Dofs[ampName][index]
            datasetPtc.kspValues[ampName] = datasetPtc.kspValues[ampName][index]
            datasetPtc.expIdMask[ampName] = datasetPtc.expIdMask[ampName][index]
            datasetPtc.covariances[ampName] = datasetPtc.covariances[ampName][index]
            datasetPtc.covariancesSqrtWeights[ampName] = datasetPtc.covariancesSqrtWeights[ampName][index]
        for key, value in datasetPtc.auxValues.items():
            datasetPtc.auxValues[key] = value[index]

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

        # Initial validation of PTC fit.
        for ampName in ampNames:
            # These may be all nan (esp. in tests) and will be filtered
            # as appropriate later.
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                noise = np.nanmedian(datasetPtc.noiseList[ampName])
                noiseFitted = np.sqrt(datasetPtc.noise[ampName])

            # Check if noise is close to noiseFitted
            if not np.isclose(noiseFitted, noise, rtol=0.05, atol=0.0, equal_nan=True):
                self.log.warning(f"Read noise from PTC fit ({noiseFitted}) is not consistent "
                                 f"with read noise measured from overscan ({noise}) for "
                                 f"amplifier {ampName}. Try adjusting the fit range.")

        if camera:
            detector = camera[detId]
        else:
            detector = None
        datasetPtc.updateMetadataFromExposures(inputCovariances)
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

        - "a" coefficients (r by r matrix), units: 1/e
        - "b" coefficients (r by r matrix), units: 1/e
        - noise matrix (r by r matrix), units: e^2
        - gain, units: e/ADU

        "b" appears in Eq. 20 only through the "ab" combination, which
        is defined in this code as "c=ab".

        Total number of parameters: #entries(a) + #entries(c) + #entries(noise)
        + 1. This is equivalent to r^2 + r^2 + r^2 + 1, where "r" is the
        maximum lag considered for the covariances calculation, and the
        extra "1" is the gain. If "b" is 0, then "c" is 0, and len(pInit) will
        have r^2 fewer entries.
        """
        matrixSide = dataset.covMatrixSide
        matrixSideFit = dataset.covMatrixSideFullCovFit
        lenParams = matrixSideFit*matrixSideFit

        for ampName in dataset.ampNames:
            lenInputTimes = len(dataset.rawExpTimes[ampName])
            # Not used when ptcFitType is 'FULLCOVARIANCE'
            dataset.ptcFitPars[ampName] = np.array([np.nan])
            dataset.ptcFitParsError[ampName] = np.array([np.nan])
            dataset.ptcFitChiSq[ampName] = np.nan

            if ampName in dataset.badAmps:
                # Bad amp
                # Entries need to have proper dimensions so read/write
                # with astropy.Table works.
                nanMatrixFit = np.full((matrixSideFit, matrixSideFit), np.nan)
                listNanMatrix = np.full((lenInputTimes, matrixSide, matrixSide), np.nan)
                listNanMatrixFit = np.full((lenInputTimes, matrixSideFit, matrixSideFit), np.nan)
                dataset.covariancesModel[ampName] = listNanMatrixFit
                dataset.covariancesSqrtWeights[ampName] = listNanMatrix
                dataset.aMatrix[ampName] = nanMatrixFit
                dataset.bMatrix[ampName] = nanMatrixFit
                dataset.covariancesModelNoB[ampName] = listNanMatrixFit
                dataset.aMatrixNoB[ampName] = nanMatrixFit
                dataset.noiseMatrix[ampName] = nanMatrixFit
                dataset.noiseMatrixNoB[ampName] = nanMatrixFit

                dataset.expIdMask[ampName] = np.repeat(False, lenInputTimes)
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

            muAtAmpMasked = muAtAmp[maskAtAmp]
            covAtAmp = dataset.covariances[ampName]
            covAtAmpMasked = np.nan_to_num(covAtAmp)[maskAtAmp]
            covSqrtWeightsAtAmp = dataset.covariancesSqrtWeights[ampName]
            covSqrtWeightsAtAmpMasked = np.nan_to_num(covSqrtWeightsAtAmp)[maskAtAmp]

            # Subtract long-range covariances
            if self.config.doSubtractLongRangeCovariances:
                startLag = self.config.startLongRangeCovariances
                covAtAmpMasked, covSqrtWeightsAtAmpMasked = self.subtractDistantOffset(
                    muAtAmpMasked, covAtAmpMasked,
                    covSqrtWeightsAtAmpMasked,
                    start=startLag,
                    degree=self.config.polyDegLongRangeCovariances)

            # In principle, we could fit to a lag smaller than the measured
            # covariances.
            r = self.config.maximumRangeCovariancesAstierFullCovFit
            covAtAmpForFitMasked = covAtAmpMasked[:, :r, :r]
            covSqrtWeightsAtAmpForFitMasked = covSqrtWeightsAtAmpMasked[:, :r, :r]

            # Initial fit, to approximate parameters, with c=0
            a0, c0, noise0, gain0 = self.initialFitFullCovariance(
                muAtAmpMasked,
                covAtAmpForFitMasked,
                covSqrtWeightsAtAmpForFitMasked
            )

            # Fit full model (Eq. 20 of Astier+19) and same model with
            # b=0 (c=0 in this code)
            pInit = np.concatenate((a0.ravel(), c0.ravel(), noise0.ravel(), np.array(gain0)), axis=None)
            functionsDict = {'fullModel': self.funcFullCovarianceModel,
                             'fullModelNoB': self.funcFullCovarianceModelNoB}
            fitResults = {'fullModel': {'a': [], 'c': [], 'noise': [], 'gain': [], 'paramsErr': []},
                          'fullModelNoB': {'a': [], 'c': [], 'noise': [], 'gain': [], 'paramsErr': []}}
            for key in functionsDict:
                params, paramsErr, _ = fitLeastSq(pInit, muAtAmpMasked,
                                                  covAtAmpForFitMasked.ravel(), functionsDict[key],
                                                  weightsY=covSqrtWeightsAtAmpForFitMasked.ravel())
                a = params[:lenParams].reshape((matrixSideFit, matrixSideFit))
                c = params[lenParams:2*lenParams].reshape((matrixSideFit, matrixSideFit))
                noise = params[2*lenParams:3*lenParams].reshape((matrixSideFit, matrixSideFit))
                gain = params[-1]

                fitResults[key]['a'] = a
                fitResults[key]['c'] = c
                fitResults[key]['noise'] = noise
                fitResults[key]['gain'] = gain
                fitResults[key]['paramsErr'] = paramsErr

            # Put the information in the PTC dataset

            # Not used when ptcFitType is 'FULLCOVARIANCE'
            dataset.ptcFitPars[ampName] = np.array([np.nan])
            dataset.ptcFitParsError[ampName] = np.array([np.nan])
            dataset.ptcFitChiSq[ampName] = np.nan

            # Save full covariances, covariances models, and their weights.
            # dataset.expIdMask is already full, but needs to be
            # converted to bool.
            dataset.expIdMask[ampName] = np.array(dataset.expIdMask[ampName], dtype=bool)
            dataset.covariances[ampName] = covAtAmp
            # We evaluate the covariance model everywhere, even the
            # masked amps.
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
            dataset.noiseMatrix[ampName] = fitResults['fullModel']['noise']
            dataset.noiseMatrixNoB[ampName] = fitResults['fullModelNoB']['noise']

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
        matrixSideFit = self.config.maximumRangeCovariancesAstierFullCovFit

        # Initialize fit parameters
        a = np.zeros((matrixSideFit, matrixSideFit))
        c = np.zeros((matrixSideFit, matrixSideFit))
        noise = np.zeros((matrixSideFit, matrixSideFit))
        gain = 1.

        # iterate the fit to account for higher orders
        # the chi2 does not necessarily go down, so one could
        # stop when it increases
        oldChi2 = 1e30
        for _ in range(5):
            model = np.nan_to_num(self.evalCovModel(mu, a, c, noise, gain, setBtoZero=True))
            # loop on lags
            for i in range(matrixSideFit):
                for j in range(matrixSideFit):
                    # fit a parabola for a given lag
                    parsFit = np.polyfit(mu, cov[:, i, j] - model[:, i, j],
                                         2, w=sqrtW[:, i, j])
                    # model equation (Eq. 20) in Astier+19, with c=a*b=0:
                    a[i, j] += parsFit[0]
                    noise[i, j] += parsFit[2]
                    if i + j == 0:
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
        matrixSideFit = self.config.maximumRangeCovariancesAstierFullCovFit
        lenParams = matrixSideFit*matrixSideFit
        aMatrix = params[:lenParams].reshape((matrixSideFit, matrixSideFit))
        cMatrix = params[lenParams:2*lenParams].reshape((matrixSideFit, matrixSideFit))
        noiseMatrix = params[2*lenParams:3*lenParams].reshape((matrixSideFit, matrixSideFit))
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
        matrixSideFit = self.config.maximumRangeCovariancesAstierFullCovFit
        lenParams = matrixSideFit*matrixSideFit
        aMatrix = params[:lenParams].reshape((matrixSideFit, matrixSideFit))
        cMatrix = params[lenParams:2*lenParams].reshape((matrixSideFit, matrixSideFit))
        noiseMatrix = params[2*lenParams:3*lenParams].reshape((matrixSideFit, matrixSideFit))
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

        - "a" coefficients (M by M matrix), units: 1/e
        - "b" coefficients (M by M matrix), units: 1/e
        - noise matrix (M by M matrix), units: e^2
        - gain, units: e/ADU

        "b" appears in Eq. 20 only through the "ab" combination, which
        is defined in this code as "c=ab".
        """
        matrixSideFit = self.config.maximumRangeCovariancesAstierFullCovFit
        sa = (matrixSideFit, matrixSideFit)
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
        a2 = a2[np.newaxis, xc:xc + matrixSideFit, yc:yc + matrixSideFit]
        a3 = a3[np.newaxis, xc:xc + matrixSideFit, yc:yc + matrixSideFit]
        ac = ac[np.newaxis, xc:xc + matrixSideFit, yc:yc + matrixSideFit]
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

    def subtractDistantOffset(self, muAtAmpMasked, covAtAmpMasked, covSqrtWeightsAtAmpMasked,
                              start, degree=1):
        """Subtract distant offset from the covariance matrices.

        Parameters
        ----------
        muAtAmpMasked : `numpy.array`
            Masked mean flux array for a particular amplifier.
        covAtAmpMasked : `numpy.array`
            Masked measured covariances for a particular amplifier.
        covSqrtWeightsAtAmpMasked : `numpy.array`
            Masked inverse covariance weights for a particular amplifier.
        start : int, optional
            The starting index to eliminate the core for the fit.
        degree : int, optional
            Degree of the polynomial fit.

        Returns
        -------
        covAtAmpMasked : `numpy.array`
            Subtracted measured covariances for a particular amplifier.
        covSqrtWeightsAtAmpMasked : `numpy.array`
            Masked inverse covariance weights for a particular amplifier.

        Notes
        -----
        Ported from https://gitlab.in2p3.fr/astier/bfptc by P. Astier.

        This function subtracts a distant offset from the
        covariance matrices using polynomial fitting. The core
        of the matrices is eliminated for the fit.

        The function modifies the internal state of the object, updating the
        covariance matrices and related attributes.
        """
        for k in range(len(muAtAmpMasked)):
            # Make a copy because it will be altered
            w = np.copy(covSqrtWeightsAtAmpMasked[k, ...])
            wShape = w.shape
            i, j = np.meshgrid(range(wShape[0]), range(wShape[1]), indexing='ij')

            # Eliminate the core for the fit
            w[:start, :start] = 0

            poly = Pol2D(i, j, covAtAmpMasked[k, ...], degree, w=w)
            back = poly.eval(i, j)

            covAtAmpMasked[k, ...] -= back

        return covAtAmpMasked, covSqrtWeightsAtAmpMasked

    # EXPAPPROXIMATION and POLYNOMIAL fit methods
    @staticmethod
    def _initialParsForPolynomial(order):
        assert order >= 2
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

        # Finally, we filter out any infinities or NaNs.
        goodPoints[(~np.isfinite(means)) | (~np.isfinite(variances))] = False

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
        the order specified in the task config (POLNOMIAL),
        or using the exponential approximation in Astier+19
        (Eq. 16, FULLCOVARIANCE).

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
        RuntimeError
            Raised if dataset.ptcFitType is None or empty.
        """
        if dataset.ptcFitType:
            ptcFitType = dataset.ptcFitType
        else:
            raise RuntimeError("ptcFitType is None of empty in PTC dataset.")
        # For FULLCOVARIANCE model fit
        matrixSideFit = dataset.covMatrixSideFullCovFit
        nanMatrixFit = np.empty((matrixSideFit, matrixSideFit))
        nanMatrixFit[:] = np.nan

        for amp in dataset.ampNames:
            lenInputTimes = len(dataset.rawExpTimes[amp])
            listNanMatrixFit = np.empty((lenInputTimes, matrixSideFit, matrixSideFit))
            listNanMatrixFit[:] = np.nan

            dataset.covariancesModel[amp] = listNanMatrixFit
            dataset.aMatrix[amp] = nanMatrixFit
            dataset.bMatrix[amp] = nanMatrixFit
            dataset.covariancesModelNoB[amp] = listNanMatrixFit
            dataset.aMatrixNoB[amp] = nanMatrixFit
            dataset.noiseMatrix[amp] = nanMatrixFit
            dataset.noiseMatrixNoB[amp] = nanMatrixFit

        def errFunc(p, x, y):
            return ptcFunc(p, x) - y

        sigmaCutPtcOutliers = self.config.sigmaCutPtcOutliers
        maxIterationsPtcOutliers = self.config.maxIterationsPtcOutliers

        for i, ampName in enumerate(dataset.ampNames):
            meanVecOriginal = dataset.rawMeans[ampName].copy()
            varVecOriginal = dataset.rawVars[ampName].copy()
            varVecOriginal = self._makeZeroSafe(varVecOriginal)

            # These must be sorted for the given amplifier.
            meanVecSort = np.argsort(meanVecOriginal)
            meanVecSorted = meanVecOriginal[meanVecSort]
            varVecSorted = varVecOriginal[meanVecSort]

            if self.config.doLegacyTurnoffSelection:
                # Discard points when the variance starts to decrease after two
                # consecutive signal levels
                goodPoints = self._getInitialGoodPoints(meanVecSorted, varVecSorted,
                                                        self.config.minVarPivotSearch,
                                                        self.config.consecutivePointsVarDecreases)
            else:
                # Make sure we have this properly sorted.
                goodPoints = dataset.expIdMask[ampName].copy()
                goodPoints = goodPoints[meanVecSort]

            # Check if all points are bad from the 'cpExtractPtcTask'
            initialExpIdMask = dataset.expIdMask[ampName].copy()

            if not (goodPoints.any() and initialExpIdMask.any()):
                msg = (f"SERIOUS: All points in goodPoints: {goodPoints} or "
                       f"in initialExpIdMask: {initialExpIdMask} are bad."
                       f"Setting {ampName} to BAD.")
                self.log.warning(msg)
                # Fill entries with NaNs
                self.fillBadAmp(dataset, ptcFitType, ampName)
                continue

            mask = goodPoints.copy()

            if ptcFitType == 'EXPAPPROXIMATION':
                ptcFunc = funcAstier
                parsIniPtc = [-1e-9, 1.0, 10.]  # a00, gain, noise^2
                # lowers and uppers obtained from BOT data studies by
                # C. Lage (UC Davis, 11/2020).
                if self.config.binSize > 1:
                    bounds = self._boundsForAstier(parsIniPtc)
                else:
                    bounds = self._boundsForAstier(parsIniPtc, lowers=[-1e-4, 0.1, -2000],
                                                   uppers=[1e-4, 10.0, 2000])
            if ptcFitType == 'POLYNOMIAL':
                ptcFunc = funcPolynomial
                parsIniPtc = self._initialParsForPolynomial(self.config.polynomialFitDegree + 1)
                bounds = self._boundsForPolynomial(parsIniPtc)

            # We perform an initial (unweighted) fit of variance vs signal
            # (after initial KS test or post-drop selection) to look for
            # outliers, particularly at the high-flux end. The initial fit
            # is performed only for points that are guaranteed to be below
            # the PTC turnoff and then extrapolated to ensure that high
            # flux points that have abnormal variance values can be properly
            # rejected in this phase without biasing the initial fit.
            # This algorithm was initially developed by Seth Digel for
            # the EO Testing pipeline.

            if self.config.scaleMaxSignalInitialPtcOutlierFit:
                approxGain = np.nanmedian(meanVecSorted/varVecSorted)
                maxADUInitialPtcOutlierFit = self.config.maxSignalInitialPtcOutlierFit/approxGain
                maxDeltaADUInitialPtcOutlierFit = self.config.maxDeltaInitialPtcOutlierFit/approxGain
                self.log.info(
                    "Using approximate gain %.3f and ADU signal cutoff of %.1f and delta %.1f "
                    "for amplifier %s",
                    approxGain,
                    maxADUInitialPtcOutlierFit,
                    maxDeltaADUInitialPtcOutlierFit,
                    ampName,
                )
            else:
                maxADUInitialPtcOutlierFit = self.config.maxSignalInitialPtcOutlierFit
                maxDeltaADUInitialPtcOutlierFit = self.config.maxDeltaInitialPtcOutlierFit

            if maxIterationsPtcOutliers == 0:
                # We are not doing any outlier rejection here, but we do want
                # an initial fit.
                res = least_squares(
                    errFunc,
                    parsIniPtc,
                    bounds=bounds,
                    args=(meanVecSorted[mask], varVecSorted[mask]),
                )
                pars = res.x
                newMask = mask.copy()
            else:
                newMask = (mask & (meanVecSorted <= maxADUInitialPtcOutlierFit))

                converged = False
                count = 0
                lastMask = mask.copy()
                while count < maxIterationsPtcOutliers:
                    res = least_squares(
                        errFunc,
                        parsIniPtc,
                        bounds=bounds,
                        args=(meanVecSorted[newMask], varVecSorted[newMask]),
                    )
                    pars = res.x

                    sigResids = (varVecSorted - ptcFunc(pars, meanVecSorted))/np.sqrt(varVecSorted)
                    # The new mask includes points where the residuals are
                    # finite, are less than the cut, and include the original
                    # mask of known points that should not be used.
                    newMask = (
                        np.isfinite(sigResids)
                        & (np.abs(np.nan_to_num(sigResids)) < sigmaCutPtcOutliers)
                        & mask
                    )
                    # Demand at least 2 points to continue.
                    if np.count_nonzero(newMask) < 2:
                        msg = (f"SERIOUS: All points after outlier rejection are bad. "
                               f"Setting {ampName} to BAD.")
                        self.log.warning(msg)
                        # Fill entries with NaNs
                        self.fillBadAmp(dataset, ptcFitType, ampName)
                        break

                    self.log.debug(
                        "Iteration %d: Removed %d points in total for %s.",
                        count,
                        np.count_nonzero(mask) - np.count_nonzero(newMask),
                        ampName,
                    )

                    # Loop over all used (True) points. If one of them follows
                    # a False point, then it must be within
                    # maxDeltaADUInitialPtcOutlierFit of a True point. If it
                    # is a large gap, everything above is marked False.
                    useMask, = np.where(newMask)
                    for useIndex, usePoint in enumerate(useMask):
                        if useIndex == 0 or newMask[usePoint - 1]:
                            # The previous point was good; continue.
                            continue
                        deltaADU = meanVecSorted[usePoint] - meanVecSorted[useMask[useIndex - 1]]
                        if deltaADU < maxDeltaADUInitialPtcOutlierFit:
                            # This jump is fine; continue.
                            continue

                        # Mark all further points bad.
                        newMask[usePoint:] = False
                        break

                    # If the mask hasn't changed then break out.
                    if np.all(newMask == lastMask):
                        self.log.debug("Convergence at iteration %d; breaking loop for %s.", count, ampName)
                        converged = True
                        break

                    lastMask = newMask.copy()

                    count += 1

            if not converged:
                self.log.warning(
                    "Outlier detection was not converged prior to %d iteration for %s",
                    count,
                    ampName
                )

            # Set the mask to the new mask, and reset the sorting.
            mask = np.zeros(len(meanVecSort), dtype=np.bool_)
            mask[meanVecSort[newMask]] = True
            maskSorted = newMask.copy()

            if not mask.any():
                # We hae already filled the bad amp above, so continue.
                continue

            dataset.expIdMask[ampName] = mask

            parsIniPtc = pars
            meanVecFinal = meanVecOriginal[mask]
            varVecFinal = varVecOriginal[mask]

            # Save the maximum point after outlier detection as the
            # PTC turnoff point. We need to pay attention to the sorting
            # here.
            dataset.ptcTurnoff[ampName] = np.max(meanVecFinal)
            # And compute the ptcTurnoffSamplingError as one half the
            # difference between the previous and next point.
            lastGoodIndex = np.where(maskSorted)[0][-1]
            ptcTurnoffLow = meanVecSorted[lastGoodIndex - 1]
            if lastGoodIndex == (len(meanVecSorted) - 1):
                # If it's the last index, just use the interval.
                ptcTurnoffSamplingError = dataset.ptcTurnoff[ampName] - ptcTurnoffLow
            elif not np.isfinite(meanVecSorted[lastGoodIndex + 1]):
                # If the next index is not finite, just use the interval.
                ptcTurnoffSamplingError = dataset.ptcTurnoff[ampName] - ptcTurnoffLow
            else:
                ptcTurnoffSamplingError = (meanVecSorted[lastGoodIndex + 1] - ptcTurnoffLow)/2.
            dataset.ptcTurnoffSamplingError[ampName] = ptcTurnoffSamplingError

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
            # Fit the PTC.
            # The variance of the variance is Var(v)=2*v^2/Npix. This is
            # already calculated in `makeCovArray` of CpPtcExtract.
            # dataset.covariancesSqrtWeights[ampName][:,0,0]
            # has 1/sqrt(Var(v)).
            weightsY = dataset.covariancesSqrtWeights[ampName][:, 0, 0][mask]
            if self.config.doFitBootstrap:
                parsFit, parsFitErr, reducedChiSqPtc = fitBootstrap(parsIniPtc, meanVecFinal,
                                                                    varVecFinal, ptcFunc,
                                                                    weightsY=weightsY)
            else:
                parsFit, parsFitErr, reducedChiSqPtc = fitLeastSq(parsIniPtc, meanVecFinal,
                                                                  varVecFinal, ptcFunc,
                                                                  weightsY=weightsY)
            dataset.ptcFitPars[ampName] = parsFit
            dataset.ptcFitParsError[ampName] = parsFitErr
            dataset.ptcFitChiSq[ampName] = reducedChiSqPtc

            dataset.finalVars[ampName] = varVecOriginal
            dataset.finalVars[ampName][~mask] = np.nan
            dataset.finalModelVars[ampName] = ptcFunc(parsFit, meanVecOriginal)
            dataset.finalModelVars[ampName][~mask] = np.nan
            dataset.finalMeans[ampName] = meanVecOriginal
            dataset.finalMeans[ampName][~mask] = np.nan

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
            dataset.badAmps = []

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
