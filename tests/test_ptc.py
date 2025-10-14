#!/usr/bin/env python

#
# LSST Data Management System
#
# Copyright 2008-2017  AURA/LSST.
#
# This product includes software developed by the
# LSST Project (http://www.lsst.org/).
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
# You should have received a copy of the LSST License Statement and
# the GNU General Public License along with this program.  If not,
# see <https://www.lsstcorp.org/LegalNotices/>.
#
"""Test cases for cp_pipe."""

import unittest
import numpy as np
import copy
import tempfile
import logging
import warnings

import lsst.log
import lsst.utils
import lsst.utils.tests

import lsst.cp.pipe as cpPipe
import lsst.ip.isr.isrMock as isrMock
from lsst.ip.isr import PhotonTransferCurveDataset, PhotodiodeCalib, AmpOffsetTask
from lsst.cp.pipe.utils import makeMockFlats, ampOffsetGainRatioFixup

from lsst.pipe.base import InMemoryDatasetHandle


class FakeCamera(list):
    def getName(self):
        return "FakeCam"


class PretendRef:
    "A class to act as a mock exposure reference"

    def __init__(self, exposure):
        self.exp = exposure

    def get(self, component=None):
        if component == "visitInfo":
            return self.exp.info.getVisitInfo()
        elif component == "detector":
            return self.exp.getDetector()
        elif component == "metadata":
            return self.exp.getMetadata()
        else:
            return self.exp


class MeasurePhotonTransferCurveTaskTestCase(lsst.utils.tests.TestCase):
    """A test case for the PTC tasks."""

    def setUp(self):
        self.defaultConfigExtract = (
            cpPipe.ptc.PhotonTransferCurveExtractPairTask.ConfigClass()
        )
        self.defaultTaskExtract = cpPipe.ptc.PhotonTransferCurveExtractTask(
            config=self.defaultConfigExtract
        )

        self.defaultConfigSolve = cpPipe.ptc.PhotonTransferCurveSolveTask.ConfigClass()
        self.defaultTaskSolve = cpPipe.ptc.PhotonTransferCurveSolveTask(
            config=self.defaultConfigSolve
        )

        self.flatMean = 2000
        self.readNoiseAdu = 10
        mockImageConfig = isrMock.IsrMock.ConfigClass()

        # flatDrop is not really relevant as we replace the data
        # but good to note it in case we change how this image is made
        mockImageConfig.flatDrop = 0.99999
        mockImageConfig.isTrimmed = True

        self.flatExp1 = isrMock.FlatMock(config=mockImageConfig).run()
        self.flatExp2 = self.flatExp1.clone()
        (shapeY, shapeX) = self.flatExp1.getDimensions()

        self.flatWidth = np.sqrt(self.flatMean) + self.readNoiseAdu

        self.rng1 = np.random.RandomState(1984)
        flatData1 = self.rng1.normal(self.flatMean, self.flatWidth, (shapeX, shapeY))
        self.rng2 = np.random.RandomState(666)
        flatData2 = self.rng2.normal(self.flatMean, self.flatWidth, (shapeX, shapeY))

        self.flatExp1.image.array[:] = flatData1
        self.flatExp2.image.array[:] = flatData2

        # create fake PTC data to see if fit works, for one amp ('amp')
        self.flux = 1000.0  # ADU/sec
        self.timeVec = np.arange(1.0, 101.0, 5)
        self.k2NonLinearity = -5e-6
        # quadratic signal-chain non-linearity
        muVec = self.flux * self.timeVec + self.k2NonLinearity * self.timeVec**2
        self.gain = 0.75  # e-/ADU
        self.c1 = 1.0 / self.gain
        self.noiseSq = 2 * self.gain  # 7.5 (e-)^2
        self.a00 = -1.2e-6
        self.c2 = -1.5e-6
        self.c3 = -4.7e-12  # tuned so that it turns over for 200k mean
        self.photoCharges = np.linspace(1e-8, 1e-5, len(self.timeVec))

        self.ampNames = [
            amp.getName() for amp in self.flatExp1.getDetector().getAmplifiers()
        ]
        self.dataset = PhotonTransferCurveDataset(self.ampNames, ptcFitType="PARTIAL")
        self.covariancesSqrtWeights = {}
        for (
            ampName
        ) in self.ampNames:  # just the expTimes and means here - vars vary per function
            self.dataset.rawExpTimes[ampName] = self.timeVec
            self.dataset.rawMeans[ampName] = muVec
            self.dataset.covariancesSqrtWeights[ampName] = np.zeros(
                (1, self.dataset.covMatrixSide, self.dataset.covMatrixSide)
            )

    def test_covAstier(self):
        """Test to check getCovariancesAstier

        We check that the gain is the same as the input gain from the
        mock data, that the covariances via FFT (as it is in
        MeasurePhotonTransferCurveTask when doCovariancesAstier=True)
        are the same as calculated in real space, and that Cov[0, 0]
        (i.e., the variances) are similar to the variances calculated
        with the standard method (when doCovariancesAstier=false),

        """
        extractConfig = self.defaultConfigExtract
        extractConfig.minNumberGoodPixelsForCovariance = 5000
        extractConfig.detectorMeasurementRegion = "FULL"
        extractConfig.doExtractPhotodiodeData = True
        extractConfig.doKsHistMeasurement = True
        extractConfig.auxiliaryHeaderKeys = ["CCOBCURR", "CCDTEMP"]
        extractPairTask = cpPipe.ptc.PhotonTransferCurveExtractPairTask(config=extractConfig)

        # Create solve task config
        solveConfig = self.defaultConfigSolve
        # Cut off the low-flux point which is a bad fit, and this
        # also exercises this functionality and makes the tests
        # run a lot faster.
        solveConfig.minMeanSignal["ALL_AMPS"] = 2000.0
        # Set the outlier fit threshold higher than the default appropriate
        # for this test dataset.
        solveConfig.maxSignalInitialPtcOutlierFit = 90000.0
        # Given the limited nature of the test data, we can only
        # reliably fit out to a 3x3 covariance matrix.
        # Improvements will be investigated on DM-46131.
        solveConfig.maximumRangeCovariancesAstierFullCovFit = 3

        inputGain = self.gain

        muStandard, varStandard = {}, {}
        nPixelCovarianceStandard = {}
        expHandles = []
        expIds = []
        pdHandles = []
        idCounter = 0
        for i, expTime in enumerate(self.timeVec):
            mockExp1, mockExp2 = makeMockFlats(
                expTime,
                gain=inputGain,
                readNoiseElectrons=3,
                expId1=idCounter,
                expId2=idCounter + 1,
                ampNames=self.ampNames,
            )
            for mockExp in [mockExp1, mockExp2]:
                md = mockExp.getMetadata()
                # These values are chosen to be easily compared after
                # processing for correct ordering.
                md['CCOBCURR'] = float(idCounter)
                md['CCDTEMP'] = float(idCounter + 1)
                mockExp.setMetadata(md)
                md['SEQFILE'] = 'a_seqfile'
                md['SEQNAME'] = 'a_seqfile'
                md['SEQCKSUM'] = 'deadbeef'

            mockExpRef1 = PretendRef(mockExp1)
            mockExpRef2 = PretendRef(mockExp2)
            expHandles.extend((mockExpRef1, mockExpRef2))
            expIds.append(idCounter)
            expIds.append(idCounter + 1)
            maskValue = mockExp1.mask.getPlaneBitMask(extractPairTask.config.maskNameList)
            for ampNumber, ampName in enumerate(self.ampNames):
                # cov has (i, j, var, cov, npix)
                (
                    im1Area,
                    im2Area,
                    imStatsCtrl,
                    mu1,
                    mu2,
                ) = extractPairTask.getImageAreasMasksStats(mockExp1, mockExp2)
                nPixelCovariance = ((im1Area.mask.array & maskValue) == 0).sum()
                nPixelCovarianceStandard.setdefault(ampName, []).append(nPixelCovariance)
                muDiff, varDiff, covAstier, rowMeanVariance = extractPairTask.measureMeanVarCov(
                    im1Area, im2Area, imStatsCtrl, mu1, mu2
                )
                muStandard.setdefault(ampName, []).append(muDiff)
                varStandard.setdefault(ampName, []).append(varDiff)

            # Make a photodiode dataset to integrate.
            timeSamples = np.linspace(0, 20.0, 100)
            currentSamples1 = np.zeros(100)
            currentSamples1[50] = -1.0*self.photoCharges[i]
            currentSamples2 = np.zeros(100)
            currentSamples2[50] = -1.0*(self.photoCharges[i] + 1e-12)

            pdCalib1 = PhotodiodeCalib(timeSamples=timeSamples, currentSamples=currentSamples1)
            pdCalib1.currentScale = -1.0
            pdCalib1.integrationMethod = "CHARGE_SUM"

            pdCalib2 = PhotodiodeCalib(timeSamples=timeSamples, currentSamples=currentSamples2)
            pdCalib2.currentScale = -1.0
            pdCalib2.integrationMethod = "CHARGE_SUM"

            pdHandles.append(
                InMemoryDatasetHandle(
                    pdCalib1,
                    dataId={"exposure": idCounter},
                )
            )
            pdHandles.append(
                InMemoryDatasetHandle(
                    pdCalib2,
                    dataId={"exposure": idCounter + 1},
                )
            )
            idCounter += 2

        # Divide out the pairs.
        outputCovariances = []
        for i in range(len(expHandles) // 2):
            inputExp = [expHandles[2*i], expHandles[2*i + 1]]
            inputDims = [expIds[2*i], expIds[2*i + 1]]
            inputPhotodiodeData = [pdHandles[2*i], pdHandles[2*i + 1]]
            results = extractPairTask.run(
                inputExp=inputExp,
                inputDims=inputDims,
                inputPhotodiodeData=inputPhotodiodeData,
            )
            outputCovariances.append(results.outputCovariance)

        # Force the last PTC dataset to have a NaN, and ensure that the
        # task runs (DM-38029).  This is a minor perturbation and does not
        # affect the output comparison.
        outputCovariances[-1].rawMeans["C:0,0"] = np.array([np.nan])
        outputCovariances[-1].rawVars["C:0,0"] = np.array([np.nan])

        # Force the next-to-last PTC dataset to have a decreased variance to
        # ensure that the outlier fit rejection works.
        rawVar = outputCovariances[-2].rawVars["C:0,0"]
        outputCovariances[-2].rawVars["C:0,0"] = rawVar * 0.9

        # Reorganize the outputCovariances so we can confirm they come
        # out sorted afterwards.
        outputCovariancesRev = outputCovariances[::-1]

        # Some expected values for noise matrix, just to check that
        # it was calculated.
        expectedNoiseMatrix = {
            "FULLCOVARIANCE": np.array(
                [[8.99474598, 9.94916264, -27.90587299],
                 [-2.95079527, -17.11827641, -47.88156244],
                 [5.24915021, -3.25786165, 26.09634067]],
            ),
            "FULLCOVARIANCE_NO_B": np.array(
                [[8.71049338, 12.48584043, -37.06585088],
                 [-4.80523971, -23.29102809, -66.37815343],
                 [7.48654766, -4.10168337, 35.64469824]],
            ),
        }

        for fitType in ["FULLCOVARIANCE", "FULLCOVARIANCE_NO_B"]:
            solveConfig.ptcFitType = fitType
            solveTask = cpPipe.ptc.PhotonTransferCurveSolveTask(config=solveConfig)

            # Ensure no warnings re: read noise mismatches are logged.
            with self.assertNoLogs(level=logging.WARNING):
                resultsSolve = solveTask.run(
                    outputCovariancesRev, camera=FakeCamera([self.flatExp1.getDetector()])
                )

            ptc = resultsSolve.outputPtcDataset

            # Confirm that metadata keywords are set.
            for key in ["SEQFILE", "SEQNAME", "SEQCKSUM"]:
                self.assertEqual(ptc.metadata[key], expHandles[0].get().metadata[key])
            self.assertEqual(ptc.metadata["INSTRUME"], FakeCamera().getName())
            det = expHandles[0].get().getDetector()
            self.assertEqual(ptc.metadata["DETECTOR"], det.getId())
            self.assertEqual(ptc.metadata["DET_NAME"], det.getName())
            self.assertEqual(ptc.metadata["DET_SER"], det.getSerial())

            for amp in self.ampNames:
                self.assertAlmostEqual(ptc.gain[amp], inputGain, places=2)
                self.assertFloatsAlmostEqual(
                    np.asarray(varStandard[amp])[ptc.expIdMask[amp]] / ptc.finalVars[amp][ptc.expIdMask[amp]],
                    1.0,
                    rtol=1e-4,
                )
                np.testing.assert_array_equal(ptc.nPixelCovariances[amp], nPixelCovarianceStandard[amp][0])

                # Check that the PTC turnoff is correctly computed.
                # This will be different for the C:0,0 amp.
                if amp == "C:0,0":
                    self.assertAlmostEqual(ptc.ptcTurnoff[amp], ptc.rawMeans[amp][-3])
                else:
                    self.assertAlmostEqual(ptc.ptcTurnoff[amp], ptc.rawMeans[amp][-1])

                # Test that all the quantities are correctly ordered and
                # have not accidentally been masked.
                for i, extractPtc in enumerate(outputCovariances):
                    self.assertFloatsAlmostEqual(
                        extractPtc.rawExpTimes[ampName][0],
                        ptc.rawExpTimes[ampName][i],
                    )
                    self.assertFloatsAlmostEqual(
                        extractPtc.rawMeans[ampName][0],
                        ptc.rawMeans[ampName][i],
                    )
                    self.assertFloatsAlmostEqual(
                        extractPtc.rawVars[ampName][0],
                        ptc.rawVars[ampName][i],
                    )
                    self.assertFloatsAlmostEqual(
                        extractPtc.rawDeltas[ampName][0],
                        ptc.rawDeltas[ampName][i],
                    )
                    # Ensure these are set to a non-zero value.
                    self.assertFloatsNotEqual(
                        extractPtc.rawDeltas[ampName][0],
                        0.0,
                    )
                    self.assertFloatsAlmostEqual(
                        extractPtc.photoCharges[ampName][0],
                        ptc.photoCharges[ampName][i],
                    )
                    self.assertFloatsAlmostEqual(
                        extractPtc.photoChargeDeltas[ampName][0],
                        ptc.photoChargeDeltas[ampName][i],
                    )
                    # Ensure these are set to a non-zero value.
                    self.assertFloatsNotEqual(
                        extractPtc.photoChargeDeltas[ampName][0],
                        0.0,
                    )
                    self.assertFloatsAlmostEqual(
                        extractPtc.histVars[ampName][0],
                        ptc.histVars[ampName][i],
                    )
                    self.assertFloatsAlmostEqual(
                        extractPtc.histChi2Dofs[ampName][0],
                        ptc.histChi2Dofs[ampName][i],
                    )
                    self.assertFloatsAlmostEqual(
                        extractPtc.kspValues[ampName][0],
                        ptc.kspValues[ampName][i],
                    )
                    self.assertFloatsAlmostEqual(
                        extractPtc.covariances[ampName][0],
                        ptc.covariances[ampName][i],
                    )
                    self.assertFloatsAlmostEqual(
                        extractPtc.covariancesSqrtWeights[ampName][0],
                        ptc.covariancesSqrtWeights[ampName][i],
                    )
                    self.assertFloatsAlmostEqual(
                        ptc.noiseMatrix[ampName],
                        expectedNoiseMatrix[fitType],
                        atol=1e-8,
                        rtol=None,
                    )
                    self.assertFloatsAlmostEqual(
                        ptc.ampOffsets[ampName],
                        0.0,
                    )
                    self.assertFloatsAlmostEqual(
                        ptc.noise[ampName],
                        np.nanmedian(ptc.noiseList[ampName]) * ptc.gain[ampName],
                        rtol=0.05,
                    )
                    # If the noise error is greater than the noise,
                    # something is seriously wrong. Possibly some
                    # kind of gain application mismatch.
                    self.assertLess(
                        ptc.noiseErr[ampName],
                        ptc.noise[ampName],
                    )

                mask = ptc.getGoodPoints(amp)

                values = (
                    ptc.covariancesModel[amp][mask, 0, 0] - ptc.covariances[amp][mask, 0, 0]
                ) / ptc.covariancesModel[amp][mask, 0, 0]
                np.testing.assert_array_less(np.abs(values), 2e-3)

                if ptc.ptcFitType == "FULLCOVARIANCE":
                    values = (
                        ptc.covariancesModel[amp][mask, 0, 1] - ptc.covariances[amp][mask, 0, 1]
                    ) / ptc.covariancesModel[amp][mask, 0, 1]
                    np.testing.assert_array_less(np.abs(values), 0.3)

                    values = (
                        ptc.covariancesModel[amp][mask, 1, 0] - ptc.covariances[amp][mask, 1, 0]
                    ) / ptc.covariancesModel[amp][mask, 1, 0]
                    np.testing.assert_array_less(np.abs(values), 0.3)

            # And test that the auxiliary values are there and
            # correctly ordered.
            self.assertIn('CCOBCURR', ptc.auxValues)
            self.assertIn('CCDTEMP', ptc.auxValues)
            firstExpIds = np.array([i for i, _ in ptc.inputExpIdPairs['C:0,0']], dtype=np.float64)
            self.assertFloatsAlmostEqual(ptc.auxValues['CCOBCURR'], firstExpIds)
            self.assertFloatsAlmostEqual(ptc.auxValues['CCDTEMP'], firstExpIds + 1)

            expIdsUsed = ptc.getExpIdsUsed("C:0,0")
            # Check that these are the same as the inputs, paired up, with the
            # first two (low flux) and final four (outliers, nans) removed.
            self.assertTrue(
                np.all(expIdsUsed == np.array(expIds).reshape(len(expIds) // 2, 2)[1:-2])
            )

            goodAmps = ptc.getGoodAmps()
            self.assertEqual(goodAmps, self.ampNames)

            # Check that every possibly modified field has the same length.
            covShape = None
            covSqrtShape = None
            covModelShape = None

            for ampName in self.ampNames:
                if covShape is None:
                    covShape = ptc.covariances[ampName].shape
                    covSqrtShape = ptc.covariancesSqrtWeights[ampName].shape
                    covModelShape = ptc.covariancesModel[ampName].shape
                else:
                    self.assertEqual(ptc.covariances[ampName].shape, covShape)
                    self.assertEqual(
                        ptc.covariancesSqrtWeights[ampName].shape, covSqrtShape
                    )
                    self.assertEqual(ptc.covariancesModel[ampName].shape, covModelShape)

                # Check if evalPtcModel produces expected values
                nanMask = ~np.isnan(ptc.finalMeans[ampName])
                means = ptc.finalMeans[ampName][nanMask]
                covModel = ptc.covariancesModel[ampName][nanMask]
                covariancesModel = ptc.evalPtcModel(means)[ampName]
                self.assertFloatsAlmostEqual(covariancesModel, covModel, atol=1e-12)

            # And check that this is serializable
            with tempfile.NamedTemporaryFile(suffix=".fits") as f:
                usedFilename = ptc.writeFits(f.name)
                fromFits = PhotonTransferCurveDataset.readFits(usedFilename)
            self.assertEqual(fromFits, ptc)

    def ptcFitAndCheckPtc(
        self,
        fitType=None,
        doFitBootstrap=False,
    ):
        localDataset = copy.deepcopy(self.dataset)
        localDataset.ptcFitType = fitType
        configSolve = copy.copy(self.defaultConfigSolve)
        if doFitBootstrap:
            configSolve.doFitBootstrap = True

        if fitType == "EXPAPPROXIMATION":
            g = self.gain
            for ampName in self.ampNames:
                localDataset.rawVars[ampName] = [
                    (
                        0.5 / (self.a00 * g**2) * (np.exp(2 * self.a00 * mu * g) - 1)
                        + self.noiseSq / (g * g)
                    )
                    for mu in localDataset.rawMeans[ampName]
                ]
        else:
            raise RuntimeError(
                "Fit type must be 'EXPAPPROXIMATION'"
            )

        # Initialize mask and covariance weights that will be used in fits.
        # Covariance weights values empirically determined from one of
        # the cases in test_covAstier.
        matrixSize = localDataset.covMatrixSide
        maskLength = len(localDataset.rawMeans[ampName])
        for ampName in self.ampNames:
            localDataset.expIdMask[ampName] = np.repeat(True, maskLength)
            localDataset.covariancesSqrtWeights[ampName] = np.repeat(
                np.ones((matrixSize, matrixSize)), maskLength
            ).reshape((maskLength, matrixSize, matrixSize))
            localDataset.covariancesSqrtWeights[ampName][:, 0, 0] = [
                0.07980188,
                0.01339653,
                0.0073118,
                0.00502802,
                0.00383132,
                0.00309475,
                0.00259572,
                0.00223528,
                0.00196273,
                0.00174943,
                0.00157794,
                0.00143707,
                0.00131929,
                0.00121935,
                0.0011334,
                0.00105893,
                0.00099357,
                0.0009358,
                0.00088439,
                0.00083833,
            ]

        solveTask = cpPipe.ptc.PhotonTransferCurveSolveTask(config=configSolve)

        if fitType == "EXPAPPROXIMATION":
            localDataset = solveTask.fitPtc(localDataset)
        else:
            localDataset = solveTask.fitDataFullCovariance(localDataset)

        # Check entries in localDataset, which was modified by the function
        for ampName in self.ampNames:
            self.assertEqual(fitType, localDataset.ptcFitType)
            self.assertAlmostEqual(self.gain, localDataset.gain[ampName])
            if fitType == "EXPAPPROXIMATION":
                self.assertAlmostEqual(
                    self.a00, localDataset.ptcFitPars[ampName][0]
                )
                # Noise already in electrons
                self.assertAlmostEqual(
                    np.sqrt(self.noiseSq), localDataset.noise[ampName]
                )
                # If the noise error is greater than the noise, something
                # is seriously wrong. Possibly some kind of gain application
                # mismatch.
                self.assertLess(
                    localDataset.noiseErr[ampName],
                    np.sqrt(self.noiseSq)
                )

            # Check if evalColModel produces expected values
            if not doFitBootstrap:
                nanMask = ~np.isnan(localDataset.finalMeans[ampName])
                means = localDataset.finalMeans[ampName][nanMask]
                model = localDataset.finalModelVars[ampName][nanMask]
                evalVarModel = localDataset.evalPtcModel(means)[ampName]
                self.assertFloatsEqual(evalVarModel, model)

    def test_lsstcam_samples(self):
        for dense in [False, True]:
            for mode in ["normal", "upturn", "dip"]:
                for doModelPtcRolloff in [False, True]:
                    rawMeans, rawVars, ptcTurnoff = self._getSampleMeanAndVar(dense=dense, mode=mode)

                    # We only need a single amp.
                    ampName = self.ampNames[0]
                    dataset = PhotonTransferCurveDataset([ampName], ptcFitType="EXPAPPROXIMATION")
                    dataset.rawExpTimes[ampName] = np.arange(len(rawMeans), dtype=np.float64) + 1.0
                    dataset.rawMeans[ampName] = rawMeans
                    dataset.rawVars[ampName] = rawVars
                    dataset.covariancesSqrtWeights[ampName] = np.repeat(
                        np.ones((dataset.covMatrixSide, dataset.covMatrixSide)), len(rawMeans)
                    ).reshape((len(rawMeans), dataset.covMatrixSide, dataset.covMatrixSide))
                    dataset.covariancesSqrtWeights[ampName][:, 0, 0] = np.sqrt(rawVars)
                    mask = np.ones(len(rawMeans), dtype=np.bool_)
                    mask[~np.isfinite(rawMeans) | ~np.isfinite(rawVars)] = False
                    dataset.expIdMask[ampName] = mask

                    configSolve = copy.copy(self.defaultConfigSolve)
                    configSolve.doModelPtcRolloff = doModelPtcRolloff
                    configSolve.doFitBootstrap = False

                    solveTask = cpPipe.ptc.PhotonTransferCurveSolveTask(config=configSolve)

                    initialFitDataset = solveTask.fitPtc(dataset, computePtcTurnoff=True)

                    # Model the PTC rolloff
                    if doModelPtcRolloff:
                        initialFitDataset = solveTask.fitPtcRolloff(initialFitDataset)

                    solvedDataset = solveTask.fitPtc(initialFitDataset, computePtcTurnoff=False)
                    # Check that the ptcTurnoff is what is expected.
                    self.assertFloatsAlmostEqual(
                        solvedDataset.ptcTurnoff[ampName],
                        ptcTurnoff,
                        msg=(
                            f"Dense: {dense}; Mode: {mode}, "
                            f"doModelPtcRolloff: {doModelPtcRolloff}; Amp: {ampName}"
                        ),
                    )

                    # Check that no values above the
                    # turnoff/rolloff are "good".
                    maxMean = solvedDataset.ptcTurnoff[ampName]
                    if doModelPtcRolloff:
                        # Check that the PTC rolloff is less
                        # than the PTC turnoff and within 10%
                        # of the turnoff.
                        if dense:
                            # Check that
                            self.assertLess(
                                solvedDataset.ptcRolloff[ampName],
                                solvedDataset.ptcTurnoff[ampName],
                                msg=(
                                    f"Dense: {dense}; Mode: {mode}, "
                                    f"doModelPtcRolloff: {doModelPtcRolloff}; "
                                    f"Amp: {ampName}"
                                ),
                            )
                            self.assertFloatsAlmostEqual(
                                solvedDataset.ptcRolloff[ampName],
                                solvedDataset.ptcTurnoff[ampName],
                                atol=0.1 * solvedDataset.ptcTurnoff[ampName],
                                msg=(
                                    f"Dense: {dense}; Mode: {mode}, "
                                    f"doModelPtcRolloff: {doModelPtcRolloff}; "
                                    f"Amp: {ampName}"
                                ),
                            )
                        else:
                            # In all other cases, it should not be able to
                            # find points beyond the turnoff, and the
                            # rolloff should be set to the PTC turnoff.
                            self.assertEqual(
                                solvedDataset.ptcRolloff[ampName],
                                solvedDataset.ptcTurnoff[ampName],
                                msg=(
                                    f"Dense: {dense}; Mode: {mode}, "
                                    f"doModelPtcRolloff: {doModelPtcRolloff}; "
                                    f"Amp: {ampName}"
                                ),
                            )

                        # If that passes set the maxMean to the PTC Rolloff
                        maxMean = solvedDataset.ptcRolloff[ampName]

                    above = (solvedDataset.finalMeans[ampName] > maxMean)
                    self.assertEqual(
                        np.sum(above),
                        0,
                        msg=(
                            f"Dense: {dense}; Mode: {mode}, "
                            f"doModelPtcRolloff: {doModelPtcRolloff}; Amp: {ampName}"
                        ),
                    )

                    # Check the sampling error on the turnoff.
                    turnoffIdx = np.argwhere(
                        solvedDataset.rawMeans[ampName] == solvedDataset.ptcTurnoff[ampName]
                    )
                    samplingError = (rawMeans[turnoffIdx + 1] - rawMeans[turnoffIdx - 1])/2.
                    self.assertFloatsAlmostEqual(
                        solvedDataset.ptcTurnoffSamplingError[ampName],
                        samplingError,
                        msg=f"Dense: {dense}; Mode: {mode}",
                    )

    def test_ptcFit(self):
        for fitType in ["EXPAPPROXIMATION"]:
            self.ptcFitAndCheckPtc(
                fitType=fitType
            )

    def test_meanVarMeasurement(self):
        task = self.defaultTaskExtract
        im1Area, im2Area, imStatsCtrl, mu1, mu2 = task.getImageAreasMasksStats(
            self.flatExp1, self.flatExp2
        )
        mu, varDiff, _, _ = task.measureMeanVarCov(im1Area, im2Area, imStatsCtrl, mu1, mu2)

        self.assertLess(self.flatWidth - np.sqrt(varDiff), 1)
        self.assertLess(self.flatMean - mu, 1)

    def test_meanVarMeasurementWithNans(self):
        task = self.defaultTaskExtract

        flatExp1 = self.flatExp1.clone()
        flatExp2 = self.flatExp2.clone()

        flatExp1.image.array[20:30, :] = np.nan
        flatExp2.image.array[20:30, :] = np.nan

        im1Area, im2Area, imStatsCtrl, mu1, mu2 = task.getImageAreasMasksStats(
            flatExp1, flatExp2
        )
        mu, varDiff, _, _ = task.measureMeanVarCov(im1Area, im2Area, imStatsCtrl, mu1, mu2)

        expectedMu1 = np.nanmean(flatExp1.image.array)
        expectedMu2 = np.nanmean(flatExp2.image.array)
        expectedMu = 0.5 * (expectedMu1 + expectedMu2)

        # Now the variance of the difference. First, create the diff image.
        im1 = flatExp1.maskedImage
        im2 = flatExp2.maskedImage

        temp = im2.clone()
        temp *= expectedMu1
        diffIm = im1.clone()
        diffIm *= expectedMu2
        diffIm -= temp
        diffIm /= expectedMu

        # Divide by two as it is what measureMeanVarCov returns
        # (variance of difference)
        expectedVar = 0.5 * np.nanvar(diffIm.image.array)

        # Check that the standard deviations and the emans agree to
        # less than 1 ADU
        self.assertLess(np.sqrt(expectedVar) - np.sqrt(varDiff), 1)
        self.assertLess(expectedMu - mu, 1)

    def test_meanVarMeasurementAllNan(self):
        task = self.defaultTaskExtract
        flatExp1 = self.flatExp1.clone()
        flatExp2 = self.flatExp2.clone()

        flatExp1.image.array[:, :] = np.nan
        flatExp2.image.array[:, :] = np.nan

        im1Area, im2Area, imStatsCtrl, mu1, mu2 = task.getImageAreasMasksStats(
            flatExp1, flatExp2
        )
        mu, varDiff, covDiff, rowMeanVariance = task.measureMeanVarCov(
            im1Area, im2Area, imStatsCtrl, mu1, mu2
        )

        self.assertTrue(np.isnan(mu))
        self.assertTrue(np.isnan(varDiff))
        self.assertTrue(covDiff is None)
        self.assertTrue(np.isnan(rowMeanVariance))

    def test_meanVarMeasurementTooFewPixels(self):
        task = self.defaultTaskExtract
        flatExp1 = self.flatExp1.clone()
        flatExp2 = self.flatExp2.clone()

        flatExp1.image.array[0:190, :] = np.nan
        flatExp2.image.array[0:190, :] = np.nan

        bit = flatExp1.mask.getMaskPlaneDict()["NO_DATA"]
        flatExp1.mask.array[0:190, :] &= 2**bit
        flatExp2.mask.array[0:190, :] &= 2**bit

        im1Area, im2Area, imStatsCtrl, mu1, mu2 = task.getImageAreasMasksStats(
            flatExp1, flatExp2
        )
        with self.assertLogs(level=logging.WARNING) as cm:
            mu, varDiff, covDiff, rowMeanVariance = task.measureMeanVarCov(
                im1Area, im2Area, imStatsCtrl, mu1, mu2
            )
        self.assertIn("Number of good points", cm.output[0])

        self.assertTrue(np.isnan(mu))
        self.assertTrue(np.isnan(varDiff))
        self.assertTrue(covDiff is None)
        self.assertTrue(np.isnan(rowMeanVariance))

    def test_meanVarMeasurementTooNarrowStrip(self):
        # We need a new config to make sure the second covariance cut is
        # triggered.
        config = cpPipe.ptc.PhotonTransferCurveExtractTask.ConfigClass()
        config.minNumberGoodPixelsForCovariance = 10
        task = cpPipe.ptc.PhotonTransferCurveExtractTask(config=config)
        flatExp1 = self.flatExp1.clone()
        flatExp2 = self.flatExp2.clone()

        flatExp1.image.array[0:195, :] = np.nan
        flatExp2.image.array[0:195, :] = np.nan
        flatExp1.image.array[:, 0:195] = np.nan
        flatExp2.image.array[:, 0:195] = np.nan

        bit = flatExp1.mask.getMaskPlaneDict()["NO_DATA"]
        flatExp1.mask.array[0:195, :] &= 2**bit
        flatExp2.mask.array[0:195, :] &= 2**bit
        flatExp1.mask.array[:, 0:195] &= 2**bit
        flatExp2.mask.array[:, 0:195] &= 2**bit

        im1Area, im2Area, imStatsCtrl, mu1, mu2 = task.getImageAreasMasksStats(
            flatExp1, flatExp2
        )
        with self.assertLogs(level=logging.WARNING) as cm:
            mu, varDiff, covDiff, rowMeanVariance = task.measureMeanVarCov(
                im1Area, im2Area, imStatsCtrl, mu1, mu2
            )
        self.assertIn("Not enough pixels", cm.output[0])

        self.assertTrue(np.isnan(mu))
        self.assertTrue(np.isnan(varDiff))
        self.assertTrue(covDiff is None)
        self.assertTrue(np.isnan(rowMeanVariance))

    def test_makeZeroSafe(self):
        noZerosArray = [1.0, 20, -35, 45578.98, 90.0, 897, 659.8]
        someZerosArray = [1.0, 20, 0, 0, 90, 879, 0]
        allZerosArray = [0.0, 0.0, 0, 0, 0.0, 0, 0]

        substituteValue = 1e-10

        expectedSomeZerosArray = [
            1.0,
            20,
            substituteValue,
            substituteValue,
            90,
            879,
            substituteValue,
        ]
        expectedAllZerosArray = np.repeat(substituteValue, len(allZerosArray))

        measuredSomeZerosArray = self.defaultTaskSolve._makeZeroSafe(
            someZerosArray, substituteValue=substituteValue
        )
        measuredAllZerosArray = self.defaultTaskSolve._makeZeroSafe(
            allZerosArray, substituteValue=substituteValue
        )
        measuredNoZerosArray = self.defaultTaskSolve._makeZeroSafe(
            noZerosArray, substituteValue=substituteValue
        )

        for exp, meas in zip(expectedSomeZerosArray, measuredSomeZerosArray):
            self.assertEqual(exp, meas)
        for exp, meas in zip(expectedAllZerosArray, measuredAllZerosArray):
            self.assertEqual(exp, meas)
        for exp, meas in zip(noZerosArray, measuredNoZerosArray):
            self.assertEqual(exp, meas)

    def runGetGainFromFlatPair(self, correctionType="NONE"):
        extractConfig = self.defaultConfigExtract
        extractConfig.gainCorrectionType = correctionType
        extractConfig.minNumberGoodPixelsForCovariance = 5000
        extractTask = cpPipe.ptc.PhotonTransferCurveExtractTask(config=extractConfig)

        expDict = {}
        expIds = []
        idCounter = 0
        inputGain = self.gain  # 1.5 e/ADU
        for expTime in self.timeVec:
            # Approximation works better at low flux, e.g., < 10000 ADU
            mockExp1, mockExp2 = makeMockFlats(
                expTime,
                gain=inputGain,
                readNoiseElectrons=np.sqrt(self.noiseSq),
                fluxElectrons=100,
                expId1=idCounter,
                expId2=idCounter + 1,
                ampNames=self.ampNames,
            )
            mockExpRef1 = PretendRef(mockExp1)
            mockExpRef2 = PretendRef(mockExp2)
            expDict[expTime] = ((mockExpRef1, idCounter), (mockExpRef2, idCounter + 1))
            expIds.append(idCounter)
            expIds.append(idCounter + 1)
            idCounter += 2

        # This will test the full extract task.
        resultsExtract = extractTask.run(
            inputExp=expDict,
            inputDims=expIds,
        )
        for exposurePair in resultsExtract.outputCovariances:
            for ampName in self.ampNames:
                if exposurePair.gain[ampName] is np.nan:
                    continue
                self.assertAlmostEqual(
                    exposurePair.gain[ampName], inputGain, delta=0.04
                )

        # Run through the solver and check that the gains from flat pairs
        # are recorded in the gainList.
        solveConfig = self.defaultConfigSolve
        solveConfig.ptcFitType = "EXPAPPROXIMATION"
        solveTask = cpPipe.ptc.PhotonTransferCurveSolveTask(config=solveConfig)

        resultsSolve = solveTask.run(
            resultsExtract.outputCovariances,
            camera=FakeCamera([self.flatExp1.getDetector()]),
        )
        ptc = resultsSolve.outputPtcDataset

        for i in range(len(ptc.inputExpIdPairs)):
            for ampName in self.ampNames:
                if np.isnan(ptc.gainList[ampName][i]):
                    continue
                self.assertAlmostEqual(
                    ptc.gainList[ampName][i], inputGain, delta=0.04,
                )
                self.assertAlmostEqual(
                    ptc.noiseList[ampName][i], np.sqrt(self.noiseSq) / self.gain,
                )

    def test_getGainFromFlatPair(self):
        for gainCorrectionType in [
            "NONE",
            "SIMPLE",
            "FULL",
        ]:
            self.runGetGainFromFlatPair(gainCorrectionType)

    def test_ptcFitBootstrap(self):
        """Test the bootstrap fit option for the PTC"""
        for fitType in ['EXPAPPROXIMATION']:
            self.ptcFitAndCheckPtc(fitType=fitType, doFitBootstrap=True)

    def test_ampOffsetGainRatioFixup(self):
        """Test the ampOffsetGainRatioFixup code via
        PhotonTransferCurveFixupGainRatiosTask."""
        # TODO DM-52883: Remove tests for deprecated task.
        rng = np.random.RandomState(12345)

        gainsTruth = rng.normal(loc=1.7, scale=0.05, size=len(self.ampNames))
        gainsMedian = np.median(gainsTruth)

        gainsMeasured = rng.normal(loc=gainsTruth, scale=0.02, size=len(self.ampNames))

        # Make one of the measured gains a much larger outlier to ensure
        # the code can handle a wacky measurement.
        weird_amp_index = 1
        gainsMeasured[weird_amp_index] = 1.2

        # We have a perfectly flat illuminated detector (in electrons)
        nFlat = 20
        meansElectron = gainsMedian * np.linspace(500.0, 30000.0, nFlat)

        # Set up the fixup task.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fixupConfig = cpPipe.ptc.PhotonTransferCurveFixupGainRatiosConfig()
            fixupConfig.ampOffsetGainRatioMinAdu = 1000.0
            fixupConfig.ampOffsetGainRatioMaxAdu = 20000.0
            fixupTask = cpPipe.ptc.PhotonTransferCurveFixupGainRatiosTask(config=fixupConfig)

        for testMode in ["full", "badamp"]:
            badAmp = None

            ptc = PhotonTransferCurveDataset(self.ampNames, ptcFitType="FULLCOVARIANCE")

            # Build the PTC to test amp offsets.
            for i, ampName in enumerate(self.ampNames):
                ptc.gain[ampName] = gainsMeasured[i]
                ptc.gainUnadjusted[ampName] = gainsMeasured[i]

                if testMode == "badamp" and i == 3:
                    ptc.gain[ampName] = np.nan
                    ptc.gainUnadjusted[ampName] = np.nan
                    badAmp = ampName

                # The measured amp means are given by the true gains.
                ptc.finalMeans[ampName] = meansElectron / gainsTruth[i]
                ptc.expIdMask[ampName] = np.ones(nFlat, dtype=np.bool_)

                # Fill the amp offsets temporarily.
                ptc.ampOffsets[ampName] = np.zeros(len(meansElectron))

            # Build each toy flat and measure the amp offsets.
            config = AmpOffsetTask.ConfigClass()
            config.ampEdgeMaxOffset = 100000.0
            config.ampEdgeWidth = 16
            config.ampEdgeInset = 10
            config.doBackground = False
            config.doDetection = False
            config.doApplyAmpOffset = False

            ampOffset = AmpOffsetTask(config=config)

            detector = self.flatExp1.getDetector()
            metadatas = []
            for i in range(nFlat):
                exp = self.flatExp1.clone()
                for amp in detector:
                    amp_name = amp.getName()
                    exp[amp.getBBox()].image.array[:, :] = ptc.finalMeans[amp_name][i]
                    md = exp.metadata

                    md[f"LSST ISR FINAL MEDIAN {amp_name}"] = ptc.finalMeans[amp_name][i]
                    md[f"LSST ISR FINAL STDEV {amp_name}"] = np.sqrt(ptc.finalMeans[amp_name][i])
                    md[f"LSST ISR READNOISE {amp_name}"] = 0.0

                ampOffset.run(exp)
                metadatas.append(exp.metadata)

            result = fixupTask.run(inputPtc=ptc, exposureMetadata=metadatas)
            ptc = result.outputPtc

            # Check that the flats are flat after adjustment.
            for i in range(nFlat):
                for j, amp in enumerate(detector):
                    gain = ptc.gain[amp.getName()]
                    exp[amp.getBBox()].image.array[:, :] = ptc.finalMeans[amp.getName()][i] * gain

                testImage = exp.image.array / np.nanmedian(exp.image.array)
                if testMode == "badamp":
                    # Force the bad values to 1.0
                    testImage[~np.isfinite(testImage)] = 1.0

                self.assertFloatsAlmostEqual(testImage.ravel(), 1.0, rtol=1e-6)

            # Confirm that the median gain is reasonably unchanged.
            gainsAdjusted = np.array([ptc.gain[ampName] for ampName in self.ampNames])
            self.assertFloatsAlmostEqual(np.nanmedian(gainsAdjusted), np.nanmedian(gainsMeasured), rtol=0.007)

            # Confirm that the median of the corrections is 1.0
            gainCorrections = np.array([ptc.gain[ampName]/ptc.gainUnadjusted[ampName]
                                        for ampName in self.ampNames])
            self.assertFloatsAlmostEqual(np.nanmedian(gainCorrections), 1.0, rtol=1e-7)

            # Check that the gain ratios are matched as expected.
            for i, ampName1 in enumerate(self.ampNames):
                for j, ampName2 in enumerate(self.ampNames):
                    ratioTruth = gainsTruth[i] / gainsTruth[j]
                    ratio = ptc.gain[ampName1] / ptc.gain[ampName2]

                    if ampName1 != badAmp and ampName2 != badAmp:
                        self.assertFloatsAlmostEqual(ratio, ratioTruth, rtol=1e-7)

        # Check that everything is logged correctly when things are very bad.
        for ampName in self.ampNames[2:]:
            ptc.ampOffsets[ampName][:] = np.nan

        with self.assertLogs(level=logging.WARNING) as cm:
            ampOffsetGainRatioFixup(ptc, 1000.0, 20000.0)
        self.assertIn("Not enough good amp offset measurements", cm.output[0])

        for ampName in self.ampNames[1:]:
            ptc.gain[ampName] = np.nan

        with self.assertLogs(level=logging.WARNING) as cm:
            ampOffsetGainRatioFixup(ptc, 1000.0, 20000.0)
        self.assertIn("Cannot apply ampOffsetGainRatioFixup", cm.output[0])

    def test_maskVignetteFunctionRegion(self):
        # The function coefficients are chosen to trigger on some amps.
        # The amps with no masked pixels are C:1,0 and C:1,3 due to the
        # curvature.
        extractConfig = self.defaultConfigExtract
        extractConfig.doVignetteFunctionRegionSelection = True
        extractConfig.vignetteFunctionPolynomialCoeffs = [-100.0, -18.0, 1.0]
        extractConfig.vignetteFunctionRegionSelectionMinimumPixels = 2_000

        task = cpPipe.ptc.PhotonTransferCurveExtractPairTask(config=extractConfig)
        exposures = [self.flatExp1, self.flatExp2]
        with self.assertNoLogs(level=logging.WARNING):
            task._maskVignetteFunctionRegion(exposures)

        bitmask = self.flatExp1.mask.getPlaneBitMask("SUSPECT")
        for amp in self.flatExp1.getDetector():
            bbox = amp.getBBox()
            nMasked = ((self.flatExp1.mask[bbox].array & bitmask) > 0).sum()
            if amp.getName() not in ["C:1,0", "C:1,3"]:
                self.assertGreater(nMasked, 0, msg=f"Missing masked pixels in {amp.getName()}")
            else:
                self.assertEqual(nMasked, 0, msg=f"Unexpected masked pixels in {amp.getName()}")

        # Check what happens if we ask for too many pixels.
        extractConfig = self.defaultConfigExtract
        extractConfig.doVignetteFunctionRegionSelection = True
        extractConfig.vignetteFunctionPolynomialCoeffs = [-100.0, -18.0, 1.0]
        extractConfig.vignetteFunctionRegionSelectionMinimumPixels = 10_000

        self.flatExp1.mask.array[:, :] = 0
        self.flatExp2.mask.array[:, :] = 0
        task = cpPipe.ptc.PhotonTransferCurveExtractPairTask(config=extractConfig)
        exposures = [self.flatExp1, self.flatExp2]
        with self.assertLogs(level=logging.WARNING):
            task._maskVignetteFunctionRegion(exposures)

        # This should not have any masked pixels because we essentially
        # insisted that the full amp be selected.
        for amp in self.flatExp1.getDetector():
            self.assertEqual(nMasked, 0, msg=f"Unexpected masked pixels in {amp.getName()}")

    def _getSampleMeanAndVar(self, dense=False, mode="normal"):
        """Get sample mean/var vectors and ptcTurnoff from LSSTCam data.

        The ptcTurnoff values here were obtained by looking at the data
        and what the code does. These may be changed in the future if
        we decide on a different way of computing the turnoff.

        Parameters
        ----------
        dense : `bool`, optional
            Return a dense PTC?
        mode : `str`, optional
            The ptc type, should be "normal", "upturn", or "dip".

        Returns
        -------
        rawMeans : `np.ndarray`
            Array of raw mean values.
        rawVars : `np.ndarray`
            Array of raw variance values.
        ptcTurnoff : `float`
            PTC turnoff determined from fitting by eye.
        """
        if dense and mode == "normal":
            # Taken from dense run 13591, detector 94, amplifier C02
            ptcTurnoff = 92239.4794
            rawMeans = np.array([
                3.72806883e+01, 3.86850679e+01, 4.09319941e+01, 4.30288886e+01,
                4.59546561e+01, 4.77510985e+01, 5.06130505e+01, 5.35060747e+01,
                5.61016577e+01, 5.95244630e+01, 6.21722799e+01, 6.59958508e+01,
                7.32513785e+01, 7.91695151e+01, 8.19214651e+01, 8.66175502e+01,
                9.06656207e+01, 9.57917501e+01, 1.00925700e+02, 1.07661026e+02,
                1.12211207e+02, 1.18038278e+02, 1.24776787e+02, 1.33127394e+02,
                1.39086972e+02, 1.47962442e+02, 1.55007859e+02, 1.64972119e+02,
                1.72224552e+02, 1.81663660e+02, 1.92083706e+02, 2.01818483e+02,
                2.12893098e+02, 2.25865898e+02, 2.36777769e+02, 2.50805751e+02,
                2.63582968e+02, 2.78848126e+02, 2.93430577e+02, 3.10974538e+02,
                3.28617662e+02, 3.44447122e+02, 3.63779087e+02, 3.85447934e+02,
                4.04865571e+02, 4.26619625e+02, 4.50642349e+02, 4.75572605e+02,
                5.00924590e+02, 5.29463072e+02, 5.56894775e+02, 5.89400667e+02,
                6.22194672e+02, 6.55411968e+02, 6.90233178e+02, 7.28617776e+02,
                7.70386654e+02, 8.11275732e+02, 8.56899058e+02, 9.04075885e+02,
                9.53041280e+02, 1.00428165e+03, 1.05984066e+03, 1.11936964e+03,
                1.17993499e+03, 1.24493657e+03, 1.31333429e+03, 1.38477571e+03,
                1.46288220e+03, 1.54131684e+03, 1.62718454e+03, 1.71685496e+03,
                1.91104434e+03, 1.91128408e+03, 2.01496942e+03, 2.01510504e+03,
                2.12643136e+03, 2.12665024e+03, 2.24351981e+03, 2.36534141e+03,
                2.49658670e+03, 2.77859948e+03, 2.92932915e+03, 3.21897902e+03,
                3.39732732e+03, 3.58413491e+03, 3.78057120e+03, 3.98731229e+03,
                4.20691253e+03, 4.43947873e+03, 4.68353906e+03, 4.94068272e+03,
                5.20957520e+03, 5.21201592e+03, 5.49832249e+03, 5.80143284e+03,
                6.12091976e+03, 6.81333865e+03, 7.18560833e+03, 7.76381518e+03,
                8.33998116e+03, 8.91777512e+03, 9.49514657e+03, 1.00697547e+04,
                1.06495655e+04, 1.12197718e+04, 1.18024551e+04, 1.23803888e+04,
                1.29546730e+04, 1.35302383e+04, 1.41095292e+04, 1.46864242e+04,
                1.52554127e+04, 1.58424382e+04, 1.64174609e+04, 1.69934436e+04,
                1.75654540e+04, 1.81523967e+04, 1.87298339e+04, 1.93014337e+04,
                2.04450305e+04, 2.10377447e+04, 2.16102067e+04, 2.16123196e+04,
                2.21851791e+04, 2.21921696e+04, 2.33433082e+04, 2.39171545e+04,
                2.44958327e+04, 2.50717007e+04, 2.56486425e+04, 2.62185599e+04,
                2.68048566e+04, 2.73814219e+04, 2.79519282e+04, 2.85325879e+04,
                2.91077169e+04, 2.96844973e+04, 3.02554411e+04, 3.14125545e+04,
                3.25683638e+04, 3.31324135e+04, 3.33243063e+04, 3.38868072e+04,
                3.44574798e+04, 3.50295131e+04, 3.55959947e+04, 3.61681182e+04,
                3.67288051e+04, 3.73019527e+04, 3.78659087e+04, 3.84175071e+04,
                3.90035052e+04, 3.95715617e+04, 4.06979100e+04, 4.12778387e+04,
                4.18444142e+04, 4.24067921e+04, 4.29743477e+04, 4.35479754e+04,
                4.41086362e+04, 4.46748906e+04, 4.57912581e+04, 4.63620286e+04,
                4.69293084e+04, 4.80512660e+04, 4.86149041e+04, 4.91798146e+04,
                5.03083769e+04, 5.08678825e+04, 5.14329108e+04, 5.20101245e+04,
                5.25674618e+04, 5.31040686e+04, 5.36908774e+04, 5.42697605e+04,
                5.47843148e+04, 5.53743324e+04, 5.59452125e+04, 5.70753786e+04,
                5.76356662e+04, 5.81990397e+04, 5.87644952e+04, 5.98894309e+04,
                6.04630169e+04, 6.10270338e+04, 6.15960865e+04, 6.21427559e+04,
                6.27197146e+04, 6.32781202e+04, 6.38170695e+04, 6.44264810e+04,
                6.49720024e+04, 6.55524996e+04, 6.66814853e+04, 6.72313467e+04,
                6.78274362e+04, 6.83859289e+04, 6.89427312e+04, 6.95198678e+04,
                7.00847030e+04, 7.06620283e+04, 7.12327432e+04, 7.17769841e+04,
                7.23610736e+04, 7.29237272e+04, 7.34938353e+04, 7.40400864e+04,
                7.46399787e+04, 7.52085787e+04, 7.57175033e+04, 7.63346271e+04,
                7.69141477e+04, 7.74690838e+04, 7.80403707e+04, 7.85672045e+04,
                7.91728742e+04, 7.97519543e+04, 8.03063026e+04, 8.08869068e+04,
                8.14577271e+04, 8.20105730e+04, 8.25824739e+04, 8.31490218e+04,
                8.37308110e+04, 8.42758180e+04, 8.48493221e+04, 8.54126656e+04,
                8.59826959e+04, 8.65501659e+04, 8.70696403e+04, 8.76932979e+04,
                8.82464074e+04, 8.93795020e+04, 8.99537321e+04, 9.05097100e+04,
                9.10881246e+04, 9.22394794e+04, 9.27870984e+04, 9.33315296e+04,
                9.39230922e+04, 9.44926266e+04, 9.56192429e+04, 9.61483057e+04,
                9.72621754e+04, 9.78325850e+04, 9.83972793e+04, 9.89552969e+04,
                9.94332509e+04, 1.00047506e+05, 1.00569417e+05, 1.01105830e+05,
                1.01634936e+05, 1.02121705e+05, 1.02641415e+05, 1.03123767e+05,
                1.03599733e+05, 1.04062370e+05, 1.04907216e+05, 1.05256436e+05,
                1.05665701e+05, 1.06007083e+05, 1.06318115e+05, 1.06593953e+05,
                1.06893542e+05, 1.07121150e+05, 1.07346243e+05, 1.07625934e+05,
                1.08044429e+05, 1.08167080e+05, 1.08308995e+05, 1.08399830e+05,
                1.08526582e+05, 1.08620505e+05, 1.08713803e+05, 1.08784455e+05,
                1.08862350e+05, 1.08937801e+05, 1.08989964e+05, 1.09060085e+05,
                1.09162026e+05, 1.09220593e+05, 1.09273416e+05, 1.09333464e+05,
                1.09372863e+05, 1.09468530e+05, 1.09513735e+05, 1.09553322e+05,
                1.09594566e+05, 1.09641845e+05, 1.09679236e+05, 1.09723860e+05,
                1.09753056e+05, 1.09826579e+05, 1.09872135e+05, 1.09895670e+05,
                1.09902080e+05, 1.09943174e+05, 1.10002465e+05, 1.10019329e+05,
                1.10068730e+05, 1.10095848e+05, 1.10145789e+05, 1.10193658e+05,
                1.10213581e+05, 1.10248075e+05, 1.10291320e+05, 1.10316883e+05,
                1.10358975e+05, 1.10350689e+05, 1.10391552e+05, 1.10400507e+05,
                1.10409523e+05, 1.10409459e+05, 1.10412119e+05, 1.10409689e+05,
                1.10414648e+05, 1.10391061e+05, 1.10420513e+05, 1.10424437e+05,
                1.10435588e+05, 1.10438362e+05, 1.10443905e+05, 1.10450180e+05,
                1.10452714e+05, 1.10449100e+05, 1.10466222e+05, 1.10458324e+05,
                1.10478871e+05, 1.10486620e+05, 1.10485105e+05, 1.10489515e+05,
                1.10491243e+05, 1.10499598e+05, 1.10501449e+05, 1.10501009e+05,
                1.10515083e+05, 1.10533044e+05, 1.10531740e+05, 1.10548636e+05,
                1.10558815e+05, 1.10560570e+05, 1.10569349e+05, 1.10572778e+05,
                1.10588800e+05, 1.10593287e+05, 1.10596037e+05, 1.10584233e+05,
                1.10611084e+05, 1.10616525e+05, 1.10621289e+05, 1.10631455e+05,
                1.10639320e+05, 1.10647906e+05, 1.10646043e+05, 1.10635789e+05,
                1.10635305e+05, 1.10652093e+05, 1.10638755e+05, 1.10655426e+05,
                1.10651182e+05, 1.10653604e+05, 1.10665648e+05, 1.10655049e+05,
                1.10650105e+05, 1.10655658e+05, 1.10660937e+05, 1.10673337e+05,
                1.10673003e+05, 1.10655815e+05, 1.10661151e+05, 1.10662036e+05,
                1.10676951e+05, 1.10667334e+05, 1.10679642e+05, 1.10678497e+05])
            rawVars = np.array([
                4.47787277e+01, 4.64353847e+01, 4.85247580e+01, 4.93948408e+01,
                5.15948411e+01, 5.30389406e+01, 5.58813000e+01, 5.88525810e+01,
                6.11585484e+01, 6.41104167e+01, 6.56152371e+01, 6.82093774e+01,
                7.14063291e+01, 7.45066055e+01, 7.64755940e+01, 7.93540481e+01,
                8.18907727e+01, 8.66402959e+01, 8.97887750e+01, 9.68782459e+01,
                9.95221383e+01, 1.02767714e+02, 1.08068716e+02, 1.16108758e+02,
                1.20246097e+02, 1.26018524e+02, 1.30397686e+02, 1.36138570e+02,
                1.41942986e+02, 1.46772865e+02, 1.53919845e+02, 1.60521709e+02,
                1.68353264e+02, 1.78909843e+02, 1.88788533e+02, 2.00792706e+02,
                2.07611019e+02, 2.16057203e+02, 2.26987335e+02, 2.39701806e+02,
                2.51287870e+02, 2.64963149e+02, 2.81918185e+02, 2.93010615e+02,
                3.04772726e+02, 3.21723049e+02, 3.43194608e+02, 3.60791523e+02,
                3.72312673e+02, 3.92534946e+02, 4.18291901e+02, 4.40825475e+02,
                4.58818537e+02, 4.88340078e+02, 5.12852998e+02, 5.35522179e+02,
                5.69935148e+02, 5.92975103e+02, 6.31052929e+02, 6.61056947e+02,
                6.95541286e+02, 7.33787875e+02, 7.74291674e+02, 8.19211863e+02,
                8.62765375e+02, 8.98546352e+02, 9.55344080e+02, 1.00237170e+03,
                1.05926161e+03, 1.10844186e+03, 1.17604012e+03, 1.24027865e+03,
                1.37334609e+03, 1.37631570e+03, 1.44433952e+03, 1.44613237e+03,
                1.53642103e+03, 1.53144123e+03, 1.60103165e+03, 1.69486912e+03,
                1.79228744e+03, 1.97807118e+03, 2.08913779e+03, 2.29127569e+03,
                2.41035376e+03, 2.54559185e+03, 2.68285671e+03, 2.82061302e+03,
                2.98565936e+03, 3.13680879e+03, 3.29910967e+03, 3.48157975e+03,
                3.66147828e+03, 3.67694292e+03, 3.84787540e+03, 4.07925681e+03,
                4.28699288e+03, 4.75578456e+03, 5.00956756e+03, 5.39321737e+03,
                5.79686635e+03, 6.17477624e+03, 6.55132737e+03, 6.93000705e+03,
                7.31757235e+03, 7.68650298e+03, 8.05377657e+03, 8.42789583e+03,
                8.77475510e+03, 9.17341164e+03, 9.54927300e+03, 9.89596158e+03,
                1.02637150e+04, 1.06167914e+04, 1.09873368e+04, 1.13685550e+04,
                1.17026479e+04, 1.20765605e+04, 1.23946750e+04, 1.27577017e+04,
                1.34485509e+04, 1.37927193e+04, 1.42006355e+04, 1.41857628e+04,
                1.44769908e+04, 1.45162172e+04, 1.51614067e+04, 1.55571196e+04,
                1.58809609e+04, 1.61752079e+04, 1.65104318e+04, 1.68369000e+04,
                1.71408213e+04, 1.75355600e+04, 1.78226398e+04, 1.81932474e+04,
                1.84482292e+04, 1.88389901e+04, 1.91150651e+04, 1.97375210e+04,
                2.03591057e+04, 2.07155451e+04, 2.07138350e+04, 2.11073361e+04,
                2.13837002e+04, 2.16588926e+04, 2.19612146e+04, 2.22753789e+04,
                2.25303517e+04, 2.27928117e+04, 2.31125410e+04, 2.33679689e+04,
                2.37393625e+04, 2.40488975e+04, 2.45764011e+04, 2.48064050e+04,
                2.50721617e+04, 2.53965873e+04, 2.56131331e+04, 2.59277943e+04,
                2.61709392e+04, 2.64119896e+04, 2.68955682e+04, 2.72267063e+04,
                2.74887295e+04, 2.79083033e+04, 2.82310485e+04, 2.84366770e+04,
                2.90226269e+04, 2.92324828e+04, 2.94521967e+04, 2.97949458e+04,
                2.99735234e+04, 3.02749574e+04, 3.05427850e+04, 3.07518546e+04,
                3.09317879e+04, 3.12692332e+04, 3.15643550e+04, 3.21293639e+04,
                3.24007437e+04, 3.26147619e+04, 3.28979579e+04, 3.34807881e+04,
                3.37109710e+04, 3.40246283e+04, 3.43898469e+04, 3.45302267e+04,
                3.47818944e+04, 3.51593681e+04, 3.52477656e+04, 3.56490287e+04,
                3.59240161e+04, 3.61669226e+04, 3.66423677e+04, 3.68827117e+04,
                3.71622239e+04, 3.74768274e+04, 3.76872328e+04, 3.79405515e+04,
                3.81364527e+04, 3.84119822e+04, 3.86423009e+04, 3.88877574e+04,
                3.91248488e+04, 3.92990610e+04, 3.96215459e+04, 3.98297091e+04,
                4.00712366e+04, 4.04322148e+04, 4.05178037e+04, 4.07431736e+04,
                4.10193223e+04, 4.11740695e+04, 4.13790755e+04, 4.14632263e+04,
                4.18525778e+04, 4.20979322e+04, 4.23130642e+04, 4.24582022e+04,
                4.24946930e+04, 4.27860167e+04, 4.30933820e+04, 4.32894483e+04,
                4.33858206e+04, 4.34411628e+04, 4.36629531e+04, 4.39919091e+04,
                4.40293360e+04, 4.42002784e+04, 4.42310862e+04, 4.45443328e+04,
                4.45481958e+04, 4.48604145e+04, 4.48891948e+04, 4.48416001e+04,
                4.50553039e+04, 4.50927543e+04, 4.51136486e+04, 4.51126520e+04,
                4.49857031e+04, 4.48705631e+04, 4.46276191e+04, 4.43327388e+04,
                4.37938194e+04, 4.32211550e+04, 4.27332194e+04, 4.23177306e+04,
                4.14539982e+04, 4.06974694e+04, 3.96551098e+04, 3.82440396e+04,
                3.67165456e+04, 3.47661502e+04, 3.28393493e+04, 3.02395522e+04,
                2.76939927e+04, 2.48272541e+04, 1.92526286e+04, 1.63986811e+04,
                1.42095886e+04, 1.22350406e+04, 1.06293553e+04, 9.24838723e+03,
                8.09429377e+03, 7.27582028e+03, 6.47441127e+03, 5.21985266e+03,
                4.36279092e+03, 4.00953782e+03, 3.69585190e+03, 3.43405611e+03,
                3.18567804e+03, 3.01195379e+03, 2.89123841e+03, 2.68282359e+03,
                2.60091022e+03, 2.51301062e+03, 2.43890788e+03, 2.35838231e+03,
                2.26256855e+03, 2.21204078e+03, 2.20282902e+03, 2.18205860e+03,
                2.15007675e+03, 2.08949483e+03, 2.10732780e+03, 2.09600072e+03,
                2.07407783e+03, 2.11741997e+03, 2.08083045e+03, 2.06397688e+03,
                2.04765529e+03, 2.05836012e+03, 2.04093663e+03, 1.98334647e+03,
                2.01999612e+03, 1.99394914e+03, 1.96972450e+03, 1.88216817e+03,
                1.86902307e+03, 1.98290800e+03, 1.94714119e+03, 1.92046621e+03,
                1.83796316e+03, 1.87468459e+03, 1.89491779e+03, 1.83110571e+03,
                1.75025929e+03, 1.66067524e+03, 1.68052010e+03, 1.67404206e+03,
                1.65221787e+03, 1.63380745e+03, 1.61258211e+03, 1.57173282e+03,
                1.55635385e+03, 1.59914440e+03, 1.63280915e+03, 1.62416209e+03,
                1.55550327e+03, 1.59790786e+03, 1.63913552e+03, 1.62123248e+03,
                1.59571147e+03, 1.48508783e+03, 1.60763833e+03, 1.60573267e+03,
                1.66286509e+03, 1.56023986e+03, 1.60226461e+03, 1.62178972e+03,
                1.63248481e+03, 1.63982132e+03, 1.67350316e+03, 1.62120745e+03,
                1.69261497e+03, 1.67835970e+03, 1.71259617e+03, 1.66567572e+03,
                1.73107044e+03, 1.60808418e+03, 1.62957233e+03, 1.65949976e+03,
                1.62773379e+03, 1.74438786e+03, 1.61948665e+03, 1.83335873e+03,
                1.65774501e+03, 1.65605436e+03, 1.72054542e+03, 1.68968657e+03,
                1.73787963e+03, 1.74437326e+03, 1.71408015e+03, 1.93871967e+03,
                1.87508731e+03, 1.70631906e+03, 1.92219749e+03, 1.72410949e+03,
                1.86172638e+03, 1.88692937e+03, 1.77337717e+03, 1.95303579e+03,
                1.86292107e+03, 1.87948406e+03, 1.91966097e+03, 1.76238101e+03,
                1.59694869e+03, 1.89306746e+03, 1.84068186e+03, 1.86018512e+03,
                1.81578862e+03, 1.83413719e+03, 1.80771751e+03, 1.79214601e+03])
        elif dense and mode == "upturn":
            # Taken from dense run 13591, detector 73, amplifier C07
            ptcTurnoff = 86917.0045
            rawMeans = np.array([
                2.87448603e+01, 3.03861239e+01, 3.20301466e+01, 3.35831624e+01,
                3.51161960e+01, 3.69857753e+01, 3.91580507e+01, 4.12179225e+01,
                4.36662086e+01, 4.58399375e+01, 4.80202775e+01, 5.10071275e+01,
                5.68263968e+01, 5.99352524e+01, 6.34210182e+01, 6.68834309e+01,
                7.01195882e+01, 7.43313750e+01, 7.80758357e+01, 8.27309929e+01,
                8.73124933e+01, 9.17856382e+01, 9.70086101e+01, 1.02546128e+02,
                1.08169091e+02, 1.13738150e+02, 1.20058621e+02, 1.26792703e+02,
                1.33422104e+02, 1.40893068e+02, 1.48727555e+02, 1.56662520e+02,
                1.65129261e+02, 1.74130948e+02, 1.83878223e+02, 1.93976598e+02,
                2.04599993e+02, 2.16044601e+02, 2.27838642e+02, 2.40360862e+02,
                2.53772624e+02, 2.67443055e+02, 2.82013487e+02, 2.97705423e+02,
                3.14297001e+02, 3.31260825e+02, 3.49758342e+02, 3.68963998e+02,
                3.88634613e+02, 4.10112249e+02, 4.32301833e+02, 4.56407219e+02,
                4.81690100e+02, 5.08370293e+02, 5.35921852e+02, 5.65828928e+02,
                5.96685398e+02, 6.29615188e+02, 6.64802708e+02, 7.00563142e+02,
                7.39711995e+02, 7.79679559e+02, 8.22530629e+02, 8.67814522e+02,
                9.15951880e+02, 9.65946953e+02, 1.01948611e+03, 1.07504268e+03,
                1.13509200e+03, 1.19727942e+03, 1.26351949e+03, 1.33211624e+03,
                1.48320209e+03, 1.48384430e+03, 1.56467822e+03, 1.56485206e+03,
                1.65060918e+03, 1.65147654e+03, 1.74229201e+03, 1.83706229e+03,
                1.93811946e+03, 2.15719262e+03, 2.27546155e+03, 2.50056748e+03,
                2.63872700e+03, 2.78379012e+03, 2.93745238e+03, 3.09833186e+03,
                3.27013288e+03, 3.44990418e+03, 3.63968250e+03, 3.83921968e+03,
                4.04711266e+03, 4.05013387e+03, 4.27266981e+03, 4.50809670e+03,
                4.75635569e+03, 5.29310771e+03, 5.58427306e+03, 6.03306708e+03,
                6.48160541e+03, 6.92983470e+03, 7.37874615e+03, 7.82619855e+03,
                8.27493037e+03, 8.72049208e+03, 9.17227788e+03, 9.62234260e+03,
                1.00674503e+04, 1.05166951e+04, 1.09652006e+04, 1.14145457e+04,
                1.18581337e+04, 1.23138210e+04, 1.27597933e+04, 1.32094995e+04,
                1.36537794e+04, 1.41079990e+04, 1.45579666e+04, 1.50042564e+04,
                1.58932106e+04, 1.63527856e+04, 1.67995854e+04, 1.68014549e+04,
                1.72456751e+04, 1.72511595e+04, 1.81472850e+04, 1.85967925e+04,
                1.90437339e+04, 1.94933795e+04, 1.99423590e+04, 2.03854630e+04,
                2.08408610e+04, 2.12909264e+04, 2.17356857e+04, 2.21873924e+04,
                2.26346682e+04, 2.30822802e+04, 2.35248689e+04, 2.44292445e+04,
                2.53306721e+04, 2.57709779e+04, 2.59292740e+04, 2.63672474e+04,
                2.68118979e+04, 2.72586202e+04, 2.76977437e+04, 2.81455545e+04,
                2.85840092e+04, 2.90309073e+04, 2.94738200e+04, 2.99007273e+04,
                3.03581316e+04, 3.08035397e+04, 3.16820715e+04, 3.21348581e+04,
                3.25795035e+04, 3.30160938e+04, 3.34614077e+04, 3.39098464e+04,
                3.43488841e+04, 3.47938018e+04, 3.56683528e+04, 3.61120617e+04,
                3.65571740e+04, 3.74373798e+04, 3.78795706e+04, 3.83213509e+04,
                3.92106205e+04, 3.96436191e+04, 4.00888939e+04, 4.05395713e+04,
                4.09729696e+04, 4.13953138e+04, 4.18558141e+04, 4.23098575e+04,
                4.27184098e+04, 4.31754813e+04, 4.36191519e+04, 4.45068672e+04,
                4.49463419e+04, 4.53858299e+04, 4.58261673e+04, 4.67014085e+04,
                4.71515103e+04, 4.75882657e+04, 4.80322253e+04, 4.84586953e+04,
                4.89039488e+04, 4.93442503e+04, 4.97561230e+04, 5.02341725e+04,
                5.06588581e+04, 5.11035621e+04, 5.19836216e+04, 5.24056394e+04,
                5.28686518e+04, 5.33018757e+04, 5.37265598e+04, 5.41807663e+04,
                5.46136176e+04, 5.50609130e+04, 5.55007300e+04, 5.59337737e+04,
                5.63788843e+04, 5.68094757e+04, 5.72570533e+04, 5.76741866e+04,
                5.81383020e+04, 5.85826153e+04, 5.89753856e+04, 5.94617576e+04,
                5.99083532e+04, 6.03419978e+04, 6.07815876e+04, 6.11916990e+04,
                6.16679074e+04, 6.21070006e+04, 6.25498996e+04, 6.30000329e+04,
                6.34432415e+04, 6.38632967e+04, 6.43145751e+04, 6.47608638e+04,
                6.52134266e+04, 6.56348895e+04, 6.60841662e+04, 6.65220989e+04,
                6.69687269e+04, 6.74171257e+04, 6.78295038e+04, 6.83037417e+04,
                6.87381890e+04, 6.96234966e+04, 7.00749883e+04, 7.05130803e+04,
                7.09702156e+04, 7.18669367e+04, 7.23008238e+04, 7.27326728e+04,
                7.31879639e+04, 7.36444652e+04, 7.45313822e+04, 7.49512933e+04,
                7.58422569e+04, 7.63002282e+04, 7.67542279e+04, 7.71971910e+04,
                7.75899806e+04, 7.80880043e+04, 7.85207688e+04, 7.89717058e+04,
                7.94257076e+04, 7.98485141e+04, 8.03023282e+04, 8.07414358e+04,
                8.11910981e+04, 8.16398955e+04, 8.25250712e+04, 8.29401845e+04,
                8.34236581e+04, 8.38641888e+04, 8.42971844e+04, 8.47295006e+04,
                8.52114910e+04, 8.56328503e+04, 8.60862279e+04, 8.69170045e+04,
                8.78165034e+04, 8.82689547e+04, 8.87058177e+04, 8.90804208e+04,
                8.95362169e+04, 8.99496410e+04, 9.03236457e+04, 9.06958198e+04,
                9.11193888e+04, 9.14557890e+04, 9.17189073e+04, 9.21437093e+04,
                9.28449356e+04, 9.31434208e+04, 9.33064667e+04, 9.34994900e+04,
                9.37926812e+04, 9.41725477e+04, 9.45383949e+04, 9.46575696e+04,
                9.49209096e+04, 9.49952393e+04, 9.50539106e+04, 9.52347786e+04,
                9.52460768e+04, 9.54989344e+04, 9.56298553e+04, 9.58490767e+04,
                9.62648021e+04, 9.61075892e+04, 9.63538129e+04, 9.63559508e+04,
                9.64708846e+04, 9.64756350e+04, 9.66256538e+04, 9.64809810e+04,
                9.65366387e+04, 9.68214859e+04, 9.69137722e+04, 9.68210311e+04,
                9.71873201e+04, 9.71342102e+04, 9.71078451e+04, 9.71069892e+04,
                9.75296912e+04, 9.74999383e+04, 9.73480623e+04, 9.76524037e+04,
                9.74981803e+04, 9.76922174e+04, 9.76337785e+04, 9.75835125e+04,
                9.75674694e+04, 9.75367570e+04, 9.77438281e+04, 9.76871410e+04,
                9.78773702e+04, 9.78321613e+04, 9.77808877e+04, 9.77359422e+04,
                9.79278808e+04, 9.79329075e+04, 9.77675254e+04, 9.78952318e+04,
                9.77270368e+04, 9.79509533e+04, 9.77760093e+04, 9.80460286e+04,
                9.81360499e+04, 9.79637125e+04, 9.79664967e+04, 9.78575509e+04,
                9.80556911e+04, 9.81417648e+04, 9.80352945e+04, 9.81329700e+04,
                9.81188437e+04, 9.79678779e+04, 9.79598626e+04, 9.80560275e+04,
                9.79542087e+04, 9.79987939e+04, 9.81534114e+04, 9.83899105e+04,
                9.84065000e+04, 9.82811922e+04, 9.82840840e+04, 9.82784134e+04,
                9.82871998e+04, 9.84461381e+04, 9.84682635e+04, 9.85020571e+04,
                np.nan, np.nan, np.nan, np.nan,
                np.nan, np.nan, np.nan, np.nan,
                np.nan, np.nan, np.nan, np.nan,
                np.nan, np.nan, np.nan, np.nan])
            rawVars = np.array([
                4.87069525e+01, 4.96960311e+01, 5.04910053e+01, 5.17128726e+01,
                5.25955479e+01, 5.38353690e+01, 5.53191132e+01, 5.66941478e+01,
                5.82926522e+01, 5.98938552e+01, 6.12951161e+01, 6.37287578e+01,
                6.80636332e+01, 7.04028909e+01, 7.29998462e+01, 7.48586667e+01,
                7.66622012e+01, 7.82391455e+01, 7.99899592e+01, 8.21891213e+01,
                8.44852817e+01, 8.74729321e+01, 9.05085253e+01, 9.45153523e+01,
                9.84325033e+01, 1.02490363e+02, 1.05547104e+02, 1.08782968e+02,
                1.12593458e+02, 1.17458820e+02, 1.22086446e+02, 1.27863525e+02,
                1.34404768e+02, 1.42189276e+02, 1.48361964e+02, 1.52241417e+02,
                1.56903695e+02, 1.63487388e+02, 1.71250175e+02, 1.79776726e+02,
                1.87500190e+02, 1.95127335e+02, 2.05237738e+02, 2.17941739e+02,
                2.27601893e+02, 2.34445474e+02, 2.46136553e+02, 2.58638437e+02,
                2.70612240e+02, 2.88057360e+02, 3.01436607e+02, 3.11174829e+02,
                3.27060261e+02, 3.47777215e+02, 3.68419809e+02, 3.80518820e+02,
                3.97377177e+02, 4.25381795e+02, 4.42049448e+02, 4.59452324e+02,
                4.92530031e+02, 5.14495733e+02, 5.34196108e+02, 5.72228281e+02,
                5.92914611e+02, 6.29229419e+02, 6.63254402e+02, 7.10047069e+02,
                7.31110299e+02, 7.74635732e+02, 8.06624321e+02, 8.55180585e+02,
                9.44435441e+02, 9.43183163e+02, 1.00413513e+03, 9.99872981e+02,
                1.05311443e+03, 1.05562621e+03, 1.10199374e+03, 1.15531802e+03,
                1.22522414e+03, 1.36520090e+03, 1.42667128e+03, 1.56495960e+03,
                1.65397576e+03, 1.74424087e+03, 1.84231293e+03, 1.94740328e+03,
                2.03765806e+03, 2.14414450e+03, 2.27212692e+03, 2.38935653e+03,
                2.50602833e+03, 2.51661588e+03, 2.64775661e+03, 2.78538088e+03,
                2.93987651e+03, 3.26826498e+03, 3.44330599e+03, 3.69978469e+03,
                3.97061173e+03, 4.24728849e+03, 4.50971380e+03, 4.77595875e+03,
                5.05755732e+03, 5.30327839e+03, 5.58773282e+03, 5.81966058e+03,
                6.10932503e+03, 6.36332817e+03, 6.61863136e+03, 6.90433479e+03,
                7.13413117e+03, 7.43562156e+03, 7.66582807e+03, 7.94754672e+03,
                8.18686206e+03, 8.45779399e+03, 8.71402609e+03, 8.95964156e+03,
                9.47591061e+03, 9.76959463e+03, 9.95917651e+03, 9.97584309e+03,
                1.02555717e+04, 1.02417814e+04, 1.07674493e+04, 1.09981921e+04,
                1.12352859e+04, 1.15551940e+04, 1.17333125e+04, 1.19690736e+04,
                1.22534101e+04, 1.24640828e+04, 1.27500496e+04, 1.29625045e+04,
                1.32703257e+04, 1.34926746e+04, 1.37251524e+04, 1.42302648e+04,
                1.46793658e+04, 1.49722647e+04, 1.49709034e+04, 1.52549107e+04,
                1.55088982e+04, 1.57314736e+04, 1.59816003e+04, 1.61640169e+04,
                1.64548981e+04, 1.66413940e+04, 1.68686050e+04, 1.71125336e+04,
                1.73511674e+04, 1.75817535e+04, 1.80470011e+04, 1.82624491e+04,
                1.84524739e+04, 1.87019552e+04, 1.89575762e+04, 1.91930866e+04,
                1.94004074e+04, 1.96460005e+04, 2.00479314e+04, 2.02955099e+04,
                2.04822881e+04, 2.09531956e+04, 2.11697303e+04, 2.13633602e+04,
                2.18346507e+04, 2.19539919e+04, 2.22252821e+04, 2.25155821e+04,
                2.26702123e+04, 2.28414558e+04, 2.30830649e+04, 2.32726468e+04,
                2.34988786e+04, 2.37395021e+04, 2.39101878e+04, 2.43317884e+04,
                2.45771060e+04, 2.47054420e+04, 2.49622825e+04, 2.52797662e+04,
                2.55292494e+04, 2.57138336e+04, 2.59366428e+04, 2.61122036e+04,
                2.63896695e+04, 2.64562386e+04, 2.66907602e+04, 2.69102279e+04,
                2.71833821e+04, 2.73303887e+04, 2.77661838e+04, 2.79006930e+04,
                2.81874904e+04, 2.83607526e+04, 2.86045106e+04, 2.87425416e+04,
                2.89632711e+04, 2.91293885e+04, 2.93514145e+04, 2.96850582e+04,
                2.98715928e+04, 3.01274873e+04, 3.03033141e+04, 3.04572121e+04,
                3.07470469e+04, 3.10501508e+04, 3.11450218e+04, 3.14122871e+04,
                3.16022955e+04, 3.18325755e+04, 3.21339141e+04, 3.23047828e+04,
                3.24219992e+04, 3.28022196e+04, 3.30193533e+04, 3.32907509e+04,
                3.34017470e+04, 3.36009585e+04, 3.39733107e+04, 3.40535547e+04,
                3.42458795e+04, 3.44664410e+04, 3.47643491e+04, 3.50165080e+04,
                3.51727941e+04, 3.53854796e+04, 3.55711407e+04, 3.58190707e+04,
                3.60454029e+04, 3.63969548e+04, 3.66329292e+04, 3.68771986e+04,
                3.71032547e+04, 3.73983989e+04, 3.76425541e+04, 3.79247481e+04,
                3.80882104e+04, 3.82317170e+04, 3.86471187e+04, 3.89086176e+04,
                3.92339037e+04, 3.95436246e+04, 3.96460919e+04, 3.98028414e+04,
                3.99125771e+04, 4.01715067e+04, 4.03722901e+04, 4.05626968e+04,
                4.06933838e+04, 4.10184964e+04, 4.11922744e+04, 4.13557454e+04,
                4.16157584e+04, 4.16356931e+04, 4.21157583e+04, 4.21816345e+04,
                4.24543184e+04, 4.26313857e+04, 4.27797954e+04, 4.29586701e+04,
                4.31317246e+04, 4.31196229e+04, 4.30508028e+04, 4.29439940e+04,
                4.14814510e+04, 4.17104998e+04, 4.12950588e+04, 4.00762620e+04,
                3.95745952e+04, 3.79603760e+04, 3.68764412e+04, 3.59324071e+04,
                3.41774826e+04, 3.35146378e+04, 3.01416949e+04, 3.10962443e+04,
                2.93723225e+04, 2.88267934e+04, 2.93057515e+04, 2.78885387e+04,
                2.97481790e+04, 2.89003096e+04, 2.98475325e+04, 3.15227067e+04,
                3.15157345e+04, 3.32593077e+04, 3.30733852e+04, 3.37623662e+04,
                3.58784503e+04, 3.75160267e+04, 3.82803313e+04, 3.89001912e+04,
                4.17283047e+04, 4.20086699e+04, 4.22993774e+04, 4.34867653e+04,
                4.17054368e+04, 4.52190044e+04, 4.55586401e+04, 4.39034690e+04,
                4.49939183e+04, 4.75700720e+04, 4.78978238e+04, 4.62120749e+04,
                4.85317658e+04, 4.97305999e+04, 4.70691830e+04, 4.77667215e+04,
                5.19257836e+04, 5.01885056e+04, 4.83671721e+04, 5.21924342e+04,
                5.10329762e+04, 5.25552111e+04, 5.17040972e+04, 5.11958488e+04,
                5.19165668e+04, 4.88842210e+04, 5.17286013e+04, 5.13182361e+04,
                5.16342048e+04, 5.17850241e+04, 5.16475818e+04, 5.00165868e+04,
                5.19391672e+04, 5.21771011e+04, 5.01633917e+04, 5.19270231e+04,
                4.93839429e+04, 4.96516804e+04, 4.93706114e+04, 5.25025570e+04,
                5.43045043e+04, 5.27477605e+04, 5.23609667e+04, 4.89296194e+04,
                5.26232764e+04, 5.24395882e+04, 4.90790820e+04, 5.19268918e+04,
                5.21608822e+04, 4.89292620e+04, 4.86074588e+04, 4.93274381e+04,
                4.84901815e+04, 4.91868029e+04, 5.25082453e+04, 5.44279458e+04,
                5.55797293e+04, 5.20003094e+04, 5.19304424e+04, 5.17391458e+04,
                5.24042964e+04, 5.53566164e+04, 5.53202737e+04, 5.58931187e+04,
                np.nan, np.nan, np.nan, np.nan,
                np.nan, np.nan, np.nan, np.nan,
                np.nan, np.nan, np.nan, np.nan,
                np.nan, np.nan, np.nan, np.nan])
        elif dense and mode == "dip":
            # Taken from dense run 13591, detector 46, amplifier C16
            ptcTurnoff = 78592.643
            rawMeans = np.array(
                [3.46319253e+01, 3.66430301e+01, 3.86622690e+01, 4.05725802e+01,
                 4.25802432e+01, 4.40269809e+01, 4.80275071e+01, 5.05005628e+01,
                 5.24638704e+01, 5.51726716e+01, 5.80431294e+01, 6.18363693e+01,
                 6.76872491e+01, 7.54744430e+01, 7.59445874e+01, 8.07200670e+01,
                 8.39801977e+01, 8.89822266e+01, 9.34165914e+01, 1.00259640e+02,
                 1.04115753e+02, 1.09746401e+02, 1.16495426e+02, 1.24702365e+02,
                 1.29617918e+02, 1.38517235e+02, 1.44598282e+02, 1.55474182e+02,
                 1.60372893e+02, 1.68952583e+02, 1.79169853e+02, 1.87803850e+02,
                 1.97901901e+02, 2.12236200e+02, 2.20703858e+02, 2.33721460e+02,
                 2.45632287e+02, 2.60078861e+02, 2.73309812e+02, 2.91303550e+02,
                 3.08138900e+02, 3.20634527e+02, 3.39157572e+02, 3.61285287e+02,
                 3.77306857e+02, 3.97532592e+02, 4.19725123e+02, 4.42637799e+02,
                 4.66432222e+02, 4.92760774e+02, 5.18439719e+02, 5.49043359e+02,
                 5.80725784e+02, 6.10607872e+02, 6.42130865e+02, 6.78015970e+02,
                 7.19634201e+02, 7.56125967e+02, 7.97647549e+02, 8.44250515e+02,
                 8.88164221e+02, 9.35554605e+02, 9.87551014e+02, 1.04304478e+03,
                 1.09859274e+03, 1.15934514e+03, 1.22290957e+03, 1.28928061e+03,
                 1.36205339e+03, 1.43517932e+03, 1.51527821e+03, 1.59829513e+03,
                 1.77964089e+03, 1.78170273e+03, 1.87682761e+03, 1.87684189e+03,
                 1.98075997e+03, 1.98113130e+03, 2.08945214e+03, 2.20305152e+03,
                 2.32556464e+03, 2.58962409e+03, 2.72814882e+03, 2.99794092e+03,
                 3.16447505e+03, 3.33946592e+03, 3.52147949e+03, 3.71394579e+03,
                 3.91907040e+03, 4.13438382e+03, 4.36189594e+03, 4.60043753e+03,
                 4.85130253e+03, 4.85473652e+03, 5.12061926e+03, 5.40324180e+03,
                 5.70061253e+03, 6.34723550e+03, 6.69436828e+03, 7.23257375e+03,
                 7.76958269e+03, 8.30697359e+03, 8.84549436e+03, 9.38149894e+03,
                 9.92214412e+03, 1.04540863e+04, 1.09984280e+04, 1.15336125e+04,
                 1.20730516e+04, 1.26066598e+04, 1.31475457e+04, 1.36835704e+04,
                 1.42142220e+04, 1.47607783e+04, 1.52995772e+04, 1.58349683e+04,
                 1.63699779e+04, 1.69169118e+04, 1.74527183e+04, 1.79882677e+04,
                 1.90590055e+04, 1.96042099e+04, 2.01412931e+04, 2.01415525e+04,
                 2.06773002e+04, 2.06836599e+04, 2.17563308e+04, 2.22945949e+04,
                 2.28326494e+04, 2.33730224e+04, 2.39083711e+04, 2.44415981e+04,
                 2.49887745e+04, 2.55254849e+04, 2.60582748e+04, 2.66026296e+04,
                 2.71388848e+04, 2.76745959e+04, 2.82098429e+04, 2.92893984e+04,
                 3.03685478e+04, 3.08997866e+04, 3.10729575e+04, 3.16013527e+04,
                 3.21337166e+04, 3.26668783e+04, 3.31930745e+04, 3.37295906e+04,
                 3.42575950e+04, 3.47930111e+04, 3.53194913e+04, 3.58360666e+04,
                 3.63796428e+04, 3.69119003e+04, 3.79618999e+04, 3.85046991e+04,
                 3.90341729e+04, 3.95622456e+04, 4.00906725e+04, 4.06248524e+04,
                 4.11506714e+04, 4.16820728e+04, 4.27281690e+04, 4.32583738e+04,
                 4.37923105e+04, 4.48436280e+04, 4.53713042e+04, 4.58983070e+04,
                 4.69589412e+04, 4.74759399e+04, 4.80036320e+04, 4.85414573e+04,
                 4.90619668e+04, 4.95708602e+04, 5.01124419e+04, 5.06550432e+04,
                 5.11459970e+04, 5.16864720e+04, 5.22155963e+04, 5.32732837e+04,
                 5.37950395e+04, 5.43202717e+04, 5.48449050e+04, 5.58936896e+04,
                 5.64268047e+04, 5.69454503e+04, 5.74788533e+04, 5.79886165e+04,
                 5.85227316e+04, 5.90467563e+04, 5.95529633e+04, 6.01107425e+04,
                 6.06190504e+04, 6.11482815e+04, 6.22041090e+04, 6.27192672e+04,
                 6.32698650e+04, 6.37882800e+04, 6.43035282e+04, 6.48401842e+04,
                 6.53646656e+04, 6.59030142e+04, 6.64333556e+04, 6.69479267e+04,
                 6.74848468e+04, 6.80079972e+04, 6.85445388e+04, 6.90509171e+04,
                 6.96092229e+04, 7.01385610e+04, 7.06212128e+04, 7.11939523e+04,
                 7.17266360e+04, 7.22471223e+04, 7.27780652e+04, 7.32761803e+04,
                 7.38361226e+04, 7.43656998e+04, 7.48934046e+04, 7.54352821e+04,
                 7.59644966e+04, 7.64724502e+04, 7.70119423e+04, 7.75439221e+04,
                 7.80842181e+04, 7.85926430e+04, 7.91251970e+04, 7.96491204e+04,
                 8.01869810e+04, 8.07146073e+04, 8.12202954e+04, 8.17785695e+04,
                 8.22963087e+04, 8.33550384e+04, 8.38994212e+04, 8.44155447e+04,
                 8.49563789e+04, 8.60281829e+04, 8.65433251e+04, 8.70627645e+04,
                 8.76007341e+04, 8.81511125e+04, 8.92018550e+04, 8.97044207e+04,
                 9.07592037e+04, 9.13032564e+04, 9.18479184e+04, 9.23763279e+04,
                 9.28541040e+04, 9.34276276e+04, 9.39508203e+04, 9.44807722e+04,
                 9.50201270e+04, 9.55260238e+04, 9.60584060e+04, 9.65900977e+04,
                 9.71082260e+04, 9.76499634e+04, 9.86927409e+04, 9.92001749e+04,
                 9.97605950e+04, 1.00281826e+05, 1.00799323e+05, 1.01317520e+05,
                 1.01880214e+05, 1.02375482e+05, 1.02921781e+05, 1.03914664e+05,
                 1.04996919e+05, 1.05531177e+05, 1.06045662e+05, 1.06553960e+05,
                 1.07074502e+05, 1.07620081e+05, 1.08135030e+05, 1.08625699e+05,
                 1.09176443e+05, 1.09713812e+05, 1.10216327e+05, 1.10733831e+05,
                 1.11710810e+05, 1.12193799e+05, 1.12690561e+05, 1.13133767e+05,
                 1.13607554e+05, 1.14479014e+05, 1.14907269e+05, 1.15294108e+05,
                 1.15680838e+05, 1.16050756e+05, 1.16395663e+05, 1.16738698e+05,
                 1.17036744e+05, 1.17606634e+05, 1.17864854e+05, 1.18107937e+05,
                 1.18338933e+05, 1.18484492e+05, 1.18751107e+05, 1.18925783e+05,
                 1.19127596e+05, 1.19257553e+05, 1.19452513e+05, 1.19608706e+05,
                 1.19753398e+05, 1.19883813e+05, 1.20021670e+05, 1.20155507e+05,
                 1.20406612e+05, 1.20605600e+05, 1.20753078e+05, 1.20866466e+05,
                 np.nan, np.nan, np.nan, np.nan,
                 np.nan, np.nan, np.nan, np.nan,
                 np.nan, np.nan, np.nan, np.nan,
                 np.nan, np.nan, np.nan, np.nan,
                 np.nan, np.nan, np.nan, np.nan,
                 np.nan, np.nan, np.nan, np.nan,
                 np.nan, np.nan, np.nan, np.nan,
                 np.nan, np.nan, np.nan, np.nan,
                 np.nan, np.nan, np.nan, np.nan,
                 np.nan, np.nan, np.nan, np.nan,
                 np.nan, np.nan, np.nan, np.nan,
                 np.nan, np.nan, np.nan, np.nan,
                 np.nan, np.nan, np.nan, np.nan,
                 np.nan, np.nan, np.nan, np.nan,
                 np.nan, np.nan, np.nan, np.nan,
                 np.nan, np.nan, np.nan, np.nan])
            rawVars = np.array(
                [3.70611558e+01, 3.83737395e+01, 3.99468861e+01, 4.10014338e+01,
                 4.23251266e+01, 4.33678155e+01, 4.59071461e+01, 4.73626968e+01,
                 4.87148063e+01, 5.03835098e+01, 5.25225011e+01, 5.49763960e+01,
                 5.89144406e+01, 6.46901096e+01, 6.46921274e+01, 6.77000641e+01,
                 7.03062675e+01, 7.37864491e+01, 7.78022510e+01, 8.44978037e+01,
                 8.74723267e+01, 9.13765683e+01, 9.51046462e+01, 9.90038919e+01,
                 1.01309767e+02, 1.06498820e+02, 1.10600587e+02, 1.18170347e+02,
                 1.21408878e+02, 1.28214192e+02, 1.36647653e+02, 1.42148979e+02,
                 1.48491678e+02, 1.59805162e+02, 1.65823057e+02, 1.73305507e+02,
                 1.80522937e+02, 1.87449934e+02, 1.96245721e+02, 2.08352732e+02,
                 2.21201432e+02, 2.32523118e+02, 2.44009905e+02, 2.56991468e+02,
                 2.68047397e+02, 2.79329180e+02, 2.97712914e+02, 3.15304298e+02,
                 3.27789122e+02, 3.44980267e+02, 3.61446379e+02, 3.84494985e+02,
                 4.05677044e+02, 4.22749240e+02, 4.44709272e+02, 4.77813761e+02,
                 5.01815280e+02, 5.21252820e+02, 5.49793847e+02, 5.80608964e+02,
                 6.10172377e+02, 6.44282320e+02, 6.76618722e+02, 7.14200019e+02,
                 7.52150585e+02, 7.90093939e+02, 8.43584018e+02, 8.74257958e+02,
                 9.25137165e+02, 9.77217130e+02, 1.02268591e+03, 1.07911844e+03,
                 1.20579007e+03, 1.21094193e+03, 1.26536689e+03, 1.26783114e+03,
                 1.33667935e+03, 1.34071853e+03, 1.40959965e+03, 1.49236868e+03,
                 1.56214455e+03, 1.73911080e+03, 1.84452009e+03, 2.01282493e+03,
                 2.11914383e+03, 2.23859893e+03, 2.35101284e+03, 2.48665441e+03,
                 2.61201878e+03, 2.74350197e+03, 2.89732268e+03, 3.05252644e+03,
                 3.22199981e+03, 3.22767156e+03, 3.38954315e+03, 3.58149621e+03,
                 3.76614023e+03, 4.19955412e+03, 4.39939398e+03, 4.74116692e+03,
                 5.08774295e+03, 5.43362489e+03, 5.76587686e+03, 6.09147076e+03,
                 6.46434875e+03, 6.77118428e+03, 7.10758829e+03, 7.41292668e+03,
                 7.75391200e+03, 8.07133480e+03, 8.40331959e+03, 8.71723040e+03,
                 9.03873483e+03, 9.36871022e+03, 9.69206829e+03, 9.99510413e+03,
                 1.03183250e+04, 1.06519986e+04, 1.09717629e+04, 1.12984284e+04,
                 1.19017371e+04, 1.22009054e+04, 1.25073348e+04, 1.24828917e+04,
                 1.28123892e+04, 1.27965364e+04, 1.34024560e+04, 1.36798307e+04,
                 1.40186807e+04, 1.43485421e+04, 1.45916270e+04, 1.48802247e+04,
                 1.51990105e+04, 1.54775801e+04, 1.57746393e+04, 1.60211115e+04,
                 1.63459546e+04, 1.66499888e+04, 1.68983125e+04, 1.74427484e+04,
                 1.80456894e+04, 1.82652348e+04, 1.83690177e+04, 1.86208031e+04,
                 1.89412363e+04, 1.91602864e+04, 1.94785697e+04, 1.97091334e+04,
                 1.99827654e+04, 2.02492520e+04, 2.04397095e+04, 2.07912874e+04,
                 2.09363768e+04, 2.12446667e+04, 2.17637411e+04, 2.19817593e+04,
                 2.22395694e+04, 2.24653630e+04, 2.27254307e+04, 2.29678916e+04,
                 2.32155716e+04, 2.34915971e+04, 2.39308788e+04, 2.41386350e+04,
                 2.43300723e+04, 2.47776731e+04, 2.49855450e+04, 2.52822560e+04,
                 2.57310565e+04, 2.58705668e+04, 2.61365827e+04, 2.63745002e+04,
                 2.64761919e+04, 2.67763879e+04, 2.69546269e+04, 2.72247330e+04,
                 2.73854794e+04, 2.75964799e+04, 2.78536768e+04, 2.83219440e+04,
                 2.84473603e+04, 2.86066766e+04, 2.88858064e+04, 2.93640042e+04,
                 2.95802005e+04, 2.97953000e+04, 2.99784114e+04, 3.01818213e+04,
                 3.04537120e+04, 3.07186903e+04, 3.09159639e+04, 3.11340544e+04,
                 3.13467981e+04, 3.16593731e+04, 3.19829978e+04, 3.22855092e+04,
                 3.25921574e+04, 3.27154499e+04, 3.28698071e+04, 3.31481419e+04,
                 3.34606399e+04, 3.37260080e+04, 3.38331508e+04, 3.40882291e+04,
                 3.43298166e+04, 3.44626343e+04, 3.47352389e+04, 3.48844453e+04,
                 3.50976243e+04, 3.53422532e+04, 3.54703155e+04, 3.57850397e+04,
                 3.60218788e+04, 3.61282209e+04, 3.63718905e+04, 3.64159312e+04,
                 3.67264583e+04, 3.68810487e+04, 3.70236104e+04, 3.73025531e+04,
                 3.74696213e+04, 3.74774745e+04, 3.76885954e+04, 3.77461428e+04,
                 3.77311980e+04, 3.76449769e+04, 3.71874494e+04, 3.67403027e+04,
                 3.58397731e+04, 3.44249565e+04, 3.25971681e+04, 3.12036933e+04,
                 2.94985871e+04, 2.71719366e+04, 2.69809068e+04, 2.72483125e+04,
                 2.83267080e+04, 3.15350383e+04, 3.33307911e+04, 3.48927220e+04,
                 3.65827728e+04, 3.79316518e+04, 3.97114600e+04, 4.02038182e+04,
                 4.10252529e+04, 4.14747301e+04, 4.16607368e+04, 4.18321712e+04,
                 4.18311337e+04, 4.21298732e+04, 4.21384783e+04, 4.21559876e+04,
                 4.23156351e+04, 4.22011377e+04, 4.22592203e+04, 4.20913479e+04,
                 4.15029042e+04, 4.09470822e+04, 3.85767751e+04, 3.64665853e+04,
                 3.45138148e+04, 3.21709915e+04, 2.96629755e+04, 2.69203438e+04,
                 2.43537600e+04, 2.23428644e+04, 2.01581545e+04, 1.62544599e+04,
                 1.34824506e+04, 1.18721963e+04, 1.05452615e+04, 9.46843007e+03,
                 8.23841467e+03, 7.23997272e+03, 6.36586857e+03, 5.65708162e+03,
                 4.99963764e+03, 4.48251363e+03, 4.09078951e+03, 3.72817220e+03,
                 3.21067198e+03, 3.03262162e+03, 2.89743032e+03, 2.77343484e+03,
                 2.69667301e+03, 2.60188277e+03, 2.55172731e+03, 2.52540578e+03,
                 2.56130915e+03, 2.52293303e+03, 2.50717444e+03, 2.51044664e+03,
                 2.51531215e+03, 2.53768779e+03, 2.56201926e+03, 2.56217908e+03,
                 2.60528686e+03, 2.61531849e+03, 2.62971142e+03, 2.66896083e+03,
                 2.69706371e+03, 2.73927492e+03, 2.74999442e+03, 2.75180096e+03,
                 2.78694836e+03, 2.80704605e+03, 2.83222240e+03, 2.86654398e+03,
                 2.90678852e+03, 2.95139745e+03, 2.95942911e+03, 3.00791154e+03,
                 np.nan, np.nan, np.nan, np.nan,
                 np.nan, np.nan, np.nan, np.nan,
                 np.nan, np.nan, np.nan, np.nan,
                 np.nan, np.nan, np.nan, np.nan,
                 np.nan, np.nan, np.nan, np.nan,
                 np.nan, np.nan, np.nan, np.nan,
                 np.nan, np.nan, np.nan, np.nan,
                 np.nan, np.nan, np.nan, np.nan,
                 np.nan, np.nan, np.nan, np.nan,
                 np.nan, np.nan, np.nan, np.nan,
                 np.nan, np.nan, np.nan, np.nan,
                 np.nan, np.nan, np.nan, np.nan,
                 np.nan, np.nan, np.nan, np.nan,
                 np.nan, np.nan, np.nan, np.nan,
                 np.nan, np.nan, np.nan, np.nan,
                 np.nan, np.nan, np.nan, np.nan])
        elif not dense and mode == "normal":
            # Taken from B protocol run 13557, detector 94, amplifier C02
            ptcTurnoff = 92173.9596
            rawMeans = np.array([
                7.35242203e+01, 9.29532799e+01, 1.12149391e+02, 1.39776279e+02,
                1.74220020e+02, 2.15692281e+02, 2.68814926e+02, 3.34446868e+02,
                4.14359931e+02, 5.14691775e+02, 6.40642848e+02, 7.95596873e+02,
                9.90753280e+02, 1.22885224e+03, 1.52684706e+03, 1.89907469e+03,
                2.35701770e+03, 2.93106806e+03, 3.59466174e+03, 4.46720318e+03,
                5.55232373e+03, 6.90091293e+03, 8.57660238e+03, 1.06583769e+04,
                1.32457052e+04, 1.64618675e+04, 2.04605037e+04, 2.54301936e+04,
                3.15926481e+04, 3.87784321e+04, 4.81318315e+04, 5.97145570e+04,
                7.41859291e+04, 9.21739596e+04, 1.08484387e+05, 1.10456942e+05,
                1.10677645e+05, 1.10691134e+05, 1.10693202e+05, 1.10692292e+05,
                1.10699619e+05, 1.10703407e+05, 1.10704764e+05])
            rawVars = np.array([
                71.66098099, 85.85187266, 100.07444552, 121.59045422,
                143.54998243, 171.39262057, 210.51220342, 257.40796789,
                313.11677494, 381.91333248, 476.46542075, 583.72632638,
                729.26397749, 888.46194891, 1101.58606157, 1363.39263745,
                1686.28916615, 2096.14920835, 2552.12897667, 3159.70329711,
                3894.29254595, 4800.37079501, 5940.17760433, 7310.2253724,
                9015.14374818, 11022.81586492, 13509.18049715, 16391.89059679,
                19905.15871788, 23628.84875886, 28128.21785629, 33476.91933085,
                40023.07406229, 45198.48421179, 3390.76561947, 1601.38678043,
                1890.12715102, 1805.34862035, 1904.84156084, 1903.84859699,
                1857.15885789, 1889.45649514, 1892.01672711])
        elif not dense and mode == "upturn":
            # Taken from B protocol run 13557, detector 73, amplifier C07
            ptcTurnoff = 71832.2986
            rawMeans = np.array([
                5.65607783e+01, 7.04062631e+01, 8.74047057e+01, 1.08812996e+02,
                1.35333046e+02, 1.67636667e+02, 2.08445200e+02, 2.59202618e+02,
                3.21602277e+02, 3.99545078e+02, 4.96963648e+02, 6.17791959e+02,
                7.67578708e+02, 9.54040040e+02, 1.18587264e+03, 1.47297327e+03,
                1.83109922e+03, 2.27527491e+03, 2.79309339e+03, 3.47193365e+03,
                4.31520802e+03, 5.36244611e+03, 6.66482002e+03, 8.28357528e+03,
                1.02962387e+04, 1.27972565e+04, 1.59069356e+04, 1.97746199e+04,
                2.45737678e+04, 3.01873089e+04, 3.75075197e+04, 4.65677579e+04,
                5.77977537e+04, 7.18322986e+04, 8.92553708e+04, 9.74408608e+04,
                np.nan, np.nan, np.nan, np.nan,
                np.nan, np.nan, np.nan])
            rawVars = np.array([
                68.2700079, 76.51661413, 84.52771312, 98.70960506,
                113.95644441, 137.2544902, 158.39159813, 190.7985178,
                229.92447148, 279.52534113, 338.90663196, 414.97072284,
                510.16012626, 622.00951125, 763.81268011, 938.07098006,
                1149.75983481, 1424.88125193, 1745.36382679, 2165.89927287,
                2666.43505016, 3303.63407541, 4094.4066631, 5072.7831761,
                6238.35971003, 7697.01580574, 9479.41363292, 11654.65158713,
                14268.54471588, 17316.88016853, 21012.8242229, 25275.74644538,
                30725.05799423, 37363.24467221, 39830.3337401, 48083.26241611,
                np.nan, np.nan, np.nan, np.nan,
                np.nan, np.nan, np.nan])
        elif not dense and mode == "dip":
            # Taken from B protocol run 13557, detector 46, amplifier C16
            ptcTurnoff = 87557.8289
            rawMeans = np.array(
                [6.94187281e+01, 8.92438988e+01, 1.05958756e+02, 1.32423251e+02,
                 1.64877564e+02, 2.04367731e+02, 2.55076247e+02, 3.16940500e+02,
                 3.92525299e+02, 4.87631833e+02, 6.06783494e+02, 7.53540982e+02,
                 9.39742534e+02, 1.16367754e+03, 1.44670172e+03, 1.80047945e+03,
                 2.23357272e+03, 2.77872101e+03, 3.40646845e+03, 4.23306945e+03,
                 5.26085005e+03, 6.53969254e+03, 8.12914423e+03, 1.01016373e+04,
                 1.25546410e+04, 1.56072062e+04, 1.93987924e+04, 2.41153677e+04,
                 2.99669024e+04, 3.67956267e+04, 4.56879445e+04, 5.66901053e+04,
                 7.04360489e+04, 8.75578289e+04, 1.08738204e+05, np.nan,
                 np.nan, np.nan, np.nan, np.nan,
                 np.nan, np.nan, np.nan])
            rawVars = np.array(
                [61.7553328, 75.60175792, 88.98931208, 106.95977,
                 129.78637495, 153.5130247, 192.25182776, 233.4904232,
                 287.02595043, 349.57554198, 428.71515892, 528.0987701,
                 656.97011401, 820.88213784, 1003.62289116, 1243.67283887,
                 1543.24210179, 1905.00050382, 2321.63329694, 2889.08652787,
                 3574.52610263, 4393.02299998, 5427.50087472, 6691.62708604,
                 8226.42941205, 10104.75767742, 12365.56291744, 15029.66357377,
                 18233.5785582, 21684.66746226, 25811.97832361, 30686.30643311,
                 36834.73747467, 41993.20346722, 30079.64884394, np.nan,
                 np.nan, np.nan, np.nan, np.nan,
                 np.nan, np.nan, np.nan])
        else:
            raise RuntimeError("Illegal mode")

        return rawMeans, rawVars, ptcTurnoff


class MeasurePhotonTransferCurveDatasetTestCase(lsst.utils.tests.TestCase):
    def setUp(self):
        self.ptcData = PhotonTransferCurveDataset(["C00", "C01"], " ")
        self.ptcData.inputExpIdPairs = {
            "C00": [(123, 234), (345, 456), (567, 678)],
            "C01": [(123, 234), (345, 456), (567, 678)],
        }

    def test_generalBehaviour(self):
        test = PhotonTransferCurveDataset(["C00", "C01"], " ")
        test.inputExpIdPairs = {
            "C00": [(123, 234), (345, 456), (567, 678)],
            "C01": [(123, 234), (345, 456), (567, 678)],
        }


class TestMemory(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
