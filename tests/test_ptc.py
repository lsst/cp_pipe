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

import lsst.utils
import lsst.utils.tests

import lsst.cp.pipe as cpPipe
import lsst.ip.isr.isrMock as isrMock
from lsst.ip.isr import PhotonTransferCurveDataset, PhotodiodeCalib
from lsst.cp.pipe.utils import makeMockFlats

from lsst.pipe.base import InMemoryDatasetHandle, TaskMetadata


class FakeCamera(list):
    def getName(self):
        return "FakeCam"


class PretendRef:
    "A class to act as a mock exposure reference"

    def __init__(self, exposure):
        self.exp = exposure

    def get(self, component=None):
        if component == "visitInfo":
            return self.exp.getVisitInfo()
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
            cpPipe.ptc.PhotonTransferCurveExtractTask.ConfigClass()
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

        # ISR metadata
        self.metadataContents = TaskMetadata()
        self.metadataContents["isr"] = {}
        # Overscan readout noise [in ADU]
        for amp in self.ampNames:
            self.metadataContents["isr"][f"RESIDUAL STDEV {amp}"] = (
                np.sqrt(self.noiseSq) / self.gain
            )

    def test_covAstier(self):
        """Test to check getCovariancesAstier

        We check that the gain is the same as the imput gain from the
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
        extractConfig.auxiliaryHeaderKeys = ["CCOBCURR", "CCDTEMP"]
        extractTask = cpPipe.ptc.PhotonTransferCurveExtractTask(config=extractConfig)

        solveConfig = self.defaultConfigSolve
        solveConfig.ptcFitType = "FULLCOVARIANCE"
        # Cut off the low-flux point which is a bad fit, and this
        # also exercises this functionality and makes the tests
        # run a lot faster.
        solveConfig.minMeanSignal["ALL_AMPS"] = 2000.0
        # Set the outlier fit threshold higher than the default appropriate
        # for this test dataset.
        solveConfig.maxSignalInitialPtcOutlierFit = 90000.0
        solveTask = cpPipe.ptc.PhotonTransferCurveSolveTask(config=solveConfig)

        inputGain = self.gain

        muStandard, varStandard = {}, {}
        expDict = {}
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
            )
            for mockExp in [mockExp1, mockExp2]:
                md = mockExp.getMetadata()
                # These values are chosen to be easily compared after
                # processing for correct ordering.
                md['CCOBCURR'] = float(idCounter)
                md['CCDTEMP'] = float(idCounter + 1)
                mockExp.setMetadata(md)

            mockExpRef1 = PretendRef(mockExp1)
            mockExpRef2 = PretendRef(mockExp2)
            expDict[expTime] = ((mockExpRef1, idCounter), (mockExpRef2, idCounter + 1))
            expIds.append(idCounter)
            expIds.append(idCounter + 1)
            for ampNumber, ampName in enumerate(self.ampNames):
                # cov has (i, j, var, cov, npix)
                (
                    im1Area,
                    im2Area,
                    imStatsCtrl,
                    mu1,
                    mu2,
                ) = extractTask.getImageAreasMasksStats(mockExp1, mockExp2)
                muDiff, varDiff, covAstier = extractTask.measureMeanVarCov(
                    im1Area, im2Area, imStatsCtrl, mu1, mu2
                )
                muStandard.setdefault(ampName, []).append(muDiff)
                varStandard.setdefault(ampName, []).append(varDiff)

            # Make a photodiode dataset to integrate.
            timeSamples = np.linspace(0, 20.0, 100)
            currentSamples = np.zeros(100)
            currentSamples[50] = -1.0*self.photoCharges[i]

            pdCalib = PhotodiodeCalib(timeSamples=timeSamples, currentSamples=currentSamples)
            pdCalib.currentScale = -1.0
            pdCalib.integrationMethod = "CHARGE_SUM"

            pdHandles.append(
                InMemoryDatasetHandle(
                    pdCalib,
                    dataId={"exposure": idCounter},
                )
            )
            pdHandles.append(
                InMemoryDatasetHandle(
                    pdCalib,
                    dataId={"exposure": idCounter + 1},
                )
            )
            idCounter += 2

        resultsExtract = extractTask.run(
            inputExp=expDict,
            inputDims=expIds,
            taskMetadata=[self.metadataContents for x in expIds],
            inputPhotodiodeData=pdHandles,
        )

        # Force the last PTC dataset to have a NaN, and ensure that the
        # task runs (DM-38029).  This is a minor perturbation and does not
        # affect the output comparison. Note that we use index -2 because
        # these datasets are in pairs of [real, dummy] to match the inputs
        # to the extract task.
        resultsExtract.outputCovariances[-2].rawMeans["C:0,0"] = np.array([np.nan])
        resultsExtract.outputCovariances[-2].rawVars["C:0,0"] = np.array([np.nan])

        # Force the next-to-last PTC dataset to have a decreased variance to
        # ensure that the outlier fit rejection works. Note that we use
        # index -4 because these datasets are in pairs of [real, dummy] to
        # match the inputs to the extract task.
        rawVar = resultsExtract.outputCovariances[-4].rawVars["C:0,0"]
        resultsExtract.outputCovariances[-4].rawVars["C:0,0"] = rawVar * 0.9

        # Reorganize the outputCovariances so we can confirm they come
        # out sorted afterwards.
        outputCovariancesRev = resultsExtract.outputCovariances[::-1]

        resultsSolve = solveTask.run(
            outputCovariancesRev, camera=FakeCamera([self.flatExp1.getDetector()])
        )

        ptc = resultsSolve.outputPtcDataset

        # Some expected values for noise matrix, just to check that
        # it was calculated.
        noiseMatrixNoBExpected = {
            (0, 0): 6.53126505,
            (1, 1): -23.20924747,
            (2, 2): 35.69834113,
        }
        noiseMatrixExpected = {
            (0, 0): 29.37146918,
            (1, 1): -14.6849025,
            (2, 2): 24.7328517,
        }

        noiseMatrixExpected = np.array(
            [
                [
                    29.37146918,
                    9.2760363,
                    -29.08907932,
                    33.65818827,
                    -52.65710984,
                    -18.5821773,
                    -46.26896286,
                    65.01049736,
                ],
                [
                    -3.62427987,
                    -14.6849025,
                    -46.55230305,
                    -1.30410627,
                    6.44903599,
                    18.11796075,
                    -22.72874074,
                    20.90219857,
                ],
                [
                    5.09203058,
                    -4.40097862,
                    24.7328517,
                    39.2847586,
                    -21.46132351,
                    8.12179783,
                    6.23585617,
                    -2.09949622,
                ],
                [
                    35.79204016,
                    -6.50205005,
                    3.37910363,
                    15.22335662,
                    -19.29035067,
                    9.66065941,
                    7.47510934,
                    20.25962845,
                ],
                [
                    -36.23187633,
                    -22.72307472,
                    16.29140749,
                    -13.09493835,
                    3.32091085,
                    52.4380977,
                    -8.06428902,
                    -22.66669839,
                ],
                [
                    -27.93122896,
                    15.37016686,
                    9.18835073,
                    -24.48892946,
                    8.14480304,
                    22.38983222,
                    22.36866891,
                    -0.38803439,
                ],
                [
                    17.13962665,
                    -28.33153763,
                    -17.79744334,
                    -18.57064463,
                    7.69408833,
                    8.48265396,
                    18.0447022,
                    -16.97496022,
                ],
                [
                    10.09078383,
                    -26.61613002,
                    10.48504889,
                    15.33196998,
                    -23.35165517,
                    -24.53098643,
                    -18.21201067,
                    17.40755051,
                ],
            ]
        )

        noiseMatrixNoBExpected = np.array(
            [
                [
                    6.53126505,
                    12.14827594,
                    -37.11919923,
                    41.18675353,
                    -85.1613845,
                    -28.45801954,
                    -61.24442999,
                    88.76480122,
                ],
                [
                    -4.64541165,
                    -23.20924747,
                    -66.08733987,
                    -0.87558055,
                    12.20111853,
                    24.84795549,
                    -34.92458788,
                    24.42745014,
                ],
                [
                    7.66734507,
                    -4.51403645,
                    35.69834113,
                    52.73693356,
                    -30.85044089,
                    10.86761771,
                    10.8503068,
                    -2.18908327,
                ],
                [
                    50.9901156,
                    -7.34803977,
                    5.33443765,
                    21.60899396,
                    -25.06129827,
                    15.14015505,
                    10.94263771,
                    29.23975515,
                ],
                [
                    -48.66912069,
                    -31.58003774,
                    21.81305735,
                    -13.08993444,
                    8.17275394,
                    74.85293723,
                    -11.18403252,
                    -31.7799437,
                ],
                [
                    -38.55206382,
                    22.92982676,
                    13.39861008,
                    -33.3307362,
                    8.65362238,
                    29.18775548,
                    31.78433947,
                    1.27923706,
                ],
                [
                    23.33663918,
                    -41.74105625,
                    -26.55920751,
                    -24.71611677,
                    12.13343146,
                    11.25763907,
                    21.79131019,
                    -26.579393,
                ],
                [
                    11.44334226,
                    -34.9759641,
                    13.96449509,
                    19.64121933,
                    -36.09794843,
                    -34.27205933,
                    -25.16574105,
                    23.80460972,
                ],
            ]
        )

        for amp in self.ampNames:
            self.assertAlmostEqual(ptc.gain[amp], inputGain, places=2)
            for v1, v2 in zip(varStandard[amp], ptc.finalVars[amp]):
                self.assertAlmostEqual(v1 / v2, 1.0, places=1)

            # Check that the PTC turnoff is correctly computed.
            # This will be different for the C:0,0 amp.
            if amp == "C:0,0":
                self.assertAlmostEqual(ptc.ptcTurnoff[amp], ptc.rawMeans[ampName][-3])
            else:
                self.assertAlmostEqual(ptc.ptcTurnoff[amp], ptc.rawMeans[ampName][-1])

            # Test that all the quantities are correctly ordered and have
            # not accidentally been masked. We check every other output ([::2])
            # because these datasets are in pairs of [real, dummy] to
            # match the inputs to the extract task.
            for i, extractPtc in enumerate(resultsExtract.outputCovariances[::2]):
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
                    extractPtc.photoCharges[ampName][0],
                    ptc.photoCharges[ampName][i],
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
                    ptc.noiseMatrix[ampName], noiseMatrixExpected, atol=1e-8, rtol=None
                )
                self.assertFloatsAlmostEqual(
                    ptc.noiseMatrixNoB[ampName],
                    noiseMatrixNoBExpected,
                    atol=1e-8,
                    rtol=None,
                )

            mask = ptc.getGoodPoints(amp)

            values = (
                ptc.covariancesModel[amp][mask, 0, 0] - ptc.covariances[amp][mask, 0, 0]
            ) / ptc.covariancesModel[amp][mask, 0, 0]
            np.testing.assert_array_less(np.abs(values), 2e-3)

            values = (
                ptc.covariancesModel[amp][mask, 1, 1] - ptc.covariances[amp][mask, 1, 1]
            ) / ptc.covariancesModel[amp][mask, 1, 1]
            np.testing.assert_array_less(np.abs(values), 0.2)

            values = (
                ptc.covariancesModel[amp][mask, 1, 2] - ptc.covariances[amp][mask, 1, 2]
            ) / ptc.covariancesModel[amp][mask, 1, 2]
            np.testing.assert_array_less(np.abs(values), 0.2)

        # And test that the auxiliary values are there and correctly ordered.
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
        covModelNoBShape = None

        for ampName in self.ampNames:
            if covShape is None:
                covShape = ptc.covariances[ampName].shape
                covSqrtShape = ptc.covariancesSqrtWeights[ampName].shape
                covModelShape = ptc.covariancesModel[ampName].shape
                covModelNoBShape = ptc.covariancesModelNoB[ampName].shape
            else:
                self.assertEqual(ptc.covariances[ampName].shape, covShape)
                self.assertEqual(
                    ptc.covariancesSqrtWeights[ampName].shape, covSqrtShape
                )
                self.assertEqual(ptc.covariancesModel[ampName].shape, covModelShape)
                self.assertEqual(
                    ptc.covariancesModelNoB[ampName].shape, covModelNoBShape
                )

        # And check that this is serializable
        with tempfile.NamedTemporaryFile(suffix=".fits") as f:
            usedFilename = ptc.writeFits(f.name)
            fromFits = PhotonTransferCurveDataset.readFits(usedFilename)
        self.assertEqual(fromFits, ptc)

    def ptcFitAndCheckPtc(
        self,
        order=None,
        fitType=None,
        doFitBootstrap=False,
        doLegacy=False,
    ):
        localDataset = copy.deepcopy(self.dataset)
        localDataset.ptcFitType = fitType
        configSolve = copy.copy(self.defaultConfigSolve)
        if doFitBootstrap:
            configSolve.doFitBootstrap = True

        configSolve.doLegacyTurnoffSelection = doLegacy

        if fitType == "POLYNOMIAL":
            if order not in [2, 3]:
                RuntimeError("Enter a valid polynomial order for this test: 2 or 3")
            if order == 2:
                for ampName in self.ampNames:
                    localDataset.rawVars[ampName] = [
                        self.noiseSq + self.c1 * mu + self.c2 * mu**2
                        for mu in localDataset.rawMeans[ampName]
                    ]
                configSolve.polynomialFitDegree = 2
            if order == 3:
                for ampName in self.ampNames:
                    localDataset.rawVars[ampName] = [
                        self.noiseSq
                        + self.c1 * mu
                        + self.c2 * mu**2
                        + self.c3 * mu**3
                        for mu in localDataset.rawMeans[ampName]
                    ]
                configSolve.polynomialFitDegree = 3
        elif fitType == "EXPAPPROXIMATION":
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
                "Enter a fit function type: 'POLYNOMIAL' or 'EXPAPPROXIMATION'"
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

        localDataset = solveTask.fitMeasurementsToModel(localDataset)

        # check entries in localDataset, which was modified by the function
        for ampName in self.ampNames:
            self.assertEqual(fitType, localDataset.ptcFitType)
            self.assertAlmostEqual(self.gain, localDataset.gain[ampName])
            if fitType == "POLYNOMIAL":
                self.assertAlmostEqual(self.c1, localDataset.ptcFitPars[ampName][1])
                self.assertAlmostEqual(
                    np.sqrt(self.noiseSq) * self.gain, localDataset.noise[ampName]
                )
            if fitType == "EXPAPPROXIMATION":
                self.assertAlmostEqual(
                    self.a00, localDataset.ptcFitPars[ampName][0]
                )
                # noise already in electrons for 'EXPAPPROXIMATION' fit
                self.assertAlmostEqual(
                    np.sqrt(self.noiseSq), localDataset.noise[ampName]
                )

    def test_ptcFit(self):
        for doLegacy in [False, True]:
            for fitType, order in [
                ("POLYNOMIAL", 2),
                ("POLYNOMIAL", 3),
                ("EXPAPPROXIMATION", None),
            ]:
                self.ptcFitAndCheckPtc(
                    fitType=fitType,
                    order=order,
                    doLegacy=doLegacy,
                )

    def test_meanVarMeasurement(self):
        task = self.defaultTaskExtract
        im1Area, im2Area, imStatsCtrl, mu1, mu2 = task.getImageAreasMasksStats(
            self.flatExp1, self.flatExp2
        )
        mu, varDiff, _ = task.measureMeanVarCov(im1Area, im2Area, imStatsCtrl, mu1, mu2)

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
        mu, varDiff, _ = task.measureMeanVarCov(im1Area, im2Area, imStatsCtrl, mu1, mu2)

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
        mu, varDiff, covDiff = task.measureMeanVarCov(
            im1Area, im2Area, imStatsCtrl, mu1, mu2
        )

        self.assertTrue(np.isnan(mu))
        self.assertTrue(np.isnan(varDiff))
        self.assertTrue(covDiff is None)

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
            mu, varDiff, covDiff = task.measureMeanVarCov(
                im1Area, im2Area, imStatsCtrl, mu1, mu2
            )
        self.assertIn("Number of good points", cm.output[0])

        self.assertTrue(np.isnan(mu))
        self.assertTrue(np.isnan(varDiff))
        self.assertTrue(covDiff is None)

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
            mu, varDiff, covDiff = task.measureMeanVarCov(
                im1Area, im2Area, imStatsCtrl, mu1, mu2
            )
        self.assertIn("Not enough pixels", cm.output[0])

        self.assertTrue(np.isnan(mu))
        self.assertTrue(np.isnan(varDiff))
        self.assertTrue(covDiff is None)

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

    def test_getInitialGoodPoints(self):
        xs = [1, 2, 3, 4, 5, 6]
        ys = [2 * x for x in xs]
        points = self.defaultTaskSolve._getInitialGoodPoints(
            xs, ys, minVarPivotSearch=0.0, consecutivePointsVarDecreases=2
        )
        assert np.all(points) == np.all(np.array([True for x in xs]))

        ys[4] = 7  # Variance decreases in two consecutive points after ys[3]=8
        ys[5] = 6
        points = self.defaultTaskSolve._getInitialGoodPoints(
            xs, ys, minVarPivotSearch=0.0, consecutivePointsVarDecreases=2
        )
        assert np.all(points) == np.all(np.array([True, True, True, True, False]))

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
            )
            mockExpRef1 = PretendRef(mockExp1)
            mockExpRef2 = PretendRef(mockExp2)
            expDict[expTime] = ((mockExpRef1, idCounter), (mockExpRef2, idCounter + 1))
            expIds.append(idCounter)
            expIds.append(idCounter + 1)
            idCounter += 2

        resultsExtract = extractTask.run(
            inputExp=expDict,
            inputDims=expIds,
            taskMetadata=[self.metadataContents for x in expIds],
        )
        for exposurePair in resultsExtract.outputCovariances:
            for ampName in self.ampNames:
                if exposurePair.gain[ampName] is np.nan:
                    continue
                self.assertAlmostEqual(
                    exposurePair.gain[ampName], inputGain, delta=0.04
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
        for (fitType, order) in [('POLYNOMIAL', 2), ('POLYNOMIAL', 3), ('EXPAPPROXIMATION', None)]:
            self.ptcFitAndCheckPtc(fitType=fitType, order=order, doFitBootstrap=True)


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
