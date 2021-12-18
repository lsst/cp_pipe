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

from __future__ import absolute_import, division, print_function
import unittest
import numpy as np
import copy

import lsst.utils
import lsst.utils.tests

import lsst.cp.pipe as cpPipe
import lsst.ip.isr.isrMock as isrMock
from lsst.ip.isr import PhotonTransferCurveDataset
from lsst.cp.pipe.utils import (funcPolynomial, makeMockFlats)


class FakeCamera(list):
    def getName(self):
        return "FakeCam"


class PretendRef():
    "A class to act as a mock exposure reference"
    def __init__(self, exposure):
        self.exp = exposure

    def get(self, component=None):
        if component == 'visitInfo':
            return self.exp.getVisitInfo()
        elif component == 'detector':
            return self.exp.getDetector()
        else:
            return self.exp


class MeasurePhotonTransferCurveTaskTestCase(lsst.utils.tests.TestCase):
    """A test case for the PTC task."""

    def setUp(self):
        self.defaultConfig = cpPipe.ptc.MeasurePhotonTransferCurveTask.ConfigClass()
        self.defaultTask = cpPipe.ptc.MeasurePhotonTransferCurveTask(config=self.defaultConfig)

        self.defaultConfigExtract = cpPipe.ptc.PhotonTransferCurveExtractTask.ConfigClass()
        self.defaultTaskExtract = cpPipe.ptc.PhotonTransferCurveExtractTask(config=self.defaultConfigExtract)

        self.defaultConfigSolve = cpPipe.ptc.PhotonTransferCurveSolveTask.ConfigClass()
        self.defaultTaskSolve = cpPipe.ptc.PhotonTransferCurveSolveTask(config=self.defaultConfigSolve)

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
        self.flux = 1000.  # ADU/sec
        self.timeVec = np.arange(1., 101., 5)
        self.k2NonLinearity = -5e-6
        # quadratic signal-chain non-linearity
        muVec = self.flux*self.timeVec + self.k2NonLinearity*self.timeVec**2
        self.gain = 1.5  # e-/ADU
        self.c1 = 1./self.gain
        self.noiseSq = 5*self.gain  # 7.5 (e-)^2
        self.a00 = -1.2e-6
        self.c2 = -1.5e-6
        self.c3 = -4.7e-12  # tuned so that it turns over for 200k mean

        self.ampNames = [amp.getName() for amp in self.flatExp1.getDetector().getAmplifiers()]
        self.dataset = PhotonTransferCurveDataset(self.ampNames, " ")  # pack raw data for fitting

        for ampName in self.ampNames:  # just the expTimes and means here - vars vary per function
            self.dataset.rawExpTimes[ampName] = self.timeVec
            self.dataset.rawMeans[ampName] = muVec

    def test_covAstier(self):
        """Test to check getCovariancesAstier

        We check that the gain is the same as the imput gain from the
        mock data, that the covariances via FFT (as it is in
        MeasurePhotonTransferCurveTask when doCovariancesAstier=True)
        are the same as calculated in real space, and that Cov[0, 0]
        (i.e., the variances) are similar to the variances calculated
        with the standard method (when doCovariancesAstier=false),

        """
        task = self.defaultTask
        extractConfig = self.defaultConfigExtract
        extractConfig.minNumberGoodPixelsForCovariance = 5000
        extractConfig.detectorMeasurementRegion = 'FULL'
        extractTask = cpPipe.ptc.PhotonTransferCurveExtractTask(config=extractConfig)

        solveConfig = self.defaultConfigSolve
        solveConfig.ptcFitType = 'FULLCOVARIANCE'
        solveTask = cpPipe.ptc.PhotonTransferCurveSolveTask(config=solveConfig)

        inputGain = 0.75

        muStandard, varStandard = {}, {}
        expDict = {}
        expIds = []
        idCounter = 0
        for expTime in self.timeVec:
            mockExp1, mockExp2 = makeMockFlats(expTime, gain=inputGain,
                                               readNoiseElectrons=3, expId1=idCounter,
                                               expId2=idCounter+1)
            mockExpRef1 = PretendRef(mockExp1)
            mockExpRef2 = PretendRef(mockExp2)
            expDict[expTime] = ((mockExpRef1, idCounter), (mockExpRef2, idCounter+1))
            expIds.append(idCounter)
            expIds.append(idCounter+1)
            for ampNumber, ampName in enumerate(self.ampNames):
                # cov has (i, j, var, cov, npix)
                muDiff, varDiff, covAstier = task.extract.measureMeanVarCov(mockExp1, mockExp2)
                muStandard.setdefault(ampName, []).append(muDiff)
                varStandard.setdefault(ampName, []).append(varDiff)
                # Calculate covariances in an independent way: direct space
                _, _, covsDirect = task.extract.measureMeanVarCov(mockExp1, mockExp2, covAstierRealSpace=True)

                # Test that the arrays "covs" (FFT) and "covDirect"
                # (direct space) are the same
                for row1, row2 in zip(covAstier, covsDirect):
                    for a, b in zip(row1, row2):
                        self.assertAlmostEqual(a, b)
            idCounter += 2
        resultsExtract = extractTask.run(inputExp=expDict, inputDims=expIds)
        resultsSolve = solveTask.run(resultsExtract.outputCovariances)

        for amp in self.ampNames:
            self.assertAlmostEqual(resultsSolve.outputPtcDataset.gain[amp], inputGain, places=2)
            for v1, v2 in zip(varStandard[amp], resultsSolve.outputPtcDataset.finalVars[amp]):
                self.assertAlmostEqual(v1/v2, 1.0, places=1)

    def ptcFitAndCheckPtc(self, order=None, fitType=None, doTableArray=False, doFitBootstrap=False):
        localDataset = copy.copy(self.dataset)
        localDataset.ptcFitType = fitType
        configSolve = copy.copy(self.defaultConfigSolve)
        configLin = cpPipe.linearity.LinearitySolveTask.ConfigClass()
        placesTests = 6
        if doFitBootstrap:
            configSolve.doFitBootstrap = True
            # Bootstrap method in cp_pipe/utils.py does multiple fits
            # in the precense of noise.  Allow for more margin of
            # error.
            placesTests = 3

        if fitType == 'POLYNOMIAL':
            if order not in [2, 3]:
                RuntimeError("Enter a valid polynomial order for this test: 2 or 3")
            if order == 2:
                for ampName in self.ampNames:
                    localDataset.rawVars[ampName] = [self.noiseSq + self.c1*mu + self.c2*mu**2 for
                                                     mu in localDataset.rawMeans[ampName]]
                configSolve.polynomialFitDegree = 2
            if order == 3:
                for ampName in self.ampNames:
                    localDataset.rawVars[ampName] = [self.noiseSq + self.c1*mu + self.c2*mu**2 + self.c3*mu**3
                                                     for mu in localDataset.rawMeans[ampName]]
                configSolve.polynomialFitDegree = 3
        elif fitType == 'EXPAPPROXIMATION':
            g = self.gain
            for ampName in self.ampNames:
                localDataset.rawVars[ampName] = [(0.5/(self.a00*g**2)*(np.exp(2*self.a00*mu*g)-1)
                                                 + self.noiseSq/(g*g))
                                                 for mu in localDataset.rawMeans[ampName]]
        else:
            RuntimeError("Enter a fit function type: 'POLYNOMIAL' or 'EXPAPPROXIMATION'")

        configLin.maxLookupTableAdu = 200000  # Max ADU in input mock flats
        configLin.maxLinearAdu = 100000
        configLin.minLinearAdu = 50000
        if doTableArray:
            configLin.linearityType = "LookupTable"
        else:
            configLin.linearityType = "Polynomial"
        solveTask = cpPipe.ptc.PhotonTransferCurveSolveTask(config=configSolve)
        linearityTask = cpPipe.linearity.LinearitySolveTask(config=configLin)

        if doTableArray:
            # Non-linearity
            numberAmps = len(self.ampNames)
            # localDataset: PTC dataset
            # (`lsst.ip.isr.ptcDataset.PhotonTransferCurveDataset`)
            localDataset = solveTask.fitPtc(localDataset)
            # linDataset here is a lsst.pipe.base.Struct
            linDataset = linearityTask.run(localDataset,
                                           dummy=[1.0],
                                           camera=FakeCamera([self.flatExp1.getDetector()]),
                                           inputDims={'detector': 0})
            linDataset = linDataset.outputLinearizer
        else:
            localDataset = solveTask.fitPtc(localDataset)
            linDataset = linearityTask.run(localDataset,
                                           dummy=[1.0],
                                           camera=FakeCamera([self.flatExp1.getDetector()]),
                                           inputDims={'detector': 0})
            linDataset = linDataset.outputLinearizer
        if doTableArray:
            # check that the linearizer table has been filled out properly
            for i in np.arange(numberAmps):
                tMax = (configLin.maxLookupTableAdu)/self.flux
                timeRange = np.linspace(0., tMax, configLin.maxLookupTableAdu)
                signalIdeal = timeRange*self.flux
                signalUncorrected = funcPolynomial(np.array([0.0, self.flux, self.k2NonLinearity]),
                                                   timeRange)
                linearizerTableRow = signalIdeal - signalUncorrected
                self.assertEqual(len(linearizerTableRow), len(linDataset.tableData[i, :]))
                for j in np.arange(len(linearizerTableRow)):
                    self.assertAlmostEqual(linearizerTableRow[j], linDataset.tableData[i, :][j],
                                           places=placesTests)
        else:
            # check entries in localDataset, which was modified by the function
            for ampName in self.ampNames:
                maskAmp = localDataset.expIdMask[ampName]
                finalMuVec = localDataset.rawMeans[ampName][maskAmp]
                finalTimeVec = localDataset.rawExpTimes[ampName][maskAmp]
                linearPart = self.flux*finalTimeVec
                inputFracNonLinearityResiduals = 100*(linearPart - finalMuVec)/linearPart
                self.assertEqual(fitType, localDataset.ptcFitType)
                self.assertAlmostEqual(self.gain, localDataset.gain[ampName])
                if fitType == 'POLYNOMIAL':
                    self.assertAlmostEqual(self.c1, localDataset.ptcFitPars[ampName][1])
                    self.assertAlmostEqual(np.sqrt(self.noiseSq)*self.gain, localDataset.noise[ampName])
                if fitType == 'EXPAPPROXIMATION':
                    self.assertAlmostEqual(self.a00, localDataset.ptcFitPars[ampName][0])
                    # noise already in electrons for 'EXPAPPROXIMATION' fit
                    self.assertAlmostEqual(np.sqrt(self.noiseSq), localDataset.noise[ampName])

            # check entries in returned dataset (a dict of , for nonlinearity)
            for ampName in self.ampNames:
                maskAmp = localDataset.expIdMask[ampName]
                finalMuVec = localDataset.rawMeans[ampName][maskAmp]
                finalTimeVec = localDataset.rawExpTimes[ampName][maskAmp]
                linearPart = self.flux*finalTimeVec
                inputFracNonLinearityResiduals = 100*(linearPart - finalMuVec)/linearPart

                # Nonlinearity fit parameters
                # Polynomial fits are now normalized to unit flux scaling
                self.assertAlmostEqual(0.0, linDataset.fitParams[ampName][0], places=1)
                self.assertAlmostEqual(1.0, linDataset.fitParams[ampName][1],
                                       places=5)

                # Non-linearity coefficient for linearizer
                squaredCoeff = self.k2NonLinearity/(self.flux**2)
                self.assertAlmostEqual(squaredCoeff, linDataset.fitParams[ampName][2],
                                       places=placesTests)
                self.assertAlmostEqual(-squaredCoeff, linDataset.linearityCoeffs[ampName][2],
                                       places=placesTests)

                linearPartModel = linDataset.fitParams[ampName][1]*finalTimeVec*self.flux
                outputFracNonLinearityResiduals = 100*(linearPartModel - finalMuVec)/linearPartModel
                # Fractional nonlinearity residuals
                self.assertEqual(len(outputFracNonLinearityResiduals), len(inputFracNonLinearityResiduals))
                for calc, truth in zip(outputFracNonLinearityResiduals, inputFracNonLinearityResiduals):
                    self.assertAlmostEqual(calc, truth, places=3)

    def test_ptcFit(self):
        for createArray in [True, False]:
            for (fitType, order) in [('POLYNOMIAL', 2), ('POLYNOMIAL', 3), ('EXPAPPROXIMATION', None)]:
                self.ptcFitAndCheckPtc(fitType=fitType, order=order, doTableArray=createArray)

    def test_meanVarMeasurement(self):
        task = self.defaultTaskExtract
        mu, varDiff, _ = task.measureMeanVarCov(self.flatExp1, self.flatExp2)

        self.assertLess(self.flatWidth - np.sqrt(varDiff), 1)
        self.assertLess(self.flatMean - mu, 1)

    def test_meanVarMeasurementWithNans(self):
        task = self.defaultTaskExtract
        self.flatExp1.image.array[20:30, :] = np.nan
        self.flatExp2.image.array[20:30, :] = np.nan

        mu, varDiff, _ = task.measureMeanVarCov(self.flatExp1, self.flatExp2)

        expectedMu1 = np.nanmean(self.flatExp1.image.array)
        expectedMu2 = np.nanmean(self.flatExp2.image.array)
        expectedMu = 0.5*(expectedMu1 + expectedMu2)

        # Now the variance of the difference. First, create the diff image.
        im1 = self.flatExp1.maskedImage
        im2 = self.flatExp2.maskedImage

        temp = im2.clone()
        temp *= expectedMu1
        diffIm = im1.clone()
        diffIm *= expectedMu2
        diffIm -= temp
        diffIm /= expectedMu

        # Divide by two as it is what measureMeanVarCov returns
        # (variance of difference)
        expectedVar = 0.5*np.nanvar(diffIm.image.array)

        # Check that the standard deviations and the emans agree to
        # less than 1 ADU
        self.assertLess(np.sqrt(expectedVar) - np.sqrt(varDiff), 1)
        self.assertLess(expectedMu - mu, 1)

    def test_meanVarMeasurementAllNan(self):
        task = self.defaultTaskExtract
        self.flatExp1.image.array[:, :] = np.nan
        self.flatExp2.image.array[:, :] = np.nan

        mu, varDiff, covDiff = task.measureMeanVarCov(self.flatExp1, self.flatExp2)

        self.assertTrue(np.isnan(mu))
        self.assertTrue(np.isnan(varDiff))
        self.assertTrue(covDiff is None)

    def test_makeZeroSafe(self):
        noZerosArray = [1., 20, -35, 45578.98, 90.0, 897, 659.8]
        someZerosArray = [1., 20, 0, 0, 90, 879, 0]
        allZerosArray = [0., 0.0, 0, 0, 0.0, 0, 0]

        substituteValue = 1e-10

        expectedSomeZerosArray = [1., 20, substituteValue, substituteValue, 90, 879, substituteValue]
        expectedAllZerosArray = np.repeat(substituteValue, len(allZerosArray))

        measuredSomeZerosArray = self.defaultTaskSolve._makeZeroSafe(someZerosArray,
                                                                     substituteValue=substituteValue)
        measuredAllZerosArray = self.defaultTaskSolve._makeZeroSafe(allZerosArray,
                                                                    substituteValue=substituteValue)
        measuredNoZerosArray = self.defaultTaskSolve._makeZeroSafe(noZerosArray,
                                                                   substituteValue=substituteValue)

        for exp, meas in zip(expectedSomeZerosArray, measuredSomeZerosArray):
            self.assertEqual(exp, meas)
        for exp, meas in zip(expectedAllZerosArray, measuredAllZerosArray):
            self.assertEqual(exp, meas)
        for exp, meas in zip(noZerosArray, measuredNoZerosArray):
            self.assertEqual(exp, meas)

    def test_getInitialGoodPoints(self):
        xs = [1, 2, 3, 4, 5, 6]
        ys = [2*x for x in xs]
        points = self.defaultTaskSolve._getInitialGoodPoints(xs, ys, minVarPivotSearch=0.)
        assert np.all(points) == np.all(np.array([True for x in xs]))

        ys[4] = 7  # Variance decreases in two consecutive points after ys[3]=8
        ys[5] = 6
        points = self.defaultTaskSolve._getInitialGoodPoints(xs, ys, minVarPivotSearch=0.)
        assert np.all(points) == np.all(np.array([True, True, True, True, False]))

    def test_getExpIdsUsed(self):
        localDataset = copy.copy(self.dataset)

        for pair in [(12, 34), (56, 78), (90, 10)]:
            localDataset.inputExpIdPairs["C:0,0"].append(pair)
        localDataset.expIdMask["C:0,0"] = np.array([True, False, True])
        self.assertTrue(np.all(localDataset.getExpIdsUsed("C:0,0") == [(12, 34), (90, 10)]))

        localDataset.expIdMask["C:0,0"] = np.array([True, False, True, True])  # wrong length now
        with self.assertRaises(AssertionError):
            localDataset.getExpIdsUsed("C:0,0")

    def test_getGoodAmps(self):
        dataset = self.dataset

        self.assertTrue(dataset.ampNames == self.ampNames)
        dataset.badAmps.append("C:0,1")
        self.assertTrue(dataset.getGoodAmps() == [amp for amp in self.ampNames if amp != "C:0,1"])


class MeasurePhotonTransferCurveDatasetTestCase(lsst.utils.tests.TestCase):
    def setUp(self):
        self.ptcData = PhotonTransferCurveDataset(['C00', 'C01'], " ")
        self.ptcData.inputExpIdPairs = {'C00': [(123, 234), (345, 456), (567, 678)],
                                        'C01': [(123, 234), (345, 456), (567, 678)]}

    def test_generalBehaviour(self):
        test = PhotonTransferCurveDataset(['C00', 'C01'], " ")
        test.inputExpIdPairs = {'C00': [(123, 234), (345, 456), (567, 678)],
                                'C01': [(123, 234), (345, 456), (567, 678)]}


class TestMemory(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
