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
from lsst.cp.pipe.ptc import PhotonTransferCurveDataset
from lsst.cp.pipe.astierCovPtcUtils import fitData
from lsst.cp.pipe.utils import (funcPolynomial, makeMockFlats)


class MeasurePhotonTransferCurveTaskTestCase(lsst.utils.tests.TestCase):
    """A test case for the PTC task."""

    def setUp(self):
        self.defaultConfig = cpPipe.ptc.MeasurePhotonTransferCurveTask.ConfigClass()
        self.defaultTask = cpPipe.ptc.MeasurePhotonTransferCurveTask(config=self.defaultConfig)

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
        timeVec = np.arange(1., 201.)
        self.k2NonLinearity = -5e-6
        muVec = self.flux*timeVec + self.k2NonLinearity*timeVec**2   # quadratic signal-chain non-linearity
        self.gain = 1.5  # e-/ADU
        self.c1 = 1./self.gain
        self.noiseSq = 5*self.gain  # 7.5 (e-)^2
        self.a00 = -1.2e-6
        self.c2 = -1.5e-6
        self.c3 = -4.7e-12  # tuned so that it turns over for 200k mean

        self.ampNames = [amp.getName() for amp in self.flatExp1.getDetector().getAmplifiers()]
        self.dataset = PhotonTransferCurveDataset(self.ampNames, " ")  # pack raw data for fitting

        for ampName in self.ampNames:  # just the expTimes and means here - vars vary per function
            self.dataset.rawExpTimes[ampName] = timeVec
            self.dataset.rawMeans[ampName] = muVec

    def test_covAstier(self):
        """Test to check getCovariancesAstier

        We check that the gain is the same as the imput gain from the mock data, that
        the covariances via FFT (as it is in MeasurePhotonTransferCurveTask when
        doCovariancesAstier=True) are the same as calculated in real space, and that
        Cov[0, 0] (i.e., the variances) are similar to the variances calculated with the standard
        method (when doCovariancesAstier=false),
        """
        localDataset = copy.copy(self.dataset)
        config = copy.copy(self.defaultConfig)
        task = cpPipe.ptc.MeasurePhotonTransferCurveTask(config=config)

        expTimes = np.arange(5, 170, 5)
        tupleRecords = []
        allTags = []
        muStandard, varStandard = {}, {}
        for expTime in expTimes:
            mockExp1, mockExp2 = makeMockFlats(expTime, gain=0.75)
            tupleRows = []

            for ampNumber, amp in enumerate(self.ampNames):
                # cov has (i, j, var, cov, npix)
                muDiff, varDiff, covAstier = task.measureMeanVarCov(mockExp1, mockExp2)
                muStandard.setdefault(amp, []).append(muDiff)
                varStandard.setdefault(amp, []).append(varDiff)
                # Calculate covariances in an independent way: direct space
                _, _, covsDirect = task.measureMeanVarCov(mockExp1, mockExp2, covAstierRealSpace=True)

                # Test that the arrays "covs" (FFT) and "covDirect" (direct space) are the same
                for row1, row2 in zip(covAstier, covsDirect):
                    for a, b in zip(row1, row2):
                        self.assertAlmostEqual(a, b)
                tupleRows += [(muDiff, ) + covRow + (ampNumber, expTime, amp) for covRow in covAstier]
                tags = ['mu', 'i', 'j', 'var', 'cov', 'npix', 'ext', 'expTime', 'ampName']
            allTags += tags
            tupleRecords += tupleRows
        covariancesWithTags = np.core.records.fromrecords(tupleRecords, names=allTags)
        covFits, _ = fitData(covariancesWithTags)
        localDataset = task.getOutputPtcDataCovAstier(localDataset, covFits)
        # Chek the gain and that the ratio of the variance caclulated via cov Astier (FFT) and
        # that calculated with the standard PTC calculation (afw) is close to 1.
        for amp in self.ampNames:
            self.assertAlmostEqual(localDataset.gain[amp], 0.75, places=2)
            for v1, v2 in zip(varStandard[amp], localDataset.finalVars[amp][0]):
                v2 *= (0.75**2)  # convert to electrons
                self.assertAlmostEqual(v1/v2, 1.0, places=1)

    def ptcFitAndCheckPtc(self, order=None, fitType='', doTableArray=False, doFitBootstrap=False):
        localDataset = copy.copy(self.dataset)
        config = copy.copy(self.defaultConfig)
        placesTests = 6
        if doFitBootstrap:
            config.doFitBootstrap = True
            # Bootstrap method in cp_pipe/utils.py does multiple fits in the precense of noise.
            # Allow for more margin of error.
            placesTests = 3

        if fitType == 'POLYNOMIAL':
            if order not in [2, 3]:
                RuntimeError("Enter a valid polynomial order for this test: 2 or 3")
            if order == 2:
                for ampName in self.ampNames:
                    localDataset.rawVars[ampName] = [self.noiseSq + self.c1*mu + self.c2*mu**2 for
                                                     mu in localDataset.rawMeans[ampName]]
                config.polynomialFitDegree = 2
            if order == 3:
                for ampName in self.ampNames:
                    localDataset.rawVars[ampName] = [self.noiseSq + self.c1*mu + self.c2*mu**2 + self.c3*mu**3
                                                     for mu in localDataset.rawMeans[ampName]]
                config.polynomialFitDegree = 3
        elif fitType == 'EXPAPPROXIMATION':
            g = self.gain
            for ampName in self.ampNames:
                localDataset.rawVars[ampName] = [(0.5/(self.a00*g**2)*(np.exp(2*self.a00*mu*g)-1) +
                                                 self.noiseSq/(g*g)) for mu in localDataset.rawMeans[ampName]]
        else:
            RuntimeError("Enter a fit function type: 'POLYNOMIAL' or 'EXPAPPROXIMATION'")

        config.maxAduForLookupTableLinearizer = 200000  # Max ADU in input mock flats
        task = cpPipe.ptc.MeasurePhotonTransferCurveTask(config=config)

        if doTableArray:
            # Non-linearity
            numberAmps = len(self.ampNames)
            numberAduValues = config.maxAduForLookupTableLinearizer
            lookupTableArray = np.zeros((numberAmps, numberAduValues), dtype=np.float32)
            # localDataset: PTC dataset (lsst.cp.pipe.ptc.PhotonTransferCurveDataset)
            localDataset = task.fitPtc(localDataset, ptcFitType=fitType)
            # linDataset: Dictionary of `lsst.cp.pipe.ptc.LinearityResidualsAndLinearizersDataset`
            linDataset = task.fitNonLinearity(localDataset, tableArray=lookupTableArray)
        else:
            localDataset = task.fitPtc(localDataset, ptcFitType=fitType)
            linDataset = task.fitNonLinearity(localDataset)

        if doTableArray:
            # check that the linearizer table has been filled out properly
            for i in np.arange(numberAmps):
                tMax = (config.maxAduForLookupTableLinearizer)/self.flux
                timeRange = np.linspace(0., tMax, config.maxAduForLookupTableLinearizer)
                signalIdeal = timeRange*self.flux
                signalUncorrected = funcPolynomial(np.array([0.0, self.flux, self.k2NonLinearity]),
                                                   timeRange)
                linearizerTableRow = signalIdeal - signalUncorrected
                self.assertEqual(len(linearizerTableRow), len(lookupTableArray[i, :]))
                for j in np.arange(len(linearizerTableRow)):
                    self.assertAlmostEqual(linearizerTableRow[j], lookupTableArray[i, :][j],
                                           places=placesTests)

        # check entries in localDataset, which was modified by the function
        for ampName in self.ampNames:
            maskAmp = localDataset.visitMask[ampName]
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
            maskAmp = localDataset.visitMask[ampName]
            finalMuVec = localDataset.rawMeans[ampName][maskAmp]
            finalTimeVec = localDataset.rawExpTimes[ampName][maskAmp]
            linearPart = self.flux*finalTimeVec
            inputFracNonLinearityResiduals = 100*(linearPart - finalMuVec)/linearPart

            # Nonlinearity fit parameters
            self.assertLess(np.fabs(linDataset[ampName].meanSignalVsTimePolyFitPars[0]), 0.01)
            self.assertAlmostEqual(self.flux, linDataset[ampName].meanSignalVsTimePolyFitPars[1],
                                   places=placesTests)
            self.assertAlmostEqual(self.k2NonLinearity, linDataset[ampName].meanSignalVsTimePolyFitPars[2],
                                   places=placesTests)

            # Non-linearity coefficient for linearizer
            self.assertAlmostEqual(-self.k2NonLinearity/(self.flux**2),
                                   linDataset[ampName].quadraticPolynomialLinearizerCoefficient,
                                   places=placesTests)

            linearPartModel = linDataset[ampName].meanSignalVsTimePolyFitPars[1]*finalTimeVec
            outputFracNonLinearityResiduals = 100*(linearPartModel - finalMuVec)/linearPartModel
            # Fractional nonlinearity residuals
            self.assertEqual(len(outputFracNonLinearityResiduals), len(inputFracNonLinearityResiduals))
            for calc, truth in zip(outputFracNonLinearityResiduals, inputFracNonLinearityResiduals):
                self.assertAlmostEqual(calc, truth, places=placesTests)

            # check calls to calculateLinearityResidualAndLinearizers
            datasetLinResAndLinearizers = task.calculateLinearityResidualAndLinearizers(
                localDataset.rawExpTimes[ampName], localDataset.rawMeans[ampName])

            self.assertAlmostEqual(-self.k2NonLinearity/(self.flux**2),
                                   datasetLinResAndLinearizers.quadraticPolynomialLinearizerCoefficient,
                                   places=placesTests)
            self.assertAlmostEqual(0.0, datasetLinResAndLinearizers.meanSignalVsTimePolyFitPars[0],
                                   places=placesTests)
            self.assertAlmostEqual(self.flux, datasetLinResAndLinearizers.meanSignalVsTimePolyFitPars[1],
                                   places=placesTests)
            self.assertAlmostEqual(self.k2NonLinearity,
                                   datasetLinResAndLinearizers.meanSignalVsTimePolyFitPars[2],
                                   places=placesTests)

    def test_ptcFitBootstrap(self):
        """Test the bootstrap fit option for the PTC"""
        for (fitType, order) in [('POLYNOMIAL', 2), ('POLYNOMIAL', 3), ('EXPAPPROXIMATION', None)]:
            self.ptcFitAndCheckPtc(fitType=fitType, order=order, doTableArray=False, doFitBootstrap=True)

    def test_ptcFit(self):
        for createArray in [True, False]:
            for (fitType, order) in [('POLYNOMIAL', 2), ('POLYNOMIAL', 3), ('EXPAPPROXIMATION', None)]:
                self.ptcFitAndCheckPtc(fitType=fitType, order=order, doTableArray=createArray)

    def test_meanVarMeasurement(self):
        task = self.defaultTask
        mu, varDiff, _ = task.measureMeanVarCov(self.flatExp1, self.flatExp2)

        self.assertLess(self.flatWidth - np.sqrt(varDiff), 1)
        self.assertLess(self.flatMean - mu, 1)

    def test_meanVarMeasurementWithNans(self):
        task = self.defaultTask
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

        # Dive by two as it is what measureMeanVarCov returns (variance of difference)
        expectedVar = 0.5*np.nanvar(diffIm.image.array)

        # Check that the standard deviations and the emans agree to less than 1 ADU
        self.assertLess(np.sqrt(expectedVar) - np.sqrt(varDiff), 1)
        self.assertLess(expectedMu - mu, 1)

    def test_meanVarMeasurementAllNan(self):
        task = self.defaultTask
        self.flatExp1.image.array[:, :] = np.nan
        self.flatExp2.image.array[:, :] = np.nan

        mu, varDiff, covDiff = task.measureMeanVarCov(self.flatExp1, self.flatExp2)

        self.assertTrue(np.isnan(mu))
        self.assertTrue(np.isnan(varDiff))
        self.assertTrue(covDiff is None)

    def test_getInitialGoodPoints(self):
        xs = [1, 2, 3, 4, 5, 6]
        ys = [2*x for x in xs]
        points = self.defaultTask._getInitialGoodPoints(xs, ys, 0.1, 0.25)
        assert np.all(points) == np.all(np.array([True for x in xs]))

        ys[-1] = 30
        points = self.defaultTask._getInitialGoodPoints(xs, ys, 0.1, 0.25)
        assert np.all(points) == np.all(np.array([True, True, True, True, False]))

        ys = [2*x for x in xs]
        newYs = copy.copy(ys)
        results = [False, True, True, False, False]
        for i, factor in enumerate([-0.5, -0.1, 0, 0.1, 0.5]):
            newYs[-1] = ys[-1] + (factor*ys[-1])
            points = self.defaultTask._getInitialGoodPoints(xs, newYs, 0.05, 0.25)
            assert (np.all(points[0:-1]) == True)  # noqa: E712 - flake8 is wrong here because of numpy.bool
            assert points[-1] == results[i]

    def test_getVisitsUsed(self):
        localDataset = copy.copy(self.dataset)

        for pair in [(12, 34), (56, 78), (90, 10)]:
            localDataset.inputVisitPairs["C:0,0"].append(pair)
        localDataset.visitMask["C:0,0"] = np.array([True, False, True])
        self.assertTrue(np.all(localDataset.getVisitsUsed("C:0,0") == [(12, 34), (90, 10)]))

        localDataset.visitMask["C:0,0"] = np.array([True, False, True, True])  # wrong length now
        with self.assertRaises(AssertionError):
            localDataset.getVisitsUsed("C:0,0")

    def test_getGoodAmps(self):
        dataset = self.dataset

        self.assertTrue(dataset.ampNames == self.ampNames)
        dataset.badAmps.append("C:0,1")
        self.assertTrue(dataset.getGoodAmps() == [amp for amp in self.ampNames if amp != "C:0,1"])


class MeasurePhotonTransferCurveDatasetTestCase(lsst.utils.tests.TestCase):
    def setUp(self):
        self.ptcData = PhotonTransferCurveDataset(['C00', 'C01'], " ")
        self.ptcData.inputVisitPairs = {'C00': [(123, 234), (345, 456), (567, 678)],
                                        'C01': [(123, 234), (345, 456), (567, 678)]}

    def test_generalBehaviour(self):
        test = PhotonTransferCurveDataset(['C00', 'C01'], " ")
        test.inputVisitPairs = {'C00': [(123, 234), (345, 456), (567, 678)],
                                'C01': [(123, 234), (345, 456), (567, 678)]}

        with self.assertRaises(AttributeError):
            test.newItem = 1


class TestMemory(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
