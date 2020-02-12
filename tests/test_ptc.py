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


class MeasurePhotonTransferCurveTaskTestCase(lsst.utils.tests.TestCase):
    """A test case for the PTC task."""

    def setUp(self):
        self.defaultConfig = cpPipe.ptc.MeasurePhotonTransferCurveTask.ConfigClass()
        self.defaultConfig.isr.doFlat = False
        self.defaultConfig.isr.doFringe = False
        self.defaultConfig.isr.doCrosstalk = False
        self.defaultConfig.isr.doAddDistortionModel = False
        self.defaultConfig.isr.doUseOpticsTransmission = False
        self.defaultConfig.isr.doUseFilterTransmission = False
        self.defaultConfig.isr.doUseSensorTransmission = False
        self.defaultConfig.isr.doUseAtmosphereTransmission = False
        self.defaultConfig.isr.doAttachTransmissionCurve = False

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
        flux = 1000  # ADU/sec
        timeVec = np.arange(1., 201.)
        muVec = flux*timeVec  # implies that signal-chain non-linearity is zero
        self.gain = 1.5  # e-/ADU
        self.c1 = 1./self.gain
        self.noiseSq = 5*self.gain  # 7.5 (e-)^2
        self.a00 = -1.2e-6
        self.c2 = -1.5e-6
        self.c3 = -4.7e-12  # tuned so that it turns over for 200k mean

        self.ampNames = [amp.getName() for amp in self.flatExp1.getDetector().getAmplifiers()]
        self.dataset = PhotonTransferCurveDataset(self.ampNames)  # pack raw data for fitting

        for ampName in self.ampNames:  # just the expTimes and means here - vars vary per function
            self.dataset.rawExpTimes[ampName] = timeVec
            self.dataset.rawMeans[ampName] = muVec

    def test_ptcFitQuad(self):
        localDataset = copy.copy(self.dataset)
        for ampName in self.ampNames:
            localDataset.rawVars[ampName] = [self.noiseSq + self.c1*mu + self.c2*mu**2 for
                                             mu in localDataset.rawMeans[ampName]]

        config = copy.copy(self.defaultConfig)
        config.polynomialFitDegree = 2
        task = cpPipe.ptc.MeasurePhotonTransferCurveTask(config=config)

        numberAmps = len(self.ampNames)
        numberAduValues = config.maxAduForLookupTableLinearizer
        lookupTableArray = np.zeros((numberAmps, numberAduValues), dtype=np.float32)

        task.fitPtcAndNonLinearity(localDataset, lookupTableArray, ptcFitType='POLYNOMIAL')

        for ampName in self.ampNames:
            self.assertAlmostEqual(self.gain, localDataset.gain[ampName])
            self.assertAlmostEqual(np.sqrt(self.noiseSq)*self.gain, localDataset.noise[ampName])
            # Linearity residual should be zero
            self.assertTrue(localDataset.nonLinearityResiduals[ampName].all() == 0)

    def test_ptcFitCubic(self):
        localDataset = copy.copy(self.dataset)
        for ampName in self.ampNames:
            localDataset.rawVars[ampName] = [self.noiseSq + self.c1*mu + self.c2*mu**2 + self.c3*mu**3 for
                                             mu in localDataset.rawMeans[ampName]]

        config = copy.copy(self.defaultConfig)
        config.polynomialFitDegree = 3

        task = cpPipe.ptc.MeasurePhotonTransferCurveTask(config=config)

        numberAmps = len(self.ampNames)
        numberAduValues = config.maxAduForLookupTableLinearizer
        lookupTableArray = np.zeros((numberAmps, numberAduValues), dtype=np.float32)

        task.fitPtcAndNonLinearity(localDataset, lookupTableArray, ptcFitType='POLYNOMIAL')

        for ampName in self.ampNames:
            self.assertAlmostEqual(self.gain, localDataset.gain[ampName])
            self.assertAlmostEqual(np.sqrt(self.noiseSq)*self.gain, localDataset.noise[ampName])
            # Linearity residual should be zero
            self.assertTrue(localDataset.nonLinearityResiduals[ampName].all() == 0)

    def test_ptcFitAstier(self):
        localDataset = copy.copy(self.dataset)
        g = self.gain  # next line is too long without this shorthand!
        for ampName in self.ampNames:
            localDataset.rawVars[ampName] = [(0.5/(self.a00*g**2)*(np.exp(2*self.a00*mu*g)-1) +
                                              self.noiseSq/(g*g)) for mu in localDataset.rawMeans[ampName]]

        config = copy.copy(self.defaultConfig)
        task = cpPipe.ptc.MeasurePhotonTransferCurveTask(config=config)

        numberAmps = len(self.ampNames)
        numberAduValues = config.maxAduForLookupTableLinearizer
        lookupTableArray = np.zeros((numberAmps, numberAduValues), dtype=np.float32)

        task.fitPtcAndNonLinearity(localDataset, lookupTableArray, ptcFitType='ASTIERAPPROXIMATION')

        for ampName in self.ampNames:
            self.assertAlmostEqual(self.gain, localDataset.gain[ampName])
            #  noise already comes out of the fit in electrons with Astier
            self.assertAlmostEqual(np.sqrt(self.noiseSq), localDataset.noise[ampName])
            # Linearity residual should be zero
            self.assertTrue(localDataset.nonLinearityResiduals[ampName].all() == 0)

    def test_meanVarMeasurement(self):
        task = self.defaultTask
        mu, varDiff = task.measureMeanVarPair(self.flatExp1, self.flatExp2)

        self.assertLess(self.flatWidth - np.sqrt(varDiff), 1)
        self.assertLess(self.flatMean - mu, 1)

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
        self.ptcData = PhotonTransferCurveDataset(['C00', 'C01'])
        self.ptcData.inputVisitPairs = {'C00': [(123, 234), (345, 456), (567, 678)],
                                        'C01': [(123, 234), (345, 456), (567, 678)]}

    def test_generalBehaviour(self):
        test = PhotonTransferCurveDataset(['C00', 'C01'])
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
