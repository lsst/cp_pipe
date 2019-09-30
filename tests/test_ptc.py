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
        timeVec = np.arange(0., 201.)
        muVec = flux*timeVec  # implies that signal-chain non-linearity is zero
        self.gain = 1.5  # e-/ADU
        c1 = 1./self.gain
        self.noiseSq = 5*self.gain  # 7.5 (e-)^2
        a00 = -1.2e-6
        c2 = -1.5e-6
        c3 = 1.7e-11

        self.fitVectorsQuadDict = {'amp': ([], [], [])}
        self.fitVectorsCubicDict = {'amp': ([], [], [])}
        self.fitVectorsAstierDict = {'amp': ([], [], [])}

        for (t, mu) in zip(timeVec, muVec):
            varQuad = self.noiseSq + c1*mu + c2*mu**2
            varCubic = self.noiseSq + c1*mu + c2*mu**2 + c3*mu**3
            varAstier = (0.5/(a00*self.gain*self.gain)*(np.exp(2*a00*mu*self.gain)-1) +
                         self.noiseSq/(self.gain*self.gain))

            self.fitVectorsQuadDict['amp'][0].append(t)
            self.fitVectorsQuadDict['amp'][1].append(mu)
            self.fitVectorsQuadDict['amp'][2].append(varQuad)

            self.fitVectorsCubicDict['amp'][0].append(t)
            self.fitVectorsCubicDict['amp'][1].append(mu)
            self.fitVectorsCubicDict['amp'][2].append(varCubic)

            self.fitVectorsAstierDict['amp'][0].append(t)
            self.fitVectorsAstierDict['amp'][1].append(mu)
            self.fitVectorsAstierDict['amp'][2].append(varAstier)

    def test_ptcFitQuad(self):
        config = copy.copy(self.defaultConfig)
        config.polynomialFitDegree = 2
        task = cpPipe.ptc.MeasurePhotonTransferCurveTask(config=config)

        _, nlDict, gainDict, noiseDict = task.fitPtcAndNl(self.fitVectorsQuadDict,
                                                          ptcFitType='POLYNOMIAL')

        self.assertAlmostEqual(self.gain, gainDict['amp'][0])
        self.assertAlmostEqual(np.sqrt(self.noiseSq)*self.gain, noiseDict['amp'][0])
        # Linearity residual should be zero
        # nlDict[amp] = (timeVecFinal, meanVecFinal, linResidual, parsFit, parsFitErr)
        self.assertTrue(nlDict['amp'][2].all() == 0)

    def test_ptcFitCubic(self):
        config = copy.copy(self.defaultConfig)
        config.polynomialFitDegree = 3
        task = cpPipe.ptc.MeasurePhotonTransferCurveTask(config=config)
        _, nlDict, gainDict, noiseDict = task.fitPtcAndNl(self.fitVectorsCubicDict,
                                                          ptcFitType='POLYNOMIAL')
        self.assertAlmostEqual(self.gain, gainDict['amp'][0])
        self.assertAlmostEqual(np.sqrt(self.noiseSq)*self.gain, noiseDict['amp'][0])
        self.assertTrue(nlDict['amp'][2].all() == 0)

    def test_ptcFitAstier(self):
        task = self.defaultTask

        _, nlDict, gainDict, noiseDict = task.fitPtcAndNl(self.fitVectorsAstierDict,
                                                          ptcFitType='ASTIERAPPROXIMATION')

        self.assertAlmostEqual(self.gain, gainDict['amp'][0])
        #  noise already comes out of the fit in electrons
        self.assertAlmostEqual(np.sqrt(self.noiseSq), noiseDict['amp'][0])
        self.assertTrue(nlDict['amp'][2].all() == 0)

    def test_meanVarMeasurement(self):
        task = self.defaultTask
        mu, varDiff = task.measureMeanVarPair(self.flatExp1, self.flatExp2)

        self.assertLess(self.flatWidth - np.sqrt(varDiff), 1)
        self.assertLess(self.flatMean - mu, 1)


class TestMemory(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
