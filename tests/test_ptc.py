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
# import copy

import lsst.utils
import lsst.utils.tests

import lsst.cp.pipe as cpPipe
import lsst.ip.isr.isrMock as isrMock
# from lsst.afw.geom import Box2I, Point2I, Extent2I


class FindDefectsTaskTestCase(lsst.utils.tests.TestCase):
    """A test case for the defect finding task."""

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

        self.flatExp = isrMock.FlatMock(config=mockImageConfig).run()
        (shapeY, shapeX) = self.flatExp.getDimensions()

        flatWidth = np.sqrt(self.flatMean) + self.readNoiseAdu
        self.rng = np.random.RandomState(0)
        flatData = self.rng.normal(self.flatMean, flatWidth, (shapeX, shapeY))

        self.flatExp.image.array[:] = flatData

    def tearDown(self):
        del self


class TestMemory(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
