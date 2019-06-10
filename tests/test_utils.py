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

import lsst.utils
import lsst.utils.tests

import lsst.ip.isr.isrMock as isrMock
import lsst.ip.isr as ipIsr
import lsst.afw.image as afwImage
from lsst.afw.geom import Box2I, Point2I, Extent2I
import lsst.log as lsstLog

import lsst.cp.pipe.utils as cpUtils


class UtilsTestCase(lsst.utils.tests.TestCase):
    """A test case for the utility functions for cp_pipe."""

    def setUp(self):

        mockImageConfig = isrMock.IsrMock.ConfigClass()

        # flatDrop is not really relevant as we replace the data
        # but good to note it in case we change how this image is made
        mockImageConfig.flatDrop = 0.99999
        mockImageConfig.isTrimmed = True

        self.flatExp = isrMock.FlatMock(config=mockImageConfig).run()
        (shapeY, shapeX) = self.flatExp.getDimensions()

        self.rng = np.random.RandomState(0)
        self.flatMean = 1000
        self.flatWidth = np.sqrt(self.flatMean)
        flatData = self.rng.normal(self.flatMean, self.flatWidth, (shapeX, shapeY))
        self.flatExp.image.array[:] = flatData

    def test_countMaskedPixels(self):
        exp = self.flatExp.clone()
        mi = exp.maskedImage
        self.assertEqual(cpUtils.countMaskedPixels(mi, 'NO_DATA'), 0)
        self.assertEqual(cpUtils.countMaskedPixels(mi, 'BAD'), 0)

        NODATABIT = mi.mask.getPlaneBitMask("NO_DATA")
        noDataBox = Box2I(Point2I(31, 49), Extent2I(3, 6))
        mi.mask[noDataBox] |= NODATABIT

        self.assertEqual(cpUtils.countMaskedPixels(mi, 'NO_DATA'), noDataBox.getArea())
        self.assertEqual(cpUtils.countMaskedPixels(mi, 'BAD'), 0)

        mi.mask[noDataBox] ^= NODATABIT  # XOR to reset what we did
        self.assertEqual(cpUtils.countMaskedPixels(mi, 'NO_DATA'), 0)

    def test_parseCmdlineNumberString(self):
        parsed = cpUtils.parseCmdlineNumberString('1..5')
        self.assertEqual(parsed, [1, 2, 3, 4, 5])

        parsed = cpUtils.parseCmdlineNumberString('1..5:2^123..126')
        self.assertEqual(parsed, [1, 3, 5, 123, 124, 125, 126])

        parsed = cpUtils.parseCmdlineNumberString('12^23^34^43^987')
        self.assertEqual(parsed, [12, 23, 34, 43, 987])

        parsed = cpUtils.parseCmdlineNumberString('12^23^34^43^987..990')
        self.assertEqual(parsed, [12, 23, 34, 43, 987, 988, 989, 990])

    def test_checkExpLengthEqual(self):
        exp1 = self.flatExp.clone()
        exp2 = self.flatExp.clone()

        self.assertTrue(cpUtils.checkExpLengthEqual(exp1, exp2))

        visitInfo = afwImage.VisitInfo(exposureTime=-1, darkTime=1)
        exp2.getInfo().setVisitInfo(visitInfo)
        self.assertFalse(cpUtils.checkExpLengthEqual(exp1, exp2))

        with self.assertRaises(RuntimeError):
            cpUtils.checkExpLengthEqual(exp1, exp2, raiseWithMessage=True)

    def test_validateIsrConfig(self):

        mand = ['doAssembleCcd']
        forb = ['doFlat', 'doFringe']
        des = ['doBias', 'doDark']
        undes = ['doLinearize', 'doBrighterFatter']

        passingConfig = ipIsr.IsrTask.ConfigClass()
        for item in (mand + des):
            setattr(passingConfig, item, True)
        for item in (forb + undes):
            setattr(passingConfig, item, False)

        task = ipIsr.IsrTask(config=passingConfig)

        with self.assertRaises(TypeError):
            cpUtils.validateIsrConfig(None, mand, forb, des, undes)
            cpUtils.validateIsrConfig(passingConfig, mand, forb, des, undes)

        with self.assertRaises(RuntimeError):  # mand/forb swapped
            cpUtils.validateIsrConfig(task, forb, mand, des, undes)

        with self.assertRaises(RuntimeError):  # raise for missing mandatory
            cpUtils.validateIsrConfig(task, mand + ['test'], forb, des, undes)

        logName = 'testLogger'
        with lsstLog.UsePythonLogging():  # otherwise none of this is caught
            with self.assertLogs(logName, level='INFO'):  # not found info-logs for (un)desirable
                cpUtils.validateIsrConfig(task, mand, forb, des + ['test'], undes, logName=logName)
                cpUtils.validateIsrConfig(task, mand, forb, des, undes + ['test'], logName=logName)

            with self.assertLogs(logName, "WARN"):  # not found warnings for forbidden
                cpUtils.validateIsrConfig(task, mand, forb + ['test'], des, undes, logName=logName)


class TestMemory(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
