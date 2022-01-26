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
"""Test cases for lsst.cp.pipe.utils"""

from __future__ import absolute_import, division, print_function
import unittest
import numpy as np

import lsst.utils
import lsst.utils.tests

from lsst.geom import Box2I, Point2I, Extent2I
from lsst.ip.isr import isrMock

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


class TestMemory(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
