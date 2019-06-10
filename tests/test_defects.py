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
from lsst.afw.geom import Box2I, Point2I, Extent2I
from lsst.cp.pipe.utils import countMaskedPixels


class FindDefectsTaskTestCase(lsst.utils.tests.TestCase):
    """A test case for the defect finding task."""

    def setUp(self):
        self.defaultConfig = cpPipe.defects.FindDefectsTask.ConfigClass()

        for config in [self.defaultConfig.isrForDarks, self.defaultConfig.isrForFlats]:
            config.doCrosstalk = False
            config.doAddDistortionModel = False
            config.doUseOpticsTransmission = False
            config.doUseFilterTransmission = False
            config.doUseSensorTransmission = False
            config.doUseAtmosphereTransmission = False
            config.doAttachTransmissionCurve = False

        self.flatMean = 2000
        self.darkMean = 1
        self.readNoiseAdu = 10
        self.nSigmaBright = 8
        self.nSigmaDark = 8

        mockImageConfig = isrMock.IsrMock.ConfigClass()

        # flatDrop is not really relevant as we replace the data
        # but good to note it in case we change how this image is made
        mockImageConfig.flatDrop = 0.99999
        mockImageConfig.isTrimmed = True

        self.flatExp = isrMock.FlatMock(config=mockImageConfig).run()
        (shapeY, shapeX) = self.flatExp.getDimensions()

        # x, y, size tuples
        # always put edge defects at the start and change the value of nEdge
        self.brightDefects = [(0, 15, 3), (100, 123, 1), (77, 90, 3)]
        self.darkDefects = [(25, 0, 1), (33, 62, 2), (52, 21, 2)]
        nEdge = 1  # NB update if more edge defects are included
        self.noEdges = slice(nEdge, None)
        self.onlyEdges = slice(0, nEdge)

        self.darkBBoxes = [Box2I(Point2I(x, y), Extent2I(s, s)) for (x, y, s) in self.darkDefects]
        self.brightBBoxes = [Box2I(Point2I(x, y), Extent2I(s, s)) for (x, y, s) in self.brightDefects]

        flatWidth = np.sqrt(self.flatMean) + self.readNoiseAdu
        darkWidth = self.readNoiseAdu
        self.rng = np.random.RandomState(0)
        flatData = self.rng.normal(self.flatMean, flatWidth, (shapeX, shapeY))
        darkData = self.rng.normal(self.darkMean, darkWidth, (shapeX, shapeY))

        # NB darks and flats have same defects applied deliberately to both
        for defect in self.brightDefects:
            y, x, s = defect
            flatData[x:x+s, y:y+s] += self.nSigmaBright * flatWidth  # are these actually the numbers we want?
            darkData[x:x+s, y:y+s] += self.nSigmaBright * darkWidth  # are these actually the numbers we want?

        for defect in self.darkDefects:
            y, x, s = defect
            flatData[x:x+s, y:y+s] -= self.nSigmaDark * flatWidth  # are these actually the numbers we want?
            darkData[x:x+s, y:y+s] -= self.nSigmaDark * darkWidth  # are these actually the numbers we want?

        self.darkExp = self.flatExp.clone()
        self.spareImage = self.flatExp.clone()  # for testing edge bits and misc

        self.flatExp.image.array[:] = flatData
        self.darkExp.image.array[:] = darkData

        self.defaultTask = cpPipe.defects.FindDefectsTask(config=self.defaultConfig)

    def tearDown(self):
        del self

    def test_defectFindingAllSensor(self):
        config = copy.copy(self.defaultConfig)
        config.nPixBorderUpDown = 0
        config.nPixBorderLeftRight = 0
        task = cpPipe.defects.FindDefectsTask(config=config)

        defects = task.findHotAndColdPixels(self.flatExp, 'flat')

        allBBoxes = self.darkBBoxes + self.brightBBoxes

        for defect in defects:
            self.assertIn(defect.getBBox(), allBBoxes)

    def test_defectFindingEdgeIgnore(self):
        task = cpPipe.defects.FindDefectsTask(config=self.defaultConfig)
        defects = task.findHotAndColdPixels(self.flatExp, 'flat')

        shouldBeFound = self.darkBBoxes[self.noEdges] + self.brightBBoxes[self.noEdges]
        for defect in defects:
            self.assertIn(defect.getBBox(), shouldBeFound)

        shouldBeMissed = self.darkBBoxes[self.onlyEdges] + self.brightBBoxes[self.onlyEdges]
        for defect in defects:
            self.assertNotIn(defect.getBBox(), shouldBeMissed)

    def test_postProcessDefectSets(self):
        defects = self.defaultTask.findHotAndColdPixels(self.flatExp, 'flat')

        # minimal test
        merged = self.defaultTask._postProcessDefectSets([defects], self.flatExp.getDimensions())
        self.assertEqual(defects, merged)

        # should always be true regardless of config
        merged = self.defaultTask._postProcessDefectSets([defects, defects], self.flatExp.getDimensions())
        self.assertEqual(defects, merged)

        # now start manipulating defect lists
        config = copy.copy(self.defaultConfig)
        config.combinationMode = 'FRACTION'
        config.combinationFraction = 0.85
        task = cpPipe.defects.FindDefectsTask(config=config)
        merged = task._postProcessDefectSets([defects, defects], self.flatExp.getDimensions())

        defectList = [defects]*10  # 10 identical defect sets
        # remove one defect from one of them, should still be over threshold
        defectList[7] = defectList[7][:-1]
        merged = task._postProcessDefectSets(defectList, self.flatExp.getDimensions())
        self.assertEqual(defects, merged)

        # remove another and should be under threshold
        defectList[3] = defectList[3][:-1]
        merged = task._postProcessDefectSets(defectList, self.flatExp.getDimensions())
        self.assertNotEqual(defects, merged)

        # now test the AND and OR modes - this can be done with a kwarg instead
        # of having to reinstantiate with a new config
        defectList = [defects]*10  # 10 identical defect sets
        merged = task._postProcessDefectSets(defectList, self.flatExp.getDimensions(), mode='AND')
        self.assertEqual(defects, merged)

        defectList[7] = defectList[7][:-1]
        merged = task._postProcessDefectSets(defectList, self.flatExp.getDimensions(), mode='AND')
        self.assertNotEqual(defects, merged)

        merged = task._postProcessDefectSets(defectList, self.flatExp.getDimensions(), mode='OR')
        self.assertEqual(defects, merged)

    def test_pixelCounting(self):
        config = copy.copy(self.defaultConfig)
        config.nPixBorderUpDown = 0
        config.nPixBorderLeftRight = 0
        task = cpPipe.defects.FindDefectsTask(config=config)
        defects = task.findHotAndColdPixels(self.flatExp, 'flat')

        defectArea = 0
        for defect in defects:
            defectArea += defect.getBBox().getArea()

        crossCheck = 0
        for x, y, s in self.brightDefects:
            crossCheck += s**2
        for x, y, s in self.darkDefects:
            crossCheck += s**2

        self.assertEqual(defectArea, task._nPixFromDefects(defects))  # the real function being tested
        self.assertEqual(defectArea, crossCheck)  # but calculate the sum two ways

    def test_getNumGoodPixels(self):
        testImage = self.flatExp.clone()
        mi = testImage.maskedImage

        imageSize = testImage.getBBox().getArea()
        nGood = self.defaultTask._getNumGoodPixels(mi)

        self.assertEqual(imageSize, nGood)

        NODATABIT = mi.mask.getPlaneBitMask("NO_DATA")

        noDataBox = Box2I(Point2I(31, 49), Extent2I(3, 6))
        testImage.mask[noDataBox] |= NODATABIT

        self.assertEqual(imageSize - noDataBox.getArea(), self.defaultTask._getNumGoodPixels(mi))
        self.assertEqual(imageSize, self.defaultTask._getNumGoodPixels(mi, 'BAD'))  # check for misfire

        testImage.mask[noDataBox] ^= NODATABIT  # XOR to reset what we did
        self.assertEqual(imageSize, nGood)

        BADBIT = mi.mask.getPlaneBitMask("BAD")
        badBox = Box2I(Point2I(85, 98), Extent2I(4, 7))
        testImage.mask[badBox] |= BADBIT

        self.assertEqual(imageSize - badBox.getArea(), self.defaultTask._getNumGoodPixels(mi, 'BAD'))

    def test_edgeMasking(self):
        testImage = self.flatExp.clone()
        mi = testImage.maskedImage

        self.assertEqual(countMaskedPixels(mi, 'EDGE'), 0)
        self.defaultTask._setEdgeBits(mi)

        hEdge = self.defaultConfig.nPixBorderLeftRight
        vEdge = self.defaultConfig.nPixBorderUpDown
        xSize, ySize = mi.getDimensions()

        nEdge = xSize*vEdge*2 + ySize*hEdge*2 - hEdge*vEdge*4

        self.assertEqual(countMaskedPixels(mi, 'EDGE'), nEdge)

    def testImport(self):
        import lsst.cp.pipe as cpPipe  # noqa: F401

    def testClassInstantiation(self):

        defectConfig = cpPipe.defects.FindDefectsTask.ConfigClass()

        for config in [defectConfig.isrForDarks, defectConfig.isrForFlats]:
            config.doCrosstalk = False
            config.doAddDistortionModel = False
            config.doUseOpticsTransmission = False
            config.doUseFilterTransmission = False
            config.doUseSensorTransmission = False
            config.doUseAtmosphereTransmission = False
            config.doAttachTransmissionCurve = False

        defectTask = cpPipe.defects.FindDefectsTask(config=defectConfig)

        del defectTask


class TestMemory(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
