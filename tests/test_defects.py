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
"""Test cases for lsst.cp.pipe.FindDefectsTask."""

import unittest
import numpy as np
import copy

import lsst.utils
import lsst.utils.tests

import lsst.cp.pipe as cpPipe
from lsst.cp.pipe.utils import countMaskedPixels
from lsst.ip.isr import isrMock
from lsst.geom import Box2I, Point2I, Extent2I
import lsst.meas.algorithms as measAlg


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

        self.brightDefects = [(0, 15, 3, 3), (100, 123, 1, 1),
                              (15, 1, 1, 50),
                              (20, 1, 1, 25),
                              (25, 1, 1, 8),
                              (30, 1, 1, 2), (30, 5, 1, 3), (30, 11, 1, 5), (30, 19, 1, 5),
                              (30, 27, 1, 4), (30, 34, 1, 15),
                              (35, 1, 1, 2), (35, 5, 1, 3), (35, 11, 1, 2),
                              (40, 1, 1, 2), (40, 5, 1, 3), (40, 19, 1, 5), (40, 27, 1, 4), (40, 34, 1, 15),
                              (45, 10, 1, 2), (45, 30, 1, 3),
                              [50, 10, 1, 1], [50, 12, 1, 1], [50, 14, 1, 1], [50, 16, 1, 1],
                              [50, 18, 1, 1], [50, 20, 1, 1], [50, 22, 1, 1], [50, 24, 1, 1],
                              [50, 26, 1, 1], [50, 28, 1, 1], [50, 30, 1, 1], [50, 32, 1, 1],
                              [50, 34, 1, 1], [50, 36, 1, 1], [50, 38, 1, 1], [50, 40, 1, 1],
                              [55, 20, 1, 1], [55, 22, 1, 1], [55, 24, 1, 1], [55, 26, 1, 1],
                              [55, 28, 1, 1], [55, 30, 1, 1],
                              (60, 1, 1, 18), (60, 20, 1, 10), (61, 2, 2, 2), (61, 6, 2, 8),
                              (70, 1, 1, 18), (70, 20, 1, 10), (68, 2, 2, 2), (68, 6, 2, 8),
                              (75, 1, 1, 18), (75, 20, 1, 10), (73, 2, 2, 2), (73, 6, 2, 8), (76, 2, 2, 2),
                              (76, 6, 2, 8),
                              (80, 1, 1, 18), (80, 20, 1, 10), (81, 2, 2, 2), (81, 8, 2, 8),
                              (87, 1, 1, 18), (87, 20, 1, 10), (85, 2, 2, 2), (85, 8, 2, 8),
                              (93, 1, 1, 12), (93, 15, 1, 20), (91, 2, 2, 2), (91, 7, 2, 2),
                              (94, 2, 2, 2), (94, 7, 2, 2),
                              (91, 18, 2, 3), (91, 24, 2, 3), (94, 18, 2, 3), (94, 24, 2, 3)]

        self.darkDefects = [(5, 0, 1, 1), (7, 62, 2, 2)]

        nEdge = 1  # NOTE: update if more edge defects are included
        self.noEdges = slice(nEdge, None)
        self.onlyEdges = slice(0, nEdge)

        self.darkBBoxes = [Box2I(Point2I(x, y), Extent2I(sx, sy)) for (x, y, sx, sy) in self.darkDefects]
        self.brightBBoxes = [Box2I(Point2I(x, y), Extent2I(sx, sy)) for (x, y, sx, sy) in self.brightDefects]

        flatWidth = np.sqrt(self.flatMean) + self.readNoiseAdu
        darkWidth = self.readNoiseAdu
        self.rng = np.random.RandomState(0)
        flatData = self.rng.normal(self.flatMean, flatWidth, (shapeX, shapeY))
        darkData = self.rng.normal(self.darkMean, darkWidth, (shapeX, shapeY))

        # NOTE: darks and flats have same defects applied deliberately to both
        for defect in self.brightDefects:
            y, x, sy, sx = defect
            # are these actually the numbers we want?
            flatData[x:x+sx, y:y+sy] += self.nSigmaBright * flatWidth
            darkData[x:x+sx, y:y+sy] += self.nSigmaBright * darkWidth

        for defect in self.darkDefects:
            y, x, sy, sx = defect
            # are these actually the numbers we want?
            flatData[x:x+sx, y:y+sy] -= self.nSigmaDark * flatWidth
            darkData[x:x+sx, y:y+sy] -= self.nSigmaDark * darkWidth

        self.darkExp = self.flatExp.clone()
        self.spareImage = self.flatExp.clone()  # for testing edge bits and misc

        self.flatExp.image.array[:] = flatData
        self.darkExp.image.array[:] = darkData

        self.defaultTask = cpPipe.defects.FindDefectsTask(config=self.defaultConfig)

        self.allDefectsList = measAlg.Defects()

        self.brightDefectsList = measAlg.Defects()
        for d in self.brightBBoxes:
            self.brightDefectsList.append(d)
            self.allDefectsList.append(d)

        self.darkDefectsList = measAlg.Defects()
        for d in self.darkBBoxes:
            self.darkDefectsList.append(d)
            self.allDefectsList.append(d)

    def test_maskBlocksIfIntermitentBadPixelsInColumnTest1(self):
        # continuous bad column

        config = copy.copy(self.defaultConfig)
        config.badOnAndOffPixelColumnThreshold = 10
        config.goodPixelColumnGapThreshold = 5
        config.nPixBorderUpDown = 0
        config.nPixBorderLeftRight = 0

        task = cpPipe.defects.FindDefectsTask(config=config)

        expectedDefects = [Box2I(corner = Point2I(15, 1), dimensions = Extent2I(1, 50))]
        defects = self.allDefectsList

        defectsWithColumns = task.maskBlocksIfIntermitentBadPixelsInColumn(defects)

        boxesMeasured = []
        for defect in defectsWithColumns:
            boxesMeasured.append(defect.getBBox())

        for boxInput in expectedDefects:
            self.assertIn(boxInput, boxesMeasured)

    def test_maskBlocksIfIntermitentBadPixelsInColumnTest2(self):
        # Test 2: n contiguous bad pixels in a column where n >= threshold

        config = copy.copy(self.defaultConfig)
        config.badOnAndOffPixelColumnThreshold = 10
        config.goodPixelColumnGapThreshold = 5
        config.nPixBorderUpDown = 0
        config.nPixBorderLeftRight = 0

        task = cpPipe.defects.FindDefectsTask(config=config)

        expectedDefects = [Box2I(corner = Point2I(20, 1), dimensions = Extent2I(1, 25))]
        defects = self.allDefectsList

        defectsWithColumns = task.maskBlocksIfIntermitentBadPixelsInColumn(defects)

        boxesMeasured = []
        for defect in defectsWithColumns:
            boxesMeasured.append(defect.getBBox())

        for boxInput in expectedDefects:
            self.assertIn(boxInput, boxesMeasured)

    def test_maskBlocksIfIntermitentBadPixelsInColumnTest3(self):
        # Test 3: n contiguous bad pixels in a column where n < threshold

        config = copy.copy(self.defaultConfig)
        config.badOnAndOffPixelColumnThreshold = 10
        config.goodPixelColumnGapThreshold = 5
        config.nPixBorderUpDown = 0
        config.nPixBorderLeftRight = 0

        task = cpPipe.defects.FindDefectsTask(config=config)

        expectedDefects = [Box2I(corner = Point2I(25, 1), dimensions = Extent2I(1, 8))]
        defects = self.allDefectsList

        defectsWithColumns = task.maskBlocksIfIntermitentBadPixelsInColumn(defects)

        boxesMeasured = []
        for defect in defectsWithColumns:
            boxesMeasured.append(defect.getBBox())

        for boxInput in expectedDefects:
            self.assertIn(boxInput, boxesMeasured)

    def test_maskBlocksIfIntermitentBadPixelsInColumnTest4(self):
        # Test 4: n discontiguous bad pixels in a column where n >= threshold,
        # gap < "good" threshold  (n=34 >= 10)

        config = copy.copy(self.defaultConfig)
        config.badOnAndOffPixelColumnThreshold = 10
        config.goodPixelColumnGapThreshold = 5
        config.nPixBorderUpDown = 0
        config.nPixBorderLeftRight = 0

        task = cpPipe.defects.FindDefectsTask(config=config)

        expectedDefects = [Box2I(corner = Point2I(30, 1), dimensions = Extent2I(1, 48))]
        defects = self.allDefectsList

        defectsWithColumns = task.maskBlocksIfIntermitentBadPixelsInColumn(defects)

        boxesMeasured = []
        for defect in defectsWithColumns:
            boxesMeasured.append(defect.getBBox())

        for boxInput in expectedDefects:
            self.assertIn(boxInput, boxesMeasured)

    def test_maskBlocksIfIntermitentBadPixelsInColumnTest5(self):
        # Test 5: n discontiguous bad pixels in a column where n < threshold,
        # gap < "good" threshold (n=7<10)
        # bad_test5 = np.array([(35, 1, 1, 2), (35, 5, 1, 3), (35, 11, 1, 2)])

        config = copy.copy(self.defaultConfig)
        config.badOnAndOffPixelColumnThreshold = 10
        config.goodPixelColumnGapThreshold = 5
        config.nPixBorderUpDown = 0
        config.nPixBorderLeftRight = 0

        task = cpPipe.defects.FindDefectsTask(config=config)

        expectedDefects = [Box2I(corner = Point2I(35, 1), dimensions = Extent2I(1, 2)),
                           Box2I(corner = Point2I(35, 5), dimensions = Extent2I(1, 3)),

                           Box2I(corner = Point2I(35, 11), dimensions = Extent2I(1, 2))]
        defects = self.allDefectsList

        defectsWithColumns = task.maskBlocksIfIntermitentBadPixelsInColumn(defects)

        boxesMeasured = []
        for defect in defectsWithColumns:
            boxesMeasured.append(defect.getBBox())

        for boxInput in expectedDefects:
            self.assertIn(boxInput, boxesMeasured)

    def test_maskBlocksIfIntermitentBadPixelsInColumnTest6(self):
        # Test 6: n discontiguous bad pixels in a column where n >= threshold, gap >= "good" threshold
        # n=34 bad pixels total, 1 "good" gap big enough (13>=5 good pixels, from y=6 (1+5) to y=19)

        config = copy.copy(self.defaultConfig)
        config.badOnAndOffPixelColumnThreshold = 10
        config.goodPixelColumnGapThreshold = 5
        config.nPixBorderUpDown = 0
        config.nPixBorderLeftRight = 0

        task = cpPipe.defects.FindDefectsTask(config=config)

        expectedDefects = [Box2I(corner = Point2I(40, 1), dimensions = Extent2I(1, 7)),
                           Box2I(corner = Point2I(40, 19), dimensions = Extent2I(1, 30))]
        defects = self.allDefectsList

        defectsWithColumns = task.maskBlocksIfIntermitentBadPixelsInColumn(defects)

        boxesMeasured = []
        for defect in defectsWithColumns:
            boxesMeasured.append(defect.getBBox())

        for boxInput in expectedDefects:
            self.assertIn(boxInput, boxesMeasured)

    def test_maskBlocksIfIntermitentBadPixelsInColumnTest7(self):
        # Test 7: n discontiguous bad pixels in a column where n < threshold, gap >= "good" threshold
        # 5<10 bad pixels total, 1 "good" gap big enough (29>=5 good pixels, from y =12 (10+2) to y=30)

        config = copy.copy(self.defaultConfig)
        config.badOnAndOffPixelColumnThreshold = 10
        config.goodPixelColumnGapThreshold = 5
        config.nPixBorderUpDown = 0
        config.nPixBorderLeftRight = 0

        task = cpPipe.defects.FindDefectsTask(config=config)

        expectedDefects = [Box2I(corner = Point2I(45, 10), dimensions = Extent2I(1, 2)),
                           Box2I(corner = Point2I(45, 30), dimensions = Extent2I(1, 3))]
        defects = self.allDefectsList

        defectsWithColumns = task.maskBlocksIfIntermitentBadPixelsInColumn(defects)

        boxesMeasured = []
        for defect in defectsWithColumns:
            boxesMeasured.append(defect.getBBox())

        for boxInput in expectedDefects:
            self.assertIn(boxInput, boxesMeasured)

    def test_maskBlocksIfIntermitentBadPixelsInColumnTest8(self):
        # Test 8: n discontiguous bad pixels, every other pixel is bad, n >= threshold
        # n discontiguous bad pixels, every other pixel is bad, n >= threshold (n = 15  >= 10)

        config = copy.copy(self.defaultConfig)
        config.badOnAndOffPixelColumnThreshold = 10
        config.goodPixelColumnGapThreshold = 5
        config.nPixBorderUpDown = 0
        config.nPixBorderLeftRight = 0

        task = cpPipe.defects.FindDefectsTask(config=config)

        expectedDefects = [Box2I(corner = Point2I(50, 10), dimensions = Extent2I(1, 31))]
        defects = self.allDefectsList

        defectsWithColumns = task.maskBlocksIfIntermitentBadPixelsInColumn(defects)

        boxesMeasured = []
        for defect in defectsWithColumns:
            boxesMeasured.append(defect.getBBox())

        for boxInput in expectedDefects:
            self.assertIn(boxInput, boxesMeasured)

    def test_maskBlocksIfIntermitentBadPixelsInColumnTest9(self):
        # Test 9: n discontiguous bad pixels, every other pixel is bad, n < threshold
        # n discontiguous bad pixels, every other pixel is bad, n < threshold (n = 5 < 10)

        config = copy.copy(self.defaultConfig)
        config.badOnAndOffPixelColumnThreshold = 10
        config.goodPixelColumnGapThreshold = 5
        config.nPixBorderUpDown = 0
        config.nPixBorderLeftRight = 0

        task = cpPipe.defects.FindDefectsTask(config=config)

        expectedDefects = [Box2I(corner = Point2I(55, 20), dimensions = Extent2I(1, 1)),
                           Box2I(corner = Point2I(55, 22), dimensions = Extent2I(1, 1)),
                           Box2I(corner = Point2I(55, 24), dimensions = Extent2I(1, 1)),
                           Box2I(corner = Point2I(55, 26), dimensions = Extent2I(1, 1)),
                           Box2I(corner = Point2I(55, 28), dimensions = Extent2I(1, 1)),
                           Box2I(corner = Point2I(55, 30), dimensions = Extent2I(1, 1))]
        defects = self.allDefectsList

        defectsWithColumns = task.maskBlocksIfIntermitentBadPixelsInColumn(defects)

        boxesMeasured = []
        for defect in defectsWithColumns:
            boxesMeasured.append(defect.getBBox())

        for boxInput in expectedDefects:
            self.assertIn(boxInput, boxesMeasured)

    def test_maskBlocksIfIntermitentBadPixelsInColumnTest10(self):
        # Test 10: n discontiguous bad pixels in column with "blobs" of "m" bad pixels to one side, m >
        # threshold, # gaps between blobs < "good" threshold.
        # expected_test10 = np.array([(60,1,1,31), (61, 2, 2, 14)])

        config = copy.copy(self.defaultConfig)
        config.badOnAndOffPixelColumnThreshold = 10
        config.goodPixelColumnGapThreshold = 5
        config.nPixBorderUpDown = 0
        config.nPixBorderLeftRight = 0

        task = cpPipe.defects.FindDefectsTask(config=config)

        expectedDefects = [Box2I(corner = Point2I(60, 1), dimensions = Extent2I(1, 29)),
                           Box2I(corner = Point2I(61, 2), dimensions = Extent2I(1, 12))]
        defects = self.allDefectsList

        defectsWithColumns = task.maskBlocksIfIntermitentBadPixelsInColumn(defects)

        boxesMeasured = []
        for defect in defectsWithColumns:
            boxesMeasured.append(defect.getBBox())

        for boxInput in expectedDefects:
            self.assertIn(boxInput, boxesMeasured)

    def test_maskBlocksIfIntermitentBadPixelsInColumnTest11(self):
        # Test 11: n discontiguous bad pixels in column with "blobs" of "m" bad pixels to other side, m >
        # threshold, gaps between blobs < "good" threshold.
        # bad_test11 = np.array([(70, 1, 1, 16), (70, 20, 1, 10), (68, 2, 2, 2), (68, 6, 2, 8) ])
        # expected_test11 = np.array([(70,1,1,31), (68, 2, 2, 14)])

        config = copy.copy(self.defaultConfig)
        config.badOnAndOffPixelColumnThreshold = 10
        config.goodPixelColumnGapThreshold = 5
        config.nPixBorderUpDown = 0
        config.nPixBorderLeftRight = 0

        task = cpPipe.defects.FindDefectsTask(config=config)

        expectedDefects = [Box2I(corner = Point2I(70, 1), dimensions = Extent2I(1, 29)),
                           Box2I(corner = Point2I(68, 2), dimensions = Extent2I(1, 12))]
        defects = self.allDefectsList

        defectsWithColumns = task.maskBlocksIfIntermitentBadPixelsInColumn(defects)

        boxesMeasured = []
        for defect in defectsWithColumns:
            boxesMeasured.append(defect.getBBox())

        for boxInput in expectedDefects:
            self.assertIn(boxInput, boxesMeasured)

    def test_maskBlocksIfIntermitentBadPixelsInColumnTest12(self):
        # Test 12: n discontiguous bad pixels in column with "blobs" of "m" bad pixels to both sides, m >
        # threshold, gaps between blobs < "good" threshold.

        config = copy.copy(self.defaultConfig)
        config.badOnAndOffPixelColumnThreshold = 10
        config.goodPixelColumnGapThreshold = 5
        config.nPixBorderUpDown = 0
        config.nPixBorderLeftRight = 0

        task = cpPipe.defects.FindDefectsTask(config=config)

        expectedDefects = [Box2I(corner = Point2I(75, 1), dimensions = Extent2I(1, 29)),
                           Box2I(corner = Point2I(73, 2), dimensions = Extent2I(1, 12)),
                           Box2I(corner = Point2I(76, 2), dimensions = Extent2I(1, 12))]
        defects = self.allDefectsList

        defectsWithColumns = task.maskBlocksIfIntermitentBadPixelsInColumn(defects)

        boxesMeasured = []
        for defect in defectsWithColumns:
            boxesMeasured.append(defect.getBBox())

        for boxInput in expectedDefects:
            self.assertIn(boxInput, boxesMeasured)

    def test_maskBlocksIfIntermitentBadPixelsInColumnTest13(self):
        # Tests 13, 14, 15: same as tests 10, 11, 12 but with gaps between blobs > = "good" threshold.
        # bad_test13 = np.array([(80, 1, 1, 12), (80, 20, 1, 10), (81, 2, 2, 2), (81, 8, 2, 2) ])
        # expected_test13 = np.array([(80, 1, 1, 30), (81, 2, 2, 2), (81, 8, 2, 2) ])

        config = copy.copy(self.defaultConfig)
        config.badOnAndOffPixelColumnThreshold = 10
        config.goodPixelColumnGapThreshold = 5
        config.nPixBorderUpDown = 0
        config.nPixBorderLeftRight = 0

        task = cpPipe.defects.FindDefectsTask(config=config)

        expectedDefects = [Box2I(corner = Point2I(80, 1), dimensions = Extent2I(1, 29)),
                           Box2I(corner = Point2I(81, 2), dimensions = Extent2I(1, 2)),
                           Box2I(corner = Point2I(81, 8), dimensions = Extent2I(1, 8))]
        defects = self.allDefectsList

        defectsWithColumns = task.maskBlocksIfIntermitentBadPixelsInColumn(defects)

        boxesMeasured = []
        for defect in defectsWithColumns:
            boxesMeasured.append(defect.getBBox())

        for boxInput in expectedDefects:
            self.assertIn(boxInput, boxesMeasured)

    def test_maskBlocksIfIntermitentBadPixelsInColumnTest14(self):
        # Tests 13, 14, 15: same as tests 10, 11, 12 but with gaps between blobs > = "good" threshold.
        # bad_test14 = np.array([(87, 1, 1, 12), (87, 20, 1, 10), (85, 2, 2, 2), (85, 8, 2, 2) ])
        # expected_test14 = np.array([(87, 1, 1, 30), (85, 2, 2, 2), (85, 8, 2, 2) ])

        config = copy.copy(self.defaultConfig)
        config.badOnAndOffPixelColumnThreshold = 10
        config.goodPixelColumnGapThreshold = 5
        config.nPixBorderUpDown = 0
        config.nPixBorderLeftRight = 0

        task = cpPipe.defects.FindDefectsTask(config=config)

        expectedDefects = [Box2I(corner = Point2I(87, 1), dimensions = Extent2I(1, 29)),
                           Box2I(corner = Point2I(85, 2), dimensions = Extent2I(1, 2)),
                           Box2I(corner = Point2I(85, 8), dimensions = Extent2I(1, 8))]
        defects = self.allDefectsList

        defectsWithColumns = task.maskBlocksIfIntermitentBadPixelsInColumn(defects)

        boxesMeasured = []
        for defect in defectsWithColumns:
            boxesMeasured.append(defect.getBBox())

        for boxInput in expectedDefects:
            self.assertIn(boxInput, boxesMeasured)

    def test_maskBlocksIfIntermitentBadPixelsInColumnTest15(self):
        # Tests 13, 14, 15: same as tests 10, 11, 12 but with gaps between blobs > = "good" threshold.
        # bad_test15 = np.array([ (93, 1, 1, 12), (93, 15, 1, 20), (91, 2, 2, 2), (91, 8, 2, 2), (94, 2, 2,
        # 2), (94, 8, 2, 2),(91, 18, 2, 3), (91, 24, 2, 3), (94, 18, 2, 3), (94, 24, 2, 3)])
        # expected_test15 = np.array([ (93, 1, 1, 35), (91, 2, 2, 10), (91, 18, 2, 27), (94,
        #                       2, 2, 10), (94, 18, 2, 27)  ])

        config = copy.copy(self.defaultConfig)
        config.badOnAndOffPixelColumnThreshold = 10
        config.goodPixelColumnGapThreshold = 5
        config.nPixBorderUpDown = 0
        config.nPixBorderLeftRight = 0

        task = cpPipe.defects.FindDefectsTask(config=config)

        expectedDefects = [Box2I(corner = Point2I(93, 1), dimensions = Extent2I(1, 34)),
                           Box2I(corner = Point2I(91, 2), dimensions = Extent2I(1, 7)),
                           Box2I(corner = Point2I(91, 18), dimensions = Extent2I(1, 9)),
                           Box2I(corner = Point2I(94, 2), dimensions = Extent2I(1, 7)),
                           Box2I(corner = Point2I(94, 18), dimensions = Extent2I(1, 9))]
        defects = self.allDefectsList

        defectsWithColumns = task.maskBlocksIfIntermitentBadPixelsInColumn(defects)

        boxesMeasured = []
        for defect in defectsWithColumns:
            boxesMeasured.append(defect.getBBox())

        for boxInput in expectedDefects:
            self.assertIn(boxInput, boxesMeasured)

    def test_defectFindingAllSensor(self):
        config = copy.copy(self.defaultConfig)
        config.nPixBorderLeftRight = 0
        config.nPixBorderUpDown = 0

        task = cpPipe.defects.FindDefectsTask(config=config)

        defects = task.findHotAndColdPixels(self.flatExp, 'flat')

        allBBoxes = self.darkBBoxes + self.brightBBoxes

        boxesMeasured = []
        for defect in defects:
            boxesMeasured.append(defect.getBBox())

        for expectedBBox in allBBoxes:
            self.assertIn(expectedBBox, boxesMeasured)

    def test_defectFindingEdgeIgnore(self):
        config = copy.copy(self.defaultConfig)
        config.nPixBorderUpDown = 0
        task = cpPipe.defects.FindDefectsTask(config=config)
        defects = task.findHotAndColdPixels(self.flatExp, 'flat')

        shouldBeFound = self.darkBBoxes[self.noEdges] + self.brightBBoxes[self.noEdges]

        boxesMeasured = []
        for defect in defects:
            boxesMeasured.append(defect.getBBox())

        for expectedBBox in shouldBeFound:
            self.assertIn(expectedBBox, boxesMeasured)

        shouldBeMissed = self.darkBBoxes[self.onlyEdges] + self.brightBBoxes[self.onlyEdges]
        for boxMissed in shouldBeMissed:
            self.assertNotIn(boxMissed, boxesMeasured)

    def test_postProcessDefectSets(self):
        """Tests the way in which the defect sets merge.

        There is potential for logic errors in their combination
        so several combinations of defects and combination methods
        are tested here."""
        defects = self.defaultTask.findHotAndColdPixels(self.flatExp, 'flat')

        # defect list has length one
        merged = self.defaultTask._postProcessDefectSets([defects], self.flatExp.getDimensions(), 'FRACTION')
        self.assertEqual(defects, merged)

        # should always be true regardless of config
        # defect list now has length 2
        merged = self.defaultTask._postProcessDefectSets([defects, defects], self.flatExp.getDimensions(),
                                                         'FRACTION')
        self.assertEqual(defects, merged)

        # now start manipulating defect lists
        config = copy.copy(self.defaultConfig)
        config.combinationMode = 'FRACTION'
        config.combinationFraction = 0.85
        task = cpPipe.defects.FindDefectsTask(config=config)
        merged = task._postProcessDefectSets([defects, defects], self.flatExp.getDimensions(), 'FRACTION')

        defectList = [defects]*10  # 10 identical defect sets
        # remove one defect from one of them, should still be over threshold
        defectList[7] = defectList[7][:-1]
        merged = task._postProcessDefectSets(defectList, self.flatExp.getDimensions(), 'FRACTION')
        self.assertEqual(defects, merged)

        # remove another and should be under threshold
        defectList[3] = defectList[3][:-1]
        merged = task._postProcessDefectSets(defectList, self.flatExp.getDimensions(), 'FRACTION')
        self.assertNotEqual(defects, merged)

        # now test the AND and OR modes
        defectList = [defects]*10  # 10 identical defect sets
        merged = task._postProcessDefectSets(defectList, self.flatExp.getDimensions(), 'AND')
        self.assertEqual(defects, merged)

        defectList[7] = defectList[7][:-1]
        merged = task._postProcessDefectSets(defectList, self.flatExp.getDimensions(), 'AND')
        self.assertNotEqual(defects, merged)

        merged = task._postProcessDefectSets(defectList, self.flatExp.getDimensions(), 'OR')
        self.assertEqual(defects, merged)

    def test_pixelCounting(self):
        """Test that the number of defective pixels identified is as expected."""
        config = copy.copy(self.defaultConfig)
        config.nPixBorderUpDown = 0
        config.nPixBorderLeftRight = 0
        task = cpPipe.defects.FindDefectsTask(config=config)
        defects = task.findHotAndColdPixels(self.flatExp, 'flat')

        defectArea = 0
        for defect in defects:
            defectArea += defect.getBBox().getArea()

        # The columnar code will cover blocks of a column
        # with on-and-off pixels, thus creating more bad pixels
        # that what initially placed in self.brightDefects and self.darkDefects.
        # Thus, defectArea should be >= crossCheck.
        crossCheck = 0
        for x, y, sx, sy in self.brightDefects:
            crossCheck += sx*sy
        for x, y, sx, sy in self.darkDefects:
            crossCheck += sx*sy

        # Test the result of _nPixFromDefects()
        # via two different ways of calculating area.
        self.assertEqual(defectArea, task._nPixFromDefects(defects))
        # defectArea should be >= crossCheck
        self.assertGreaterEqual(defectArea, crossCheck)

    def test_getNumGoodPixels(self):
        """Test the the number of pixels in the image not masked is as expected."""
        testImage = self.flatExp.clone()
        mi = testImage.maskedImage

        imageSize = testImage.getBBox().getArea()
        nGood = self.defaultTask._getNumGoodPixels(mi)

        self.assertEqual(imageSize, nGood)

        NODATABIT = mi.mask.getPlaneBitMask("NO_DATA")

        noDataBox = Box2I(Point2I(31, 49), Extent2I(3, 6))
        testImage.mask[noDataBox] |= NODATABIT

        self.assertEqual(imageSize - noDataBox.getArea(), self.defaultTask._getNumGoodPixels(mi))
        # check for misfire; we're setting NO_DATA here, not BAD
        self.assertEqual(imageSize, self.defaultTask._getNumGoodPixels(mi, 'BAD'))

        testImage.mask[noDataBox] ^= NODATABIT  # XOR to reset what we did
        self.assertEqual(imageSize, nGood)

        BADBIT = mi.mask.getPlaneBitMask("BAD")
        badBox = Box2I(Point2I(85, 98), Extent2I(4, 7))
        testImage.mask[badBox] |= BADBIT

        self.assertEqual(imageSize - badBox.getArea(), self.defaultTask._getNumGoodPixels(mi, 'BAD'))

    def test_edgeMasking(self):
        """Check that the right number of edge pixels are masked by _setEdgeBits()"""
        testImage = self.flatExp.clone()
        mi = testImage.maskedImage

        self.assertEqual(countMaskedPixels(mi, 'EDGE'), 0)
        self.defaultTask._setEdgeBits(mi)

        hEdge = self.defaultConfig.nPixBorderLeftRight
        vEdge = self.defaultConfig.nPixBorderUpDown
        xSize, ySize = mi.getDimensions()

        nEdge = xSize*vEdge*2 + ySize*hEdge*2 - hEdge*vEdge*4

        self.assertEqual(countMaskedPixels(mi, 'EDGE'), nEdge)


class TestMemory(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
