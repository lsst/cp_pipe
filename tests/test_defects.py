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
"""Test cases for lsst.cp.pipe.defects.MeasureDefectsTask."""

import unittest
import numpy as np
import copy

import lsst.utils
import lsst.utils.tests

import lsst.ip.isr as ipIsr
import lsst.cp.pipe as cpPipe
from lsst.ip.isr import isrMock, countMaskedPixels
from lsst.geom import Box2I, Point2I, Extent2I
from lsst.daf.base import PropertyList


class MeasureDefectsTaskTestCase(lsst.utils.tests.TestCase):
    """A test case for the defect finding task."""

    def setUp(self):
        self.defaultConfig = cpPipe.defects.MeasureDefectsTask.ConfigClass()

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

        self.brightDefects = [(0, 15, 3, 3), (100, 123, 1, 1)]

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

        self.defaultTask = cpPipe.defects.MeasureDefectsTask()

        self.allDefectsList = ipIsr.Defects()
        self.brightDefectsList = ipIsr.Defects()
        self.darkDefectsList = ipIsr.Defects()

        # Set image types, the defects code will use them.
        metaDataFlat = PropertyList()
        metaDataFlat["IMGTYPE"] = "FLAT"
        self.flatExp.setMetadata(metaDataFlat)

        metaDataDark = PropertyList()
        metaDataDark["IMGTYPE"] = "DARK"
        self.darkExp.setMetadata(metaDataDark)

        with self.allDefectsList.bulk_update():
            with self.brightDefectsList.bulk_update():
                for d in self.brightBBoxes:
                    self.brightDefectsList.append(d)
                    self.allDefectsList.append(d)

            with self.darkDefectsList.bulk_update():
                for d in self.darkBBoxes:
                    self.darkDefectsList.append(d)
                    self.allDefectsList.append(d)

    def check_maskBlocks(self, inputDefects, expectedDefects):
        """A helper function for the tests of
        maskBlocksIfIntermitentBadPixelsInColumn.

        """
        config = copy.copy(self.defaultConfig)
        config.badOnAndOffPixelColumnThreshold = 10
        config.goodPixelColumnGapThreshold = 5
        config.nPixBorderUpDown = 0
        config.nPixBorderLeftRight = 0

        task = self.defaultTask
        task.config = config

        defectsWithColumns = task.maskBlocksIfIntermitentBadPixelsInColumn(inputDefects)
        boxesMeasured = []
        for defect in defectsWithColumns:
            boxesMeasured.append(defect.getBBox())

        for boxInput in expectedDefects:
            self.assertIn(boxInput, boxesMeasured)

        # Check that the code did not mask anything extra by
        # looking in both the input list and "expanded-column" list.
        unionInputExpectedBoxes = []
        for defect in inputDefects:
            unionInputExpectedBoxes.append(defect.getBBox())
        for defect in expectedDefects:
            unionInputExpectedBoxes.append(defect)

        # Check that code doesn't mask more than it is supposed to.
        for boxMeas in boxesMeasured:
            self.assertIn(boxMeas, unionInputExpectedBoxes)

    def test_maskBlocks_full_column(self):
        """A test for maskBlocksIfIntermitentBadPixelsInColumn.

        Tests that a contigous bad column does not get split by the
        code.

        The mock flat has a size of 200X204 pixels. This column has a
        maximum length of 50 pixels, otherwise there would be a split
        along the mock amp boundary.

        Plots can be found in DM-19903 on Jira.

        """

        defects = self.allDefectsList
        defects.append(Box2I(corner=Point2I(15, 1), dimensions=Extent2I(1, 50)))
        expectedDefects = [Box2I(corner=Point2I(15, 1), dimensions=Extent2I(1, 50))]

        self.check_maskBlocks(defects, expectedDefects)

    def test_maskBlocks_long_column(self):
        """A test for maskBlocksIfIntermitentBadPixelsInColumn.

        Tests that a contigous bad column with Npix >=
        badOnAndOffPixelColumnThreshold (10) does not get split by the
        code.

        Plots can be found in DM-19903 on Jira.

        """

        expectedDefects = [Box2I(corner=Point2I(20, 1), dimensions=Extent2I(1, 25))]
        defects = self.allDefectsList
        defects.append(Box2I(corner=Point2I(20, 1), dimensions=Extent2I(1, 25)))

        self.check_maskBlocks(defects, expectedDefects)

    def test_maskBlocks_short_column(self):
        """A test for maskBlocksIfIntermitentBadPixelsInColumn.

        Tests that a contigous bad column Npix <
        badOnAndOffPixelColumnThreshold (10) does not get split by the
        code.

        Plots can be found in DM-19903 on Jira.

        """

        expectedDefects = [Box2I(corner=Point2I(25, 1), dimensions=Extent2I(1, 8))]
        defects = self.allDefectsList
        defects.append(Box2I(corner=Point2I(25, 1), dimensions=Extent2I(1, 8)))

        self.check_maskBlocks(defects, expectedDefects)

    def test_maskBlocks_discontigous_to_single_block(self):
        """A test for maskBlocksIfIntermitentBadPixelsInColumn.

        Npix discontiguous bad pixels in a column where Npix >=
        badOnAndOffPixelColumnThreshold (10) and gaps of good pixels <
        goodPixelColumnGapThreshold (5). Under these conditions, the
        whole block of bad pixels (including good gaps) should be
        masked.

        Plots can be found in DM-19903 on Jira.

        """

        expectedDefects = [Box2I(corner=Point2I(30, 1), dimensions=Extent2I(1, 48))]
        defects = self.allDefectsList
        badPixels = [Box2I(corner=Point2I(30, 1), dimensions=Extent2I(1, 2)),
                     Box2I(corner=Point2I(30, 5), dimensions=Extent2I(1, 3)),
                     Box2I(corner=Point2I(30, 11), dimensions=Extent2I(1, 5)),
                     Box2I(corner=Point2I(30, 19), dimensions=Extent2I(1, 5)),
                     Box2I(corner=Point2I(30, 27), dimensions=Extent2I(1, 4)),
                     Box2I(corner=Point2I(30, 34), dimensions=Extent2I(1, 15))]
        with defects.bulk_update():
            for badBox in badPixels:
                defects.append(badBox)

        self.check_maskBlocks(defects, expectedDefects)

    def test_maskBlocks_discontigous_less_than_thresholds(self):
        """A test for maskBlocksIfIntermitentBadPixelsInColumn.

        Npix discontiguous bad pixels in a column where Npix <
        badOnAndOffPixelColumnThreshold (10) and gaps of good pixels <
        goodPixelColumnGapThreshold (5). Under these conditions, the
        expected defect boxes should be the same as the input boxes.

        Plots can be found in DM-19903 on Jira.

        """

        expectedDefects = [Box2I(corner=Point2I(35, 1), dimensions=Extent2I(1, 2)),
                           Box2I(corner=Point2I(35, 5), dimensions=Extent2I(1, 3)),
                           Box2I(corner=Point2I(35, 11), dimensions=Extent2I(1, 2))]
        defects = self.allDefectsList
        badPixels = [Box2I(corner=Point2I(35, 1), dimensions=Extent2I(1, 2)),
                     Box2I(corner=Point2I(35, 5), dimensions=Extent2I(1, 3)),
                     Box2I(corner=Point2I(35, 11), dimensions=Extent2I(1, 2))]
        with defects.bulk_update():
            for badBox in badPixels:
                defects.append(badBox)

        self.check_maskBlocks(defects, expectedDefects)

    def test_maskBlocks_more_than_thresholds(self):
        """A test for maskBlocksIfIntermitentBadPixelsInColumn.

        Npix discontiguous bad pixels in a column where Npix <
        badOnAndOffPixelColumnThreshold (10) and gaps of good pixels <
        goodPixelColumnGapThreshold (5).  Npix=34 (> 10) bad pixels
        total, 1 "good" gap with 13 pixels big enough (13 >= 5 good
        pixels, from y=6 (1+5) to y=19).

        Plots can be found in DM-19903 on Jira.

        """

        expectedDefects = [Box2I(corner=Point2I(40, 1), dimensions=Extent2I(1, 7)),
                           Box2I(corner=Point2I(40, 19), dimensions=Extent2I(1, 30))]
        defects = self.allDefectsList
        badPixels = [Box2I(corner=Point2I(40, 1), dimensions=Extent2I(1, 2)),
                     Box2I(corner=Point2I(40, 5), dimensions=Extent2I(1, 3)),
                     Box2I(corner=Point2I(40, 19), dimensions=Extent2I(1, 5)),
                     Box2I(corner=Point2I(40, 27), dimensions=Extent2I(1, 4)),
                     Box2I(corner=Point2I(40, 34), dimensions=Extent2I(1, 15))]
        with defects.bulk_update():
            for badBox in badPixels:
                defects.append(badBox)

        self.check_maskBlocks(defects, expectedDefects)

    def test_maskBlocks_not_enough_bad_pixels_in_column(self):
        """A test for maskBlocksIfIntermitentBadPixelsInColumn.

        Npix discontiguous bad pixels in a column where Npix <
        badOnAndOffPixelColumnThreshold (10) and and gaps of good
        pixels > goodPixelColumnGapThreshold (5). Since Npix <
        badOnAndOffPixelColumnThreshold, then it doesn't matter that
        the number of good pixels in gap >
        goodPixelColumnGapThreshold. 5<10 bad pixels total, 1 "good"
        gap big enough (29>=5 good pixels, from y =12 (10+2) to y=30)

        Plots can be found in DM-19903 on Jira.

        """

        expectedDefects = [Box2I(corner=Point2I(45, 10), dimensions=Extent2I(1, 2)),
                           Box2I(corner=Point2I(45, 30), dimensions=Extent2I(1, 3))]
        defects = self.allDefectsList
        badPixels = [Box2I(corner=Point2I(45, 10), dimensions=Extent2I(1, 2)),
                     Box2I(corner=Point2I(45, 30), dimensions=Extent2I(1, 3))]
        with defects.bulk_update():
            for badBox in badPixels:
                defects.append(badBox)

        self.check_maskBlocks(defects, expectedDefects)

    def test_maskBlocks_every_other_pixel_bad_greater_than_threshold(self):
        """A test for maskBlocksIfIntermitentBadPixelsInColumn.

        Npix discontiguous bad pixels in a column where Npix >
        badOnAndOffPixelColumnThreshold (10) and every other pixel is
        bad.

        Plots can be found in DM-19903 on Jira.

        """

        expectedDefects = [Box2I(corner=Point2I(50, 10), dimensions=Extent2I(1, 31))]
        defects = self.allDefectsList
        badPixels = [Box2I(corner=Point2I(50, 10), dimensions=Extent2I(1, 1)),
                     Box2I(corner=Point2I(50, 12), dimensions=Extent2I(1, 1)),
                     Box2I(corner=Point2I(50, 14), dimensions=Extent2I(1, 1)),
                     Box2I(corner=Point2I(50, 16), dimensions=Extent2I(1, 1)),
                     Box2I(corner=Point2I(50, 18), dimensions=Extent2I(1, 1)),
                     Box2I(corner=Point2I(50, 20), dimensions=Extent2I(1, 1)),
                     Box2I(corner=Point2I(50, 22), dimensions=Extent2I(1, 1)),
                     Box2I(corner=Point2I(50, 24), dimensions=Extent2I(1, 1)),
                     Box2I(corner=Point2I(50, 26), dimensions=Extent2I(1, 1)),
                     Box2I(corner=Point2I(50, 28), dimensions=Extent2I(1, 1)),
                     Box2I(corner=Point2I(50, 30), dimensions=Extent2I(1, 1)),
                     Box2I(corner=Point2I(50, 32), dimensions=Extent2I(1, 1)),
                     Box2I(corner=Point2I(50, 34), dimensions=Extent2I(1, 1)),
                     Box2I(corner=Point2I(50, 36), dimensions=Extent2I(1, 1)),
                     Box2I(corner=Point2I(50, 38), dimensions=Extent2I(1, 1)),
                     Box2I(corner=Point2I(50, 40), dimensions=Extent2I(1, 1))]
        with defects.bulk_update():
            for badBox in badPixels:
                defects.append(badBox)

        self.check_maskBlocks(defects, expectedDefects)

    def test_maskBlocks_every_other_pixel_bad_less_than_threshold(self):
        """A test for maskBlocksIfIntermitentBadPixelsInColumn.

        Npix discontiguous bad pixels in a column where Npix >
        badOnAndOffPixelColumnThreshold (10) and every other pixel is
        bad.

        Plots can be found in DM-19903 on Jira.

        """

        expectedDefects = [Box2I(corner=Point2I(55, 20), dimensions=Extent2I(1, 1)),
                           Box2I(corner=Point2I(55, 22), dimensions=Extent2I(1, 1)),
                           Box2I(corner=Point2I(55, 24), dimensions=Extent2I(1, 1)),
                           Box2I(corner=Point2I(55, 26), dimensions=Extent2I(1, 1)),
                           Box2I(corner=Point2I(55, 28), dimensions=Extent2I(1, 1)),
                           Box2I(corner=Point2I(55, 30), dimensions=Extent2I(1, 1))]
        defects = self.allDefectsList
        badPixels = [Box2I(corner=Point2I(55, 20), dimensions=Extent2I(1, 1)),
                     Box2I(corner=Point2I(55, 22), dimensions=Extent2I(1, 1)),
                     Box2I(corner=Point2I(55, 24), dimensions=Extent2I(1, 1)),
                     Box2I(corner=Point2I(55, 26), dimensions=Extent2I(1, 1)),
                     Box2I(corner=Point2I(55, 28), dimensions=Extent2I(1, 1)),
                     Box2I(corner=Point2I(55, 30), dimensions=Extent2I(1, 1))]
        with defects.bulk_update():
            for badBox in badPixels:
                defects.append(badBox)

        self.check_maskBlocks(defects, expectedDefects)

    def test_maskBlocks_blobs_one_side_good_less_than_threshold(self):
        """A test for maskBlocksIfIntermitentBadPixelsInColumn.

        Npix discontiguous bad pixels in column with "blobs" of "m"
        bad pixels to one side, m > badOnAndOffPixelColumnThreshold
        (10), number of good pixel in gaps between blobs <
        goodPixelColumnGapThreshold (5).

        Plots can be found in DM-19903 on Jira.

        """

        expectedDefects = [Box2I(corner=Point2I(60, 1), dimensions=Extent2I(1, 29)),
                           Box2I(corner=Point2I(61, 2), dimensions=Extent2I(2, 12))]
        defects = self.allDefectsList
        badPixels = [Box2I(corner=Point2I(60, 1), dimensions=Extent2I(1, 18)),
                     Box2I(corner=Point2I(60, 20), dimensions=Extent2I(1, 10)),
                     Box2I(corner=Point2I(61, 2), dimensions=Extent2I(2, 2)),
                     Box2I(corner=Point2I(61, 6), dimensions=Extent2I(2, 8))]
        with defects.bulk_update():
            for badBox in badPixels:
                defects.append(badBox)

        self.check_maskBlocks(defects, expectedDefects)

    def test_maskBlocks_blobs_other_side_good_less_than_threshold(self):
        """A test for maskBlocksIfIntermitentBadPixelsInColumn.

        Npix discontiguous bad pixels in column with "blobs" of "m"
        bad pixels to the other side, m >
        badOnAndOffPixelColumnThreshold (10), number of good pixel in
        gaps between blobs < goodPixelColumnGapThreshold (5).

        Plots can be found in DM-19903 on Jira.

        """

        expectedDefects = [Box2I(corner=Point2I(70, 1), dimensions=Extent2I(1, 29)),
                           Box2I(corner=Point2I(68, 2), dimensions=Extent2I(2, 12))]
        defects = self.allDefectsList
        badPixels = [Box2I(corner=Point2I(70, 1), dimensions=Extent2I(1, 18)),
                     Box2I(corner=Point2I(70, 20), dimensions=Extent2I(1, 10)),
                     Box2I(corner=Point2I(68, 2), dimensions=Extent2I(2, 2)),
                     Box2I(corner=Point2I(68, 6), dimensions=Extent2I(2, 8))]
        with defects.bulk_update():
            for badBox in badPixels:
                defects.append(badBox)

        self.check_maskBlocks(defects, expectedDefects)

    def test_maskBlocks_blob_both_sides_good_less_than_threshold(self):
        """A test for maskBlocksIfIntermitentBadPixelsInColumn.

        Npix discontiguous bad pixels in column with "blobs" of "m"
        bad pixels to both sides, m > badOnAndOffPixelColumnThreshold
        (10), number of good pixel in gaps between blobs <
        goodPixelColumnGapThreshold (5).

        Plots can be found in DM-19903 on Jira.

        """

        expectedDefects = [Box2I(corner=Point2I(75, 1), dimensions=Extent2I(1, 29)),
                           Box2I(corner=Point2I(73, 2), dimensions=Extent2I(2, 12)),
                           Box2I(corner=Point2I(76, 2), dimensions=Extent2I(2, 12))]
        defects = self.allDefectsList
        badPixels = [Box2I(corner=Point2I(75, 1), dimensions=Extent2I(1, 18)),
                     Box2I(corner=Point2I(75, 20), dimensions=Extent2I(1, 10)),
                     Box2I(corner=Point2I(73, 2), dimensions=Extent2I(2, 2)),
                     Box2I(corner=Point2I(73, 6), dimensions=Extent2I(2, 8)),
                     Box2I(corner=Point2I(76, 2), dimensions=Extent2I(2, 2)),
                     Box2I(corner=Point2I(76, 6), dimensions=Extent2I(2, 8))]
        with defects.bulk_update():
            for badBox in badPixels:
                defects.append(badBox)

        self.check_maskBlocks(defects, expectedDefects)

    def test_maskBlocks_blob_one_side_good_greater_than_threshold(self):
        """A test for maskBlocksIfIntermitentBadPixelsInColumn.

        Npix discontiguous bad pixels in column with "blobs" of "m"
        bad pixels to one side, m > badOnAndOffPixelColumnThreshold
        (10), number of good pixel in gaps between blobs >
        goodPixelColumnGapThreshold (5).

        Plots can be found in DM-19903 on Jira.

        """

        expectedDefects = [Box2I(corner=Point2I(80, 1), dimensions=Extent2I(1, 29)),
                           Box2I(corner=Point2I(81, 2), dimensions=Extent2I(2, 2)),
                           Box2I(corner=Point2I(81, 8), dimensions=Extent2I(2, 8))]
        defects = self.allDefectsList
        badPixels = [Box2I(corner=Point2I(80, 1), dimensions=Extent2I(1, 18)),
                     Box2I(corner=Point2I(80, 20), dimensions=Extent2I(1, 10)),
                     Box2I(corner=Point2I(81, 2), dimensions=Extent2I(2, 2)),
                     Box2I(corner=Point2I(81, 8), dimensions=Extent2I(2, 8))]
        with defects.bulk_update():
            for badBox in badPixels:
                defects.append(badBox)

        self.check_maskBlocks(defects, expectedDefects)

    def test_maskBlocks_other_side_good_greater_than_threshold(self):
        """A test for maskBlocksIfIntermitentBadPixelsInColumn.

        Npix discontiguous bad pixels in column with "blobs" of "m"
        bad pixels to the other side, m >
        badOnAndOffPixelColumnThreshold (10), number of good pixel in
        gaps between blobs > goodPixelColumnGapThreshold (5).

        Plots can be found in DM-19903 on Jira.

        """

        expectedDefects = [Box2I(corner=Point2I(87, 1), dimensions=Extent2I(1, 29)),
                           Box2I(corner=Point2I(85, 2), dimensions=Extent2I(2, 2)),
                           Box2I(corner=Point2I(85, 8), dimensions=Extent2I(2, 8))]
        defects = self.allDefectsList
        badPixels = [Box2I(corner=Point2I(87, 1), dimensions=Extent2I(1, 18)),
                     Box2I(corner=Point2I(87, 20), dimensions=Extent2I(1, 10)),
                     Box2I(corner=Point2I(85, 2), dimensions=Extent2I(2, 2)),
                     Box2I(corner=Point2I(85, 8), dimensions=Extent2I(2, 8))]
        with defects.bulk_update():
            for badBox in badPixels:
                defects.append(badBox)

        self.check_maskBlocks(defects, expectedDefects)

    def test_maskBlocks_both_sides_good_greater_than_threshold(self):
        """A test for maskBlocksIfIntermitentBadPixelsInColumn.

        Npix discontiguous bad pixels in column with "blobs" of "m"
        bad pixels to both sides, m > badOnAndOffPixelColumnThreshold
        (10), number of good pixel in gaps between blobs >
        goodPixelColumnGapThreshold (5).

        Plots can be found in DM-19903 on Jira.

        """

        expectedDefects = [Box2I(corner=Point2I(93, 1), dimensions=Extent2I(1, 34)),
                           Box2I(corner=Point2I(91, 2), dimensions=Extent2I(2, 7)),
                           Box2I(corner=Point2I(91, 18), dimensions=Extent2I(2, 9)),
                           Box2I(corner=Point2I(94, 2), dimensions=Extent2I(2, 7)),
                           Box2I(corner=Point2I(94, 18), dimensions=Extent2I(2, 9))]
        defects = self.allDefectsList
        badPixels = [Box2I(corner=Point2I(93, 1), dimensions=Extent2I(1, 12)),
                     Box2I(corner=Point2I(93, 15), dimensions=Extent2I(1, 20)),
                     Box2I(corner=Point2I(91, 2), dimensions=Extent2I(2, 2)),
                     Box2I(corner=Point2I(91, 7), dimensions=Extent2I(2, 2)),
                     Box2I(corner=Point2I(94, 2), dimensions=Extent2I(2, 2)),
                     Box2I(corner=Point2I(94, 7), dimensions=Extent2I(2, 2)),
                     Box2I(corner=Point2I(91, 18), dimensions=Extent2I(2, 3)),
                     Box2I(corner=Point2I(91, 24), dimensions=Extent2I(2, 3)),
                     Box2I(corner=Point2I(94, 18), dimensions=Extent2I(2, 3)),
                     Box2I(corner=Point2I(94, 24), dimensions=Extent2I(2, 3))]
        with defects.bulk_update():
            for badBox in badPixels:
                defects.append(badBox)

        self.check_maskBlocks(defects, expectedDefects)

    def test_maskBlocks_y_out_of_order_dm38103(self):
        """A test for maskBlocksIfIntermitentBadPixelsInColumn, y out of order.

        This test is a variant of
        notest_maskBlocks_every_other_pixel_bad_greater_than_threshold with
        an extra out-of-y-order bad pixel to trigger DM-38103.
        """
        expectedDefects = [Box2I(corner=Point2I(50, 110), dimensions=Extent2I(1, 31))]
        defects = self.allDefectsList
        badPixels = [Box2I(corner=Point2I(50, 110), dimensions=Extent2I(1, 1)),
                     Box2I(corner=Point2I(50, 112), dimensions=Extent2I(1, 1)),
                     Box2I(corner=Point2I(50, 114), dimensions=Extent2I(1, 1)),
                     Box2I(corner=Point2I(50, 116), dimensions=Extent2I(1, 1)),
                     Box2I(corner=Point2I(50, 118), dimensions=Extent2I(1, 1)),
                     Box2I(corner=Point2I(50, 120), dimensions=Extent2I(1, 1)),
                     Box2I(corner=Point2I(50, 122), dimensions=Extent2I(1, 1)),
                     Box2I(corner=Point2I(50, 124), dimensions=Extent2I(1, 1)),
                     Box2I(corner=Point2I(50, 126), dimensions=Extent2I(1, 1)),
                     Box2I(corner=Point2I(50, 128), dimensions=Extent2I(1, 1)),
                     Box2I(corner=Point2I(50, 130), dimensions=Extent2I(1, 1)),
                     Box2I(corner=Point2I(50, 132), dimensions=Extent2I(1, 1)),
                     Box2I(corner=Point2I(50, 134), dimensions=Extent2I(1, 1)),
                     Box2I(corner=Point2I(50, 136), dimensions=Extent2I(1, 1)),
                     Box2I(corner=Point2I(50, 138), dimensions=Extent2I(1, 1)),
                     Box2I(corner=Point2I(50, 140), dimensions=Extent2I(1, 1)),
                     # This last point is out of order in y.
                     Box2I(corner=Point2I(50, 100), dimensions=Extent2I(1, 1))]

        # Force defect normalization off in order to trigger DM-38301, because
        # defects.fromFootprintList() which is called by _findHotAndColdPixels
        # does not do normalization.
        defects._bulk_update = True
        for badBox in badPixels:
            defects.append(badBox)
        defects._bulk_update = False

        self.check_maskBlocks(defects, expectedDefects)

    def test_defectFindingAllSensor(self):
        config = copy.copy(self.defaultConfig)
        config.nPixBorderLeftRight = 0
        config.nPixBorderUpDown = 0

        task = self.defaultTask
        task.config = config

        defects = task._findHotAndColdPixels(self.flatExp)

        allBBoxes = self.darkBBoxes + self.brightBBoxes

        boxesMeasured = []
        for defect in defects:
            boxesMeasured.append(defect.getBBox())

        for expectedBBox in allBBoxes:
            self.assertIn(expectedBBox, boxesMeasured)

    def test_defectFindingEdgeIgnore(self):
        config = copy.copy(self.defaultConfig)
        config.nPixBorderUpDown = 0
        task = self.defaultTask
        task.config = config
        defects = task._findHotAndColdPixels(self.flatExp)

        shouldBeFound = self.darkBBoxes[self.noEdges] + self.brightBBoxes[self.noEdges]

        boxesMeasured = []
        for defect in defects:
            boxesMeasured.append(defect.getBBox())

        for expectedBBox in shouldBeFound:
            self.assertIn(expectedBBox, boxesMeasured)

        shouldBeMissed = self.darkBBoxes[self.onlyEdges] + self.brightBBoxes[self.onlyEdges]
        for boxMissed in shouldBeMissed:
            self.assertNotIn(boxMissed, boxesMeasured)

    def valueThreshold(self, fileType, saturateAmpInFlat=False):
        """Helper function to loop over flats and darks
        to test thresholdType = 'VALUE'."""
        config = copy.copy(self.defaultConfig)
        config.thresholdType = 'VALUE'
        task = self.defaultTask
        task.config = config

        for amp in self.flatExp.getDetector():
            if amp.getName() == 'C:0,0':
                regionC00 = amp.getBBox()

        if fileType == 'dark':
            exp = self.darkExp
            shouldBeFound = self.brightBBoxes[self.noEdges]
        else:
            exp = self.flatExp
            if saturateAmpInFlat:
                exp.maskedImage[regionC00].image.array[:] = 0.0
                # Amp C:0,0: minimum=(0, 0), maximum=(99, 50)
                x = self.defaultConfig.nPixBorderUpDown
                y = self.defaultConfig.nPixBorderLeftRight
                width, height = regionC00.getEndX() - x, regionC00.getEndY() - y
                # Defects code will mark whole saturated amp as defect box.
                shouldBeFound = [Box2I(corner=Point2I(x, y), dimensions=Extent2I(width, height))]
            else:
                shouldBeFound = self.darkBBoxes[self.noEdges]
                # Change the default a bit so it works for the
                # existing simulated defects.
                task.config.fracThresholdFlat = 0.9

        defects = task._findHotAndColdPixels(exp)

        boxesMeasured = []
        for defect in defects:
            boxesMeasured.append(defect.getBBox())

        for expectedBBox in shouldBeFound:
            self.assertIn(expectedBBox, boxesMeasured)

    def test_valueThreshold(self):
        for fileType in ['dark', 'flat']:
            self.valueThreshold(fileType)
        # stdDev = 0.0
        self.valueThreshold('flat', saturateAmpInFlat=True)

    def test_pixelCounting(self):
        """Test that the number of defective pixels identified is as expected.
        """
        config = copy.copy(self.defaultConfig)
        config.nPixBorderUpDown = 0
        config.nPixBorderLeftRight = 0
        task = self.defaultTask
        task.config = config
        defects = task._findHotAndColdPixels(self.flatExp)

        defectArea = 0
        for defect in defects:
            defectArea += defect.getBBox().getArea()

        # The columnar code will cover blocks of a column with
        # on-and-off pixels, thus creating more bad pixels that what
        # initially placed in self.brightDefects and self.darkDefects.
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
        """Test the the number of pixels in the image not masked is as
        expected.
        """
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
        """Check that the right number of edge pixels are masked by
        _setEdgeBits().
        """
        testImage = self.flatExp.clone()
        mi = testImage.maskedImage

        self.assertEqual(countMaskedPixels(mi, 'EDGE'), 0)
        self.defaultTask._setEdgeBits(mi)

        hEdge = self.defaultConfig.nPixBorderLeftRight
        vEdge = self.defaultConfig.nPixBorderUpDown
        xSize, ySize = mi.getDimensions()

        nEdge = xSize*vEdge*2 + ySize*hEdge*2 - hEdge*vEdge*4

        self.assertEqual(countMaskedPixels(mi, 'EDGE'), nEdge)

    def test_badImage(self):
        """Check that fully-bad images do not fail.
        """
        testImage = self.flatExp.clone()
        testImage.image.array[:, :] = 125000

        config = copy.copy(self.defaultConfig)
        # Do not exclude any pixels, so the areas match.
        config.nPixBorderUpDown = 0
        config.nPixBorderLeftRight = 0

        task = self.defaultTask
        task.config = config
        defects = task._findHotAndColdPixels(testImage)

        defectArea = 0
        for defect in defects:
            defectArea += defect.getBBox().getArea()
        self.assertEqual(defectArea, testImage.getBBox().getArea())


class TestMemory(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
