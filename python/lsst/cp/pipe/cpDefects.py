# This file is part of cp_pipe.
#
# Developed for the LSST Data Management System.
# This product includes software developed by the LSST Project
# (https://www.lsst.org).
# See the COPYRIGHT file at the top-level directory of this distribution
# for details of code ownership.
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
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
#

__all__ = ['MeasureDefectsTaskConfig', 'MeasureDefectsTask',
           'MergeDefectsTaskConfig', 'MergeDefectsTask',
           'MeasureDefectsCombinedTaskConfig', 'MeasureDefectsCombinedTask',
           'MeasureDefectsCombinedWithFilterTaskConfig', 'MeasureDefectsCombinedWithFilterTask',
           'MergeDefectsCombinedTaskConfig', 'MergeDefectsCombinedTask', ]

import numpy as np

import lsst.pipe.base as pipeBase
import lsst.pipe.base.connectionTypes as cT

from lsstDebug import getDebugFrame
import lsst.pex.config as pexConfig

import lsst.afw.image as afwImage
import lsst.afw.math as afwMath
import lsst.afw.detection as afwDetection
import lsst.afw.display as afwDisplay
from lsst.afw import cameraGeom
from lsst.geom import Box2I, Point2I, Extent2I
from lsst.meas.algorithms import SourceDetectionTask
from lsst.ip.isr import Defects, countMaskedPixels
from lsst.pex.exceptions import InvalidParameterError


class MeasureDefectsConnections(pipeBase.PipelineTaskConnections,
                                dimensions=("instrument", "exposure", "detector")):
    inputExp = cT.Input(
        name="defectExps",
        doc="Input ISR-processed exposures to measure.",
        storageClass="Exposure",
        dimensions=("instrument", "detector", "exposure"),
        multiple=False
    )
    camera = cT.PrerequisiteInput(
        name='camera',
        doc="Camera associated with this exposure.",
        storageClass="Camera",
        dimensions=("instrument", ),
        isCalibration=True,
    )

    outputDefects = cT.Output(
        name="singleExpDefects",
        doc="Output measured defects.",
        storageClass="Defects",
        dimensions=("instrument", "detector", "exposure"),
    )


class MeasureDefectsTaskConfig(pipeBase.PipelineTaskConfig,
                               pipelineConnections=MeasureDefectsConnections):
    """Configuration for measuring defects from a list of exposures
    """

    thresholdType = pexConfig.ChoiceField(
        dtype=str,
        doc=("Defects threshold type: ``STDEV`` or ``VALUE``. If ``VALUE``, cold pixels will be found "
             "in flats, and hot pixels in darks. If ``STDEV``, cold and hot pixels will be found "
             "in flats, and hot pixels in darks."),
        default='STDEV',
        allowed={'STDEV': "Use a multiple of the image standard deviation to determine detection threshold.",
                 'VALUE': "Use pixel value to determine detection threshold."},
    )
    doVampirePixels = pexConfig.Field(
        dtype=bool,
        doc=("Search for vampire pixels (bright pixels surrounded by ring of low flux) in ComCam "
             "flatBootstrap and mask the area arount them."),
        default=False,
    )
    thresholdVampirePixels = pexConfig.Field(
        dtype=float,
        doc=("Pixel value threshold to find bright pixels in ComCam flatBootstrap."),
        default=1.9,
    )
    radiusVampirePixels = pexConfig.Field(
        dtype=int,
        doc=("Radius (in pixels) of the area to mask around ComCam flatBootstrap bright pixels."),
        default=8,
    )
    darkCurrentThreshold = pexConfig.Field(
        dtype=float,
        doc=("If thresholdType=``VALUE``, dark current threshold (in e-/sec) to define "
             "hot/bright pixels in dark images. Unused if thresholdType==``STDEV``."),
        default=5,
    )
    biasThreshold = pexConfig.Field(
        dtype=float,
        doc=("If thresholdType==``VALUE``, bias threshold (in ADU) to define "
             "hot/bright pixels in bias frame. Unused if thresholdType==``STDEV``."),
        default=1000.0,
    )
    fracThresholdFlat = pexConfig.Field(
        dtype=float,
        doc=("If thresholdType=``VALUE``, fractional threshold to define cold/dark "
             "pixels in flat images (fraction of the mean value per amplifier)."
             "Unused if thresholdType==``STDEV``."),
        default=0.8,
    )
    nSigmaBright = pexConfig.Field(
        dtype=float,
        doc=("If thresholdType=``STDEV``, number of sigma above mean for bright/hot "
             "pixel detection. The default value was found to be "
             "appropriate for some LSST sensors in DM-17490. "
             "Unused if thresholdType==``VALUE``"),
        default=4.8,
    )
    nSigmaDark = pexConfig.Field(
        dtype=float,
        doc=("If thresholdType=``STDEV``, number of sigma below mean for dark/cold pixel "
             "detection. The default value was found to be "
             "appropriate for some LSST sensors in DM-17490. "
             "Unused if thresholdType==``VALUE``"),
        default=-5.0,
    )
    nPixBorderUpDown = pexConfig.Field(
        dtype=int,
        doc="Number of pixels to exclude from top & bottom of image when looking for defects.",
        default=0,
    )
    nPixBorderLeftRight = pexConfig.Field(
        dtype=int,
        doc="Number of pixels to exclude from left & right of image when looking for defects.",
        default=0,
    )
    badOnAndOffPixelColumnThreshold = pexConfig.Field(
        dtype=int,
        doc=("If BPC is the set of all the bad pixels in a given column (not necessarily consecutive) "
             "and the size of BPC is at least 'badOnAndOffPixelColumnThreshold', all the pixels between the "
             "pixels that satisfy minY (BPC) and maxY (BPC) will be marked as bad, with 'Y' being the long "
             "axis of the amplifier (and 'X' the other axis, which for a column is a constant for all "
             "pixels in the set BPC). If there are more than 'goodPixelColumnGapThreshold' consecutive "
             "non-bad pixels in BPC, an exception to the above is made and those consecutive "
             "'goodPixelColumnGapThreshold' are not marked as bad."),
        default=50,
    )
    goodPixelColumnGapThreshold = pexConfig.Field(
        dtype=int,
        doc=("Size, in pixels, of usable consecutive pixels in a column with on and off bad pixels (see "
             "'badOnAndOffPixelColumnThreshold')."),
        default=30,
    )
    badPixelsToFillColumnThreshold = pexConfig.Field(
        dtype=float,
        doc=("If the number of bad pixels in an amplifier column is above this threshold "
             "then the full amplifier column will be marked bad.  This operation is performed after "
             "any merging of blinking columns performed with badOnAndOffPixelColumnThreshold. If this"
             "value is less than 0 then no bad column filling will be performed."),
        default=-1,
    )
    saturatedColumnMask = pexConfig.Field(
        dtype=str,
        default="SAT",
        doc="Saturated mask plane for dilation.",
    )
    saturatedColumnDilationRadius = pexConfig.Field(
        dtype=int,
        doc=("Dilation radius (along rows) to use to expand saturated columns "
             "to mitigate glow."),
        default=0,
    )
    saturatedPixelsToFillColumnThreshold = pexConfig.Field(
        dtype=int,
        doc=("If the number of saturated pixels in an amplifier column is above this threshold "
             "then the full amplifier column will be marked bad. If this value is less than 0"
             "then no saturated column filling will be performed."),
        default=-1,
    )

    def validate(self):
        super().validate()
        if self.nSigmaBright < 0.0:
            raise ValueError("nSigmaBright must be above 0.0.")
        if self.nSigmaDark > 0.0:
            raise ValueError("nSigmaDark must be below 0.0.")


class MeasureDefectsTask(pipeBase.PipelineTask):
    """Measure the defects from one exposure.
    """

    ConfigClass = MeasureDefectsTaskConfig
    _DefaultName = 'cpDefectMeasure'

    def run(self, inputExp, camera):
        """Measure one exposure for defects.

        Parameters
        ----------
        inputExp : `lsst.afw.image.Exposure`
             Exposure to examine.
        camera : `lsst.afw.cameraGeom.Camera`
             Camera to use for metadata.

        Returns
        -------
        results : `lsst.pipe.base.Struct`
             Results struct containing:

             ``outputDefects``
                 The defects measured from this exposure
                 (`lsst.ip.isr.Defects`).
        """
        detector = inputExp.getDetector()
        try:
            filterName = inputExp.getFilter().physicalLabel
        except AttributeError:
            filterName = None

        defects = self._findHotAndColdPixels(inputExp)

        datasetType = inputExp.getMetadata().get('IMGTYPE', 'UNKNOWN')
        msg = "Found %s defects containing %s pixels in %s"
        self.log.info(msg, len(defects), self._nPixFromDefects(defects), datasetType)

        defects.updateMetadataFromExposures([inputExp])
        defects.updateMetadata(camera=camera, detector=detector, filterName=filterName,
                               setCalibId=True, setDate=True,
                               cpDefectGenImageType=datasetType)

        return pipeBase.Struct(
            outputDefects=defects,
        )

    @staticmethod
    def _nPixFromDefects(defects):
        """Count pixels in a defect.

        Parameters
        ----------
        defects : `lsst.ip.isr.Defects`
            Defects to measure.

        Returns
        -------
        nPix : `int`
            Number of defect pixels.
        """
        nPix = 0
        for defect in defects:
            nPix += defect.getBBox().getArea()
        return nPix

    def getVampirePixels(self, ampImg):
        """Find vampire pixels (bright pixels in flats) and get footprint of
        extended area around them,

        Parameters
        ----------
        ampImg : `lsst.afw.image._maskedImage.MaskedImageF`
            The amplifier masked image to do the vampire pixels search on.

        Returns
        -------
        fs_grow : `lsst.afw.detection._detection.FootprintSet`
            The footprint set of areas around vampire pixels in the amplifier.
        """

        # Find bright pixels
        thresh = afwDetection.Threshold(self.config.thresholdVampirePixels)
        # Bright pixels footprint grown by a radius of radiusVampire pixels
        fs = afwDetection.FootprintSet(ampImg, thresh)
        fs_grow = afwDetection.FootprintSet(fs, rGrow=self.config.radiusVampirePixels, isotropic=True)

        return fs_grow

    def _findHotAndColdPixels(self, exp):
        """Find hot and cold pixels in an image.

        Using config-defined thresholds on a per-amp basis, mask
        pixels that are nSigma above threshold in dark frames (hot
        pixels), or nSigma away from the clipped mean in flats (hot &
        cold pixels).

        Parameters
        ----------
        exp : `lsst.afw.image.exposure.Exposure`
            The exposure in which to find defects.

        Returns
        -------
        defects : `lsst.ip.isr.Defects`
            The defects found in the image.
        """
        self._setEdgeBits(exp)
        maskedIm = exp.maskedImage

        # the detection polarity for afwDetection, True for positive,
        # False for negative, and therefore True for darks as they only have
        # bright pixels, and both for flats, as they have bright and dark pix
        footprintList = []

        hotPixelCount = {}
        coldPixelCount = {}

        for amp in exp.getDetector():
            ampName = amp.getName()

            hotPixelCount[ampName] = 0
            coldPixelCount[ampName] = 0

            ampImg = maskedIm[amp.getBBox()].clone()

            # crop ampImage depending on where the amp lies in the image
            if self.config.nPixBorderLeftRight:
                if ampImg.getX0() == 0:
                    ampImg = ampImg[self.config.nPixBorderLeftRight:, :, afwImage.LOCAL]
                else:
                    ampImg = ampImg[:-self.config.nPixBorderLeftRight, :, afwImage.LOCAL]
            if self.config.nPixBorderUpDown:
                if ampImg.getY0() == 0:
                    ampImg = ampImg[:, self.config.nPixBorderUpDown:, afwImage.LOCAL]
                else:
                    ampImg = ampImg[:, :-self.config.nPixBorderUpDown, afwImage.LOCAL]

            if self._getNumGoodPixels(ampImg) == 0:  # amp contains no usable pixels
                continue

            if self.config.doVampirePixels:
                # This is only applied in LSSTComCam flatBootstrap pipeline
                footprintSet_VampirePixel = self.getVampirePixels(ampImg)
                footprintSet_VampirePixel.setMask(maskedIm.mask, ("BAD"))

            # Remove a background estimate
            meanClip = afwMath.makeStatistics(ampImg, afwMath.MEANCLIP, ).getValue()
            ampImg -= meanClip

            # Determine thresholds
            stDev = afwMath.makeStatistics(ampImg, afwMath.STDEVCLIP, ).getValue()
            expTime = exp.getInfo().getVisitInfo().getExposureTime()
            datasetType = exp.getMetadata().get('IMGTYPE', 'UNKNOWN')
            if np.isnan(expTime):
                self.log.warning("expTime=%s for AMP %s in %s. Setting expTime to 1 second",
                                 expTime, ampName, datasetType)
                expTime = 1.
            thresholdType = self.config.thresholdType
            if thresholdType == 'VALUE':
                # LCA-128 and eoTest: bright/hot pixels in dark images are
                # defined as any pixel with more than 5 e-/s of dark current.
                # We scale by the exposure time.
                if datasetType.lower() == 'dark':
                    # hot pixel threshold
                    valueThreshold = self.config.darkCurrentThreshold*expTime/amp.getGain()
                elif datasetType.lower() == 'bias':
                    # hot pixel threshold, no exposure time.
                    valueThreshold = self.config.biasThreshold
                else:
                    # LCA-128 and eoTest: dark/cold pixels in flat images as
                    # defined as any pixel with photoresponse <80% of
                    # the mean (at 500nm).

                    # We subtracted the mean above, so the threshold will be
                    # negative cold pixel threshold.
                    valueThreshold = (self.config.fracThresholdFlat-1)*meanClip
                # Find equivalent sigma values.
                if stDev == 0.0:
                    self.log.warning("stDev=%s for AMP %s in %s. Setting nSigma to inf.",
                                     stDev, ampName, datasetType)
                    nSigmaList = [np.inf]
                else:
                    nSigmaList = [valueThreshold/stDev]
            else:
                hotPixelThreshold = self.config.nSigmaBright
                coldPixelThreshold = self.config.nSigmaDark
                if datasetType.lower() == 'dark':
                    nSigmaList = [hotPixelThreshold]
                    valueThreshold = stDev*hotPixelThreshold
                elif datasetType.lower() == 'bias':
                    self.log.warning(
                        "Bias frame detected, but thresholdType == STDEV; not looking for defects.",
                    )
                    return Defects.fromFootprintList([])
                else:
                    nSigmaList = [hotPixelThreshold, coldPixelThreshold]
                    valueThreshold = [x*stDev for x in nSigmaList]

            self.log.info("Image type: %s. Amp: %s. Threshold Type: %s. Sigma values and Pixel"
                          "Values (hot and cold pixels thresholds): %s, %s",
                          datasetType, ampName, thresholdType, nSigmaList, valueThreshold)
            mergedSet = None
            for sigma in nSigmaList:
                nSig = np.abs(sigma)
                self.debugHistogram('ampFlux', ampImg, nSig, exp)
                polarity = {-1: False, 1: True}[np.sign(sigma)]

                threshold = afwDetection.createThreshold(nSig, 'stdev', polarity=polarity)

                try:
                    footprintSet = afwDetection.FootprintSet(ampImg, threshold)
                except InvalidParameterError:
                    # This occurs if the image sigma value is 0.0.
                    # Let's mask the whole area.
                    minValue = np.nanmin(ampImg.image.array) - 1.0
                    threshold = afwDetection.createThreshold(minValue, 'value', polarity=True)
                    footprintSet = afwDetection.FootprintSet(ampImg, threshold)

                footprintSet.setMask(maskedIm.mask, ("DETECTED" if polarity else "DETECTED_NEGATIVE"))

                if mergedSet is None:
                    mergedSet = footprintSet
                else:
                    mergedSet.merge(footprintSet)

                if polarity:
                    # hot pixels
                    for fp in footprintSet.getFootprints():
                        hotPixelCount[ampName] += fp.getArea()
                else:
                    # cold pixels
                    for fp in footprintSet.getFootprints():
                        coldPixelCount[ampName] += fp.getArea()

            if self.config.doVampirePixels:
                # Count the number of pixels masked
                vampirePixelCount = 0
                for fp in footprintSet_VampirePixel.getFootprints():
                    vampirePixelCount += fp.getArea()
                self.log.info("%s Vampire pixels are masked", vampirePixelCount)
                # Add vampire pixels to footprint set
                mergedSet.merge(footprintSet_VampirePixel)

            footprintList += mergedSet.getFootprints()

            self.debugView('defectMap', ampImg,
                           Defects.fromFootprintList(mergedSet.getFootprints()), exp.getDetector())

        defects = Defects.fromFootprintList(footprintList)
        defects = self.dilateSaturatedColumns(exp, defects)
        defects, _ = self.maskBlocksIfIntermitentBadPixelsInColumn(defects)
        defects, count = self.maskBadColumns(exp, defects)
        # We want this to reflect the number of completely bad columns.
        defects.updateCounters(columns=count, hot=hotPixelCount, cold=coldPixelCount)

        return defects

    @staticmethod
    def _getNumGoodPixels(maskedIm, badMaskString="NO_DATA"):
        """Return the number of non-bad pixels in the image."""
        nPixels = maskedIm.mask.array.size
        nBad = countMaskedPixels(maskedIm, badMaskString)
        return nPixels - nBad

    def _setEdgeBits(self, exposureOrMaskedImage, maskplaneToSet='EDGE'):
        """Set edge bits on an exposure or maskedImage.

        Raises
        ------
        TypeError
            Raised if parameter ``exposureOrMaskedImage`` is an invalid type.
        """
        if isinstance(exposureOrMaskedImage, afwImage.Exposure):
            mi = exposureOrMaskedImage.maskedImage
        elif isinstance(exposureOrMaskedImage, afwImage.MaskedImage):
            mi = exposureOrMaskedImage
        else:
            t = type(exposureOrMaskedImage)
            raise TypeError(f"Function supports exposure or maskedImage but not {t}")

        MASKBIT = mi.mask.getPlaneBitMask(maskplaneToSet)
        if self.config.nPixBorderLeftRight:
            mi.mask[: self.config.nPixBorderLeftRight, :, afwImage.LOCAL] |= MASKBIT
            mi.mask[-self.config.nPixBorderLeftRight:, :, afwImage.LOCAL] |= MASKBIT
        if self.config.nPixBorderUpDown:
            mi.mask[:, : self.config.nPixBorderUpDown, afwImage.LOCAL] |= MASKBIT
            mi.mask[:, -self.config.nPixBorderUpDown:, afwImage.LOCAL] |= MASKBIT

    def maskBlocksIfIntermitentBadPixelsInColumn(self, defects):
        """Mask blocks in a column if there are on-and-off bad pixels

        If there's a column with on and off bad pixels, mask all the
        pixels in between, except if there is a large enough gap of
        consecutive good pixels between two bad pixels in the column.

        Parameters
        ----------
        defects : `lsst.ip.isr.Defects`
            The defects found in the image so far

        Returns
        -------
        defects : `lsst.ip.isr.Defects`
            If the number of bad pixels in a column is not larger or
            equal than self.config.badPixelColumnThreshold, the input
            list is returned. Otherwise, the defects list returned
            will include boxes that mask blocks of on-and-of pixels.
        badColumnCount : `int`
            Number of bad columns partially masked.
        """
        badColumnCount = 0
        # Get the (x, y) values of each bad pixel in amp.
        coordinates = []
        for defect in defects:
            bbox = defect.getBBox()
            x0, y0 = bbox.getMinX(), bbox.getMinY()
            deltaX0, deltaY0 = bbox.getDimensions()
            for j in np.arange(y0, y0+deltaY0):
                for i in np.arange(x0, x0 + deltaX0):
                    coordinates.append((i, j))

        x, y = [], []
        for coordinatePair in coordinates:
            x.append(coordinatePair[0])
            y.append(coordinatePair[1])

        x = np.array(x)
        y = np.array(y)
        # Find the defects with same "x" (vertical) coordinate (column).
        unique, counts = np.unique(x, return_counts=True)
        multipleX = []
        for (a, b) in zip(unique, counts):
            if b >= self.config.badOnAndOffPixelColumnThreshold:
                multipleX.append(a)
        if len(multipleX) != 0:
            defects = self._markBlocksInBadColumn(x, y, multipleX, defects)
            badColumnCount += 1

        return defects, badColumnCount

    def dilateSaturatedColumns(self, exp, defects):
        """Dilate saturated columns by a configurable amount.

        Parameters
        ----------
        exp : `lsst.afw.image.exposure.Exposure`
            The exposure in which to find defects.
        defects : `lsst.ip.isr.Defects`
            The defects found in the image so far

        Returns
        -------
        defects : `lsst.ip.isr.Defects`
            The expanded defects.
        """
        if self.config.saturatedColumnDilationRadius <= 0:
            # This is a no-op.
            return defects

        mask = afwImage.Mask.getPlaneBitMask(self.config.saturatedColumnMask)

        satY, satX = np.where((exp.mask.array & mask) > 0)

        if len(satX) == 0:
            # No saturated pixels, nothing to do.
            return defects

        radius = self.config.saturatedColumnDilationRadius

        with defects.bulk_update():
            for index in range(len(satX)):
                minX = np.clip(satX[index] - radius, 0, None)
                maxX = np.clip(satX[index] + radius, None, exp.image.array.shape[1] - 1)
                s = Box2I(minimum=Point2I(minX, satY[index]),
                          maximum=Point2I(maxX, satY[index]))
                defects.append(s)

        return defects

    def maskBadColumns(self, exp, defects):
        """Mask full amplifier columns if they are sufficiently bad.

        Parameters
        ----------
        defects : `lsst.ip.isr.Defects`
            The defects found in the image so far

        Returns
        -------
        exp : `lsst.afw.image.exposure.Exposure`
            The exposure in which to find defects.
        defects : `lsst.ip.isr.Defects`
            If the number of bad pixels in a column is not larger or
            equal than self.config.badPixelColumnThreshold, the input
            list is returned. Otherwise, the defects list returned
            will include boxes that mask blocks of on-and-of pixels.
        badColumnCount : `int`
            Number of bad columns masked.
        """
        # Render the defects into an image.
        defectImage = afwImage.ImageI(exp.getBBox())

        for defect in defects:
            defectImage[defect.getBBox()] = 1

        badColumnCount = 0

        if self.config.badPixelsToFillColumnThreshold > 0:
            with defects.bulk_update():
                for amp in exp.getDetector():
                    subImage = defectImage[amp.getBBox()].array
                    nInCol = np.sum(subImage, axis=0)

                    badColIndices, = (nInCol >= self.config.badPixelsToFillColumnThreshold).nonzero()
                    badColumns = badColIndices + amp.getBBox().getMinX()

                    for badColumn in badColumns:
                        s = Box2I(minimum=Point2I(badColumn, amp.getBBox().getMinY()),
                                  maximum=Point2I(badColumn, amp.getBBox().getMaxY()))
                        defects.append(s)

                    badColumnCount += len(badColIndices)

        if self.config.saturatedPixelsToFillColumnThreshold > 0:
            mask = afwImage.Mask.getPlaneBitMask(self.config.saturatedColumnMask)

            with defects.bulk_update():
                for amp in exp.getDetector():
                    subMask = exp.mask[amp.getBBox()].array
                    # Turn all the SAT bits into 1s
                    subMask &= mask
                    subMask[subMask > 0] = 1

                    nInCol = np.sum(subMask, axis=0)

                    badColIndices, = (nInCol >= self.config.saturatedPixelsToFillColumnThreshold).nonzero()
                    badColumns = badColIndices + amp.getBBox().getMinX()

                    for badColumn in badColumns:
                        s = Box2I(minimum=Point2I(badColumn, amp.getBBox().getMinY()),
                                  maximum=Point2I(badColumn, amp.getBBox().getMaxY()))
                        defects.append(s)

                    badColumnCount += len(badColIndices)

        return defects, badColumnCount

    def _markBlocksInBadColumn(self, x, y, multipleX, defects):
        """Mask blocks in a column if number of on-and-off bad pixels is above
        threshold.

        This function is called if the number of on-and-off bad pixels
        in a column is larger or equal than
        self.config.badOnAndOffPixelColumnThreshold.

        Parameters
        ---------
        x : `list`
            Lower left x coordinate of defect box. x coordinate is
            along the short axis if amp.
        y : `list`
            Lower left y coordinate of defect box. x coordinate is
            along the long axis if amp.
        multipleX : list
            List of x coordinates in amp. with multiple bad pixels
            (i.e., columns with defects).
        defects : `lsst.ip.isr.Defects`
            The defcts found in the image so far

        Returns
        -------
        defects : `lsst.ip.isr.Defects`
            The defects list returned that will include boxes that
            mask blocks of on-and-of pixels.
        """
        with defects.bulk_update():
            goodPixelColumnGapThreshold = self.config.goodPixelColumnGapThreshold
            for x0 in multipleX:
                index = np.where(x == x0)
                multipleY = y[index]  # multipleY and multipleX are in 1-1 correspondence.
                multipleY.sort()  # Ensure that the y values are sorted to look for gaps.
                minY, maxY = np.min(multipleY), np.max(multipleY)
                # Next few lines: don't mask pixels in column if gap
                # of good pixels between two consecutive bad pixels is
                # larger or equal than 'goodPixelColumnGapThreshold'.
                diffIndex = np.where(np.diff(multipleY) >= goodPixelColumnGapThreshold)[0]
                if len(diffIndex) != 0:
                    limits = [minY]  # put the minimum first
                    for gapIndex in diffIndex:
                        limits.append(multipleY[gapIndex])
                        limits.append(multipleY[gapIndex+1])
                    limits.append(maxY)  # maximum last
                    for i in np.arange(0, len(limits)-1, 2):
                        s = Box2I(minimum=Point2I(x0, limits[i]), maximum=Point2I(x0, limits[i+1]))
                        defects.append(s)
                else:  # No gap is large enough
                    s = Box2I(minimum=Point2I(x0, minY), maximum=Point2I(x0, maxY))
                    defects.append(s)
        return defects

    def debugView(self, stepname, ampImage, defects, detector):  # pragma: no cover
        """Plot the defects found by the task.

        Parameters
        ----------
        stepname : `str`
            Debug frame to request.
        ampImage : `lsst.afw.image.MaskedImage`
            Amplifier image to display.
        defects : `lsst.ip.isr.Defects`
            The defects to plot.
        detector : `lsst.afw.cameraGeom.Detector`
            Detector holding camera geometry.
        """
        frame = getDebugFrame(self._display, stepname)
        if frame:
            disp = afwDisplay.Display(frame=frame)
            disp.scale('asinh', 'zscale')
            disp.setMaskTransparency(80)
            disp.setMaskPlaneColor("BAD", afwDisplay.RED)

            maskedIm = ampImage.clone()
            defects.maskPixels(maskedIm, "BAD")

            mpDict = maskedIm.mask.getMaskPlaneDict()
            for plane in mpDict.keys():
                if plane in ['BAD']:
                    continue
                disp.setMaskPlaneColor(plane, afwDisplay.IGNORE)

            disp.setImageColormap('gray')
            disp.mtv(maskedIm)
            cameraGeom.utils.overlayCcdBoxes(detector, isTrimmed=True, display=disp)
            prompt = "Press Enter to continue [c]... "
            while True:
                ans = input(prompt).lower()
                if ans in ('', 'c', ):
                    break

    def debugHistogram(self, stepname, ampImage, nSigmaUsed, exp):
        """Make a histogram of the distribution of pixel values for
        each amp.

        The main image data histogram is plotted in blue.  Edge
        pixels, if masked, are in red.  Note that masked edge pixels
        do not contribute to the underflow and overflow numbers.

        Note that this currently only supports the 16-amp LSST
        detectors.

        Parameters
        ----------
        stepname : `str`
            Debug frame to request.
        ampImage : `lsst.afw.image.MaskedImage`
            Amplifier image to display.
        nSigmaUsed : `float`
            The number of sigma used for detection
        exp : `lsst.afw.image.exposure.Exposure`
            The exposure in which the defects were found.
        """
        frame = getDebugFrame(self._display, stepname)
        if frame:
            import matplotlib.pyplot as plt

            detector = exp.getDetector()
            nX = np.floor(np.sqrt(len(detector)))
            nY = len(detector) // nX
            fig, ax = plt.subplots(nrows=int(nY), ncols=int(nX), sharex='col', sharey='row', figsize=(13, 10))

            expTime = exp.getInfo().getVisitInfo().getExposureTime()

            for (amp, a) in zip(reversed(detector), ax.flatten()):
                mi = exp.maskedImage[amp.getBBox()]

                # normalize by expTime as we plot in ADU/s and don't
                # always work with master calibs
                mi.image.array /= expTime
                stats = afwMath.makeStatistics(mi, afwMath.MEANCLIP | afwMath.STDEVCLIP)
                mean, sigma = stats.getValue(afwMath.MEANCLIP), stats.getValue(afwMath.STDEVCLIP)
                # Get array of pixels
                EDGEBIT = exp.maskedImage.mask.getPlaneBitMask("EDGE")
                imgData = mi.image.array[(mi.mask.array & EDGEBIT) == 0].flatten()
                edgeData = mi.image.array[(mi.mask.array & EDGEBIT) != 0].flatten()

                thrUpper = mean + nSigmaUsed*sigma
                thrLower = mean - nSigmaUsed*sigma

                nRight = len(imgData[imgData > thrUpper])
                nLeft = len(imgData[imgData < thrLower])

                nsig = nSigmaUsed + 1.2  # add something small so the edge of the plot is out from level used
                leftEdge = mean - nsig * nSigmaUsed*sigma
                rightEdge = mean + nsig * nSigmaUsed*sigma
                nbins = np.linspace(leftEdge, rightEdge, 1000)
                ey, bin_borders, patches = a.hist(edgeData, histtype='step', bins=nbins,
                                                  lw=1, edgecolor='red')
                y, bin_borders, patches = a.hist(imgData, histtype='step', bins=nbins,
                                                 lw=3, edgecolor='blue')

                # Report number of entries in over- and under-flow
                # bins, i.e. off the edges of the histogram
                nOverflow = len(imgData[imgData > rightEdge])
                nUnderflow = len(imgData[imgData < leftEdge])

                # Put v-lines and textboxes in
                a.axvline(thrUpper, c='k')
                a.axvline(thrLower, c='k')
                msg = f"{amp.getName()}\nmean:{mean: .2f}\n$\\sigma$:{sigma: .2f}"
                a.text(0.65, 0.6, msg, transform=a.transAxes, fontsize=11)
                msg = f"nLeft:{nLeft}\nnRight:{nRight}\nnOverflow:{nOverflow}\nnUnderflow:{nUnderflow}"
                a.text(0.03, 0.6, msg, transform=a.transAxes, fontsize=11.5)

                # set axis limits and scales
                a.set_ylim([1., 1.7*np.max(y)])
                lPlot, rPlot = a.get_xlim()
                a.set_xlim(np.array([lPlot, rPlot]))
                a.set_yscale('log')
                a.set_xlabel("ADU/s")
            fig.show()
            prompt = "Press Enter or c to continue [chp]..."
            while True:
                ans = input(prompt).lower()
                if ans in ("", " ", "c",):
                    break
                elif ans in ("p", ):
                    import pdb
                    pdb.set_trace()
                elif ans in ("h", ):
                    print("[h]elp [c]ontinue [p]db")
            plt.close()


class MeasureDefectsCombinedConnections(pipeBase.PipelineTaskConnections,
                                        dimensions=("instrument", "detector")):
    inputExp = cT.Input(
        name="dark",
        doc="Input ISR-processed combined exposure to measure.",
        storageClass="ExposureF",
        dimensions=("instrument", "detector"),
        multiple=False,
        isCalibration=True,
    )
    camera = cT.PrerequisiteInput(
        name='camera',
        doc="Camera associated with this exposure.",
        storageClass="Camera",
        dimensions=("instrument", ),
        isCalibration=True,
    )

    outputDefects = cT.Output(
        name="cpDefectsFromDark",
        doc="Output measured defects.",
        storageClass="Defects",
        dimensions=("instrument", "detector"),
    )


class MeasureDefectsCombinedTaskConfig(MeasureDefectsTaskConfig,
                                       pipelineConnections=MeasureDefectsCombinedConnections):
    """Configuration for measuring defects from combined exposures.
    """
    pass


class MeasureDefectsCombinedTask(MeasureDefectsTask):
    """Task to measure defects in combined images."""

    ConfigClass = MeasureDefectsCombinedTaskConfig
    _DefaultName = "cpDefectMeasureCombined"


class MeasureDefectsCombinedWithFilterConnections(pipeBase.PipelineTaskConnections,
                                                  dimensions=("instrument", "detector", "physical_filter")):
    """Task to measure defects in combined flats under a certain filter."""
    inputExp = cT.Input(
        name="flat",
        doc="Input ISR-processed combined exposure to measure.",
        storageClass="ExposureF",
        dimensions=("instrument", "detector", "physical_filter"),
        multiple=False,
        isCalibration=True,
    )
    camera = cT.PrerequisiteInput(
        name='camera',
        doc="Camera associated with this exposure.",
        storageClass="Camera",
        dimensions=("instrument", ),
        isCalibration=True,
    )

    outputDefects = cT.Output(
        name="cpDefectsFromFlat",
        doc="Output measured defects.",
        storageClass="Defects",
        dimensions=("instrument", "detector", "physical_filter"),
    )


class MeasureDefectsCombinedWithFilterTaskConfig(
    MeasureDefectsTaskConfig,
        pipelineConnections=MeasureDefectsCombinedWithFilterConnections):
    """Configuration for measuring defects from combined exposures.
    """
    pass


class MeasureDefectsCombinedWithFilterTask(MeasureDefectsTask):
    """Task to measure defects in combined images."""

    ConfigClass = MeasureDefectsCombinedWithFilterTaskConfig
    _DefaultName = "cpDefectMeasureWithFilterCombined"


class MergeDefectsConnections(pipeBase.PipelineTaskConnections,
                              dimensions=("instrument", "detector")):
    inputDefects = cT.Input(
        name="singleExpDefects",
        doc="Measured defect lists.",
        storageClass="Defects",
        dimensions=("instrument", "detector", "exposure",),
        multiple=True,
    )
    camera = cT.PrerequisiteInput(
        name='camera',
        doc="Camera associated with these defects.",
        storageClass="Camera",
        dimensions=("instrument", ),
        isCalibration=True,
    )

    mergedDefects = cT.Output(
        name="defects",
        doc="Final merged defects.",
        storageClass="Defects",
        dimensions=("instrument", "detector"),
        multiple=False,
        isCalibration=True,
    )


class MergeDefectsTaskConfig(pipeBase.PipelineTaskConfig,
                             pipelineConnections=MergeDefectsConnections):
    """Configuration for merging single exposure defects.
    """

    assertSameRun = pexConfig.Field(
        dtype=bool,
        doc=("Ensure that all visits are from the same run? Raises if this is not the case, or "
             "if the run key isn't found."),
        default=False,  # false because most obs_packages don't have runs. obs_lsst/ts8 overrides this.
    )
    ignoreFilters = pexConfig.Field(
        dtype=bool,
        doc=("Set the filters used in the CALIB_ID to NONE regardless of the filters on the input"
             " images. Allows mixing of filters in the input flats. Set to False if you think"
             " your defects might be chromatic and want to have registry support for varying"
             " defects with respect to filter."),
        default=True,
    )
    nullFilterName = pexConfig.Field(
        dtype=str,
        doc=("The name of the null filter if ignoreFilters is True. Usually something like NONE or EMPTY"),
        default="NONE",
    )
    combinationMode = pexConfig.ChoiceField(
        doc="Which types of defects to identify",
        dtype=str,
        default="FRACTION",
        allowed={
            "AND": "Logical AND the pixels found in each visit to form set ",
            "OR": "Logical OR the pixels found in each visit to form set ",
            "FRACTION": "Use pixels found in more than config.combinationFraction of visits ",
        }
    )
    combinationFraction = pexConfig.RangeField(
        dtype=float,
        doc=("The fraction (0..1) of visits in which a pixel was found to be defective across"
             " the visit list in order to be marked as a defect. Note, upper bound is exclusive, so use"
             " mode AND to require pixel to appear in all images."),
        default=0.7,
        min=0,
        max=1,
    )
    nPixBorderUpDown = pexConfig.Field(
        dtype=int,
        doc="Number of pixels on top & bottom of image to mask as defects if edgesAsDefects is True.",
        default=5,
    )
    nPixBorderLeftRight = pexConfig.Field(
        dtype=int,
        doc="Number of pixels on left & right of image to mask as defects if edgesAsDefects is True.",
        default=5,
    )
    edgesAsDefects = pexConfig.Field(
        dtype=bool,
        doc="Mark all edge pixels, as defined by nPixBorder[UpDown, LeftRight], as defects.",
        default=False,
    )


class MergeDefectsTask(pipeBase.PipelineTask):
    """Merge the defects from multiple exposures.
    """

    ConfigClass = MergeDefectsTaskConfig
    _DefaultName = 'cpDefectMerge'

    def run(self, inputDefects, camera):
        """Merge a list of single defects to find the common defect regions.

        Parameters
        ----------
        inputDefects : `list` [`lsst.ip.isr.Defects`]
             Partial defects from a single exposure.
        camera : `lsst.afw.cameraGeom.Camera`
             Camera to use for metadata.

        Returns
        -------
        results : `lsst.pipe.base.Struct`
             Results struct containing:

             ``mergedDefects``
                 The defects merged from the input lists
                 (`lsst.ip.isr.Defects`).
        """
        detectorId = inputDefects[0].getMetadata().get('DETECTOR', None)
        if detectorId is None:
            raise RuntimeError("Cannot identify detector id.")
        detector = camera[detectorId]

        imageTypes = set()
        for inDefect in inputDefects:
            imageType = inDefect.getMetadata().get('cpDefectGenImageType', 'UNKNOWN')
            imageTypes.add(imageType)

        # Determine common defect pixels separately for each input image type.
        splitDefects = list()
        for imageType in imageTypes:
            sumImage = afwImage.MaskedImageF(detector.getBBox())
            count = 0
            for inDefect in inputDefects:
                if imageType == inDefect.getMetadata().get('cpDefectGenImageType', 'UNKNOWN'):
                    count += 1
                    for defect in inDefect:
                        sumImage.image[defect.getBBox()] += 1.0
            sumImage /= count
            nDetected = len(np.where(sumImage.getImage().getArray() > 0)[0])
            self.log.info("Pre-merge %s pixels with non-zero detections for %s", nDetected, imageType)

            if self.config.combinationMode == 'AND':
                threshold = 1.0
            elif self.config.combinationMode == 'OR':
                threshold = 0.0
            elif self.config.combinationMode == 'FRACTION':
                threshold = self.config.combinationFraction
            else:
                raise RuntimeError(f"Got unsupported combinationMode {self.config.combinationMode}")
            indices = np.where(sumImage.getImage().getArray() > threshold)
            BADBIT = sumImage.getMask().getPlaneBitMask('BAD')
            sumImage.getMask().getArray()[indices] |= BADBIT
            self.log.info("Post-merge %s pixels marked as defects for %s", len(indices[0]), imageType)
            partialDefect = Defects.fromMask(sumImage, 'BAD')
            splitDefects.append(partialDefect)

        # Do final combination of separate image types
        finalImage = afwImage.MaskedImageF(detector.getBBox())
        for inDefect in splitDefects:
            for defect in inDefect:
                finalImage.image[defect.getBBox()] += 1
        finalImage /= len(splitDefects)
        nDetected = len(np.where(finalImage.getImage().getArray() > 0)[0])
        self.log.info("Pre-final merge %s pixels with non-zero detections", nDetected)

        # This combination is the OR of all image types
        threshold = 0.0
        indices = np.where(finalImage.getImage().getArray() > threshold)
        BADBIT = finalImage.getMask().getPlaneBitMask('BAD')
        finalImage.getMask().getArray()[indices] |= BADBIT
        self.log.info("Post-final merge %s pixels marked as defects", len(indices[0]))

        if self.config.edgesAsDefects:
            self.log.info("Masking edge pixels as defects.")
            # This code follows the pattern from isrTask.maskEdges().
            if self.config.nPixBorderLeftRight > 0:
                box = detector.getBBox()
                subImage = finalImage[box]
                box.grow(Extent2I(-self.config.nPixBorderLeftRight, 0))
                SourceDetectionTask.setEdgeBits(subImage, box, BADBIT)
            if self.config.nPixBorderUpDown > 0:
                box = detector.getBBox()
                subImage = finalImage[box]
                box.grow(Extent2I(0, -self.config.nPixBorderUpDown))
                SourceDetectionTask.setEdgeBits(subImage, box, BADBIT)

        merged = Defects.fromMask(finalImage, 'BAD')
        merged.updateMetadataFromExposures(inputDefects)
        merged.updateMetadata(camera=camera, detector=detector, filterName=None,
                              setCalibId=True, setDate=True)

        return pipeBase.Struct(
            mergedDefects=merged,
        )

# Subclass the MergeDefects task to reduce the input dimensions
# from ("instrument", "detector", "exposure") to
# ("instrument", "detector").


class MergeDefectsCombinedConnections(pipeBase.PipelineTaskConnections,
                                      dimensions=("instrument", "detector")):
    inputDarkDefects = cT.Input(
        name="cpDefectsFromDark",
        doc="Measured defect lists.",
        storageClass="Defects",
        dimensions=("instrument", "detector",),
        multiple=True,
    )
    inputBiasDefects = cT.Input(
        name="cpDefectsFromBias",
        doc="Additional measured defect lists.",
        storageClass="Defects",
        dimensions=("instrument", "detector",),
        multiple=True,
    )
    inputFlatDefects = cT.Input(
        name="cpDefectsFromFlat",
        doc="Additional measured defect lists.",
        storageClass="Defects",
        dimensions=("instrument", "detector", "physical_filter"),
        multiple=True,
    )
    inputManualDefects = cT.Input(
        name="cpManualDefects",
        doc="Additional manual defects.",
        storageClass="Defects",
        dimensions=("instrument", "detector"),
        multiple=True,
        isCalibration=True,
    )
    camera = cT.PrerequisiteInput(
        name='camera',
        doc="Camera associated with these defects.",
        storageClass="Camera",
        dimensions=("instrument", ),
        isCalibration=True,
    )

    mergedDefects = cT.Output(
        name="defects",
        doc="Final merged defects.",
        storageClass="Defects",
        dimensions=("instrument", "detector"),
        multiple=False,
        isCalibration=True,
    )

    def __init__(self, *, config=None):
        super().__init__(config=config)

        if config.doManualDefects is not True:
            del self.inputManualDefects


class MergeDefectsCombinedTaskConfig(MergeDefectsTaskConfig,
                                     pipelineConnections=MergeDefectsCombinedConnections):
    """Configuration for merging defects from combined exposure.
    """
    doManualDefects = pexConfig.Field(
        dtype=bool,
        doc="Apply manual defects?",
        default=False,
    )

    def validate(self):
        super().validate()
        if self.combinationMode != 'OR':
            raise ValueError("combinationMode must be 'OR'")


class MergeDefectsCombinedTask(MergeDefectsTask):
    """Task to measure defects in combined images."""

    ConfigClass = MergeDefectsCombinedTaskConfig
    _DefaultName = "cpMergeDefectsCombined"

    @staticmethod
    def chooseBest(inputs):
        """Select the input with the most exposures used."""
        best = 0
        if len(inputs) > 1:
            nInput = 0
            for num, exp in enumerate(inputs):
                # This technically overcounts by a factor of 3.
                N = len([k for k, v in exp.getMetadata().toDict().items() if "CPP_INPUT_" in k])
                if N > nInput:
                    best = num
                    nInput = N
        return inputs[best]

    def runQuantum(self, butlerQC, inputRefs, outputRefs):
        inputs = butlerQC.get(inputRefs)
        # Turn inputFlatDefects and inputDarkDefects into a list which
        # is what MergeDefectsTask expects.  If there are multiple,
        # use the one with the most inputs.
        tempList = [self.chooseBest(inputs['inputFlatDefects']),
                    self.chooseBest(inputs['inputDarkDefects']),
                    self.chooseBest(inputs['inputBiasDefects'])]

        if "inputManualDefects" in inputs.keys():
            tempList.extend(inputs["inputManualDefects"])

        # Rename inputDefects
        inputsCombined = {'inputDefects': tempList, 'camera': inputs['camera']}

        outputs = super().run(**inputsCombined)
        butlerQC.put(outputs, outputRefs)
