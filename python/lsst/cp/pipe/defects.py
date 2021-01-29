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
from lsst.geom import Box2I, Point2I
from lsst.meas.algorithms import SourceDetectionTask
from lsst.ip.isr import IsrTask, Defects
from .utils import countMaskedPixels
from lsst.pipe.tasks.getRepositoryData import DataRefListRunner

from ._lookupStaticCalibration import lookupStaticCalibration

__all__ = ['MeasureDefectsTaskConfig', 'MeasureDefectsTask',
           'MergeDefectsTaskConfig', 'MergeDefectsTask',
           'FindDefectsTask', 'FindDefectsTaskConfig', ]


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
        lookupFunction=lookupStaticCalibration,
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
    nSigmaBright = pexConfig.Field(
        dtype=float,
        doc=("Number of sigma above mean for bright pixel detection. The default value was found to be",
             " appropriate for some LSST sensors in DM-17490."),
        default=4.8,
    )
    nSigmaDark = pexConfig.Field(
        dtype=float,
        doc=("Number of sigma below mean for dark pixel detection. The default value was found to be",
             " appropriate for some LSST sensors in DM-17490."),
        default=-5.0,
    )
    nPixBorderUpDown = pexConfig.Field(
        dtype=int,
        doc="Number of pixels to exclude from top & bottom of image when looking for defects.",
        default=7,
    )
    nPixBorderLeftRight = pexConfig.Field(
        dtype=int,
        doc="Number of pixels to exclude from left & right of image when looking for defects.",
        default=7,
    )
    badOnAndOffPixelColumnThreshold = pexConfig.Field(
        dtype=int,
        doc=("If BPC is the set of all the bad pixels in a given column (not necessarily consecutive) ",
             "and the size of BPC is at least 'badOnAndOffPixelColumnThreshold', all the pixels between the ",
             "pixels that satisfy minY (BPC) and maxY (BPC) will be marked as bad, with 'Y' being the long ",
             "axis of the amplifier (and 'X' the other axis, which for a column is a constant for all ",
             "pixels in the set BPC). If there are more than 'goodPixelColumnGapThreshold' consecutive ",
             "non-bad pixels in BPC, an exception to the above is made and those consecutive ",
             "'goodPixelColumnGapThreshold' are not marked as bad."),
        default=50,
    )
    goodPixelColumnGapThreshold = pexConfig.Field(
        dtype=int,
        doc=("Size, in pixels, of usable consecutive pixels in a column with on and off bad pixels (see ",
             "'badOnAndOffPixelColumnThreshold')."),
        default=30,
    )

    def validate(self):
        super().validate()
        if self.nSigmaBright < 0.0:
            raise ValueError("nSigmaBright must be above 0.0.")
        if self.nSigmaDark > 0.0:
            raise ValueError("nSigmaDark must be below 0.0.")


class MeasureDefectsTask(pipeBase.PipelineTask, pipeBase.CmdLineTask):
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
             - ``outputDefects` : `lsst.ip.isr.Defects`
                 The defects measured from this exposure.
        """
        detector = inputExp.getDetector()

        filterName = inputExp.getFilterLabel().physicalLabel
        datasetType = inputExp.getMetadata().get('IMGTYPE', 'UNKNOWN')

        if datasetType.lower() == 'dark':
            nSigmaList = [self.config.nSigmaBright]
        else:
            nSigmaList = [self.config.nSigmaBright, self.config.nSigmaDark]
        defects = self.findHotAndColdPixels(inputExp, nSigmaList)

        msg = "Found %s defects containing %s pixels in %s"
        self.log.info(msg, len(defects), self._nPixFromDefects(defects), datasetType)

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

    def findHotAndColdPixels(self, exp, nSigma):
        """Find hot and cold pixels in an image.

        Using config-defined thresholds on a per-amp basis, mask
        pixels that are nSigma above threshold in dark frames (hot
        pixels), or nSigma away from the clipped mean in flats (hot &
        cold pixels).

        Parameters
        ----------
        exp : `lsst.afw.image.exposure.Exposure`
            The exposure in which to find defects.
        nSigma : `list [ `float` ]
            Detection threshold to use.  Positive for DETECTED pixels,
            negative for DETECTED_NEGATIVE pixels.

        Returns
        -------
        defects : `lsst.ip.isr.Defect`
            The defects found in the image.

        """

        self._setEdgeBits(exp)
        maskedIm = exp.maskedImage

        # the detection polarity for afwDetection, True for positive,
        # False for negative, and therefore True for darks as they only have
        # bright pixels, and both for flats, as they have bright and dark pix
        footprintList = []

        for amp in exp.getDetector():
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

            # Remove a background estimate
            ampImg -= afwMath.makeStatistics(ampImg, afwMath.MEANCLIP, ).getValue()

            mergedSet = None
            for sigma in nSigma:
                nSig = np.abs(sigma)
                self.debugHistogram('ampFlux', ampImg, nSig, exp)
                polarity = {-1: False, 1: True}[np.sign(sigma)]

                threshold = afwDetection.createThreshold(nSig, 'stdev', polarity=polarity)

                footprintSet = afwDetection.FootprintSet(ampImg, threshold)
                footprintSet.setMask(maskedIm.mask, ("DETECTED" if polarity else "DETECTED_NEGATIVE"))

                if mergedSet is None:
                    mergedSet = footprintSet
                else:
                    mergedSet.merge(footprintSet)

            footprintList += mergedSet.getFootprints()

            self.debugView('defectMap', ampImg,
                           Defects.fromFootprintList(mergedSet.getFootprints()), exp.getDetector())

        defects = Defects.fromFootprintList(footprintList)
        defects = self.maskBlocksIfIntermitentBadPixelsInColumn(defects)

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
        ---------
        defects: `lsst.ip.isr.Defect`
            The defects found in the image so far

        Returns
        ------
        defects: `lsst.ip.isr.Defect`
            If the number of bad pixels in a column is not larger or
            equal than self.config.badPixelColumnThreshold, the iput
            list is returned. Otherwise, the defects list returned
            will include boxes that mask blocks of on-and-of pixels.

        """
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

        return defects

    def _markBlocksInBadColumn(self, x, y, multipleX, defects):
        """Mask blocks in a column if number of on-and-off bad pixels is above threshold.

        This function is called if the number of on-and-off bad pixels
        in a column is larger or equal than
        self.config.badOnAndOffPixelColumnThreshold.

        Parameters
        ---------
        x: `list`
            Lower left x coordinate of defect box. x coordinate is
            along the short axis if amp.
        y: `list`
            Lower left y coordinate of defect box. x coordinate is
            along the long axis if amp.
        multipleX: list
            List of x coordinates in amp. with multiple bad pixels
            (i.e., columns with defects).
        defects: `lsst.ip.isr.Defect`
            The defcts found in the image so far

        Returns
        -------
        defects: `lsst.ip.isr.Defect`
            The defects list returned that will include boxes that
            mask blocks of on-and-of pixels.

        """
        with defects.bulk_update():
            goodPixelColumnGapThreshold = self.config.goodPixelColumnGapThreshold
            for x0 in multipleX:
                index = np.where(x == x0)
                multipleY = y[index]  # multipleY and multipleX are in 1-1 correspondence.
                minY, maxY = np.min(multipleY), np.max(multipleY)
                # Next few lines: don't mask pixels in column if gap of good pixels between
                # two consecutive bad pixels is larger or equal than 'goodPixelColumnGapThreshold'.
                diffIndex = np.where(np.diff(multipleY) >= goodPixelColumnGapThreshold)[0]
                if len(diffIndex) != 0:
                    limits = [minY]  # put the minimum first
                    for gapIndex in diffIndex:
                        limits.append(multipleY[gapIndex])
                        limits.append(multipleY[gapIndex+1])
                    limits.append(maxY)  # maximum last
                    assert len(limits)%2 == 0, 'limits is even by design, but check anyways'
                    for i in np.arange(0, len(limits)-1, 2):
                        s = Box2I(minimum=Point2I(x0, limits[i]), maximum=Point2I(x0, limits[i+1]))
                        defects.append(s)
                else:  # No gap is large enough
                    s = Box2I(minimum=Point2I(x0, minY), maximum=Point2I(x0, maxY))
                    defects.append(s)
        return defects

    def debugView(self, stepname, ampImage, defects, detector):
        # def _plotDefects(self, exp, visit, defects, imageType):  # pragma: no cover
        """Plot the defects found by the task.

        Parameters
        ----------
        exp : `lsst.afw.image.exposure.Exposure`
            The exposure in which the defects were found.
        visit : `int`
            The visit number.
        defects : `lsst.ip.isr.Defect`
            The defects to plot.
        imageType : `str`
            The type of image, either 'dark' or 'flat'.
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
        """
        Make a histogram of the distribution of pixel values for each amp.

        The main image data histogram is plotted in blue. Edge pixels,
        if masked, are in red. Note that masked edge pixels do not contribute
        to the underflow and overflow numbers.

        Note that this currently only supports the 16-amp LSST detectors.

        Parameters
        ----------
        dataRef : `lsst.daf.persistence.ButlerDataRef`
            dataRef for the detector.
        exp : `lsst.afw.image.exposure.Exposure`
            The exposure in which the defects were found.
        visit : `int`
            The visit number.
        nSigmaUsed : `float`
            The number of sigma used for detection
        """
        frame = getDebugFrame(self._display, stepname)
        if frame:
            import matplotlib.pyplot as plt

            detector = exp.getDetector()
            nX = np.floor(np.sqrt(len(detector)))
            nY = len(detector) // nX
            fig, ax = plt.subplots(nrows=nY, ncols=nX, sharex='col', sharey='row', figsize=(13, 10))

            expTime = exp.getInfo().getVisitInfo().getExposureTime()

            for (amp, a) in zip(reversed(detector), ax.flatten()):
                mi = exp.maskedImage[amp.getBBox()]

                # normalize by expTime as we plot in ADU/s and don't always work with master calibs
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

                # Report number of entries in over-and -underflow bins, i.e. off the edges of the histogram
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
        return


class MergeDefectsConnections(pipeBase.PipelineTaskConnections,
                              dimensions=("instrument", "detector")):
    inputDefects = cT.Input(
        name="singleExpDefects",
        doc="Measured defect lists.",
        storageClass="Defects",
        dimensions=("instrument", "detector", "exposure"),
        multiple=True,
    )
    camera = cT.PrerequisiteInput(
        name='camera',
        doc="Camera associated with these defects.",
        storageClass="Camera",
        dimensions=("instrument", ),
        isCalibration=True,
        lookupFunction=lookupStaticCalibration,
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
        doc=("Ensure that all visits are from the same run? Raises if this is not the case, or"
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
    edgesAsDefects = pexConfig.Field(
        dtype=bool,
        doc=("Mark all edge pixels, as defined by nPixBorder[UpDown, LeftRight], as defects."
             " Normal treatment is to simply exclude this region from the defect finding, such that no"
             " defect will be located there."),
        default=False,
    )


class MergeDefectsTask(pipeBase.PipelineTask, pipeBase.CmdLineTask):
    """Merge the defects from multiple exposures.
    """
    ConfigClass = MergeDefectsTaskConfig
    _DefaultName = 'cpDefectMerge'

    def run(self, inputDefects, camera):
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
            self.log.info("Pre-merge %s pixels with non-zero detections for %s" % (nDetected, imageType))

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
            self.log.info("Post-merge %s pixels marked as defects for %s" % (len(indices[0]), imageType))
            partialDefect = Defects.fromMask(sumImage, 'BAD')
            splitDefects.append(partialDefect)

        # Do final combination of separate image types
        finalImage = afwImage.MaskedImageF(detector.getBBox())
        for inDefect in splitDefects:
            for defect in inDefect:
                finalImage.image[defect.getBBox()] += 1
        finalImage /= len(splitDefects)
        nDetected = len(np.where(finalImage.getImage().getArray() > 0)[0])
        self.log.info("Pre-final merge %s pixels with non-zero detections" % (nDetected, ))

        # This combination is the OR of all image types
        threshold = 0.0
        indices = np.where(finalImage.getImage().getArray() > threshold)
        BADBIT = finalImage.getMask().getPlaneBitMask('BAD')
        finalImage.getMask().getArray()[indices] |= BADBIT
        self.log.info("Post-final merge %s pixels marked as defects" % (len(indices[0]), ))

        if self.config.edgesAsDefects:
            self.log.info("Masking edge pixels as defects.")
            # Do the same as IsrTask.maskEdges()
            box = detector.getBBox()
            subImage = finalImage[box]
            box.grow(-self.nPixBorder)
            SourceDetectionTask.setEdgeBits(subImage, box, BADBIT)

        merged = Defects.fromMask(finalImage, 'BAD')
        merged.updateMetadata(camera=camera, detector=detector, filterName=None,
                              setCalibId=True, setDate=True)

        return pipeBase.Struct(
            mergedDefects=merged,
        )


class FindDefectsTaskConfig(pexConfig.Config):
    measure = pexConfig.ConfigurableField(
        target=MeasureDefectsTask,
        doc="Task to measure single frame defects.",
    )
    merge = pexConfig.ConfigurableField(
        target=MergeDefectsTask,
        doc="Task to merge multiple defects together.",
    )

    isrForFlats = pexConfig.ConfigurableField(
        target=IsrTask,
        doc="Task to perform instrumental signature removal",
    )
    isrForDarks = pexConfig.ConfigurableField(
        target=IsrTask,
        doc="Task to perform instrumental signature removal",
    )
    isrMandatoryStepsFlats = pexConfig.ListField(
        dtype=str,
        doc=("isr operations that must be performed for valid results when using flats."
             " Raises if any of these are False"),
        default=['doAssembleCcd', 'doFringe']
    )
    isrMandatoryStepsDarks = pexConfig.ListField(
        dtype=str,
        doc=("isr operations that must be performed for valid results when using darks. "
             "Raises if any of these are False"),
        default=['doAssembleCcd', 'doFringe']
    )
    isrForbiddenStepsFlats = pexConfig.ListField(
        dtype=str,
        doc=("isr operations that must NOT be performed for valid results when using flats."
             " Raises if any of these are True"),
        default=['doBrighterFatter', 'doUseOpticsTransmission',
                 'doUseFilterTransmission', 'doUseSensorTransmission', 'doUseAtmosphereTransmission']
    )
    isrForbiddenStepsDarks = pexConfig.ListField(
        dtype=str,
        doc=("isr operations that must NOT be performed for valid results when using darks."
             " Raises if any of these are True"),
        default=['doBrighterFatter', 'doUseOpticsTransmission',
                 'doUseFilterTransmission', 'doUseSensorTransmission', 'doUseAtmosphereTransmission']
    )
    isrDesirableSteps = pexConfig.ListField(
        dtype=str,
        doc=("isr operations that it is advisable to perform, but are not mission-critical."
             " WARNs are logged for any of these found to be False."),
        default=['doBias']
    )

    ccdKey = pexConfig.Field(
        dtype=str,
        doc="The key by which to pull a detector from a dataId, e.g. 'ccd' or 'detector'",
        default='ccd',
    )
    imageTypeKey = pexConfig.Field(
        dtype=str,
        doc="The key for the butler to use by which to check whether images are darks or flats",
        default='imageType',
    )


class FindDefectsTask(pipeBase.CmdLineTask):
    """Task for finding defects in sensors.

    The task has two modes of operation, defect finding in raws and in
    master calibrations, which work as follows.

    Master calib defect finding
    ----------------------------

    A single visit number is supplied, for which the corresponding flat & dark
    will be used. This is because, at present at least, there is no way to pass
    a calibration exposure ID from the command line to a command line task.

    The task retrieves the corresponding dark and flat exposures for the
    supplied visit. If a flat is available the task will (be able to) look
    for both bright and dark defects. If only a dark is found then only bright
    defects will be sought.

    All pixels above/below the specified nSigma which lie with the specified
    borders for flats/darks are identified as defects.

    Raw visit defect finding
    ------------------------

    A list of exposure IDs are supplied for defect finding. The task will
    detect bright pixels in the dark frames, if supplied, and bright & dark
    pixels in the flats, if supplied, i.e. if you only supply darks you will
    only be given bright defects. This is done automatically from the imageType
    of the exposure, so the input exposure list can be a mix.

    As with the master calib detection, all pixels above/below the specified
    nSigma which lie with the specified borders for flats/darks are identified
    as defects. Then, a post-processing step is done to merge these detections,
    with pixels appearing in a fraction [0..1] of the images are kept as defects
    and those appearing below that occurrence-threshold are discarded.
    """
    ConfigClass = FindDefectsTaskConfig
    _DefaultName = "findDefects"

    RunnerClass = DataRefListRunner

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.makeSubtask("measure")
        self.makeSubtask("merge")

    @pipeBase.timeMethod
    def runDataRef(self, dataRefList):
        """Run the defect finding task.

        Find the defects, as described in the main task docstring, from a
        dataRef and a list of visit(s).

        Parameters
        ----------
        dataRefList : `list` [`lsst.daf.persistence.ButlerDataRef`]
            dataRefs for the data to be checked for defects.

        Returns
        -------
        result : `lsst.pipe.base.Struct`
            Result struct with Components:

            - ``defects`` : `lsst.ip.isr.Defect`
              The defects found by the task.
            - ``exitStatus`` : `int`
              The exit code.
        """
        dataRef = dataRefList[0]
        camera = dataRef.get("camera")

        singleExpDefects = []
        activeChip = None
        for dataRef in dataRefList:
            exposure = dataRef.get("postISRCCD")
            if activeChip:
                if exposure.getDetector().getName() != activeChip:
                    raise RuntimeError("Too many input detectors supplied!")
            else:
                activeChip = exposure.getDetector().getName()

            result = self.measure.run(exposure, camera)
            singleExpDefects.append(result.outputDefects)

        finalResults = self.merge.run(singleExpDefects, camera)
        metadata = finalResults.mergedDefects.getMetadata()
        inputDims = {'calibDate': metadata['CALIBDATE'],
                     'raftName': metadata['RAFTNAME'],
                     'detectorName': metadata['SLOTNAME'],
                     'detector': metadata['DETECTOR'],
                     'ccd': metadata['DETECTOR'],
                     'ccdnum': metadata['DETECTOR']}

        butler = dataRef.getButler()
        butler.put(finalResults.mergedDefects, "defects", inputDims)

        return finalResults
