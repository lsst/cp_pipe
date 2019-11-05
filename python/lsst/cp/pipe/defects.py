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

__all__ = ['FindDefectsTask',
           'FindDefectsTaskConfig', ]

import numpy as np
import os
import warnings

import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase
import lsst.afw.image as afwImage
import lsst.meas.algorithms as measAlg
import lsst.afw.math as afwMath
import lsst.afw.detection as afwDetection
import lsst.afw.display as afwDisplay
from lsst.afw import cameraGeom
from lsst.geom import Box2I, Point2I

from lsst.ip.isr import IsrTask
from .utils import NonexistentDatasetTaskDataIdContainer, SingleVisitListTaskRunner, countMaskedPixels, \
    validateIsrConfig


class FindDefectsTaskConfig(pexConfig.Config):
    """Config class for defect finding"""

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
        default=['doAddDistortionModel', 'doBrighterFatter', 'doUseOpticsTransmission',
                 'doUseFilterTransmission', 'doUseSensorTransmission', 'doUseAtmosphereTransmission']
    )
    isrForbiddenStepsDarks = pexConfig.ListField(
        dtype=str,
        doc=("isr operations that must NOT be performed for valid results when using darks."
             " Raises if any of these are True"),
        default=['doAddDistortionModel', 'doBrighterFatter', 'doUseOpticsTransmission',
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
    mode = pexConfig.ChoiceField(
        doc=("Use single master calibs (flat and dark) for finding defects, or a list of raw visits?"
             " If MASTER, a single visit number should be supplied, for which the corresponding master flat"
             " and dark will be used. If VISITS, the list of visits will be used, treating the flats and "
             " darks as appropriate, depending on their image types, as determined by their imageType from"
             " config.imageTypeKey"),
        dtype=str,
        default="VISITS",
        allowed={
            "VISITS": "Calculate defects from a list of raw visits",
            "MASTER": "Use the corresponding master calibs from the specified visit to measure defects",
        }
    )
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
        default=5.0,
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
        doc=("If BPC is the set of all the bad pixels in a given column (not necessarily consecutive)",
             "and the size of BPC is at least 'badOnAndOffPixelColumnThreshold', all the pixels between the",
             "pixels that satisfy minY (BPC) and maxY (BPC) will be marked as bad, with 'Y' being the long",
             "axis of the amplifier (and 'X' the other axis, which for a column is a constant for all pixels",
             "in the set BPC). If there are more than 'goodPixelColumnGapThreshold' consecutive non-bad",
             "pixels in BPC, an exception to the above is made and those consecutive",
             "'goodPixelColumnGapThreshold' are not marked as bad."),
        default=50,
    )
    goodPixelColumnGapThreshold = pexConfig.Field(
        dtype=int,
        doc=("Size, in pixels, of usable consecutive pixels in a column with on and off bad pixels (see",
             "'badOnAndOffPixelColumnThreshold')."),
        default=30,
    )
    edgesAsDefects = pexConfig.Field(
        dtype=bool,
        doc=("Mark all edge pixels, as defined by nPixBorder[UpDown, LeftRight], as defects."
             " Normal treatment is to simply exclude this region from the defect finding, such that no"
             " defect will be located there."),
        default=False,
    )
    assertSameRun = pexConfig.Field(
        dtype=bool,
        doc=("Ensure that all visits are from the same run? Raises if this is not the case, or"
             "if the run key isn't found."),
        default=False,  # false because most obs_packages don't have runs. obs_lsst/ts8 overrides this.
    )
    combinationMode = pexConfig.ChoiceField(
        doc="Which types of defects to identify",
        dtype=str,
        default="FRACTION",
        allowed={
            "AND": "Logical AND the pixels found in each visit to form set",
            "OR": "Logical OR the pixels found in each visit to form set",
            "FRACTION": "Use pixels found in more than config.combinationFraction of visits",
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
    makePlots = pexConfig.Field(
        dtype=bool,
        doc=("Plot histograms for each visit for each amp (one plot per detector) and the final"
             " defects overlaid on the sensor."),
        default=False,
    )
    writeAs = pexConfig.ChoiceField(
        doc="Write the output file as ASCII or FITS table",
        dtype=str,
        default="FITS",
        allowed={
            "ASCII": "Write the output as an ASCII file",
            "FITS": "Write the output as an FITS table",
            "BOTH": "Write the output as both a FITS table and an ASCII file",
        }
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

    RunnerClass = SingleVisitListTaskRunner
    ConfigClass = FindDefectsTaskConfig
    _DefaultName = "findDefects"

    def __init__(self, *args, **kwargs):
        pipeBase.CmdLineTask.__init__(self, *args, **kwargs)
        self.makeSubtask("isrForFlats")
        self.makeSubtask("isrForDarks")

        validateIsrConfig(self.isrForFlats, self.config.isrMandatoryStepsFlats,
                          self.config.isrForbiddenStepsFlats, self.config.isrDesirableSteps)
        validateIsrConfig(self.isrForDarks, self.config.isrMandatoryStepsDarks,
                          self.config.isrForbiddenStepsDarks, self.config.isrDesirableSteps)
        self.config.validate()
        self.config.freeze()

    @classmethod
    def _makeArgumentParser(cls):
        """Augment argument parser for the FindDefectsTask."""
        parser = pipeBase.ArgumentParser(name=cls._DefaultName)
        parser.add_argument("--visitList", dest="visitList", nargs="*",
                            help=("List of visits to use. Same for each detector."
                                  " Uses the normal 0..10:3^234 syntax"))
        parser.add_id_argument("--id", datasetType="newDefects",
                               ContainerClass=NonexistentDatasetTaskDataIdContainer,
                               help="The ccds to use, e.g. --id ccd=0..100")
        return parser

    @pipeBase.timeMethod
    def runDataRef(self, dataRef, visitList):
        """Run the defect finding task.

        Find the defects, as described in the main task docstring, from a
        dataRef and a list of visit(s).

        Parameters
        ----------
        dataRef : `lsst.daf.persistence.ButlerDataRef`
            dataRef for the detector for the visits to be fit.
        visitList : `list` [`int`]
            List of visits to be processed. If config.mode == 'VISITS' then the
            list of visits is used. If config.mode == 'MASTER' then the length
            of visitList must be one, and the corresponding master calibrations
            are used.

        Returns
        -------
        result : `lsst.pipe.base.Struct`
            Result struct with Components:

            - ``defects`` : `lsst.meas.algorithms.Defect`
              The defects found by the task.
            - ``exitStatus`` : `int`
              The exit code.
        """

        detNum = dataRef.dataId[self.config.ccdKey]
        self.log.info("Calculating defects using %s visits for detector %s" % (visitList, detNum))

        defectLists = {'dark': [], 'flat': []}

        if self.config.mode == 'MASTER':
            if len(visitList) > 1:
                raise RuntimeError(f"Must only specify one visit when using mode MASTER, got {visitList}")
            dataRef.dataId['visit'] = visitList[0]

            for datasetType in defectLists.keys():
                exp = dataRef.get(datasetType)
                defects = self.findHotAndColdPixels(exp, datasetType)

                msg = "Found %s defects containing %s pixels in master %s"
                self.log.info(msg, len(defects), self._nPixFromDefects(defects), datasetType)
                defectLists[datasetType].append(defects)
                if self.config.makePlots:
                    self._plot(dataRef, exp, visitList[0], self._getNsigmaForPlot(datasetType),
                               defects, datasetType)

        elif self.config.mode == 'VISITS':
            butler = dataRef.getButler()

            if self.config.assertSameRun:
                runs = self._getRunListFromVisits(butler, visitList)
                if len(runs) != 1:
                    raise RuntimeError(f"Got data from runs {runs} with assertSameRun==True")

            for visit in visitList:
                imageType = butler.queryMetadata('raw', self.config.imageTypeKey, dataId={'visit': visit})[0]
                imageType = imageType.lower()
                dataRef.dataId['visit'] = visit
                if imageType == 'flat':  # note different isr tasks
                    exp = self.isrForFlats.runDataRef(dataRef).exposure
                    defects = self.findHotAndColdPixels(exp, imageType)
                    defectLists['flat'].append(defects)

                elif imageType == 'dark':
                    exp = self.isrForDarks.runDataRef(dataRef).exposure
                    defects = self.findHotAndColdPixels(exp, imageType)
                    defectLists['dark'].append(defects)

                else:
                    raise RuntimeError(f"Failed on imageType {imageType}. Only flats and darks supported")

                msg = "Found %s defects containing %s pixels in visit %s"
                self.log.info(msg, len(defects), self._nPixFromDefects(defects), visit)

                if self.config.makePlots:
                    self._plot(dataRef, exp, visit, self._getNsigmaForPlot(imageType), defects, imageType)

        msg = "Combining %s defect sets from darks for detector %s"
        self.log.info(msg, len(defectLists['dark']), detNum)
        mergedDefectsFromDarks = self._postProcessDefectSets(defectLists['dark'], exp.getDimensions(),
                                                             self.config.combinationMode)
        msg = "Combining %s defect sets from flats for detector %s"
        self.log.info(msg, len(defectLists['flat']), detNum)
        mergedDefectsFromFlats = self._postProcessDefectSets(defectLists['flat'], exp.getDimensions(),
                                                             self.config.combinationMode)

        msg = "Combining bright and dark defect sets for detector %s"
        self.log.info(msg, detNum)
        brightDarkPostMerge = [mergedDefectsFromDarks, mergedDefectsFromFlats]
        allDefects = self._postProcessDefectSets(brightDarkPostMerge, exp.getDimensions(), mode='OR')

        self._writeData(dataRef, allDefects)

        self.log.info("Finished finding defects in detector %s" % detNum)
        return pipeBase.Struct(defects=allDefects, exitStatus=0)

    def _getNsigmaForPlot(self, imageType):
        assert imageType in ['flat', 'dark']
        nSig = self.config.nSigmaBright if imageType == 'flat' else self.config.nSigmaDark
        return nSig

    @staticmethod
    def _nPixFromDefects(defect):
        """Count the number of pixels in a defect object."""
        nPix = 0
        for d in defect:
            nPix += d.getBBox().getArea()
        return nPix

    def _writeData(self, dataRef, defects):
        """Write the data out to the defect file.

        Parameters
        ----------
        dataRef : `lsst.daf.persistence.ButlerDataRef`
            dataRef for the detector for defects to be written.
        defects : `lsst.meas.algorithms.Defect`
            The defects to be written.
        """
        filename = dataRef.getUri(write=True)  # does not guarantee that full path exists
        dirname = os.path.dirname(filename)
        if not os.path.exists(dirname):
            os.makedirs(dirname)

        msg = "Writing defects to %s in format: %s"
        self.log.info(msg, os.path.splitext(filename)[0], self.config.writeAs)

        if self.config.writeAs in ['FITS', 'BOTH']:
            defects.writeFits(filename)
        if self.config.writeAs in ['ASCII', 'BOTH']:
            wroteTo = defects.writeText(filename)
            assert(os.path.splitext(wroteTo)[0] == os.path.splitext(filename)[0])
        return

    @staticmethod
    def _getRunListFromVisits(butler, visitList):
        """Return the set of runs for the visits in visitList."""
        runs = set()
        for visit in visitList:
            runs.add(butler.queryMetadata('raw', 'run', dataId={'visit': visit})[0])
        return runs

    def _postProcessDefectSets(self, defectList, imageDimensions, mode):
        """Combine a list of defects to make a single defect object.

        AND, OR or use percentage of visits in which defects appear
        depending on config.

        Parameters
        ----------
        defectList : `list` [`lsst.meas.algorithms.Defect`]
            The lList of defects to merge.
        imageDimensions : `tuple` [`int`]
            The size of the image.
        mode : `str`
            The combination mode to use, either 'AND', 'OR' or 'FRACTION'

        Returns
        -------
        defects : `lsst.meas.algorithms.Defect`
            The defect set resulting from the merge.
        """
        # so that empty lists can be passed in for input data
        # where only flats or darks are supplied
        if defectList == []:
            return []

        if len(defectList) == 1:  # single input - no merging to do
            return defectList[0]

        sumImage = afwImage.MaskedImageF(imageDimensions)
        for defects in defectList:
            for defect in defects:
                sumImage.image[defect.getBBox()] += 1
        sumImage /= len(defectList)

        nDetected = len(np.where(sumImage.image.array > 0)[0])
        self.log.info("Pre-merge %s pixels with non-zero detections" % nDetected)

        if mode == 'OR':  # must appear in any
            indices = np.where(sumImage.image.array > 0)
        else:
            if mode == 'AND':  # must appear in all
                threshold = 1
            elif mode == 'FRACTION':
                threshold = self.config.combinationFraction
            else:
                raise RuntimeError(f"Got unsupported combinationMode {mode}")
            indices = np.where(sumImage.image.array >= threshold)

        BADBIT = sumImage.mask.getPlaneBitMask('BAD')
        sumImage.mask.array[indices] |= BADBIT

        self.log.info("Post-merge %s pixels marked as defects" % len(indices[0]))

        if self.config.edgesAsDefects:
            self.log.info("Masking edge pixels as defects in addition to previously identified defects")
            self._setEdgeBits(sumImage, 'BAD')

        defects = measAlg.Defects.fromMask(sumImage, 'BAD')
        return defects

    @staticmethod
    def _getNumGoodPixels(maskedIm, badMaskString="NO_DATA"):
        """Return the number of non-bad pixels in the image."""
        nPixels = maskedIm.mask.array.size
        nBad = countMaskedPixels(maskedIm, badMaskString)
        return nPixels - nBad

    def findHotAndColdPixels(self, exp, imageType, setMask=False):
        """Find hot and cold pixels in an image.

        Using config-defined thresholds on a per-amp basis, mask pixels
        that are nSigma above threshold in dark frames (hot pixels),
        or nSigma away from the clipped mean in flats (hot & cold pixels).

        Parameters
        ----------
        exp : `lsst.afw.image.exposure.Exposure`
            The exposure in which to find defects.
        imageType : `str`
            The image type, either 'dark' or 'flat'.
        setMask : `bool`
            If true, update exp with hot and cold pixels.
            hot: DETECTED
            cold: DETECTED_NEGATIVE

        Returns
        -------
        defects : `lsst.meas.algorithms.Defect`
            The defects found in the image.
        """
        assert imageType in ['flat', 'dark']

        self._setEdgeBits(exp)
        maskedIm = exp.maskedImage

        # the detection polarity for afwDetection, True for positive,
        # False for negative, and therefore True for darks as they only have
        # bright pixels, and both for flats, as they have bright and dark pix
        polarities = {'dark': [True], 'flat': [True, False]}[imageType]

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

            ampImg -= afwMath.makeStatistics(ampImg, afwMath.MEANCLIP, ).getValue()

            mergedSet = None
            for polarity in polarities:
                nSig = self.config.nSigmaBright if polarity else self.config.nSigmaDark
                threshold = afwDetection.createThreshold(nSig, 'stdev', polarity=polarity)

                footprintSet = afwDetection.FootprintSet(ampImg, threshold)
                if setMask:
                    footprintSet.setMask(maskedIm.mask, ("DETECTED" if polarity else "DETECTED_NEGATIVE"))

                if mergedSet is None:
                    mergedSet = footprintSet
                else:
                    mergedSet.merge(footprintSet)

            footprintList += mergedSet.getFootprints()

        defects = measAlg.Defects.fromFootprintList(footprintList)

        defects = self.maskBlocksIfIntermitentBadPixelsInColumn(defects)

        return defects

    def maskBlocksIfIntermitentBadPixelsInColumn(self, defects):
        """Mask blocks in a column if there are on-and-off bad pixels

        If there's a column with on and off bad pixels, mask all the pixels in between,
        except if there is a large enough gap of consecutive good pixels between two
        bad pixels in the column.

        Parameters
        ---------
        defects: `lsst.meas.algorithms.Defect`
            The defcts found in the image so far

        Returns
        ------
        defects: `lsst.meas.algorithms.Defect`
            If the number of bad pixels in a column is not larger or equal than
            self.config.badPixelColumnThreshold, the iput list is returned. Otherwise,
            the defects list returned will include boxes that mask blocks of on-and-of
            pixels.
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

        This function is called if the number of on-and-off bad pixels in a column
        is larger or equal than self.config.badOnAndOffPixelColumnThreshold.

        Parameters
        ---------
            x: list
                Lower left x coordinate of defect box. x coordinate is along the short axis if amp.

            y: list
                Lower left y coordinate of defect box. x coordinate is along the long axis if amp.

            multipleX: list
                List of x coordinates in amp. with multiple bad pixels (i.e., columns with defects).

            defects: `lsst.meas.algorithms.Defect`
                The defcts found in the image so far

        Returns
        -------
        defects: `lsst.meas.algorithms.Defect`
            The defects list returned that will include boxes that mask blocks
            of on-and-of pixels.
        """
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
                    s = Box2I(minimum = Point2I(x0, limits[i]), maximum = Point2I(x0, limits[i+1]))
                    defects.append(s)
            else:  # No gap is large enough
                s = Box2I(minimum = Point2I(x0, minY), maximum = Point2I(x0, maxY))
                defects.append(s)
        return defects

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

    def _plot(self, dataRef, exp, visit, nSig, defects, imageType):  # pragma: no cover
        """Plot the defects and pixel histograms.

        Parameters
        ----------
        dataRef : `lsst.daf.persistence.ButlerDataRef`
            dataRef for the detector.
        exp : `lsst.afw.image.exposure.Exposure`
            The exposure in which the defects were found.
        visit : `int`
            The visit number.
        nSig : `float`
            The number of sigma used for detection
        defects : `lsst.meas.algorithms.Defect`
            The defects to plot.
        imageType : `str`
            The type of image, either 'dark' or 'flat'.

        Currently only for LSST sensors. Plots are written to the path
        given by the butler for the ``cpPipePlotRoot`` dataset type.
        """
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_pdf import PdfPages

        afwDisplay.setDefaultBackend("matplotlib")
        plt.interactive(False)  # seems to need reasserting here

        dirname = dataRef.getUri(datasetType='cpPipePlotRoot', write=True)
        if not os.path.exists(dirname):
            os.makedirs(dirname)

        detNum = exp.getDetector().getId()
        nAmps = len(exp.getDetector())

        if self.config.mode == "MASTER":
            filename = f"defectPlot_det{detNum}_master-{imageType}_for-exp{visit}.pdf"
        elif self.config.mode == "VISITS":
            filename = f"defectPlot_det{detNum}_{imageType}_exp{visit}.pdf"

        filenameFull = os.path.join(dirname, filename)

        with warnings.catch_warnings():
            msg = "Matplotlib is currently using agg, which is a non-GUI backend, so cannot show the figure."
            warnings.filterwarnings("ignore", message=msg)
            with PdfPages(filenameFull) as pdfPages:
                if nAmps == 16:
                    self._plotAmpHistogram(dataRef, exp, visit, nSig)
                    pdfPages.savefig()

                self._plotDefects(exp, visit, defects, imageType)
                pdfPages.savefig()
        self.log.info("Wrote plot(s) to %s" % filenameFull)

    def _plotDefects(self, exp, visit, defects, imageType):  # pragma: no cover
        """Plot the defects found by the task.

        Parameters
        ----------
        exp : `lsst.afw.image.exposure.Exposure`
            The exposure in which the defects were found.
        visit : `int`
            The visit number.
        defects : `lsst.meas.algorithms.Defect`
            The defects to plot.
        imageType : `str`
            The type of image, either 'dark' or 'flat'.
        """
        expCopy = exp.clone()  # we mess with the copy later, so make a clone
        del exp  # del for safety - no longer needed as we have a copy so remove from scope to save mistakes
        maskedIm = expCopy.maskedImage

        defects.maskPixels(expCopy.maskedImage, "BAD")
        detector = expCopy.getDetector()

        disp = afwDisplay.Display(0, reopenPlot=True, dpi=200)

        if imageType == "flat":  # set each amp image to have a mean of 1.00
            for amp in detector:
                ampIm = maskedIm.image[amp.getBBox()]
                ampIm -= afwMath.makeStatistics(ampIm, afwMath.MEANCLIP).getValue() + 1

        mpDict = maskedIm.mask.getMaskPlaneDict()
        for plane in mpDict.keys():
            if plane in ['BAD']:
                continue
            disp.setMaskPlaneColor(plane, afwDisplay.IGNORE)

        disp.scale('asinh', 'zscale')
        disp.setMaskTransparency(80)
        disp.setMaskPlaneColor("BAD", afwDisplay.RED)

        disp.setImageColormap('gray')
        title = (f"Detector: {detector.getName()[-3:]} {detector.getSerial()}"
                 f", Type: {imageType}, visit: {visit}")
        disp.mtv(maskedIm, title=title)

        cameraGeom.utils.overlayCcdBoxes(detector, isTrimmed=True, display=disp)

    def _plotAmpHistogram(self, dataRef, exp, visit, nSigmaUsed):  # pragma: no cover
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
        import matplotlib.pyplot as plt

        detector = exp.getDetector()

        if len(detector) != 16:
            raise RuntimeError("Plotting currently only supported for 16 amp detectors")
        fig, ax = plt.subplots(nrows=4, ncols=4, sharex='col', sharey='row', figsize=(13, 10))

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
            ey, bin_borders, patches = a.hist(edgeData, histtype='step', bins=nbins, lw=1, edgecolor='red')
            y, bin_borders, patches = a.hist(imgData, histtype='step', bins=nbins, lw=3, edgecolor='blue')

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
