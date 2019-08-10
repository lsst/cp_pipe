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

__all__ = ['FindDefectsMasterTask',
           'FindDefectsMasterTaskConfig',
           'FindDefectsTask',
           'FindDefectsTaskConfig',
           ]

import numpy as np
import matplotlib.pyplot as plt         # only needed if makePlots is True
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

from lsst.ip.isr import IsrTask
from .utils import NonexistentDatasetTaskDataIdContainer, SingleVisitListTaskRunner, countMaskedPixels, \
    validateIsrConfig


class FindDefectsTaskConfig(pexConfig.Config):
    """Config class for defect finding"""

    nSigmaBright = pexConfig.Field(
        dtype=float,
        doc=("Number of sigma above mean for bright pixel detection. The default value was found to be"
             " appropriate for some LSST sensors in DM-17490."),
        default=4.8,
    )
    nSigmaFaint = pexConfig.Field(
        dtype=float,
        doc=("Number of sigma below mean for dark pixel detection. The default value was found to be"
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
    edgesAsDefects = pexConfig.Field(
        dtype=bool,
        doc=("Mark all edge pixels, as defined by nPixBorder[UpDown, LeftRight], as defects."
             " Normal treatment is to simply exclude this region from the defect finding, such that no"
             " defect will be located there."),
        default=False,
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


class FindDefectsTask(pipeBase.Task):
    """Task for finding defects in sensors.

    The task has two modes of operation, defect finding in raws and in
    master calibrations, which work as follows.

    Master calib defect finding
    ----------------------------

    Process a master dark and/or flat exposure.
    If only a flat is available the task will (be able to) look
    for both bright and dark defects. If only a dark is found then only bright
    defects will be sought.

    All pixels above/below the specified nSigma which lie with the specified
    borders for flats/darks are identified as defects.

    Raw visit defect finding
    ------------------------

    Using a list of dark Exposures and a list of flat Exposures (either or both
    of which may be empty), detect bright pixels in the dark frames, if supplied,
    and bright & dark pixels in the flats, if supplied, i.e. if you only supply
    darks you will only be given bright defects. This is done automatically from
    the imageType of the exposure, so the input exposure list can be a mix.

    As with the master calib detection, all pixels above/below the specified
    nSigma which lie with the specified borders for flats/darks are identified
    as defects. Then, a post-processing step is done to merge these detections,
    with pixels appearing in a fraction [0..1] of the images are kept as defects
    and those appearing below that occurrence-threshold are discarded.
    """

    ConfigClass = FindDefectsTaskConfig
    _DefaultName = "findDefects"

    def __init__(self, **kwargs):
        pipeBase.Task.__init__(self, **kwargs)

    def run(self, darkExposures, flatExposures, detNum=None):
        """Find the defects from a set of images of a given detector

        Parameters
        ----------
        darkExposures : dict (`int`, `lsst.afw.image.exposure.Exposure`)
            Dict of dark exposures in which to find defects, indexed
            by visit
        flatExposures : dict (`int`, `lsst.afw.image.exposure.Exposure`)
            Dict of flat exposures in which to find defects, indexed
            by visit
        detNum: `int`
            An integer identifying the detector being analysed

        Returns
        -------
        result : `lsst.pipe.base.Struct`
            Result struct with Components:

            - ``defects`` : `lsst.meas.algorithms.Defect`
              The defects found by the task
            - perExposureDefects : `dict` of lists of defect indexed by `str`
              The `str` is "dark" or "flat"; the key is a `dict` indexed by `int`
              where the `int` is a visit and the values `lsst.meas.algorithms.Defect`
              detected on that visit
            - nSigmaFaint : `float`
              The value of config.nSigmaFaint passed back up for plotting
            - nSigmaBright : `float`
              The value of config.nSigmaBright passed back up for plotting
        """
        detName = "" if detNum is None else f" detector {detNum}"

        exposures = dict(dark=darkExposures, flat=flatExposures)
        perExposureDefects = dict(dark={}, flat={})
        mergedDefects = {}

        # process all the input exposures, detecting potential defects on each;
        # then combine the results of processing the flats and biases
        dimensions = None
        for imageType in exposures:
            for visit, exp in exposures[imageType].items():
                if dimensions is None:
                    dimensions = exp.getDimensions()

                defectList = self.findHotAndColdPixels(exp, imageType)
                perExposureDefects[imageType][visit] = defectList

                self.log.info("Found %s defects containing %s pixels in visit %d%s",
                              len(defectList), self._nPixFromDefects(defectList), visit, detName)

            self.log.info("Combining %s defects from %ss for%s",
                          len(perExposureDefects[imageType]), imageType, detName)
            mergedDefects[imageType] = self._postProcessDefectSets(
                list(perExposureDefects[imageType].values()), dimensions, self.config.combinationMode)

        self.log.info("Combining %s/%s defect sets from darks/flats for%s",
                      len(mergedDefects['dark']), len(mergedDefects['flat']), detName)

        defects = self._postProcessDefectSets(list(mergedDefects.values()), dimensions, mode='OR')

        return pipeBase.Struct(defects=defects,
                               perExposureDefects=perExposureDefects,
                               nSigmaFaint=self.config.nSigmaFaint,  # needed for QA plots
                               nSigmaBright=self.config.nSigmaBright,  # needed for QA plots
                               )

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
        if len(defectList) == 0:
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
                nSig = self.config.nSigmaBright if polarity else self.config.nSigmaFaint
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
        return defects

    @staticmethod
    def _nPixFromDefects(defect):
        """Count the number of pixels in a defect object."""
        nPix = 0
        for d in defect:
            nPix += d.getBBox().getArea()
        return nPix

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


class FindDefectsMasterTaskConfig(pexConfig.Config):
    """Config class for the task driving the task which finds defects"""

    isrForFlats = pexConfig.ConfigurableField(
        target=IsrTask,
        doc="Task to perform instrumental signature removal",
    )
    isrForDarks = pexConfig.ConfigurableField(
        target=IsrTask,
        doc="Task to perform instrumental signature removal",
    )
    findDefects = pexConfig.ConfigurableField(
        target=FindDefectsTask,
        doc="Task to actually find the defects",
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
    assertSameRun = pexConfig.Field(
        dtype=bool,
        doc=("Ensure that all visits are from the same run? Raises if this is not the case, or"
             "if the run key isn't found."),
        default=False,  # false because most obs_packages don't have runs. obs_lsst/ts8 overrides this.
    )
    makePlots = pexConfig.Field(
        dtype=bool,
        doc=("Plot histograms for each visit for each amp (one plot per detector) and the final"
             " defects overlaid on the sensor."),
        default=False,
    )
    displayBackend = pexConfig.Field(
        doc="afwDisplay backend",
        dtype=str,
        default="matplotlib",
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


class FindDefectsMasterTask(pipeBase.CmdLineTask):
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
    ConfigClass = FindDefectsMasterTaskConfig
    _DefaultName = "findDefectsMaster"

    def __init__(self, *args, **kwargs):
        pipeBase.CmdLineTask.__init__(self, *args, **kwargs)
        self.makeSubtask("isrForFlats")
        self.makeSubtask("isrForDarks")
        self.makeSubtask("findDefects")

        validateIsrConfig(self.isrForFlats, self.config.isrMandatoryStepsFlats,
                          self.config.isrForbiddenStepsFlats, self.config.isrDesirableSteps)
        validateIsrConfig(self.isrForDarks, self.config.isrMandatoryStepsDarks,
                          self.config.isrForbiddenStepsDarks, self.config.isrDesirableSteps)
        self.config.validate()
        self.config.freeze()

    @classmethod
    def _makeArgumentParser(cls):
        """Augment argument parser for the FindDefectsMasterTask."""
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

        # dicts of dark and flat Exposures, indexed by visit
        exposures = dict(dark={}, flat={})

        if self.config.mode == 'MASTER':
            if len(visitList) > 1:
                raise RuntimeError(f"Must only specify one visit when using mode MASTER, got {visitList}")
            visit = visitList[0]
            dataRef.dataId['visit'] = visit

            for datasetType in exposures:
                exposures[datasetType][visit] = dataRef.get(datasetType)

        elif self.config.mode == 'VISITS':
            butler = dataRef.getButler()

            _visitList = []             # check if visits are valid
            for visit in visitList:
                runs = butler.queryMetadata('raw', 'run', dataId={'visit': visit})
                if len(runs) == 0:
                    self.log.warn("Failed to find visit %d; ignoring", visit)
                else:
                    _visitList.append(visit)
            visitList = _visitList

            if self.config.assertSameRun:
                runs = self._getRunListFromVisits(butler, visitList)
                if len(runs) != 1:
                    raise RuntimeError(f"Got data from runs {runs} with assertSameRun==True")

            for visit in visitList:
                imageType = butler.queryMetadata('raw', self.config.imageTypeKey, dataId={'visit': visit})[0]
                imageType = imageType.lower()
                dataRef.dataId['visit'] = visit

                if imageType == 'flat':  # note different isr tasks
                    isr = self.isrForFlats
                elif imageType == 'dark':
                    isr = self.isrForDarks
                else:
                    self.log.warn(f"Ignoring visit {visit} of imageType {imageType}. "
                                  "Only flats and darks supported")

                exposures[imageType][visit] = isr.runDataRef(dataRef).exposure

        # unpack exposures to make API easier to reuse
        ret = self.findDefects.run(exposures["dark"], exposures["flat"], detNum)
        defects = ret.defects
        perExposureDefects = ret.perExposureDefects

        self._writeData(dataRef, defects)

        if self.config.makePlots:
            afwDisplay.setDefaultBackend(self.config.displayBackend)
            plt.interactive(False)  # seems to need reasserting here

            dirname = dataRef.getUri(datasetType='cpPipePlotRoot', write=True)
            if not os.path.exists(dirname):
                os.makedirs(dirname)

            for datasetType in exposures:
                for visit, exp in exposures[datasetType].items():
                    self._plot(dirname, exp, visit, ret.nSigmaFaint,
                               perExposureDefects[datasetType][visit], datasetType)

                    if False:
                        import pdb
                        print("Entering pdb to wait for plot; hit 'c' to exit")
                        pdb.set_trace()

        self.log.info("Finished finding defects in detector %s" % detNum)
        return pipeBase.Struct(defects=defects, exitStatus=0)

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

    def _plot(self, dirname, exp, visit, nSig, defects, imageType):  # pragma: no cover
        """Plot the defects and pixel histograms.


        Parameters
        ----------
        dirname : `str`
            Path to where plots are written
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

        Currently only for LSST sensors.
        """
        from matplotlib.backends.backend_pdf import PdfPages

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
                    self._plotAmpHistogram(exp, visit, nSig)
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

        if afwDisplay.getDefaultBackend() == "matplotlib":
            kwargs = dict(reopenPlot=True, dpi=200)
        else:
            kwargs = {}

        disp = afwDisplay.Display(0, **kwargs)

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

    def _plotAmpHistogram(self, exp, visit, nSigmaUsed):  # pragma: no cover
        """
        Make a histogram of the distribution of pixel values for each amp.

        The main image data histogram is plotted in blue. Edge pixels,
        if masked, are in red. Note that masked edge pixels do not contribute
        to the underflow and overflow numbers.

        Note that this currently only supports the 16-amp LSST detectors.

        Parameters
        ----------
        exp : `lsst.afw.image.exposure.Exposure`
            The exposure in which the defects were found.
        visit : `int`
            The visit number.
        nSigmaUsed : `float`
            The number of sigma used for detection
        """
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

            nsig = nSigmaUsed + 1.2     # add something small so the edge of the plot is out from level used
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
            msg = f"{amp.getName()}\nmean:{mean: .1f}\n$\\sigma$:{sigma: .1f}"
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
