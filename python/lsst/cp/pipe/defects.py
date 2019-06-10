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
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import os
import warnings

import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase
import lsst.afw.image as afwImage
import lsst.meas.algorithms as measAlg
import lsst.afw.math as afwMath
import lsst.afw.detection as afwDetection
import lsst.log as lsstLog
import lsst.afw.display as afwDisplay
import lsst.afw.cameraGeom.utils as cgUtils

from lsst.ip.isr import IsrTask
from .utils import NonexistentDatasetTaskDataIdContainer, SingleVisitListTaskRunner, countMaskedPixels, \
    validateIsrConfig


class FindDefectsTaskConfig(pexConfig.Config):
    """Config class for defect finding"""

    isrForFlats = pexConfig.ConfigurableField(
        target=IsrTask,
        doc="""Task to perform instrumental signature removal""",
    )
    isrForDarks = pexConfig.ConfigurableField(
        target=IsrTask,
        doc="""Task to perform instrumental signature removal""",
    )
    isrMandatoryStepsFlats = pexConfig.ListField(
        dtype=str,
        doc="isr operations that must be performed for valid results when using flats. " +
        "Raises if any of these are False",
        default=['doAssembleCcd', 'doFringe']
    )
    isrMandatoryStepsDarks = pexConfig.ListField(
        dtype=str,
        doc="isr operations that must be performed for valid results when using darks. " +
        "Raises if any of these are False",
        default=['doAssembleCcd', 'doFringe']
    )
    isrForbiddenStepsFlats = pexConfig.ListField(
        dtype=str,
        doc="isr operations that must NOT be performed for valid results when using flats. " +
        "Raises if any of these are True",
        default=['doAddDistortionModel', 'doBrighterFatter', 'doUseOpticsTransmission',
                 'doUseFilterTransmission', 'doUseSensorTransmission', 'doUseAtmosphereTransmission']
    )
    isrForbiddenStepsDarks = pexConfig.ListField(
        dtype=str,
        doc="isr operations that must NOT be performed for valid results when using darks. " +
        "Raises if any of these are True",
        default=['doAddDistortionModel', 'doBrighterFatter', 'doUseOpticsTransmission',
                 'doUseFilterTransmission', 'doUseSensorTransmission', 'doUseAtmosphereTransmission']
    )
    isrDesirableSteps = pexConfig.ListField(
        dtype=str,
        doc="isr operations that it is advisable to perform, but are not mission-critical." +
        " WARNs are logged for any of these found to be False.",
        default=['doBias']
    )
    ccdKey = pexConfig.Field(
        dtype=str,
        doc="The key by which to pull a detector from a dataId, e.g. 'ccd' or 'detector'",
        default='ccd',
    )
    imageTypeKey = pexConfig.Field(
        dtype=str,
        doc="The key by which to check whether images are darks or flats",
        default='imageType',
    )
    mode = pexConfig.ChoiceField(
        doc="Use single master calibs (flat and dark) for finding defects, or a list of raw visits?" +
        " If MASTER, a single visit number should be supplied, for which the corresponding master flat" +
        " and dark will be used. If VISITS, the list of visits will be used, treating the flats and darks" +
        " as appropriate, depending on their image types, as determined by their imageType from" +
        " config.imageTypeKey",
        dtype=str,
        default="VISITS",
        allowed={
            "VISITS": "Calculate defects from a list of raw visits",
            "MASTER": "Use the corresponding master calibs from the specified visit to measure defects",
        }
    )
    # TODO: make this option work
    # defectTypes = pexConfig.ChoiceField(
    #     doc="Which types of defects to identify",
    #     dtype=str,
    #     default="BRIGHT_AND_DARK",
    #     allowed={
    #         "BRIGHT": "Find bright pixel/column defects in the detector",
    #         "DARK": "Find dark pixel/column defects in the detector",
    #         "BRIGHT_AND_DARK": "Find both bright and dark pixel/column defects in the detector",
    #     }
    # )
    nSigmaBright = pexConfig.Field(
        dtype=float,
        doc="Number of sigma above mean for bright pixel detection.",
        default=4.8,
    )
    nSigmaDark = pexConfig.Field(
        dtype=float,
        doc="Number of sigma below mean for dark pixel detection.",
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
        doc="Mark all edge pixels, as defined by nPixBorder[UpDown, LeftRight], as defects." +
        " Normal treatment is to simply exclude this region from the defect finding, such that no" +
        " defect will be located there.",
        default=False,  # false because most obs_packages don't have runs. obs_lsst/ts8 overrides this.
    )
    assertSameRun = pexConfig.Field(
        dtype=bool,
        doc="Ensure that all visits are from the same run? Raises is this is not the case, or" +
        "if the run key isn't found.",
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
        doc="The fraction (0..1) of visits in which a pixel was found to be defective across" +
        " the visit list in order to be marked as a defect. Note, upper bound is exclusive, so use" +
        " mode AND to require pixel to appear in all images.",
        default=0.7,
        min=0,
        max=1,
    )
    makePlots = pexConfig.Field(
        dtype=bool,
        doc="Plot histograms for each visit for each amp (one plot per detector) and the final" +
        " defects overlaid on the sensor.",
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
    """TODO: Write docstring
    """

    RunnerClass = SingleVisitListTaskRunner
    ConfigClass = FindDefectsTaskConfig
    _DefaultName = "findDefects"

    def __init__(self, *args, **kwargs):
        pipeBase.CmdLineTask.__init__(self, *args, **kwargs)
        self.makeSubtask("isrForFlats")
        self.makeSubtask("isrForDarks")
        # change these back once you use the INFO level logging to check
        # that all the right things are being done, and only the right things!
        # self.isrForFlats.log.setLevel(lsstLog.WARN)
        # self.isrForDarks.log.setLevel(lsstLog.WARN)

        plt.interactive(False)  # stop windows popping up when plotting. When headless, use 'agg' backend too
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
                            help="List of visits to use. Same for each detector." +
                            " Uses the normal 0..10:3^234 syntax")
        parser.add_id_argument("--id", datasetType="newDefects",
                               ContainerClass=NonexistentDatasetTaskDataIdContainer,
                               help="The ccds to use, e.g. --id ccd=0..100")
        return parser

    @pipeBase.timeMethod
    def runDataRef(self, dataRef, visitList):
        """Run the defect finding task

        TODO: Write docstring

        Parameters
        ----------
        dataRef : list of lsst.daf.persistence.ButlerDataRef
            dataRef for the detector for the visits to be fit.
        visitList : `iterable` of `int`
            List of visit numbers to be processed
        """

        detNum = dataRef.dataId[self.config.ccdKey]
        self.log.info(f"Calculating defects using {visitList} visits for detector {detNum}")

        defectListFromDarks = []
        defectListFromFlats = []

        # TODO: Pull polarity handling out here so that the defectTypes config can work

        if self.config.mode == 'MASTER':
            if len(visitList) > 1:
                raise RuntimeError(f"Must only specify one visit when using mode MASTER, got {visitList}")
            dataRef.dataId['visit'] = visitList[0]

            exp = dataRef.get('dark')
            defects = self.findHotAndColdPixels(exp, 'dark')
            msg = f"Found {len(defects)} defects containing {self._nPixFromDefects(defects)} pixels"
            msg += " in master dark"
            self.log.info(msg)
            defectListFromDarks.append(defects)
            if self.config.makePlots:
                self._plot(dataRef, exp, visitList[0], self.config.nSigmaDark, defects, 'dark')

            exp = dataRef.get('flat')
            defects = self.findHotAndColdPixels(exp, 'flat')
            msg = f"Found {len(defects)} defects containing {self._nPixFromDefects(defects)} pixels"
            msg += " in master flat"
            self.log.info(msg)
            defectListFromFlats.append(defects)
            if self.config.makePlots:
                self._plot(dataRef, exp, visitList[0], self.config.nSigmaBright, defects, 'flat')

        elif self.config.mode == 'VISITS':
            butler = dataRef.getButler()

            if self.config.assertSameRun:
                runs = self._getRunListFromVisits(butler, visitList)
                if len(runs) != 1:
                    raise RuntimeError(f'Got data from runs {runs} with assertSameRun==True')

            for visit in visitList:
                imageType = butler.queryMetadata('raw', self.config.imageTypeKey, dataId={'visit': visit})[0]
                imageType = imageType.lower()
                dataRef.dataId['visit'] = visit

                if imageType == 'flat':  # note different isr tasks
                    exp = self.isrForFlats.runDataRef(dataRef).exposure
                    defects = self.findHotAndColdPixels(exp, imageType)
                    defectListFromFlats.append(defects)

                elif imageType == 'dark':
                    exp = self.isrForDarks.runDataRef(dataRef).exposure
                    defects = self.findHotAndColdPixels(exp, imageType)
                    defectListFromDarks.append(defects)

                else:
                    raise RuntimeError("Failed on imageType {imageType}. Only flats and darks supported")

                msg = f"Found {len(defects)} defects containing {self._nPixFromDefects(defects)} pixels"
                msg += f" in visit {visit}"
                self.log.info(msg)

                if self.config.makePlots:
                    nSig = self.config.nSigmaBright if imageType == 'flat' else self.config.nSigmaDark
                    self._plot(dataRef, exp, visit, nSig, defects, imageType)

        self.log.info(f'Combining {len(defectListFromDarks)} defect sets from darks for detector {detNum}')
        mergedDefectsFromDarks = self._postProcessDefectSets(defectListFromDarks, exp.getDimensions())
        self.log.info(f'Combining {len(defectListFromFlats)} defect sets from flats for detector {detNum}')
        mergedDefectsFromFlats = self._postProcessDefectSets(defectListFromFlats, exp.getDimensions())

        self.log.info(f'Combining bright and dark defect sets')
        brightDarkPostMerge = [mergedDefectsFromDarks, mergedDefectsFromFlats]
        allDefects = self._postProcessDefectSets(brightDarkPostMerge, exp.getDimensions(), mode='OR')

        self._writeData(dataRef, allDefects)

        self.log.info('Finished finding defects in detector %s' % detNum)
        return pipeBase.Struct(exitStatus=0)

    @staticmethod
    def _nPixFromDefects(defects):
        """TODO: Write docstring
        """
        nPix = 0
        for defect in defects:
            nPix += defect.getBBox().getArea()
        return nPix

    def _writeData(self, dataRef, defects):
        """Write the data out to the defect file.

        TODO: Add parameters
        """
        filename = dataRef.getUri(write=True)  # does not guarantee that full path exists
        dirname = os.path.dirname(filename)
        if not os.path.exists(dirname):
            os.makedirs(dirname)

        self.log.info(f'Writing defects to {filename[:-5]} in format: {self.config.writeAs}')

        if self.config.writeAs in ['FITS', 'BOTH']:
            defects.writeFits(filename)
        if self.config.writeAs in ['ASCII', 'BOTH']:
            defects.writeText(filename)
        return

    @staticmethod
    def _getRunListFromVisits(butler, visitList):
        """Return the set of runs for the visits in visitList."""
        runs = set()
        for visit in visitList:
            runs.add(butler.queryMetadata('raw', 'run', dataId={'visit': visit})[0])
        return runs

    def _postProcessDefectSets(self, defectList, imageDimensions, mode=None):
        """Combine a list of defects to make a single defect object.

        AND, OR or use percentage of visits in which defects appear
        depending on config.

        Parameters
        ----------
        defectList: `list` of `lsst.meas.algorithms.interp.Defect`
            List of defects to merge

        imageDimensions: `tuple` of `int`
            The size of the image

        Returns
        -------
        defects: `lsst.meas.algorithms.interp.Defect`
            The merged defects
        """
        # so that empty lists can be passed in for input data
        # where only flats or darks are supplied
        if not defectList:
            return []

        if len(defectList) == 1:  # single input - no merging to do
            return defectList[0]

        if mode is None:
            mode = self.config.combinationMode

        sumImage = afwImage.MaskedImageF(imageDimensions)
        for defects in defectList:
            for defect in defects:
                sumImage.image[defect.getBBox()] += 1
        sumImage /= len(defectList)

        nDetected = len(np.where(sumImage.image.array > 0)[0])
        self.log.info(f"Pre-merge {nDetected} pixels with non-zero detections")

        if mode == 'AND':  # must appear in all
            threshold = 1
        elif mode == 'OR':  # must appear in any, but can't be zero because >= operator
            threshold = 1e-9  # we will never have a significant fraction of 1e9 input visits
        elif mode == 'FRACTION':
            threshold = self.config.combinationFraction
        else:
            raise RuntimeError(f"Got unsupported combinationMode {mode}")

        indices = np.where(sumImage.image.array >= threshold)
        BADBIT = sumImage.mask.getPlaneBitMask('BAD')
        sumImage.mask.array[indices] |= BADBIT

        self.log.info(f"Post-merge {len(indices[0])} pixels marked as defects")

        if self.config.edgesAsDefects:
            self.log.info(f"Masked edge pixels as defects in addition to previous")
            self._setEdgeBits(sumImage)

        defects = measAlg.Defects.fromMask(sumImage, 'BAD')
        return defects

    @staticmethod
    def _getNumGoodPixels(maskedIm, badMaskString="NO_DATA"):
        """Return the number of non-bad pixels in the image"""
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
        setMask: `bool`
            If true, update exp with hot and cold pixels
            hot: DETECTED
            cold: DETECTED_NEGATIVE

        Returns
        -------
        defects: `lsst.meas.algorithms.interp.Defect`
            The defects
        """
        assert imageType in ['flat', 'dark']

        self._setEdgeBits(exp)
        maskedIm = exp.maskedImage

        polarities = {'dark': [True], 'flat': [True, False]}[imageType]

        footprintList = []

        for amp in exp.getDetector():
            ampImg = maskedIm[amp.getBBox()].clone()

            # XXX can we skip this if we set the stats box to ignore the EDGE bit?
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

            # XXX needs better initialisation, and config options for the MEANCLIP
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
        return defects

    def _plot(self, dataRef, exp, visit, nSig, defects, imageType):  # pragma: no cover
        """TODO: Write docstring
        """
        afwDisplay.setDefaultBackend("matplotlib")
        plt.interactive(False)  # seems to need reasserting here

        expCopy = exp.clone()

        dirname = dataRef.getUri(datasetType='cpPipePlotRoot', write=True)
        if not os.path.exists(dirname):
            os.makedirs(dirname)

        detNum = dataRef.dataId[self.config.ccdKey]

        if self.config.mode == "MASTER":
            filename = f"defectPlot_det{detNum}_master-{imageType}_for-exp{visit}.pdf"
        elif self.config.mode == "VISITS":
            filename = f"defectPlot_det{detNum}_{imageType}_exp{visit}.pdf"

        filenameFull = os.path.join(dirname, filename)
        self.log.info(f'Wrote amp histogram to {filenameFull}')

        with warnings.catch_warnings():
            msg = "Matplotlib is currently using agg, which is a non-GUI backend, so cannot show the figure."
            warnings.filterwarnings("ignore", message=msg)
            with PdfPages(filenameFull) as pdfPages:
                self._plotAmpHistogram(dataRef, expCopy, visit, nSig)  # NB: modifies exp to scale by expTime
                pdfPages.savefig()

                self._plotDefects(exp, visit, defects, imageType)
                pdfPages.savefig()

    def _plotDefects(self, exp, visit, defects, imageType):  # pragma: no cover
        """TODO: Write docstring
        """
        expCopy = exp.clone()
        del exp  # for safety - no longer needed as we have a copy
        maskedIm = expCopy.maskedImage

        defects.maskPixels(expCopy.maskedImage, "BAD")
        detector = expCopy.getDetector()

        disp = afwDisplay.Display(0, reopenPlot=True, dpi=200)  # , dpi=200)

        if imageType == "flat":  # set each amp image to have a mean of 1.00
            for amp in detector:
                ampIm = maskedIm.image[amp.getBBox()]
                ampIm -= afwMath.makeStatistics(ampIm, afwMath.MEANCLIP).getValue() + 1

        mpDict = maskedIm.mask.getMaskPlaneDict()
        for plane in mpDict.keys():
            if plane in ['BAD']:
                continue
            disp.setMaskPlaneColor(plane, afwDisplay.IGNORE)

        # if self._getNumGoodPixels(maskedIm) >= 1:
        disp.scale('asinh', 'zscale')
        disp.setMaskTransparency(80)
        disp.setMaskPlaneColor("BAD", afwDisplay.RED)

        disp.setImageColormap('gray')
        title = f"Detector: {detector.getName()[-3:]} {detector.getSerial()}"
        title += f", Type: {imageType}, visit: {visit}"
        disp.mtv(maskedIm, title=title)

        cgUtils.overlayCcdBoxes(detector, isTrimmed=True, display=disp)

    def _plotAmpHistogram(self, dataRef, exp, visit, nSigmaUsed):  # pragma: no cover
        """
        Make a histogram of the distribution of pixel values for each amp.

        Main image data histogram plotted in blue. Edge pixels, if masked, are
        in red. Note that masked edge pixels do not contribute to the
        underflow and overflow numbers.

        Parameters
        ----------
        pdfPages: `matplotlib.backends.backend_pdf.PdfPages`
            If not None, append per-amp histograms to this object
        """
        fig, ax = plt.subplots(nrows=4, ncols=4, sharex='col', sharey='row', figsize=(13, 10))

        detector = exp.getDetector()
        expTime = exp.getInfo().getVisitInfo().getExposureTime()

        for (amp, a) in zip(reversed(detector), ax.flatten()):
            mi = exp.maskedImage[amp.getBBox()]

            # normalise by expTime as we plot in ADU/s and don't always work with master calibs
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

            nsig = 6
            leftEdge = mean - nsig * nSigmaUsed*sigma
            rightEdge = mean + nsig * nSigmaUsed*sigma
            nbins = np.linspace(leftEdge, rightEdge, 1000)
            ey, bin_borders, patches = a.hist(edgeData, histtype='step', bins=nbins, lw=1, edgecolor='red')
            y, bin_borders, patches = a.hist(imgData, histtype='step', bins=nbins, lw=3, edgecolor='blue')

            # Report number of entries in over-and -underflow bins, i.e. off the edges of the histogam
            nOverflow = len(imgData[imgData > rightEdge])
            nUnderflow = len(imgData[imgData < leftEdge])

            # Put v-lines and textboxes in
            a.axvline(thrUpper, c='k')
            a.axvline(thrLower, c='k')
            msg = f"{amp.getName()}\nmean:{mean: .1f}\n$\sigma$:{sigma: .1f}"  # noqa: W605
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

    def _setEdgeBits(self, exposureOrMaskedImage):
        """Set edge bits on an exposure or maskedImage.
        """
        if isinstance(exposureOrMaskedImage, afwImage.Exposure):
            mi = exposureOrMaskedImage.maskedImage
        elif isinstance(exposureOrMaskedImage, afwImage.MaskedImage):
            mi = exposureOrMaskedImage
        else:
            t = type(exposureOrMaskedImage)
            raise RuntimeError(f"Function supports exposure or maskedImage but not {t}")

        EDGEBIT = mi.mask.getPlaneBitMask("EDGE")
        if self.config.nPixBorderLeftRight:
            mi.mask[: self.config.nPixBorderLeftRight, :, afwImage.LOCAL] |= EDGEBIT
            mi.mask[-self.config.nPixBorderLeftRight:, :, afwImage.LOCAL] |= EDGEBIT
        if self.config.nPixBorderUpDown:
            mi.mask[:, : self.config.nPixBorderUpDown, afwImage.LOCAL] |= EDGEBIT
            mi.mask[:, -self.config.nPixBorderUpDown:, afwImage.LOCAL] |= EDGEBIT
