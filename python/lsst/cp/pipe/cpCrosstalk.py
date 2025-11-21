# This file is part of cp_pipe
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
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
import itertools
import numpy as np

from collections import defaultdict
from copy import copy
from scipy.stats import median_abs_deviation

import lsst.pipe.base as pipeBase
import lsst.pipe.base.connectionTypes as cT

from lsstDebug import getDebugFrame
from lsst.afw.cameraGeom import ReadoutCorner
from lsst.afw.detection import FootprintSet, Threshold
from lsst.afw.display import getDisplay
from lsst.pex.config import ConfigurableField, Field, ListField
from lsst.ip.isr import CrosstalkCalib, IsrProvenance, growMasks
from lsst.cp.pipe.utils import (ddict2dict, sigmaClipCorrection)
from lsst.meas.algorithms import SubtractBackgroundTask

__all__ = ["CrosstalkExtractConfig", "CrosstalkExtractTask",
           "CrosstalkSolveTask", "CrosstalkSolveConfig",
           "CrosstalkFilterTask", "CrosstalkFilterConfig"]


class CrosstalkExtractConnections(pipeBase.PipelineTaskConnections,
                                  dimensions=("instrument", "exposure", "detector")):
    inputExp = cT.Input(
        name="crosstalkInputs",
        doc="Input post-ISR processed exposure to measure crosstalk from.",
        storageClass="Exposure",
        dimensions=("instrument", "exposure", "detector"),
        multiple=False,
    )
    # TODO: Depends on DM-21904.
    sourceExp = cT.Input(
        name="crosstalkSource",
        doc="Post-ISR exposure to measure for inter-chip crosstalk onto inputExp.",
        storageClass="Exposure",
        dimensions=("instrument", "exposure", "detector"),
        multiple=True,
        deferLoad=True,
        # lookupFunction=None,
    )

    outputRatios = cT.Output(
        name="crosstalkRatios",
        doc="Extracted crosstalk pixel ratios.",
        storageClass="StructuredDataDict",
        dimensions=("instrument", "exposure", "detector"),
    )
    outputFluxes = cT.Output(
        name="crosstalkFluxes",
        doc="Source pixel fluxes used in ratios.",
        storageClass="StructuredDataDict",
        dimensions=("instrument", "exposure", "detector"),
    )

    def __init__(self, *, config=None):
        super().__init__(config=config)
        # Discard sourceExp until DM-21904 allows full interchip
        # measurements.
        self.inputs.discard("sourceExp")


class CrosstalkExtractConfig(pipeBase.PipelineTaskConfig,
                             pipelineConnections=CrosstalkExtractConnections):
    """Configuration for the measurement of pixel ratios.
    """

    doMeasureInterchip = Field(
        dtype=bool,
        default=False,
        doc="Measure inter-chip crosstalk as well?",
    )
    threshold = Field(
        dtype=float,
        default=30000,
        doc="Minimum level of source pixels for which to measure crosstalk."
    )
    ignoreSaturatedPixels = Field(
        dtype=bool,
        default=False,
        doc="Should saturated pixels be ignored?"
    )
    badMask = ListField(
        dtype=str,
        default=["BAD", "INTRP"],
        doc="Mask planes to ignore when identifying source pixels."
    )
    isTrimmed = Field(
        dtype=bool,
        default=True,
        doc="Is the input exposure trimmed?"
    )
    background = ConfigurableField(
        target=SubtractBackgroundTask,
        doc="Background estimation task.",
    )
    growMaskRadius = Field(
        dtype=int,
        default=0,
        doc="Radius to grow CT_TEMP masks prior to background estimation."
    )

    def setDefaults(self):
        super().setDefaults()
        # is this really the best way to do this??
        self.background.useApprox = False
        self.background.ignoredPixelMask.append("CT_TEMP")

    def validate(self):
        super().validate()

        # Ensure the handling of the SAT mask plane is consistent
        # with the ignoreSaturatedPixels value.
        if self.ignoreSaturatedPixels:
            if 'SAT' not in self.badMask:
                self.badMask.append('SAT')
        else:
            if 'SAT' in self.badMask:
                self.badMask = [mask for mask in self.badMask if mask != 'SAT']


class CrosstalkExtractTask(pipeBase.PipelineTask):
    """Task to measure pixel ratios to find crosstalk.
    """

    ConfigClass = CrosstalkExtractConfig
    _DefaultName = 'cpCrosstalkExtract'

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.makeSubtask("background")

    def run(self, inputExp, sourceExps=[]):
        """Measure pixel ratios between amplifiers in inputExp.

        Extract crosstalk ratios between different amplifiers.

        For pixels above ``config.threshold``, we calculate the ratio
        between each background-subtracted target amp and the source
        amp. We return a list of ratios for each pixel for each
        target/source combination, as nested dictionary containing the
        ratio.

        Parameters
        ----------
        inputExp : `lsst.afw.image.Exposure`
            Input exposure to measure pixel ratios on.
        sourceExp : `list` [`lsst.afw.image.Exposure`], optional
            List of chips to use as sources to measure inter-chip
            crosstalk.

        Returns
        -------
        results : `lsst.pipe.base.Struct`
            The results struct containing:

            ``outputRatios``
                 A catalog of ratio lists.  The dictionaries are
                 indexed such that:
                 outputRatios[targetChip][sourceChip][targetAmp][sourceAmp]
                 contains the ratio list for that combination (`dict`
                 [`dict` [`dict` [`dict` [`list`]]]]).
            ``outputFluxes``
                 A catalog of flux lists.  The dictionaries are
                 indexed such that:
                 outputFluxes[sourceChip][sourceAmp] contains the flux
                 list used in the outputRatios (`dict` [`dict`
                 [`list`]]).
        """
        outputRatios = defaultdict(lambda: defaultdict(dict))
        outputFluxes = defaultdict(lambda: defaultdict(dict))

        threshold = self.config.threshold
        badPixels = list(self.config.badMask)

        targetDetector = inputExp.getDetector()
        targetChip = targetDetector.getName()

        # Always look at the target chip first, then go to any other
        # supplied exposures.
        sourceExtractExps = [inputExp]
        sourceExtractExps.extend(sourceExps)

        self.log.info("Measuring full detector background for target: %s", targetChip)
        targetIm = inputExp.getMaskedImage()

        FootprintSet(targetIm, Threshold(threshold), "DETECTED")
        detected = targetIm.getMask().getPlaneBitMask("DETECTED")
        targetIm.getMask().addMaskPlane("CT_TEMP")
        maskBit = targetIm.getMask().getPlaneBitMask("CT_TEMP")
        bg = CrosstalkCalib.calculateBackground(targetIm, badPixels + ["DETECTED"])

        # Carry over over-threshold masked pixels to other amplifiers.
        for amp in targetDetector:
            ampIm = inputExp[amp.getBBox()]
            ampName = amp.getName()

            mask = ampIm.mask.array & detected
            if np.sum(mask) == 0:
                continue

            newMask = np.where(np.bitwise_and(mask, detected), maskBit, 0)
            for ampToMask in targetDetector:
                if ampName == ampToMask.getName():
                    # The amp we're considering already
                    continue

                extractedAmp = inputExp[ampToMask.getBBox()]
                # The mask needs to be flipped to match the target
                flippedMask = self._flipMask(newMask, amp, ampToMask)

                extractedAmp.mask.array[:, :] |= flippedMask
        # Optionally dilate these masks by some amount:
        growMasks(inputExp.mask, radius=self.config.growMaskRadius,
                  maskNameList=["CT_TEMP"], maskValue="CT_TEMP")

        # We've now masked the source pixels, and any potential CT
        # pixels, so this should be just the
        # background/reflections/etc.
        backgroundModel = self.background.fitBackground(inputExp.maskedImage)
        backgroundIm = backgroundModel.getImageF()
        self.debugView('extract', inputExp)

        # Begin search for bright pixels, and their associated crosstalk
        # signals.
        for sourceExp in sourceExtractExps:
            # This loop exists to support future inter-chip searches.
            sourceDetector = sourceExp.getDetector()
            sourceChip = sourceDetector.getName()
            sourceIm = sourceExp.getMaskedImage()
            bad = sourceIm.getMask().getPlaneBitMask(badPixels)
            self.log.info("Measuring crosstalk from source: %s", sourceChip)

            if sourceExp != inputExp:
                FootprintSet(sourceIm, Threshold(threshold), "DETECTED")
                detected = sourceIm.getMask().getPlaneBitMask("DETECTED")

            # The dictionary of amp-to-amp ratios for this pair of
            # source->target detectors.
            ratioDict = defaultdict(lambda: defaultdict(np.array))
            extractedCount = 0

            for sourceAmp in sourceDetector:
                sourceAmpName = sourceAmp.getName()
                sourceAmpBBox = sourceAmp.getBBox() if self.config.isTrimmed else sourceAmp.getRawDataBBox()
                sourceAmpImage = sourceIm[sourceAmpBBox]
                sourceMask = sourceAmpImage.mask.array
                select = ((sourceMask & detected > 0)
                          & (sourceMask & bad == 0)
                          & np.isfinite(sourceAmpImage.image.array))
                count = np.sum(select)
                self.log.debug("  Source amplifier: %s", sourceAmpName)

                outputFluxes[sourceChip][sourceAmpName] = sourceAmpImage.image.array[select].tolist()

                for targetAmp in targetDetector:
                    # iterate over targetExposure
                    targetAmpName = targetAmp.getName()
                    if sourceAmpName == targetAmpName and sourceChip == targetChip:
                        ratioDict[targetAmpName][sourceAmpName] = []
                        continue

                    self.log.debug("    Target amplifier: %s", targetAmpName)

                    targetAmpImage = CrosstalkCalib.extractAmp(targetIm,
                                                               targetAmp, sourceAmp,
                                                               isTrimmed=self.config.isTrimmed)
                    targetBkgImage = CrosstalkCalib.extractAmp(backgroundIm,
                                                               targetAmp, sourceAmp,
                                                               isTrimmed=self.config.isTrimmed)

                    bg = CrosstalkCalib.calculateBackground(targetIm, badPixels + ["DETECTED"])

                    ratios = ((targetAmpImage.image.array[select] - targetBkgImage.array[select])
                              / sourceAmpImage.image.array[select])

                    ratioDict[targetAmpName][sourceAmpName] = ratios.tolist()
                    self.log.info("Amp extracted %d pixels from %s -> %s",
                                  count, sourceAmpName, targetAmpName)
                    extractedCount += count

                    self.debugPixels('pixels',
                                     sourceAmpImage.image.array[select],
                                     targetAmpImage.image.array[select] - bg,
                                     sourceAmpName, targetAmpName)

            self.log.info("Extracted %d pixels from %s -> %s (targetBG: %f)",
                          extractedCount, sourceChip, targetChip, bg)
            outputRatios[targetChip][sourceChip] = ratioDict

        return pipeBase.Struct(
            outputRatios=ddict2dict(outputRatios),
            outputFluxes=ddict2dict(outputFluxes)
        )

    def _flipMask(self, maskArray, sourceAmp, targetAmp):
        """Flip an array from a sourceAmp to match the readout order of
        targetAmp.

        Parameters
        ----------
        maskArray : `np.ndarray`
            Mask data to flip.
        sourceAmp : `lsst.afw.cameraGeom.Amplifier`
            Amplifier corresponding to the maskArray.
        targetAmp : `lsst.afw.cameraGeom.Amplifier`
            Amplifier corresponding to the output mask.

        Returns
        -------
        maskFlipped : `np.ndarray`
            The flipped mask.

        See Also
        -----
        lsst.ip.isr.CrosstalkCalib.extractAmp()
        """
        maskFlipped = copy(maskArray)
        if sourceAmp.getReadoutCorner() == targetAmp.getReadoutCorner():
            return maskFlipped

        X_FLIP = {ReadoutCorner.LL: False,
                  ReadoutCorner.LR: True,
                  ReadoutCorner.UL: False,
                  ReadoutCorner.UR: True}
        Y_FLIP = {ReadoutCorner.LL: False,
                  ReadoutCorner.LR: False,
                  ReadoutCorner.UL: True,
                  ReadoutCorner.UR: True}

        sourceAmpCorner = sourceAmp.getReadoutCorner()
        targetAmpCorner = targetAmp.getReadoutCorner()

        # Flipping is necessary only if the desired configuration doesn't match
        # what we currently have.
        xFlip = X_FLIP[targetAmpCorner] ^ X_FLIP[sourceAmpCorner]
        yFlip = Y_FLIP[targetAmpCorner] ^ Y_FLIP[sourceAmpCorner]

        if xFlip:
            maskFlipped = np.fliplr(maskFlipped)
        if yFlip:
            maskFlipped = np.flipud(maskFlipped)

        return maskFlipped

    def debugView(self, stepname, exposure):

        """Utility function to examine the image being processed.

        Parameters
        ----------
        stepname : `str`
            State of processing to view.
        exposure : `lsst.afw.image.Exposure`
            Exposure to view.
        """
        frame = getDebugFrame(self._display, stepname)
        if frame:
            display = getDisplay(frame)
            display.scale('asinh', 'zscale')
            display.mtv(exposure)

            prompt = "Press Enter to continue: "
            while True:
                ans = input(prompt).lower()
                if ans in ("", "c",):
                    break

    def debugPixels(self, stepname, pixelsIn, pixelsOut, sourceName, targetName):
        """Utility function to examine the CT ratio pixel values.

        Parameters
        ----------
        stepname : `str`
            State of processing to view.
        pixelsIn : `np.ndarray`, (N,)
            Pixel values from the potential crosstalk source.
        pixelsOut : `np.ndarray`, (N,)
            Pixel values from the potential crosstalk target.
        sourceName : `str`
            Source amplifier name
        targetName : `str`
            Target amplifier name
        """
        frame = getDebugFrame(self._display, stepname)
        if frame:
            import matplotlib.pyplot as plt
            figure = plt.figure(1)
            figure.clear()

            axes = figure.add_axes((0.1, 0.1, 0.8, 0.8))
            axes.plot(pixelsIn, pixelsOut / pixelsIn, 'k+')
            plt.xlabel("Source amplifier pixel value")
            plt.ylabel("Measured pixel ratio")
            plt.title(f"(Source {sourceName} -> Target {targetName}) median ratio: "
                      f"{(np.median(pixelsOut / pixelsIn))}")
            figure.show()

            prompt = "Press Enter to continue: "
            while True:
                ans = input(prompt).lower()
                if ans in ("", "c",):
                    break
            plt.close()


class CrosstalkSolveConnections(pipeBase.PipelineTaskConnections,
                                dimensions=("instrument", "detector")):
    inputRatios = cT.Input(
        name="crosstalkRatios",
        doc="Ratios measured for an input exposure.",
        storageClass="StructuredDataDict",
        dimensions=("instrument", "exposure", "detector"),
        multiple=True,
        deferLoad=True,
    )
    inputFluxes = cT.Input(
        name="crosstalkFluxes",
        doc="Fluxes of CT source pixels, for nonlinear fits.",
        storageClass="StructuredDataDict",
        dimensions=("instrument", "exposure", "detector"),
        multiple=True,
        deferLoad=True,
    )
    camera = cT.PrerequisiteInput(
        name="camera",
        doc="Camera the input data comes from.",
        storageClass="Camera",
        dimensions=("instrument",),
        isCalibration=True,
    )

    outputCrosstalk = cT.Output(
        name="crosstalkProposal",
        doc="Output proposed crosstalk calibration.",
        storageClass="CrosstalkCalib",
        dimensions=("instrument", "detector"),
        multiple=False,
        isCalibration=True,
    )

    def __init__(self, *, config=None):
        super().__init__(config=config)

        # if config.fluxOrder == 0 and False:
        #    self.inputs.discard("inputFluxes")


class CrosstalkSolveConfig(pipeBase.PipelineTaskConfig,
                           pipelineConnections=CrosstalkSolveConnections):
    """Configuration for the solving of crosstalk from pixel ratios.
    """

    rejIter = Field(
        dtype=int,
        default=3,
        doc="Number of rejection iterations for final coefficient calculation.",
    )
    rejSigma = Field(
        dtype=float,
        default=2.0,
        doc="Rejection threshold (sigma) for final coefficient calculation.",
    )
    fluxOrder = Field(
        dtype=int,
        default=0,
        doc="Order of source flux fit to crosstalk. 0=simple linear; 1=first order non-linear.",
    )

    rejectNegativeSolutions = Field(
        dtype=bool,
        default=False,
        doc="Should solutions with negative coefficients (which add flux to the target) be excluded?",
    )

    significanceLimit = Field(
        dtype=float,
        default=3.0,
        doc="Sigma significance level to use in marking a coefficient valid.",
    )
    doSignificanceScaling = Field(
        dtype=bool,
        default=True,
        doc="Scale error by 1/sqrt(N) in calculating significant coefficients?",
    )
    doFiltering = Field(
        dtype=bool,
        default=False,
        doc="Filter generated crosstalk to remove marginal measurements?",
    )

    unitsAreElectrons = Field(
        dtype=bool,
        default=True,
        doc="Crosstalk measurements have been done in electrons.",
    )


class CrosstalkSolveTask(pipeBase.PipelineTask):
    """Task to solve crosstalk from pixel ratios.
    """

    ConfigClass = CrosstalkSolveConfig
    _DefaultName = 'cpCrosstalkSolve'

    def runQuantum(self, butlerQC, inputRefs, outputRefs):
        """Ensure that the input and output dimensions are passed along.

        Parameters
        ----------
        butlerQC : `lsst.daf.butler.QuantumContext`
            Butler to operate on.
        inputRefs : `lsst.pipe.base.InputQuantizedConnection`
            Input data refs to load.
        ouptutRefs : `lsst.pipe.base.OutputQuantizedConnection`
            Output data refs to persist.
        """
        inputs = butlerQC.get(inputRefs)

        # Use the dimensions to set calib/provenance information.
        inputs['inputDims'] = [dict(exp.dataId.required) for exp in inputRefs.inputRatios]
        inputs['outputDims'] = dict(outputRefs.outputCrosstalk.dataId.required)

        outputs = self.run(**inputs)
        butlerQC.put(outputs, outputRefs)

    def run(self, inputRatios, inputFluxes=None, camera=None, inputDims=None, outputDims=None):
        """Combine ratios to produce crosstalk coefficients.

        Parameters
        ----------
        inputRatios : `list` [`dict` [`dict` [`dict` [`dict` [`list`]]]]]
            A list of nested dictionaries of ratios indexed by target
            and source chip, then by target and source amplifier.
        inputFluxes : `list` [`dict` [`dict` [`list`]]]
            A list of nested dictionaries of source pixel fluxes, indexed
            by source chip and amplifier.
        camera : `lsst.afw.cameraGeom.Camera`
            Input camera.
        inputDims : `list` [`lsst.daf.butler.DataCoordinate`]
            DataIds to use to construct provenance.
        outputDims : `list` [`lsst.daf.butler.DataCoordinate`]
            DataIds to use to populate the output calibration.

        Returns
        -------
        results : `lsst.pipe.base.Struct`
            The results struct containing:

            ``outputCrosstalk``
                Final crosstalk calibration
                (`lsst.ip.isr.CrosstalkCalib`).
            ``outputProvenance``
                Provenance data for the new calibration
                (`lsst.ip.isr.IsrProvenance`).

        Raises
        ------
        RuntimeError
            Raised if the input data contains multiple target detectors.
        """
        if outputDims:
            calibChip = outputDims['detector']
            instrument = outputDims['instrument']
        else:
            # calibChip needs to be set manually in Gen2.
            calibChip = None
            instrument = None

        if camera and calibChip is not None:
            calibDetector = camera[calibChip]
            ordering = [amp.getName() for amp in calibDetector]
        else:
            calibDetector = None
            ordering = None

        self.log.info("Combining measurements from %d ratios and %d fluxes",
                      len(inputRatios), len(inputFluxes) if inputFluxes else 0)

        if inputFluxes is None:
            inputFluxes = [None for exp in inputRatios]
        if inputDims is None:
            inputDims = [{} for exp in inputRatios]

        combinedRatios = defaultdict(lambda: defaultdict(list))
        combinedFluxes = defaultdict(lambda: defaultdict(list))

        for ratioRef, fluxRef, dimensions in zip(inputRatios, inputFluxes, inputDims):
            ratioDict = ratioRef.get()
            if fluxRef is not None:
                fluxDict = fluxRef.get()
            else:
                fluxDict = None
            for targetChip in ratioDict:
                if calibChip and targetChip != calibChip and targetChip != calibDetector.getName():
                    raise RuntimeError(f"Target chip: {targetChip} does not match calibration dimension: "
                                       f"{calibChip}, {calibDetector.getName()}!")

                sourceChip = targetChip
                if sourceChip in ratioDict[targetChip]:
                    ratios = ratioDict[targetChip][sourceChip]

                    for targetAmp in ratios:
                        for sourceAmp in ratios[targetAmp]:
                            if fluxDict:
                                if len(ratios[targetAmp][sourceAmp]) != len(fluxDict[sourceChip][sourceAmp]):
                                    if targetAmp != sourceAmp:
                                        # This usually triggers when
                                        # targetAmp == sourceAmp.  The
                                        # sourceAmp has flux entries,
                                        # but by definition cannot
                                        # have ratios for itself. Only
                                        # warn when this isn't the
                                        # case.
                                        self.log.warning(f"Length mismatch for ratios {len(ratios[targetAmp][sourceAmp])} "  # noqa E501
                                                         f"and fluxes {len(fluxDict[sourceChip][sourceAmp])} for "  # noqa E501
                                                         f"Source {sourceAmp} and target {targetAmp} "
                                                         f"Rejecting this {dimensions}")
                                    continue
                                combinedFluxes[targetAmp][sourceAmp].extend(fluxDict[sourceChip][sourceAmp])
                            combinedRatios[targetAmp][sourceAmp].extend(ratios[targetAmp][sourceAmp])
                # TODO: DM-21904
                # Iterating over all other entries in
                # ratioDict[targetChip] will yield inter-chip terms.

        for targetAmp in combinedRatios:
            for sourceAmp in combinedRatios[targetAmp]:
                self.log.info("Read %d pixels for %s -> %s",
                              len(combinedRatios[targetAmp][sourceAmp]),
                              sourceAmp, targetAmp)
                if len(combinedRatios[targetAmp][sourceAmp]) > 1:
                    self.debugRatios('reduce', combinedRatios, targetAmp, sourceAmp)

        if self.config.fluxOrder < 2:
            self.log.info("Fitting crosstalk coefficients with order {self.config.fluxOrder}")

            calib = self.measureCrosstalkCoefficients(combinedRatios, ordering,
                                                      combinedFluxes,
                                                      self.config.rejIter, self.config.rejSigma)
        else:
            raise NotImplementedError("Higher order non-linear crosstalk terms are not yet supported.")

        self.log.info("Number of valid coefficients: %d", np.sum(calib.coeffValid))

        if self.config.doFiltering:
            # This step will apply the calculated validity values to
            # censor poorly measured coefficients.
            self.log.info("Filtering measured crosstalk to remove invalid solutions.")
            calib = self.filterCrosstalkCalib(calib)

        # Populate the remainder of the calibration information.
        calib.hasCrosstalk = True
        calib.interChip = {}
        calib.crosstalkRatiosUnits = 'electron' if self.config.unitsAreElectrons else 'adu'
        calib.updateMetadata(
            camera=camera,
            detector=calibDetector,
            setCalibId=True,
            setCalibInfo=True,
            setDate=True,
        )

        # Make an IsrProvenance().
        provenance = IsrProvenance(calibType="CROSSTALK")
        provenance._detectorName = calibChip
        if inputDims:
            provenance.fromDataIds(inputDims)
            provenance._instrument = instrument
        provenance.updateMetadata()

        return pipeBase.Struct(
            outputCrosstalk=calib,
            outputProvenance=provenance,
        )

    def measureCrosstalkCoefficients(self, ratios, ordering, fluxes, rejIter, rejSigma):
        """Measure crosstalk coefficients from the ratios.

        Given a list of ratios for each target/source amp combination,
        we measure a sigma clipped mean and error.

        The coefficient errors returned are the standard deviation of
        the final set of clipped input ratios.

        Parameters
        ----------
        ratios : `dict` [`dict` [`numpy.ndarray`]]
           Catalog of arrays of ratios.  The ratio arrays are one-dimensional.
        ordering : `list` [`str`] or None
           List to use as a mapping between amplifier names (the
           elements of the list) and their position in the output
           calibration (the matching index of the list).  If no
           ordering is supplied, the order of the keys in the ratio
           catalog is used.
        fluxes : `dict` [`dict` [`numpy.ndarray`]]
           Catalog of arrays of fluxes.  The flux arrays are one-dimensional.
        rejIter : `int`
           Number of rejection iterations.
        rejSigma : `float`
           Rejection threshold (sigma).

        Returns
        -------
        calib : `lsst.ip.isr.CrosstalkCalib`
            The output crosstalk calibration.
        """
        calib = CrosstalkCalib(nAmp=len(ratios))

        if ordering is None:
            ordering = list(ratios.keys())

        # Calibration stores coefficients as a numpy ndarray.
        for ss, tt in itertools.product(range(calib.nAmp), range(calib.nAmp)):
            if ss == tt:
                values = []
                myfluxes = []
            else:
                # ratios is ratios[Target][Source]
                # use tt for Target, use ss for Source, to match ip_isr.
                values = np.array(ratios[ordering[tt]][ordering[ss]])
                good_values = np.abs(values) < 1.0  # Discard unreasonable values
                values = values[good_values]
                myfluxes = np.array(fluxes[ordering[tt]][ordering[ss]])
                myfluxes = myfluxes[good_values]
                if len(values) != len(myfluxes):
                    self.log.warning(f"Flux and ratio length disagree after first filter: {len(values)} {len(myfluxes)}")  # noqa E501

            # Sigma clip using the inter-quartile distance and a
            # normal distribution.
            if ss != tt:
                for rej in range(rejIter):
                    if len(values) == 0:
                        break
                    lo, med, hi = np.percentile(values, [25.0, 50.0, 75.0])
                    sigma = 0.741*(hi - lo)
                    good = np.abs(values - med) < rejSigma*sigma
                    if good.sum() == len(good) or good.sum() == 0:
                        break
                    values = values[good]
                    myfluxes = myfluxes[good]
                if len(values) != len(myfluxes):
                    self.log.warning(f"Flux and ratio length disagree after second filter: {len(values)} {len(myfluxes)}")  # noqa E501

            # Crosstalk calib is property[Source][Target].
            calib.coeffNum[ss][tt] = len(values)
            significanceThreshold = 0.0
            if len(values) == 0:
                self.log.warning("No values for matrix element %d,%d" % (ss, tt))
                calib.coeffs[ss][tt] = np.nan
                calib.coeffErr[ss][tt] = np.nan
                calib.coeffValid[ss][tt] = False
                polyfit = [0.0, 0.0]
            else:
                calib.coeffs[ss][tt] = np.mean(values)
                polyfit = np.polyfit(myfluxes, values, 1)

                if self.config.fluxOrder == 1:
                    # substitute polyfit solution.
                    calib.coeffs[ss][tt] = polyfit[1]
                    calib.coeffsSqr[ss][tt] = polyfit[0]

                if self.config.rejectNegativeSolutions and calib.coeffs[ss][tt] < 0.0:
                    calib.coeffs[ss][tt] = 0.0

                if calib.coeffNum[ss][tt] <= 1:
                    calib.coeffErr[ss][tt] = np.nan
                    calib.coeffSqr[ss][tt] = np.nan
                    calib.coeffValid[ss][tt] = False
                else:
                    correctionFactor = sigmaClipCorrection(rejSigma)
                    calib.coeffErr[ss][tt] = np.std(values) * correctionFactor

                    # Use sample stdev.
                    significanceThreshold = self.config.significanceLimit * calib.coeffErr[ss][tt]
                    if self.config.doSignificanceScaling is True:
                        # Enabling this calculates the stdev of the mean.
                        significanceThreshold /= np.sqrt(calib.coeffNum[ss][tt])
                    calib.coeffValid[ss][tt] = np.abs(calib.coeffs[ss][tt]) > significanceThreshold
                    self.debugRatios('measure', ratios, ordering[ss], ordering[tt],
                                     calib.coeffs[ss][tt], calib.coeffValid[ss][tt])
            self.log.info("Measured %s -> %s Coeff: %e Err: %e N: %d Valid: %s Limit: %e Quadratic: %s",
                          ordering[ss], ordering[tt], calib.coeffs[ss][tt], calib.coeffErr[ss][tt],
                          calib.coeffNum[ss][tt], calib.coeffValid[ss][tt], significanceThreshold,
                          polyfit)

        return calib

    @staticmethod
    def filterCrosstalkCalib(inCalib):
        """Apply valid constraints to the measured values.

        Any measured coefficient that is determined to be invalid is
        set to zero, and has the error set to nan.  The validation is
        determined by checking that the measured coefficient is larger
        than the calculated standard error of the mean.

        Parameters
        ----------
        inCalib : `lsst.ip.isr.CrosstalkCalib`
            Input calibration to filter.

        Returns
        -------
        outCalib : `lsst.ip.isr.CrosstalkCalib`
             Filtered calibration.
        """
        outCalib = CrosstalkCalib()
        outCalib.nAmp = inCalib.nAmp

        outCalib.coeffs = inCalib.coeffs
        outCalib.coeffs[~inCalib.coeffValid] = 0.0

        outCalib.coeffErr = inCalib.coeffErr
        outCalib.coeffErr[~inCalib.coeffValid] = np.nan

        outCalib.coeffNum = inCalib.coeffNum
        outCalib.coeffValid = inCalib.coeffValid

        outCalib.coeffsSqr = inCalib.coeffsSqr
        outCalib.coeffsSqr[~inCalib.coeffValid] = 0.0

        outCalib.coeffErrSqr = inCalib.coeffErrSqr
        outCalib.coeffErrSqr[~inCalib.coeffValid] = np.nan

        outCalib.ampGainRatios = inCalib.ampGainRatios
        outCalib.crosstalkRatiosUnits = inCalib.crosstalkRatiosUnits

        outCalib.fitGains = inCalib.fitGains

        return outCalib

    def debugRatios(self, stepname, ratios, i, j, coeff=0.0, valid=False):
        """Utility function to examine the final CT ratio set.

        Parameters
        ----------
        stepname : `str`
            State of processing to view.
        ratios : `dict` [`dict` [`numpy.ndarray`]]
            Array of measured CT ratios, indexed by source/victim
            amplifier.  These arrays are one-dimensional.
        i : `str`
            Index of the target amplifier.
        j : `str`
            Index of the source amplifier.
        coeff : `float`, optional
            Coefficient calculated to plot along with the simple mean.
        valid : `bool`, optional
            Validity to be added to the plot title.
        """
        frame = getDebugFrame(self._display, stepname)
        if frame:
            if i == j or ratios is None or len(ratios) < 1:
                pass

            ratioList = ratios[i][j]
            if ratioList is None or len(ratioList) < 1:
                pass

            mean = np.mean(ratioList)
            std = np.std(ratioList)
            import matplotlib.pyplot as plt
            figure = plt.figure(1)
            figure.clear()
            plt.hist(x=ratioList, bins=len(ratioList),
                     cumulative=True, color='b', density=True, histtype='step')
            plt.xlabel("Measured pixel ratio")
            plt.ylabel(f"CDF: n={len(ratioList)}")
            plt.xlim(np.percentile(ratioList, [1.0, 99]))
            plt.axvline(x=mean, color="k")
            plt.axvline(x=coeff, color='g')
            plt.axvline(x=(std / np.sqrt(len(ratioList))), color='r')
            plt.axvline(x=-(std / np.sqrt(len(ratioList))), color='r')
            plt.title(f"(Source {j} -> Target {i}) mean: {mean:.2g} coeff: {coeff:.2g} valid: {valid}")
            figure.show()

            prompt = "Press Enter to continue: "
            while True:
                ans = input(prompt).lower()
                if ans in ("", "c",):
                    break
                elif ans in ("pdb", "p",):
                    import pdb
                    pdb.set_trace()
            plt.close()


class CrosstalkFilterConnections(pipeBase.PipelineTaskConnections,
                                 dimensions=("instrument", )):
    inputCrosstalk = cT.Input(
        name="crosstalkProposal",
        doc="Input crosstalk calibrations as measured by CrosstalkSolveTask.",
        storageClass="IsrCalib",
        dimensions=("instrument", "detector"),
        isCalibration=True,
        multiple=True,
    )

    camera = cT.PrerequisiteInput(
        name="camera",
        doc="Camera containing cameraGeom information.",
        storageClass="Camera",
        dimensions=("instrument", ),
        isCalibration=True,
    )

    outputCrosstalk = cT.Output(
        name="crosstalk",
        doc="Filtered crosstalk solutions.",
        storageClass="CrosstalkCalib",
        dimensions=("instrument", "detector"),
        isCalibration=True,
        multiple=True,
    )


class CrosstalkFilterConfig(pipeBase.PipelineTaskConfig,
                            pipelineConnections=CrosstalkFilterConnections):
    """Configuration for the filtering of measured crosstalk solutions."""
    doFiltering = Field(
        dtype=bool,
        default=True,
        doc="Do filtering?  If false, then this task acts as a pass-through to rename dataset types.",
    )

    nSigmaCoeffClip = Field(
        dtype=float,
        default=3.0,
        doc="Coefficient outlier clipping significance.",
    )
    nSigmaCoeffSqrClip = Field(
        dtype=float,
        default=6.0,
        doc="Squared-term coefficient outlier clipping significance.",
    )
    unitsAreElectrons = Field(
        dtype=bool,
        default=True,
        doc="Crosstalk measurements have been done in electrons.",
    )


class CrosstalkFilterTask(pipeBase.PipelineTask):
    """Task to compare crosstalk solutions between detectors, to identify
    and remove outliers.
    """

    ConfigClass = CrosstalkFilterConfig
    _DefaultName = 'cpCrosstalkFilter'

    def runQuantum(self, butlerQC, inputRefs, outputRefs):
        """Ensure that the input and output dimensions are passed along.

        Parameters
        ----------
        butlerQC : `lsst.daf.butler.QuantumContext`
            Butler to operate on.
        inputRefs : `lsst.pipe.base.InputQuantizedConnection`
            Input data refs to load.
        ouptutRefs : `lsst.pipe.base.OutputQuantizedConnection`
            Output data refs to persist.
        """
        inputs = butlerQC.get(inputRefs)

        # Use the dimensions to set calib/provenance information.
        inputs['inputDims'] = [dict(inCT.dataId.required) for inCT in inputRefs.inputCrosstalk]
        inputs['outputDims'] = [dict(outCT.dataId.required) for outCT in outputRefs.outputCrosstalk]

        outputs = self.run(**inputs)
        butlerQC.put(outputs, outputRefs)

    def run(self, inputCrosstalk, camera, inputDims, outputDims):
        """Compare crosstalk solutions to produce filtered crosstalk
        calibrations.

        Parameters
        ----------
        inputCrosstalk : `list` [`lsst.ip.isr.CrosstalkCalib`]
            List of crosstalk solutions to filter.
        camera : `lsst.afw.cameraGeom.Camera`
            Input camera.
                inputDims : `list` [`lsst.daf.butler.DataCoordinate`]
            DataIds to use to construct provenance.
        outputDims : `list` [`lsst.daf.butler.DataCoordinate`]
            DataIds to use to populate the output calibration.

        Returns
        -------
        results : `lsst.pipe.base.Struct`
            The results struct containing:

            ``outputCrosstalk``
                Final crosstalk calibration
                (`lsst.ip.isr.CrosstalkCalib`).

        Raises
        ------
        RuntimeError
            Raised if something goes bad.  CZW/Fix me.
        """
        # These will hold all of the input data.
        itl_c0 = []
        e2v_c0 = []
        itl_c1 = []
        e2v_c1 = []
        detector_map = {}
        itl_counter = 0
        e2v_counter = 0
        for inputCT, inputDim in zip(inputCrosstalk, inputDims):
            detId = inputDim['detector']
            detector = camera[detId]

            if detector.getPhysicalType() == 'ITL':
                itl_c0.append(inputCT.coeffs)
                itl_c1.append(inputCT.coeffsSqr)

                detector_map[detId] = itl_counter
                itl_counter += 1

            elif detector.getPhysicalType() == 'E2V':
                e2v_c0.append(inputCT.coeffs)
                e2v_c1.append(inputCT.coeffsSqr)

                detector_map[detId] = e2v_counter
                e2v_counter += 1
            else:
                # This is a wavefront sensor, and we don't want to
                # filter those.
                pass

        itl_c0 = np.array(itl_c0)
        itl_c1 = np.array(itl_c1)
        e2v_c0 = np.array(e2v_c0)
        e2v_c1 = np.array(e2v_c1)

        itl_outliers = self.find_outliers(itl_c0, itl_c1)
        e2v_outliers = self.find_outliers(e2v_c0, e2v_c1)

        if self.config.doFiltering:
            itl_final = self.replace_outliers(itl_c0, itl_c1,
                                              itl_outliers.isBad, itl_outliers.median0, itl_outliers.median1)
            e2v_final = self.replace_outliers(e2v_c0, e2v_c1,
                                              e2v_outliers.isBad, e2v_outliers.median0, e2v_outliers.median1)
        outputCrosstalkList = []
        for inputCT, inputDim, outputDim in zip(inputCrosstalk, inputDims, outputDims):
            if inputDim['detector'] != outputDim['detector']:
                raise RuntimeError("Inconsistent dimension records")

            detId = inputDim['detector']
            detector = camera[detId]

            outputCT = copy(inputCT)

            if detector.getPhysicalType() == 'ITL':
                itl_index = detector_map[detId]
                if (np.any(itl_final.new_matrix0[itl_index] != outputCT.coeffs)
                        or np.any(itl_final.new_matrix1[itl_index] != outputCT.coeffsSqr)):
                    outputCT.coeffs = itl_final.new_matrix0[itl_index]
                    outputCT.coeffsSqr = itl_final.new_matrix1[itl_index]
            elif detector.getPhysicalType() == 'E2V':
                e2v_index = detector_map[detId]
                if (np.any(e2v_final.new_matrix0[e2v_index] != outputCT.coeffs)
                        or np.any(e2v_final.new_matrix1[e2v_index] != outputCT.coeffsSqr)):
                    outputCT.coeffs = e2v_final.new_matrix0[e2v_index]
                    outputCT.coeffsSqr = e2v_final.new_matrix1[e2v_index]

            outputCT.crosstalkRatiosUnits = 'electron' if self.config.unitsAreElectrons else 'adu'
            outputCT.updateMetadata(
                camera=camera,
                detector=camera[detId],
                setCalibId=True,
                setCalibInfo=True,
                setDate=True,
            )
            outputCrosstalkList.append(outputCT)
        return pipeBase.Struct(
            outputCrosstalk=outputCrosstalkList,
        )

    def find_outliers(self, matrix0, matrix1):
        """Do checks to see if an element of the matrix is out-of-family.

        Parameters
        ----------
        matrix0 : `np.array`, (Ndet, Namp, Namp)
            Matrix holding the 0th-order terms.
        matrix1 : `np.array`, (Ndet, Namp, Namp)
            Matrix holding the 1st-order terms.

        Returns
        -------
        results : `lsst.pipe.base.Struct`
            Results struct containing

            ``median0``
                Median in-family value (`np.array` (Namp, Namp)).
            ``stdev0``
                MAD effective sigma in-family value (`np.array` (Namp, Namp)).
            ``median1``
                Median in-family value (`np.array` (Namp, Namp)).
            ``stdev1``
                MAD effective sigma in-family value (`np.array` (Namp, Namp)).
            ``isBad``
                Boolean indicator that an element has been replaced
                (`np.array` (Ndet, Namp, Namp)).

        Raises
        ------
        ValueError :
            Raised if the inputs have a mismatch in size.
        """
        if matrix0.shape != matrix1.shape:
            raise ValueError("Shape disagreement!")
        if matrix0.shape[1] != matrix0.shape[2]:
            raise ValueError("Shape disagreement!")

        nAmp = matrix0.shape[1]

        median0 = np.nanmedian(matrix0, axis=0)
        median1 = np.nanmedian(matrix1, axis=0)

        stdev0 = median_abs_deviation(matrix0, axis=0,
                                      center=np.nanmedian, scale="normal", nan_policy="omit")
        stdev1 = median_abs_deviation(matrix1, axis=0,
                                      center=np.nanmedian, scale="normal", nan_policy="omit")

        isBad = np.full_like(matrix0, False, dtype=bool)

        for i in range(nAmp):
            for j in range(nAmp):
                m0 = median0[i][j]
                m1 = median1[i][j]
                s0 = stdev0[i][j]
                s1 = stdev1[i][j]

                min0 = m0 - self.config.nSigmaCoeffClip * s0
                max0 = m0 + self.config.nSigmaCoeffClip * s0

                min1 = m1 - self.config.nSigmaCoeffSqrClip * s1
                max1 = m1 + self.config.nSigmaCoeffSqrClip * s1

                bad, = np.where(
                    (matrix0[:, i, j] < min0)
                    | (matrix0[:, i, j] > max0)
                    | (matrix1[:, i, j] < min1)
                    | (matrix1[:, i, j] > max1)
                )

                if len(bad) > 0:
                    for detIdx in bad:
                        isBad[detIdx, i, j] = True
        return pipeBase.Struct(
            median0=median0,
            median1=median1,
            stdev0=stdev0,
            stdev1=stdev1,
            isBad=isBad
        )

    def replace_outliers(self, matrix0, matrix1, isBad, median0, median1):
        """Do checks to see if an element of the matrix is out-of-family.

        Parameters
        ----------
        matrix0 : `np.array`, (Ndet, Namp, Namp)
            Matrix holding the 0th-order terms.
        matrix1 : `np.array`, (Ndet, Namp, Namp)
            Matrix holding the 1st-order terms.
        isBad : `np.array`, (Ndet, Namp, Namp)
            Matrix holding the boolean "is bad".
        median0 : `np.array`, (Namp, Namp)
            Matrix of median 0th-order terms.
        median1 : `np.array`, (Namp, Namp)
            Matrix of median 1st-order terms.

        Returns
        -------
        results : `lsst.pipe.base.Struct`
            Results struct containing

            ``new_matrix0``
                Replacement matrix0, with median substitutions.
                (`np.array` (Ndet, Namp, Namp)).
            ``new_matrix1``
                Replacement matrix1, with median substitutions.
                (`np.array` (Ndet, Namp, Namp)).

        Raises
        ------
        ValueError :
            Raised if the inputs have a mismatch in size.
        """
        if matrix0.shape != matrix1.shape:
            raise ValueError("Shape disagreement!")
        if matrix0.shape != isBad.shape:
            raise ValueError("Shape disagreement!")
        if median0.shape != median1.shape:
            raise ValueError("Shape disagreement!")

        out0 = np.full_like(matrix0, 0.0)
        out1 = np.full_like(matrix1, 0.0)

        for detIdx in range(matrix0.shape[0]):
            for srcIdx in range(matrix0.shape[1]):
                for tgtIdx in range(matrix0.shape[2]):
                    if isBad[detIdx, srcIdx, tgtIdx]:
                        self.log.info(f"Setting {detIdx} {srcIdx} {tgtIdx} from "
                                      f"{matrix0[detIdx, srcIdx, tgtIdx]} to "
                                      f"{median0[srcIdx, tgtIdx]} and "
                                      f"{matrix1[detIdx, srcIdx, tgtIdx]} to "
                                      f"{median1[srcIdx, tgtIdx]}")
                        out0[detIdx, srcIdx, tgtIdx] = median0[srcIdx, tgtIdx]
                        out1[detIdx, srcIdx, tgtIdx] = median1[srcIdx, tgtIdx]
                    else:
                        out0[detIdx, srcIdx, tgtIdx] = matrix0[detIdx, srcIdx, tgtIdx]
                        out1[detIdx, srcIdx, tgtIdx] = matrix1[detIdx, srcIdx, tgtIdx]
        return pipeBase.Struct(
            new_matrix0=out0,
            new_matrix1=out1,
        )
