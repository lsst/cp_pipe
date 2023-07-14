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
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
import numpy as np
import time

import lsst.geom as geom
import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase
import lsst.pipe.base.connectionTypes as cT
import lsst.afw.math as afwMath
import lsst.afw.image as afwImage

from lsst.ip.isr.vignette import maskVignettedRegion

from astro_metadata_translator import merge_headers, ObservationGroup
from astro_metadata_translator.serialize import dates_to_fits


__all__ = ["CalibStatsConfig", "CalibStatsTask",
           "CalibCombineConfig", "CalibCombineConnections", "CalibCombineTask",
           "CalibCombineByFilterConfig", "CalibCombineByFilterConnections", "CalibCombineByFilterTask"]


# CalibStatsConfig/CalibStatsTask from pipe_base/constructCalibs.py
class CalibStatsConfig(pexConfig.Config):
    """Parameters controlling the measurement of background
    statistics.
    """

    stat = pexConfig.Field(
        dtype=str,
        default="MEANCLIP",
        doc="Statistic name to use to estimate background (from `~lsst.afw.math.Property`)",
    )
    clip = pexConfig.Field(
        dtype=float,
        default=3.0,
        doc="Clipping threshold for background",
    )
    nIter = pexConfig.Field(
        dtype=int,
        default=3,
        doc="Clipping iterations for background",
    )
    mask = pexConfig.ListField(
        dtype=str,
        default=["DETECTED", "BAD", "NO_DATA"],
        doc="Mask planes to reject",
    )


class CalibStatsTask(pipeBase.Task):
    """Measure statistics on the background

    This can be useful for scaling the background, e.g., for flats and
    fringe frames.
    """

    ConfigClass = CalibStatsConfig

    def run(self, exposureOrImage):
        """Measure a particular statistic on an image (of some sort).

        Parameters
        ----------
        exposureOrImage : `lsst.afw.image.Exposure`,
                          `lsst.afw.image.MaskedImage`, or
                          `lsst.afw.image.Image`
            Exposure or image to calculate statistics on.

        Returns
        -------
        results : `float`
           Resulting statistic value.
        """
        stats = afwMath.StatisticsControl(self.config.clip, self.config.nIter,
                                          afwImage.Mask.getPlaneBitMask(self.config.mask))
        try:
            image = exposureOrImage.getMaskedImage()
        except Exception:
            try:
                image = exposureOrImage.getImage()
            except Exception:
                image = exposureOrImage
        statType = afwMath.stringToStatisticsProperty(self.config.stat)
        return afwMath.makeStatistics(image, statType, stats).getValue()


class CalibCombineConnections(pipeBase.PipelineTaskConnections,
                              dimensions=("instrument", "detector")):
    inputExpHandles = cT.Input(
        name="cpInputs",
        doc="Input pre-processed exposures to combine.",
        storageClass="Exposure",
        dimensions=("instrument", "detector", "exposure"),
        multiple=True,
        deferLoad=True,
    )
    inputScales = cT.Input(
        name="cpScales",
        doc="Input scale factors to use.",
        storageClass="StructuredDataDict",
        dimensions=("instrument", ),
        multiple=False,
    )

    outputData = cT.Output(
        name="cpProposal",
        doc="Output combined proposed calibration to be validated and certified..",
        storageClass="ExposureF",
        dimensions=("instrument", "detector"),
        isCalibration=True,
    )

    def __init__(self, *, config=None):
        super().__init__(config=config)

        if config and config.exposureScaling != "InputList":
            self.inputs.discard("inputScales")


# CalibCombineConfig/CalibCombineTask from pipe_base/constructCalibs.py
class CalibCombineConfig(pipeBase.PipelineTaskConfig,
                         pipelineConnections=CalibCombineConnections):
    """Configuration for combining calib exposures.
    """

    calibrationType = pexConfig.Field(
        dtype=str,
        default="calibration",
        doc="Name of calibration to be generated.",
    )

    exposureScaling = pexConfig.ChoiceField(
        dtype=str,
        allowed={
            "Unity": "Do not scale inputs.  Scale factor is 1.0.",
            "ExposureTime": "Scale inputs by their exposure time.",
            "DarkTime": "Scale inputs by their dark time.",
            "MeanStats": "Scale inputs based on their mean values.",
            "InputList": "Scale inputs based on a list of values.",
        },
        default="Unity",
        doc="Scaling to be applied to each input exposure.",
    )
    scalingLevel = pexConfig.ChoiceField(
        dtype=str,
        allowed={
            "DETECTOR": "Scale by detector.",
            "AMP": "Scale by amplifier.",
        },
        default="DETECTOR",
        doc="Region to scale.",
    )
    maxVisitsToCalcErrorFromInputVariance = pexConfig.Field(
        dtype=int,
        default=5,
        doc="Maximum number of visits to estimate variance from input variance, not per-pixel spread",
    )
    subregionSize = pexConfig.ListField(
        dtype=int,
        doc="Width, height of subregion size.",
        length=2,
        # This is 200 rows for all detectors smaller than 10k in width.
        default=(10000, 200),
    )

    doVignette = pexConfig.Field(
        dtype=bool,
        default=False,
        doc="Copy vignette polygon to output and censor vignetted pixels?"
    )

    mask = pexConfig.ListField(
        dtype=str,
        default=["SAT", "DETECTED", "INTRP"],
        doc="Mask planes to respect",
    )
    combine = pexConfig.Field(
        dtype=str,
        default="MEANCLIP",
        doc="Statistic name to use for combination (from `~lsst.afw.math.Property`)",
    )
    clip = pexConfig.Field(
        dtype=float,
        default=3.0,
        doc="Clipping threshold for combination",
    )
    nIter = pexConfig.Field(
        dtype=int,
        default=3,
        doc="Clipping iterations for combination",
    )
    stats = pexConfig.ConfigurableField(
        target=CalibStatsTask,
        doc="Background statistics configuration",
    )


class CalibCombineTask(pipeBase.PipelineTask):
    """Task to combine calib exposures."""

    ConfigClass = CalibCombineConfig
    _DefaultName = "cpCombine"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.makeSubtask("stats")

    def runQuantum(self, butlerQC, inputRefs, outputRefs):
        inputs = butlerQC.get(inputRefs)

        dimensions = [expHandle.dataId.byName() for expHandle in inputRefs.inputExpHandles]
        inputs["inputDims"] = dimensions

        outputs = self.run(**inputs)
        butlerQC.put(outputs, outputRefs)

    def run(self, inputExpHandles, inputScales=None, inputDims=None):
        """Combine calib exposures for a single detector.

        Parameters
        ----------
        inputExpHandles : `list` [`lsst.daf.butler.DeferredDatasetHandle`]
            Input list of exposure handles to combine.
        inputScales : `dict` [`dict` [`dict` [`float`]]], optional
            Dictionary of scales, indexed by detector (`int`),
            amplifier (`int`), and exposure (`int`).  Used for
            'inputExps' scaling.
        inputDims : `list` [`dict`]
            List of dictionaries of input data dimensions/values.
            Each list entry should contain:

            ``"exposure"``
                exposure id value (`int`)
            ``"detector"``
                detector id value (`int`)

        Returns
        -------
        results : `lsst.pipe.base.Struct`
            The results struct containing:

            ``outputData``
                Final combined exposure generated from the inputs
                (`lsst.afw.image.Exposure`).

        Raises
        ------
        RuntimeError
            Raised if no input data is found.  Also raised if
            config.exposureScaling == InputList, and a necessary scale
            was not found.
        """
        width, height = self.getDimensions(inputExpHandles)
        stats = afwMath.StatisticsControl(self.config.clip, self.config.nIter,
                                          afwImage.Mask.getPlaneBitMask(self.config.mask))
        numExps = len(inputExpHandles)
        if numExps < 1:
            raise RuntimeError("No valid input data")
        if numExps < self.config.maxVisitsToCalcErrorFromInputVariance:
            stats.setCalcErrorFromInputVariance(True)

        inputDetector = inputExpHandles[0].get(component="detector")

        # Create output exposure for combined data.
        combined = afwImage.MaskedImageF(width, height)
        combinedExp = afwImage.makeExposure(combined)

        # Apply scaling:
        expScales = []
        if inputDims is None:
            inputDims = [dict() for i in inputExpHandles]

        for index, (expHandle, dims) in enumerate(zip(inputExpHandles, inputDims)):
            scale = 1.0
            visitInfo = expHandle.get(component="visitInfo")
            if self.config.exposureScaling == "ExposureTime":
                scale = visitInfo.getExposureTime()
            elif self.config.exposureScaling == "DarkTime":
                scale = visitInfo.getDarkTime()
            elif self.config.exposureScaling == "MeanStats":
                # Note: there may a bug freeing memory here. TBD.
                exp = expHandle.get()
                scale = self.stats.run(exp)
                del exp
            elif self.config.exposureScaling == "InputList":
                visitId = dims.get("exposure", None)
                detectorId = dims.get("detector", None)
                if visitId is None or detectorId is None:
                    raise RuntimeError(f"Could not identify scaling for input {index} ({dims})")
                if detectorId not in inputScales["expScale"]:
                    raise RuntimeError(f"Could not identify a scaling for input {index}"
                                       f" detector {detectorId}")

                if self.config.scalingLevel == "DETECTOR":
                    if visitId not in inputScales["expScale"][detectorId]:
                        raise RuntimeError(f"Could not identify a scaling for input {index}"
                                           f"detector {detectorId} visit {visitId}")
                    scale = inputScales["expScale"][detectorId][visitId]
                elif self.config.scalingLevel == "AMP":
                    scale = [inputScales["expScale"][detectorId][amp.getName()][visitId]
                             for amp in inputDetector]
                else:
                    raise RuntimeError(f"Unknown scaling level: {self.config.scalingLevel}")
            elif self.config.exposureScaling == "Unity":
                scale = 1.0
            else:
                raise RuntimeError(f"Unknown scaling type: {self.config.exposureScaling}.")

            expScales.append(scale)
            self.log.info("Scaling input %d by %s", index, scale)

        combinedExp = self.combine(combinedExp, inputExpHandles, expScales, stats)

        if isinstance(combinedExp, afwImage.Exposure):
            self.interpolateNans(combined)

            if self.config.doVignette:
                polygon = inputExpHandles[0].get(component="validPolygon")
                maskVignettedRegion(combined, polygon=polygon, vignetteValue=0.0)

                # Combine headers
                self.combineHeaders(inputExpHandles, combinedExp,
                                    calibType=self.config.calibrationType, scales=expScales)

                # Set the detector
                combinedExp.setDetector(inputDetector)

                # Do we need to set a filter?
                filterLabel = inputExpHandles[0].get(component="filter")
                self.setFilter(combinedExp, filterLabel)

        # Return
        return pipeBase.Struct(
            outputData=combinedExp,
        )

    def getDimensions(self, expHandleList):
        """Get dimensions of the inputs.

        Parameters
        ----------
        expHandleList : `list` [`lsst.daf.butler.DeferredDatasetHandle`]
            Exposure handles to check the sizes of.

        Returns
        -------
        width, height : `int`
            Unique set of input dimensions.
        """
        dimList = [expHandle.get(component="bbox").getDimensions() for expHandle in expHandleList]

        return self.getSize(dimList)

    def getSize(self, dimList):
        """Determine a consistent size, given a list of image sizes.

        Parameters
        ----------
        dimList : `list` [`tuple` [`int`, `int`]]
            List of dimensions.

        Raises
        ------
        RuntimeError
            If input dimensions are inconsistent.

        Returns
        -------
        width, height : `int`
            Common dimensions.
        """
        dim = set((w, h) for w, h in dimList)
        if len(dim) != 1:
            raise RuntimeError("Inconsistent dimensions: %s" % dim)
        return dim.pop()

    def applyScale(self, exposure, bbox=None, scale=None):
        """Apply scale to input exposure.

        This implementation applies a flux scaling: the input exposure is
        divided by the provided scale.

        Parameters
        ----------
        exposure : `lsst.afw.image.Exposure`
            Exposure to scale.
        bbox : `lsst.geom.Box2I`
            BBox matching the segment of the exposure passed in.
        scale : `float` or `list` [`float`], optional
            Constant scale to divide the exposure by.
        """
        if scale is not None:
            mi = exposure.getMaskedImage()
            if isinstance(scale, list):
                # Create a realization of the per-amp scales as an
                # image we can take a subset of.  This may be slightly
                # slower than only populating the region we care
                # about, but this avoids needing to do arbitrary
                # numbers of offsets, etc.
                scaleExp = afwImage.MaskedImageF(exposure.getDetector().getBBox())
                for amp, ampScale in zip(exposure.getDetector(), scale):
                    scaleExp.image[amp.getBBox()] = ampScale
                scale = scaleExp[bbox]
            mi /= scale

    @staticmethod
    def _subBBoxIter(bbox, subregionSize):
        """Iterate over subregions of a bbox.

        Parameters
        ----------
        bbox : `lsst.geom.Box2I`
            Bounding box over which to iterate.
        subregionSize: `lsst.geom.Extent2I`
            Size of sub-bboxes.

        Yields
        ------
        subBBox : `lsst.geom.Box2I`
            Next sub-bounding box of size ``subregionSize`` or
            smaller; each ``subBBox`` is contained within ``bbox``, so
            it may be smaller than ``subregionSize`` at the edges of
            ``bbox``, but it will never be empty.
        """
        if bbox.isEmpty():
            raise RuntimeError("bbox %s is empty" % (bbox,))
        if subregionSize[0] < 1 or subregionSize[1] < 1:
            raise RuntimeError("subregionSize %s must be nonzero" % (subregionSize,))

        for rowShift in range(0, bbox.getHeight(), subregionSize[1]):
            for colShift in range(0, bbox.getWidth(), subregionSize[0]):
                subBBox = geom.Box2I(bbox.getMin() + geom.Extent2I(colShift, rowShift), subregionSize)
                subBBox.clip(bbox)
                if subBBox.isEmpty():
                    raise RuntimeError("Bug: empty bbox! bbox=%s, subregionSize=%s, "
                                       "colShift=%s, rowShift=%s" %
                                       (bbox, subregionSize, colShift, rowShift))
                yield subBBox

    def combine(self, target, expHandleList, expScaleList, stats):
        """Combine multiple images.

        Parameters
        ----------
        target : `lsst.afw.image.Exposure`
            Output exposure to construct.
        expHandleList : `list` [`lsst.daf.butler.DeferredDatasetHandle`]
            Input exposure handles to combine.
        expScaleList : `list` [`float`]
            List of scales to apply to each input image.
        stats : `lsst.afw.math.StatisticsControl`
            Control explaining how to combine the input images.
        """
        combineType = afwMath.stringToStatisticsProperty(self.config.combine)

        subregionSizeArr = self.config.subregionSize
        subregionSize = geom.Extent2I(subregionSizeArr[0], subregionSizeArr[1])
        for subBbox in self._subBBoxIter(target.getBBox(), subregionSize):
            images = []
            for expHandle, expScale in zip(expHandleList, expScaleList):
                inputExp = expHandle.get(parameters={"bbox": subBbox})
                self.applyScale(inputExp, subBbox, expScale)
                images.append(inputExp.getMaskedImage())

            combinedSubregion = afwMath.statisticsStack(images, combineType, stats)
            target.maskedImage.assign(combinedSubregion, subBbox)
        return target

    def combineHeaders(self, expHandleList, calib, calibType="CALIB", scales=None):
        """Combine input headers to determine the set of common headers,
        supplemented by calibration inputs.  The calibration header is
        set in-place.

        Parameters
        ----------
        expHandleList : `list` [`lsst.daf.butler.DeferredDatasetHandle`]
            Input list of exposure handles to combine.
        calib : `lsst.afw.image.Exposure`
            Output calibration to construct headers for.
        calibType : `str`, optional
            OBSTYPE the output should claim.
        scales : `list` [`float`], optional
            Scale values applied to each input to record.

        Returns
        -------
        header : `lsst.daf.base.PropertyList`
            Constructed header.
        """
        # Header
        header = calib.getMetadata()
        header.set("OBSTYPE", calibType)

        # Keywords we care about
        comments = {"TIMESYS": "Time scale for all dates",
                    "DATE-OBS": "Start date of earliest input observation",
                    "MJD-OBS": "[d] Start MJD of earliest input observation",
                    "DATE-END": "End date of oldest input observation",
                    "MJD-END": "[d] End MJD of oldest input observation",
                    "MJD-AVG": "[d] MJD midpoint of all input observations",
                    "DATE-AVG": "Midpoint date of all input observations"}

        # Creation date
        now = time.localtime()
        calibDate = time.strftime("%Y-%m-%d", now)
        calibTime = time.strftime("%X %Z", now)
        header.set("CALIB_CREATION_DATE", calibDate)
        header.set("CALIB_CREATION_TIME", calibTime)

        # Merge input headers
        inputHeaders = [expHandle.get(component="metadata") for expHandle in expHandleList]
        merged = merge_headers(inputHeaders, mode="drop")

        # Scan the first header for items that were dropped due to
        # conflict, and replace them.
        for k, v in merged.items():
            if k not in header:
                md = inputHeaders[0]
                comment = md.getComment(k) if k in md else None
                header.set(k, v, comment=comment)

        # Construct list of visits
        visitInfoList = [expHandle.get(component="visitInfo") for expHandle in expHandleList]
        for i, visit in enumerate(visitInfoList):
            if visit is None:
                continue
            header.set("CPP_INPUT_%d" % (i,), visit.id)
            header.set("CPP_INPUT_DATE_%d" % (i,), str(visit.getDate()))
            header.set("CPP_INPUT_EXPT_%d" % (i,), visit.getExposureTime())
            if scales is not None:
                header.set("CPP_INPUT_SCALE_%d" % (i,), scales[i])

        # Populate a visitInfo.  Set the exposure time and dark time
        # to 0.0 or 1.0 as appropriate, and copy the instrument name
        # from one of the inputs.
        expTime = 1.0
        if self.config.connections.outputData.lower() == 'bias':
            expTime = 0.0
        inputVisitInfo = visitInfoList[0]
        visitInfo = afwImage.VisitInfo(exposureTime=expTime, darkTime=expTime,
                                       instrumentLabel=inputVisitInfo.instrumentLabel)
        calib.getInfo().setVisitInfo(visitInfo)

        # Not yet working: DM-22302
        # Create an observation group so we can add some standard headers
        # independent of the form in the input files.
        # Use try block in case we are dealing with unexpected data headers
        try:
            group = ObservationGroup(visitInfoList, pedantic=False)
        except Exception:
            self.log.warning("Exception making an obs group for headers. Continuing.")
            # Fall back to setting a DATE-OBS from the calibDate
            dateCards = {"DATE-OBS": "{}T00:00:00.00".format(calibDate)}
            comments["DATE-OBS"] = "Date of start of day of calibration creation"
        else:
            oldest, newest = group.extremes()
            dateCards = dates_to_fits(oldest.datetime_begin, newest.datetime_end)

        for k, v in dateCards.items():
            header.set(k, v, comment=comments.get(k, None))

        return header

    def interpolateNans(self, exp):
        """Interpolate over NANs in the combined image.

        NANs can result from masked areas on the CCD.  We don't want
        them getting into our science images, so we replace them with
        the median of the image.

        Parameters
        ----------
        exp : `lsst.afw.image.Exposure`
            Exp to check for NaNs.
        """
        array = exp.getImage().getArray()
        bad = np.isnan(array)
        if np.any(bad):
            median = np.median(array[np.logical_not(bad)])
            count = np.sum(bad)
            array[bad] = median
            self.log.warning("Found and fixed %s NAN pixels", count)

    @staticmethod
    def setFilter(exp, filterLabel):
        """Dummy function that will not assign a filter.

        Parameters
        ----------
        exp : `lsst.afw.image.Exposure`
            Exposure to assign filter to.
        filterLabel : `lsst.afw.image.FilterLabel`
            Filter to assign.
        """
        pass


# Create versions of the Connections, Config, and Task that support
# filter constraints.
class CalibCombineByFilterConnections(CalibCombineConnections,
                                      dimensions=("instrument", "detector", "physical_filter")):
    inputScales = cT.Input(
        name="cpFilterScales",
        doc="Input scale factors to use.",
        storageClass="StructuredDataDict",
        dimensions=("instrument", "physical_filter"),
        multiple=False,
    )

    outputData = cT.Output(
        name="cpFilterProposal",
        doc="Output combined proposed calibration to be validated and certified.",
        storageClass="ExposureF",
        dimensions=("instrument", "detector", "physical_filter"),
        isCalibration=True,
    )

    def __init__(self, *, config=None):
        super().__init__(config=config)

        if config and config.exposureScaling != "InputList":
            self.inputs.discard("inputScales")


class CalibCombineByFilterConfig(CalibCombineConfig,
                                 pipelineConnections=CalibCombineByFilterConnections):
    pass


class CalibCombineByFilterTask(CalibCombineTask):
    """Task to combine calib exposures."""

    ConfigClass = CalibCombineByFilterConfig
    _DefaultName = "cpFilterCombine"

    @staticmethod
    def setFilter(exp, filterLabel):
        """Dummy function that will not assign a filter.

        Parameters
        ----------
        exp : `lsst.afw.image.Exposure`
            Exposure to assign filter to.
        filterLabel : `lsst.afw.image.FilterLabel`
            Filter to assign.
        """
        if filterLabel:
            exp.setFilter(filterLabel)
