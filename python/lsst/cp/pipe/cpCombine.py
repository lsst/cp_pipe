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
from datetime import datetime, UTC
from operator import attrgetter

import astropy.time
import lsst.geom as geom
import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase
import lsst.pipe.base.connectionTypes as cT
import lsst.afw.math as afwMath
import lsst.afw.image as afwImage

import lsst.daf.base

from lsst.ip.isr.vignette import maskVignettedRegion

from astro_metadata_translator import merge_headers
from astro_metadata_translator.serialize import dates_to_fits
from lsst.obs.base.utils import strip_provenance_from_fits_header


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

    distributionPercentiles = pexConfig.ListField(
        dtype=float,
        default=[0, 5, 16, 50, 84, 95, 100],
        doc="Percentile levels to measure on the final combined calibration.",
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
    noGoodPixelsMask = pexConfig.Field(
        dtype=str,
        default="BAD",
        doc="Mask bit to set when there are no good input pixels.  See code comments for details.",
    )
    checkNoData = pexConfig.Field(
        dtype=bool,
        default=True,
        doc="Check that the calibration does not have NO_DATA set?",
    )
    censorMaskPlanes = pexConfig.Field(
        dtype=bool,
        default=True,
        doc="Unset mask planes other than NO_DATA in output calibration?",
    )
    setAllPixelsToAmpMean = pexConfig.Field(
        dtype=bool,
        default=False,
        doc="Return an image with every pixel equal to the amp-wise mean of the combined calib?",
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

        dimensions = [dict(expHandle.dataId.required) for expHandle in inputRefs.inputExpHandles]
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
        stats = afwMath.StatisticsControl(
            numSigmaClip=self.config.clip,
            numIter=self.config.nIter,
            andMask=afwImage.Mask.getPlaneBitMask(self.config.mask),
        )

        # Normally, this would be set to NO_DATA.  However, we want
        # pixels in the combined calibration that had no good inputs
        # to be treated as a defect (which are interpolated) and not
        # like an empty region (such as vignetted corners, which are
        # not interpolated).  To handle this, we use the config to
        # override the mask plane used (BAD by default), and censor
        # all other mask planes below.
        stats.setNoGoodPixelsMask(afwImage.Mask.getPlaneBitMask(self.config.noGoodPixelsMask))

        numExps = len(inputExpHandles)
        if numExps < 1:
            raise RuntimeError("No valid input data")
        if numExps < self.config.maxVisitsToCalcErrorFromInputVariance:
            stats.setCalcErrorFromInputVariance(True)

        inputDetector = inputExpHandles[0].get(component="detector")

        # Create output exposure for combined data.
        combined = afwImage.MaskedImageF(width, height)
        combinedExp = afwImage.makeExposure(combined)
        meanCombinedExp = combinedExp.clone()

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

                if self.config.scalingLevel in ("DETECTOR", "AMP"):
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

        self.combine(combinedExp, inputExpHandles, expScales, stats)

        # The calibration should _never_ have NO_DATA set, as this is
        # handled poorly downstream, and so we raise here after
        # explicitly checking for pixels with that bit set.
        if self.config.checkNoData:
            test = ((combinedExp.mask.array & afwImage.Mask.getPlaneBitMask("NO_DATA")) > 0)
            if (nnodata := test.sum()) > 0:
                raise RuntimeError(f"Combined calibration has {nnodata} NO_DATA pixels!")

        if self.config.censorMaskPlanes:
            # Any mask planes that are not the noGoodPixelsMask plane
            # should be cleared.  This should remove things like the
            # CROSSTALK plane from printing into the calibration.  Run
            # after the checkNoData pass so that we don't censor what
            # we want to raise on.
            mask = combinedExp.mask
            for plane, value in mask.getMaskPlaneDict().items():
                if plane != self.config.noGoodPixelsMask:
                    mask.clearMaskPlane(value)

        self.interpolateNans(combined, maskPlane=self.config.noGoodPixelsMask)

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

        # Set QA headers
        self.calibStats(combinedExp, self.config.calibrationType)

        # Optional: set every pixel in an amplifier to the amp-wise
        # mean of the combined exposure.
        if self.config.setAllPixelsToAmpMean:
            # Possibly a good option if there are
            # not enough input calibs, and the combined
            # calib is noise-dominated
            for amp in inputDetector:
                ampDataBbox = amp.getBBox()
                ampDataMean = np.mean(combinedExp.image[ampDataBbox].array)
                combinedExp.image[ampDataBbox].array = ampDataMean

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
            raise RuntimeError(f"Inconsistent dimensions: {dim}")
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
            raise RuntimeError(f"bbox {bbox} is empty")
        if subregionSize[0] < 1 or subregionSize[1] < 1:
            raise RuntimeError(f"subregionSize {subregionSize} must be nonzero")

        for rowShift in range(0, bbox.getHeight(), subregionSize[1]):
            for colShift in range(0, bbox.getWidth(), subregionSize[0]):
                subBBox = geom.Box2I(bbox.getMin() + geom.Extent2I(colShift, rowShift), subregionSize)
                subBBox.clip(bbox)
                if subBBox.isEmpty():
                    raise RuntimeError(f"Bug: empty bbox! bbox={bbox}, subregionSize={subregionSize}, "
                                       f"colShift={colShift}, rowShift={rowShift}")
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

    def combineHeaders(self, expHandleList, calib=None, calibType="CALIB", scales=None, metadata=None):
        """Combine input headers to determine the set of common headers,
        supplemented by calibration inputs.  The calibration header is
        set in-place.

        Parameters
        ----------
        expHandleList : `list` [`lsst.daf.butler.DeferredDatasetHandle`]
            Input list of exposure handles to combine.
        calib : `lsst.afw.image.Exposure`, optional
            Output calibration to construct headers for.
        calibType : `str`, optional
            OBSTYPE the output should claim.
        scales : `list` [`float`], optional
            Scale values applied to each input to record.
        metadata : `lsst.daf.base.PropertyList`, optional
            Output metadata to add headers to.

        Returns
        -------
        header : `lsst.daf.base.PropertyList`
            Constructed header.

        Raises
        ------
        RuntimeError
            Raised if neither a calib nor a metadata was supplied.
        """
        # Header
        if calib is not None:
            header = calib.getMetadata()
        elif metadata is not None:
            header = metadata
        else:
            raise RuntimeError("No calibration and no metadata passed to combineHeaders")

        header.set("OBSTYPE", calibType)

        # Keywords we care about
        comments = {"TIMESYS": "Time scale for all dates",
                    "DATE-OBS": "Start date of earliest input observation",
                    "MJD-OBS": "[d] Start MJD of earliest input observation",
                    "DATE-BEG": "Start date of earliest input observation",
                    "MJD-BEG": "[d] Start MJD of earliest input observation",
                    "DATE-END": "End date of oldest input observation",
                    "MJD-END": "[d] End MJD of oldest input observation",
                    "MJD-AVG": "[d] MJD midpoint of all input observations",
                    "DATE-AVG": "Midpoint date of all input observations"}

        # Creation date. Calibration team standard is for local time to be
        # available. Also form UTC (not TAI) version for easier comparisons
        # across multiple processing sites.
        now = datetime.now(tz=UTC)
        header.set("CALIB_CREATION_DATETIME", now.strftime("%Y-%m-%dT%T"), comment="UTC of processing")
        local_time = now.astimezone()
        calibDate = local_time.strftime("%Y-%m-%d")
        calibTime = local_time.strftime("%X %Z")
        header.set("CALIB_CREATION_DATE", calibDate, comment="Local time day of creation")
        header.set("CALIB_CREATION_TIME", calibTime, comment="Local time in day of creation")

        # Merge input headers
        inputHeaders = [expHandle.get(component="metadata") for expHandle in expHandleList]
        merged = merge_headers(inputHeaders, mode="drop")

        # Remove any left over provenance headers that weren't dropped
        # (for example if different numbers of input datasets were present).
        strip_provenance_from_fits_header(merged)

        # Add the unchanging headers from all inputs to the given header.
        for k, v in merged.items():
            if k not in header:
                # The merged header should be a PropertyList so will have
                # comment information.
                header.set(k, v, comment=merged.getComment(k))

        # Ideally we would avoid going to butler again to read VisitInfo
        # information when there is already the header available. Unfortunately
        # the FITS reader strips VisitInfo keys from the header so it is no
        # longer possible to create an ObservationInfo with valid exposure
        # time (and DATE-BEG survives only because afw calls it DATE-OBS).

        # Construct list of visits
        visitInfoList = [expHandle.get(component="visitInfo") for expHandle in expHandleList]

        # Create provenance headers.
        for i, visit in enumerate(visitInfoList):
            if visit is None:
                continue
            header.set(f"CPP_INPUT_{i}", visit.id, comment="Input exposure ID")
            header.set(
                f"CPP_INPUT_DATE_{i}",
                str(visit.getDate().toAstropy().to_value("fits")),
                comment=f"TAI date of input {i}",
            )
            header.set(f"CPP_INPUT_EXPT_{i}", visit.getExposureTime(), comment="Input exposure time")
            if scales is not None:
                header.set(f"CPP_INPUT_SCALE_{i}", scales[i], comment="Scaling applied to input")

        # Sort the inputs into date order.
        visitInfoList = sorted(visitInfoList, key=attrgetter("date"))

        def add_time_offset(dt: lsst.daf.base.DateTime, offset: float) -> astropy.time.Time:
            # Calculate a astropy time with an offset applied.
            at = dt.toAstropy()
            if offset == 0.0:
                return at
            return at + astropy.time.TimeDelta(offset, format="sec")

        earliest = add_time_offset(visitInfoList[0].date, visitInfoList[0].exposureTime / -2.0)
        newest = add_time_offset(visitInfoList[-1].date, visitInfoList[-1].exposureTime / 2.0)

        # Add standard DATE headers covering the range of inputs.
        dateCards = dates_to_fits(earliest, newest)

        for k, v in dateCards.items():
            header.set(k, v, comment=comments.get(k, None))

        # Populate a visitInfo.  Set the exposure time and dark time
        # to 0.0 or 1.0 as appropriate, and copy the instrument name
        # from one of the inputs.
        if calib:
            expTime = 1.0
            if self.config.connections.outputData.lower() == 'bias':
                expTime = 0.0
            inputVisitInfo = visitInfoList[0]
            date_avg = earliest + (newest - earliest) / 2.0
            visitInfo = afwImage.VisitInfo(
                exposureTime=expTime,
                darkTime=expTime,
                date=lsst.daf.base.DateTime(date_avg.isot, lsst.daf.base.DateTime.TAI),
                instrumentLabel=inputVisitInfo.instrumentLabel
            )
            calib.getInfo().setVisitInfo(visitInfo)

        return header

    def interpolateNans(self, exp, maskPlane="BAD"):
        """Interpolate over NANs in the combined image.

        NANs can result from masked areas on the CCD.  We don't want
        them getting into our science images, so we replace them with
        the median of the data.

        Parameters
        ----------
        exp : `lsst.afw.image.Exposure`
            Exp to check for NaNs.
        maskPlane : `str`, optional
            Mask plane to set where we have replaced a NAN.
        """
        array = exp.image.array
        mask = exp.mask.array
        variance = exp.variance.array
        badMaskValue = exp.mask.getPlaneBitMask(maskPlane)

        bad = np.isnan(array) | np.isnan(variance)

        if np.any(bad):
            median = np.median(array[~bad])
            medianVariance = np.median(variance[~bad])
            count = np.sum(bad)

            array[bad] = median
            mask[bad] |= badMaskValue
            variance[bad] = medianVariance
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

    def calibStats(self, exp, calibrationType):
        """Measure bulk statistics for the calibration.

        Parameters
        ----------
        exp : `lsst.afw.image.Exposure`
            Exposure to calculate statistics for.
        calibrationType : `str`
            Type of calibration to record in header.
        """
        metadata = exp.getMetadata()

        noGoodPixelsBit = afwImage.Mask.getPlaneBitMask(self.config.noGoodPixelsMask)

        # percentiles
        for amp in exp.getDetector():
            ampImage = exp[amp.getBBox()]
            percentileValues = np.nanpercentile(ampImage.image.array,
                                                self.config.distributionPercentiles)
            for level, value in zip(self.config.distributionPercentiles, percentileValues):
                key = f"LSST CALIB {calibrationType.upper()} {amp.getName()} DISTRIBUTION {level}-PCT"
                metadata[key] = value

            bad = ((ampImage.mask.array & noGoodPixelsBit) > 0)
            key = f"LSST CALIB {calibrationType.upper()} {amp.getName()} BADPIX-NUM"
            metadata[key] = bad.sum()
            if metadata[key] > 0:
                self.log.warning(f"Found {metadata[key]} pixels with "
                                 f"mask plane {self.config.noGoodPixelsMask} "
                                 f"for amp {amp.getName()}.")


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
