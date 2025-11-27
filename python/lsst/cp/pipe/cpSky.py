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

import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase
import lsst.pipe.base.connectionTypes as cT
from lsst.daf.base import PropertyList
from lsst.pipe.tasks.background import (
    FocalPlaneBackground,
    FocalPlaneBackgroundConfig,
    MaskObjectsTask,
    SkyMeasurementTask,
)

from .cpCombine import CalibCombineTask

__all__ = [
    "CpSkyImageTask",
    "CpSkyImageConfig",
    "CpSkyScaleMeasureTask",
    "CpSkyScaleMeasureConfig",
    "CpSkySubtractBackgroundTask",
    "CpSkySubtractBackgroundConfig",
    "CpSkyCombineTask",
    "CpSkyCombineConfig",
]


class CpSkyImageConnections(
    pipeBase.PipelineTaskConnections, dimensions=("instrument", "physical_filter", "exposure", "detector")
):
    inputExp = cT.Input(
        name="cpSkyIsrExp",
        doc="Input pre-processed exposures to combine.",
        storageClass="Exposure",
        dimensions=("instrument", "exposure", "detector"),
    )
    camera = cT.PrerequisiteInput(
        name="camera",
        doc="Input camera to use for geometry.",
        storageClass="Camera",
        dimensions=("instrument",),
        isCalibration=True,
    )

    maskedExp = cT.Output(
        name="cpSkyMaskedIsrExp",
        doc="Output masked post-ISR exposure.",
        storageClass="Exposure",
        dimensions=("instrument", "exposure", "detector"),
    )
    maskedBkg = cT.Output(
        name="cpSkyDetectorBackground",
        doc="Initial background model from one image.",
        storageClass="FocalPlaneBackground",
        dimensions=("instrument", "exposure", "detector"),
    )


class CpSkyImageConfig(pipeBase.PipelineTaskConfig, pipelineConnections=CpSkyImageConnections):
    maskTask = pexConfig.ConfigurableField(
        target=MaskObjectsTask,
        doc="Object masker to use.",
    )

    maskThresh = pexConfig.Field(
        dtype=float,
        default=3.0,
        doc="k-sigma threshold for masking pixels.",
    )
    maskList = pexConfig.ListField(
        dtype=str,
        default=["DETECTED", "BAD", "NO_DATA", "SAT"],
        doc="Mask planes to reject.",
    )

    largeScaleBackground = pexConfig.ConfigField(
        dtype=FocalPlaneBackgroundConfig,
        doc="Large-scale background configuration.",
    )


class CpSkyImageTask(pipeBase.PipelineTask):
    """Mask the detections on the postISRCCD.

    This task maps the MaskObjectsTask across all of the initial ISR
    processed cpSkyIsr images to create cpSkyMaskedIsr products for
    all (exposure, detector) values.
    """

    ConfigClass = CpSkyImageConfig
    _DefaultName = "CpSkyImage"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.makeSubtask("maskTask")

    def run(self, inputExp, camera):
        """Mask the detections on the postISRCCD.

        Parameters
        ----------
        inputExp : `lsst.afw.image.Exposure`
            An ISR processed exposure that will have detections
            masked.
        camera : `lsst.afw.cameraGeom.Camera`
            The camera geometry for this exposure.  This is needed to
            create the background model.

        Returns
        -------
        results : `lsst.pipe.base.Struct`
            The results struct containing:

            ``maskedExp`` : `lsst.afw.image.Exposure`
                The detection-masked version of the ``inputExp``.
            ``maskedBkg`` : `lsst.pipe.tasks.background.FocalPlaneBackground`
                The partial focal plane background containing only
                this exposure/detector's worth of data.
        """
        # As constructCalibs.py SkyTask.processSingleBackground()
        # Except: check if a detector is fully masked to avoid
        # self.maskTask raising.
        currentMask = inputExp.getMask()
        badMask = currentMask.getPlaneBitMask(self.config.maskList)
        if (currentMask.getArray() & badMask).all():
            self.log.warning("All pixels are masked!")
        else:
            self.maskTask.run(inputExp, self.config.maskList)

        # As constructCalibs.py SkyTask.measureBackground()
        bgModel = FocalPlaneBackground.fromCamera(self.config.largeScaleBackground, camera)
        bgModel.addCcd(inputExp)

        return pipeBase.Struct(
            maskedExp=inputExp,
            maskedBkg=bgModel,
        )


class CpSkyScaleMeasureConnections(
    pipeBase.PipelineTaskConnections, dimensions=("instrument", "physical_filter", "exposure")
):
    camera = cT.PrerequisiteInput(
        name="camera",
        doc="Input camera to use for geometry.",
        storageClass="Camera",
        dimensions=("instrument",),
        isCalibration=True,
    )
    inputBkgs = cT.Input(
        name="cpSkyDetectorBackground",
        doc="Initial background model from one exposure/detector",
        storageClass="FocalPlaneBackground",
        dimensions=("instrument", "exposure", "detector"),
        multiple=True,
    )

    outputBkg = cT.Output(
        name="cpSkyExpBackground",
        doc="Background model for a full exposure.",
        storageClass="FocalPlaneBackground",
        dimensions=("instrument", "exposure"),
    )
    outputBkgAlternate = cT.Output(
        name="cpSkyExpBackgroundAlternate",
        doc="Background model for a full exposure from an alternate physical type.",
        storageClass="FocalPlaneBackground",
        dimensions=("instrument", "exposure"),
    )
    outputScale = cT.Output(
        name="cpSkyExpScale",
        doc="Scale for the full exposure.",
        storageClass="PropertyList",
        dimensions=("instrument", "exposure"),
    )

    def __init__(self, *, config=None):
        super().__init__(config=config)
        if not config.includeAlternateBackground:
            del self.camera
            del self.outputBkgAlternate


class CpSkyScaleMeasureConfig(pipeBase.PipelineTaskConfig, pipelineConnections=CpSkyScaleMeasureConnections):
    includeAlternateBackground = pexConfig.Field(
        dtype=bool,
        default=False,
        doc="Include an alternate focal plane background for detectors of a different physical type?",
    )


class CpSkyScaleMeasureTask(pipeBase.PipelineTask):
    """Measure per-exposure scale factors and merge focal plane backgrounds.

    Merge all the per-detector partial backgrounds to a full focal
    plane background for each exposure, and measure the scale factor
    from that full background.
    """

    ConfigClass = CpSkyScaleMeasureConfig
    _DefaultName = "cpSkyScaleMeasure"

    def runQuantum(self, butlerQC, inputRefs, outputRefs):
        """Ensure that the input and output dimensions are passed along.

        Parameters
        ----------
        butlerQC : `lsst.daf.butler.QuantumContext`
            Butler to operate on.
        inputRefs : `lsst.pipe.base.InputQuantizedConnection`
            Input data refs to load.
        outputRefs : `lsst.pipe.base.OutputQuantizedConnection`
            Output data refs to persist.
        """
        inputs = butlerQC.get(inputRefs)

        inputs["inputDims"] = [dict(bkg.dataId.required) for bkg in inputRefs.inputBkgs]

        outputs = self.run(**inputs)
        butlerQC.put(outputs, outputRefs)

    def run(self, inputBkgs, camera=None, inputDims=None):
        """Merge focal plane backgrounds and measure the scale factor.

        Parameters
        ----------
        inputBkgs : `list` [`lsst.pipe.tasks.background.FocalPlaneBackground`]
            A list of all of the partial focal plane backgrounds, one
            from each detector in this exposure.
        camera : `lsst.afw.cameraGeom.Camera`, optional
            The camera geometry for this exposure.  This is needed to
            create the background model.
        inputDims : `list` [`dict`], optional
            The data IDs for each of the input backgrounds.  This is
            used to set provenance information on the output background.

        Returns
        -------
        results : `lsst.pipe.base.Struct`
            The results struct containing:

            ``outputBkg`` : `lsst.pipe.tasks.background.FocalPlaneBackground`
                The full merged background for the entire focal plane.
            ``outputScale`` : `lsst.daf.base.PropertyList`
                A metadata containing the median level of the
                background, stored in the key 'scale'.
        """
        if self.config.includeAlternateBackground and camera is not None and inputDims is not None:
            detectorTypesAll = [camera[ref["detector"]].getPhysicalType() for ref in inputDims]
        else:
            detectorTypesAll = ["homogeneous" for _ in inputBkgs]
        detectorTypes = sorted(set(detectorTypesAll))

        backgrounds = {}
        scales = []
        for detectorType in detectorTypes:
            inputBkgsSingleType = [bg for bg, dt in zip(inputBkgs, detectorTypesAll) if dt == detectorType]

            # Merge per-detector backgrounds into a full focal plane background
            background = inputBkgsSingleType[0]
            for bg in inputBkgsSingleType[1:]:
                background.merge(bg)
            backgrounds[detectorType] = background

            backgroundPixels = background.getStatsImage().getArray()
            self.log.info(
                "Background model%s min/max: %f %f. Scale: %f",
                "" if detectorType == "homogeneous" else f" ({detectorType})",
                np.min(backgroundPixels),
                np.max(backgroundPixels),
                np.median(backgroundPixels),
            )

            # TODO: Ultimately, we should modify FocalPlaneBackground to
            # store metadata directly and set up a storage class which allows
            # us to persist multiple FocalPlaneBackground types per exposure.
            # For now, we store the scale and type data in a PropertyList.
            scales.append(np.median(background.getStatsImage().getArray()))

        scaleMD = PropertyList()
        scaleMD.set("scale", float(scales[0]))
        if len(detectorTypes) > 1:
            scaleMD.set("detectorType", detectorTypes[0])  # TODO: this is not technically needed
            scaleMD.set("scaleAlternate", float(scales[1]))
            scaleMD.set("detectorTypeAlternate", detectorTypes[1])

        return pipeBase.Struct(
            outputBkg=backgrounds[detectorTypes[0]],
            outputBkgAlternate=backgrounds.get(scaleMD.get("detectorTypeAlternate")),
            outputScale=scaleMD,
        )


class CpSkySubtractBackgroundConnections(
    pipeBase.PipelineTaskConnections, dimensions=("instrument", "physical_filter", "exposure", "detector")
):
    inputExp = cT.Input(
        name="cpSkyMaskedIsrExp",
        doc="Masked post-ISR image.",
        storageClass="Exposure",
        dimensions=("instrument", "exposure", "detector"),
    )
    inputBkg = cT.Input(
        name="cpSkyExpBackground",
        doc="Background model for the full exposure.",
        storageClass="FocalPlaneBackground",
        dimensions=("instrument", "exposure"),
    )
    inputBkgAlternate = cT.Input(
        name="cpSkyExpBackgroundAlternate",
        doc="Background model for a full exposure from an alternate physical type.",
        storageClass="FocalPlaneBackground",
        dimensions=("instrument", "exposure"),
    )
    inputScale = cT.Input(
        name="cpSkyExpScale",
        doc="Scale for the full exposure.",
        storageClass="PropertyList",
        dimensions=("instrument", "exposure"),
    )

    outputBkg = cT.Output(
        name="cpSkyExpResidualBackground",
        doc="Normalized, static background.",
        storageClass="Background",
        dimensions=("instrument", "exposure", "detector"),
    )

    def __init__(self, *, config=None):
        super().__init__(config=config)
        if not config.includeAlternateBackground:
            del self.inputBkgAlternate


class CpSkySubtractBackgroundConfig(
    pipeBase.PipelineTaskConfig, pipelineConnections=CpSkySubtractBackgroundConnections
):
    sky = pexConfig.ConfigurableField(
        target=SkyMeasurementTask,
        doc="Sky measurement",
    )
    includeAlternateBackground = pexConfig.Field(
        dtype=bool,
        default=False,
        doc="Use an alternate focal plane background for detectors of a different physical type?",
    )


class CpSkySubtractBackgroundTask(pipeBase.PipelineTask):
    """Subtract per-exposure background from individual detector masked images.

    The cpSkyMaskedIsr images constructed by CpSkyImageTask have the
    scaled background constructed by CpSkyScaleMeasureTask subtracted,
    and new background models are constructed for the remaining
    signal.

    The output was called `icExpBackground` in gen2, but the product
    created here has definition clashes that prevent that from being
    reused.
    """

    ConfigClass = CpSkySubtractBackgroundConfig
    _DefaultName = "cpSkySubtractBkg"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.makeSubtask("sky")

    def runQuantum(self, butlerQC, inputRefs, outputRefs):
        """Ensure that the input and output dimensions are passed along.

        Parameters
        ----------
        butlerQC : `lsst.daf.butler.QuantumContext`
            Butler to operate on.
        inputRefs : `lsst.pipe.base.InputQuantizedConnection`
            Input data refs to load.
        outputRefs : `lsst.pipe.base.OutputQuantizedConnection`
            Output data refs to persist.
        """
        inputs = butlerQC.get(inputRefs)

        if self.config.includeAlternateBackground:
            detectorType = inputs["inputExp"].getDetector().getPhysicalType()
            scaleMD = inputs["inputScale"]
            # Swap in the alternate background and scale if appropriate
            if detectorType == scaleMD.get("detectorTypeAlternate"):
                inputs["inputBkg"] = inputs["inputBkgAlternate"]
                scaleMDAlternate = PropertyList()
                scaleMDAlternate.set("scale", scaleMD.get("scaleAlternate"))
                inputs["inputScale"] = scaleMDAlternate
            _ = inputs.pop("inputBkgAlternate", None)

        outputs = self.run(**inputs)
        butlerQC.put(outputs, outputRefs)

    def run(self, inputExp, inputBkg, inputScale):
        """Subtract per-exposure background from individual detector masked
        images.

        Parameters
        ----------
        inputExp : `lsst.afw.image.Exposure`
            The ISR processed, detection masked image.
        inputBkg : `lsst.pipe.tasks.background.FocalPlaneBackground.
            Full focal plane background for this exposure.
        inputScale : `lsst.daf.base.PropertyList`
            Metadata containing the scale factor.

        Returns
        -------
        results : `lsst.pipe.base.Struct`
            The results struct containing:

            ``outputBkg``
                Remnant sky background with the full-exposure
                component removed. (`lsst.afw.math.BackgroundList`)
        """
        # As constructCalibs.py SkyTask.processSingle()
        image = inputExp.getMaskedImage()
        detector = inputExp.getDetector()
        bbox = image.getBBox()

        scale = inputScale.get("scale")
        background = inputBkg.toCcdBackground(detector, bbox)
        image -= background.getImage()
        image /= scale

        newBackground = self.sky.measureBackground(image)
        return pipeBase.Struct(outputBkg=newBackground)


class CpSkyCombineConnections(
    pipeBase.PipelineTaskConnections, dimensions=("instrument", "physical_filter", "detector")
):
    inputBkgs = cT.Input(
        name="cpSkyExpResidualBackground",
        doc="Normalized, static background.",
        storageClass="Background",
        dimensions=("instrument", "exposure", "detector"),
        multiple=True,
    )
    inputExpHandles = cT.Input(
        name="cpSkyMaskedIsrExp",
        doc="Masked post-ISR image.",
        storageClass="Exposure",
        dimensions=("instrument", "exposure", "detector"),
        multiple=True,
        deferLoad=True,
    )

    outputCalib = cT.Output(
        name="sky",
        doc="Averaged static background.",
        storageClass="ExposureF",
        dimensions=("instrument", "detector", "physical_filter"),
        isCalibration=True,
    )


class CpSkyCombineConfig(pipeBase.PipelineTaskConfig, pipelineConnections=CpSkyCombineConnections):
    sky = pexConfig.ConfigurableField(
        target=SkyMeasurementTask,
        doc="Sky measurement",
    )


class CpSkyCombineTask(pipeBase.PipelineTask):
    """Merge per-exposure measurements into a detector level calibration.

    Each of the per-detector results from all input exposures are
    averaged to produce the final SKY calibration.

    As before, this is written to a skyCalib instead of a SKY to avoid
    definition classes in gen3.
    """

    ConfigClass = CpSkyCombineConfig
    _DefaultName = "cpSkyCombine"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.makeSubtask("sky")

    def run(self, inputBkgs, inputExpHandles):
        """Merge per-exposure measurements into a detector level calibration.

        Parameters
        ----------
        inputBkgs : `list` [`lsst.afw.math.BackgroundList`]
            Remnant backgrounds from each exposure.
        inputHandles : `list` [`lsst.daf.butler.DeferredDatasetHandles`]
            The Butler handles to the ISR processed, detection masked images.

        Returns
        -------
        results : `lsst.pipe.base.Struct`
            The results struct containing:

            `outputCalib` : `lsst.afw.image.Exposure`
                The final sky calibration product.
        """
        skyCalib = self.sky.averageBackgrounds(inputBkgs)
        skyCalib.setDetector(inputExpHandles[0].get(component="detector"))
        skyCalib.setFilter(inputExpHandles[0].get(component="filter"))

        CalibCombineTask().combineHeaders(inputExpHandles, skyCalib, calibType="SKY")

        return pipeBase.Struct(
            outputCalib=skyCalib,
        )
