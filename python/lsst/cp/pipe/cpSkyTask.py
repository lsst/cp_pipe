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
# along with this program.  If
import numpy as np

import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase
import lsst.pipe.base.connectionTypes as cT
import lsst.afw.math as afwMath
import lsst.afw.image as afwImage

from lsst.pipe.drivers.background import (FocalPlaneBackground, MaskObjectsTask, SkyMeasurementTask,
                                          FocalPlaneBackgroundConfig, BackgroundConfig)

__all__ = ["CpSkyTask", "CpSkyTaskConfig"]


class CpSkyConnections(pipeBase.PipelineTaskConnections,
                       dimensions=("instrument", "physical_filter", "detector", "visit"),
                       defaultTemplates={}):
    inputExp = cT.Input(
        name="cpSkyISR",
        doc="Input pre-processed exposures to combine.",
        storageClass="ExposureF",
        dimensions=("instrument", "visit", "detector"),
        deferLoad=False,
        multiple=False,
    )
    camera = cT.PrerequisiteInput(
        name="camera",
        doc="Input camera for full focal plane backgrounds.",
        storageClass="Camera",
        dimensions=("instrument", "calibration_label"),
    )

    outputExp = cT.Output(
        name="cpSkyProc",
        doc="Output combined proposed calibration.",
        storageClass="ExposureF",
        dimensions=("instrument", "physical_filter", "detector", "visit"),
    )

    def __init__(self, *, config=None):
        super().__init__(config=config)


class CpSkyTaskConfig(pipeBase.PipelineTaskConfig,
                      pipelineConnections=CpSkyConnections):
    maskObjects = pexConfig.ConfigurableField(
        target=MaskObjectsTask,
        doc="Object masker to use.",
    )
    maskThresh = pexConfig.Field(
        dtype=float,
        default=3.0,
        doc="k-sigma threshold for masking pixels"
    )
    mask = pexConfig.ListField(
        dtype=str,
        default=["DETECTED", "BAD", "NO_DATA", "SAT"],
        doc="Mask planes to reject.",
    )

    largeScaleBackground = pexConfig.ConfigField(
        dtype=FocalPlaneBackgroundConfig,
        doc="Large-scale background configuration"
    )
    sky = pexConfig.ConfigurableField(
        target=SkyMeasurementTask,
        doc="Sky measurement"
    )
    background = pexConfig.ConfigField(
        dtype=BackgroundConfig,
        doc="Background config.",
    )

    def setDefaults(self):
        self.largeScaleBackground.xSize = 256
        self.largeScaleBackground.ySize = 256


class CpSkyTask(pipeBase.PipelineTask,
                pipeBase.CmdLineTask):
    ConfigClass = CpSkyTaskConfig
    _DefaultName = "cpSky"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.makeSubtask("maskObjects")
        self.makeSubtask("sky")

    def run(self, inputExp, camera):
        """Preprocess ISR processed exposures for combination to final SKY calibration.

        Parameters
        ----------
        inputExp : `lsst.afw.image.Exposure`
            Pre-processed sky frame data to combine.
        camera : `lsst.afw.cameraGeom.Camera`
            CZW confirm this is needed before merge.

        Returns
        -------
        outputExp : `lsst.afw.image.Exposure`
            Sky pre-processed data.
        """
        # As constructCalibs SkyTask.processSingleBackground
        self.maskObjects.run(inputExp, self.config.mask)

        # As constructCalibs SkyTask.measureBackground
        self.config.largeScaleBackground.xSize = 256
        self.config.largeScaleBackground.ySize = 256
        bgModel = FocalPlaneBackground.fromCamera(self.config.largeScaleBackground,
                                                  camera)
        bgModel.addCcd(inputExp)

        scale = np.median(bgModel.getStatsImage().getArray())

        # As constructCalibs SkyTask.scatterProcess
        # Remeasure bkg with scaled version of the FP model removed
        mi = inputExp.getMaskedImage()

        bg = bgModel.toCcdBackground(inputExp.getDetector(), mi.getBBox())
        mi -= bg.getImage()
        mi /= scale

        # As constructCalibs SkyTask.processSingle
        # Make the equivalent of the gen2 icExpBackground product.
        stats = afwMath.StatisticsControl()
        stats.setAndMask(mi.getMask().getPlaneBitMask(self.config.background.mask))
        stats.setNanSafe(True)
        ctrl = afwMath.BackgroundControl(
            self.config.background.algorithm,
            max(int(mi.getWidth() / self.config.background.xBinSize + 0.5), 1),
            max(int(mi.getHeight() / self.config.background.yBinSize + 0.5), 1),
            "REDUCE_INTERP_ORDER",
            stats,
            self.config.background.statistic
        )

        bgNew = afwMath.makeBackground(mi, ctrl)
        icExpBkg = afwMath.BackgroundList((
            bgNew,
            afwMath.stringToInterpStyle(self.config.background.algorithm),
            afwMath.stringToUndersampleStyle("REDUCE_INTERP_ORDER"),
            afwMath.ApproximateControl.UNKNOWN,
            0, 0, False  # CZW: It'd be nice to have some idea what these are.
        ))

        bgExp = afwImage.makeExposure(icExpBkg[0][0].getStatsImage())

        # Return
        return pipeBase.Struct(
            outputExp=bgExp,
        )
