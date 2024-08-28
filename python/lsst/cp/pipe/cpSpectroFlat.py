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

__all__ = ["CpSpectroFlatTask", "CpSpectroFlatTaskConfig"]


import numpy as np
from scipy.ndimage import gaussian_filter, uniform_filter, median_filter

import lsst.afw.math as afwMath
import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase
import lsst.pipe.base.connectionTypes as cT

from lsst.meas.algorithms import SubtractBackgroundTask


class CpSpectroFlatTaskConnections(pipeBase.PipelineTaskConnections,
                                   dimensions=("instrument", "detector", "physical_filter")):
    dummyExpRef = cT.Input(
        name="raw",
        storageClass="Exposure",
        dimensions=("instrument", "detector", "exposure"),
        doc="Dummy exposure reference to constrain butler calibration lookups.",
        deferLoad=True
    )
    inputFlat = cT.PrerequisiteInput(
        name="flat",
        doc="Input white-light flat to use as basis of spectroFlat.",
        storageClass="Exposure",
        dimensions=("instrument", "detector", "physical_filter"),
        multiple=False,
        isCalibration=True,
    )
    inputPtc = cT.PrerequisiteInput(
        name="ptc",
        doc="Input PTC containing gains to scale spectroFlat by.",
        storageClass="PhotonTransferCurveDataset",
        dimensions=("instrument", "detector"),
        multiple=False,
        isCalibration=True,
    )

    outputFlat = cT.Output(
        name="spectroFlat",
        doc="Scaled and filtered spectroscopic flat.",
        storageClass="Exposure",
        dimensions=("instrument", "detector"),
        isCalibration=True,
    )


class CpSpectroFlatTaskConfig(pipeBase.PipelineTaskConfig,
                              pipelineConnections=CpSpectroFlatTaskConnections):
    """Docs"""

    inputFlatPhysicalFilter = pexConfig.Field(
        dtype=str,
        doc="Physical filter to expect in inputFlat.",
        default="empty~empty",
    )

    # makeGainFlat
    applyGainDirection = pexConfig.Field(
        dtype=bool,
        doc="Should flat be multiplied by the gains?",
        default=True,
    )
    scaleGainsByMean = pexConfig.Field(
        dtype=bool,
        doc="Scale gains by the mean?",
        default=True,
    )

    # makeOpticFlat
    smoothingAlgorithm = pexConfig.ChoiceField(
        dtype=str,
        doc="Method to use to smooth amplifiers.",
        default="mean",
        allowed={"mean": "Use scipy uniform_filter",
                 "gauss": "Use scipy gaussian_filter",
                 "median": "Use scipy median_filter"},
    )
    smoothingWindowSize = pexConfig.Field(
        dtype=int,
        default=40,
        doc="Size of smoothing window.",
    )
    smoothingMode = pexConfig.Field(
        dtype=str,
        default="mirror",
        doc="Filter mode/downstream.",
    )
    smoothingOutlierPercentile = pexConfig.Field(
        dtype=float,
        default=1.0,
        doc="Percentile of extreme values to clip from top and bottom of distribution.",
    )

    # makePixelFlat
    # photutils.background Background2D, MedianBackground
    # astropy.stats.SigmaClip
    doBackgroundRemoval = pexConfig.Field(
        dtype=bool,
        default=True,
        doc="Should amplifier backgrounds be measured and divided out?",
    )
    background = pexConfig.ConfigurableField(
        target=SubtractBackgroundTask,
        doc="Configuration for background estimate.",
    )
    backgroundSigmaClipThreshold = pexConfig.Field(
        dtype=float,
        default=3.0,
        doc="Backgroud sigma clip threshold.",
    )
    backgroundBoxSize = pexConfig.Field(
        dtype=int,
        default=20,
        doc="Size of background super-pixel.",
    )
    backgroundFilterSize = pexConfig.Field(
        dtype=int,
        default=3,
        doc="Size of smoothing filter for backgrounds.",
    )


class CpSpectroFlatTask(pipeBase.PipelineTask):
    """Docs"""

    ConfigClass = CpSpectroFlatTaskConfig
    _DefaultName = "cpSpectroFlat"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.makeSubtask("background")

    def run(self, dummyExpRef, inputFlat, inputPtc):
        # Verify input
        exposureFilter = dummyExpRef.get(component="filter")

        if exposureFilter.physicalLabel != self.config.inputFlatPhysicalFilter:
            raise RuntimeError("Did not receive expected physical_filter: "
                               f"{exposureFilter.physicalLabel} != "
                               f"{self.config.inputFlatPhysicalFilter}")

        detector = inputFlat.getDetector()
        statsControl = afwMath.StatisticsControl(self.config.backgroundSigmaClipThreshold, 5, 0x0)

        # Make gain flat
        # https://github.com/lsst/atmospec/blob/u/jneveu/doFlat/python/lsst/atmospec/spectroFlat.py#L85
        gains = inputPtc.gain
        scaledGains = {}
        gainFlat = inputFlat.clone()

        # Todo: Get temperature dependence terms.
        if self.config.scaleGainsByMean:
            gainMean = np.mean(list(gains.values()))
        else:
            gainMean = 1.0

        for amp in detector:
            bbox = amp.getBBox()
            gainFlat[bbox].maskedImage.set(gains[amp.getName()] / gainMean,
                                           0x0,
                                           0.0)
            scaledGains[amp.getName()] = gains[amp.getName()] / gainMean
        # gainFlat is now a realization of the gains into a full image.
        # scaledGains are the gains optionally scaled by mean value.

        # Make optic flat
        # https://github.com/lsst/atmospec/blob/u/jneveu/doFlat/python/lsst/atmospec/spectroFlat.py#L130
        opticFlat = inputFlat.clone()
        for amp in detector:
            # Divide amp by median
            ampExp = opticFlat.Factory(opticFlat, amp.getBBox())
            stats = afwMath.makeStatistics(ampExp.getMaskedImage(),
                                           afwMath.MEDIAN, statsControl)
            ampExp.image.array[:, :] /= stats.getValue(afwMath.MEDIAN)

            inputData = ampExp.image.array.copy()
            match self.config.smoothingAlgorithm:
                case "gauss":
                    ampExp.image.array[:, :] = gaussian_filter(inputData,
                                                               sigma=self.config.smoothingWindowSize,
                                                               mode=self.config.smoothingMode)
                case "mean":
                    # This method is easily skewed by outliers.
                    # Filter those.
                    mask1 = (inputData >= np.percentile(inputData.ravel(),
                                                        self.config.smoothingOutlierPercentile))
                    mask2 = (inputData <= np.percentile(inputData.ravel(),
                                                        100 - self.config.smoothingOutlierPercentile))
                    mask = mask1 * mask2
                    # We've already divided the image by the median,
                    # so the median image value is 1.0
                    inputData[~mask] = 1.0
                    ampExp.image.array[:, :] = uniform_filter(inputData,
                                                              size=self.config.smoothingWindowSize,
                                                              mode=self.config.smoothingMode)
                case "median":
                    ampExp.image.array[:, :] = median_filter(inputData,
                                                             size=(self.config.smoothingWindowSize,
                                                                   self.config.smoothingWindowSize),
                                                             mode=self.config.smoothingMode)
                case _:
                    raise RuntimeError(f"Unknown smoothing method: {self.config.smoothingAlgorithm}")
            ampExp.variance.array[:, :] = 1.0
            ampExp.mask.array[:, :] = 0x0

        # Make pixel flat
        # https://github.com/lsst/atmospec/blob/u/jneveu/doFlat/python/lsst/atmospec/spectroFlat.py#L18
        pixelFlat = inputFlat.clone()
        for amp in detector:
            # Divide amp by median.  Can we reuse the image above?
            ampExp = pixelFlat.Factory(pixelFlat, amp.getBBox())
            stats = afwMath.makeStatistics(ampExp.getMaskedImage(),
                                           afwMath.MEDIAN, statsControl)
            ampExp.image.array[:, :] /= stats.getValue(afwMath.MEDIAN)

            if self.config.doBackgroundRemoval:
                ampBg = self.background.run(ampExp).background
                ampBgImg = ampBg.getImage()
                ampExp.image.array[:, :] /= ampBgImg.array[:, :]

        # Make sensor flat
        # https://github.com/lsst/atmospec/blob/u/jneveu/doFlat/python/lsst/atmospec/spectroFlat.py#L230
        sensorFlat = pixelFlat.clone()
        for amp in detector:
            sensorFlatAmp = sensorFlat.Factory(sensorFlat, amp.getBBox())
            gainFlatAmp = gainFlat.Factory(gainFlat, amp.getBBox())
            sensorFlatAmp.maskedImage.scaledMultiplies(1.0, gainFlatAmp.maskedImage)

        # Add PTC/gain information to outputFlat header. This allows
        # SpectractorShim.spectractorImageFromLsstExposure to populate
        # image.gain and image.flat from our outputFlat.
        md = sensorFlat.metadata
        md["LSST CP SPECTROFLAT GAIN MEAN"] = gainMean
        for amp in detector:
            md[f"LSST CP SPECTROFLAT GAIN {amp.getName()}"] = scaledGains[amp.getName()]
        return pipeBase.Struct(
            outputFlat=sensorFlat,
        )
