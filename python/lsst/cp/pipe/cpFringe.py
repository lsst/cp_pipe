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
import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase
import lsst.pipe.base.connectionTypes as cT
import lsst.cp.pipe.cpCombine as cpCombine
import lsst.meas.algorithms as measAlg
import lsst.afw.detection as afwDet

import numpy as np
from scipy.ndimage import gaussian_filter


__all__ = ["CpFringeTask", "CpFringeTaskConfig"]


class CpFringeConnections(pipeBase.PipelineTaskConnections,
                          dimensions=("instrument", "exposure", "detector")):
    inputExp = cT.Input(
        name="cpFringeISR",
        doc="Input pre-processed exposures to combine.",
        storageClass="Exposure",
        dimensions=("instrument", "exposure", "detector"),
    )

    outputExp = cT.Output(
        name="cpFringeProc",
        doc="Output combined proposed calibration.",
        storageClass="Exposure",
        dimensions=("instrument", "exposure", "detector"),
    )


class CpFringeTaskConfig(pipeBase.PipelineTaskConfig,
                         pipelineConnections=CpFringeConnections):
    convolutionSigma = pexConfig.Field(
        dtype=float,
        default=2.0,
        doc="Convolution psf Gaussian sigma.",
    )
    subtractBackground = pexConfig.ConfigurableField(
        target=measAlg.SubtractBackgroundTask,
        doc="Background configuration",
    )
    detection = pexConfig.ConfigurableField(
        target=measAlg.SourceDetectionTask,
        doc="Detection configuration",
    )
    detectSigma = pexConfig.Field(
        dtype=float,
        default=1.0,
        doc="Detection psf Gaussian sigma.",
    )
    stats = pexConfig.ConfigurableField(
        target=cpCombine.CalibStatsTask,
        doc="Statistics task to use.",
    )

    def setDefaults(self):
        self.subtractBackground.useApprox = False
        self.subtractBackground.binSize = 200  # Based on ps1

        self.detection.reEstimateBackground = False


class CpFringeTask(pipeBase.PipelineTask):
    """Combine pre-processed fringe frames into a proposed master calibration.
    """

    ConfigClass = CpFringeTaskConfig
    _DefaultName = "cpFringe"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.makeSubtask("subtractBackground")
        self.makeSubtask("detection")
        self.makeSubtask("stats")

    def run(self, inputExp):
        """Preprocess input exposures prior to FRINGE combination.

        This task subtracts a background level, masks potential
        sources on the image, and convolves with a Gaussian kernel to
        reduce the noise in the remaining fringe signal.

        Parameters
        ----------
        inputExp : `lsst.afw.image.Exposure`
            Pre-processed fringe frame data to combine.

        Returns
        -------
        results : `lsst.pipe.base.Struct`
            The results struct containing:

            ``outputExp``
                Fringe pre-processed frame (`lsst.afw.image.Exposure`).
        """
        # Background subtract
        self.subtractBackground.run(inputExp)
        mi = inputExp.getMaskedImage()

        # Identify sources.
        fpSets = self.detection.detectFootprints(inputExp, sigma=self.config.detectSigma)
        mask = mi.getMask()
        detected = 1 << mask.addMaskPlane("DETECTED")
        for fpSet in (fpSets.positive, fpSets.negative):
            if fpSet is not None:
                afwDet.setMaskFromFootprintList(mask, fpSet.getFootprints(), detected)

        # Convolve with smoothing kernel
        # Switch to an afw version?
        mi.image.array = gaussian_filter(mi.image.array, self.config.convolutionSigma)

        # Estimate a DC offset level to potentially use during
        # combination.  Using PS name to avoid confusion with various
        # "backgrounds".  Yes, this is likely equally confusing.
        zero = self.stats.run(inputExp)
        inputExp.metadata["LSST CP FRINGE ZERO"] = zero

        # Debugging log info for me.
        imin, i25, imedian, i75, imax = np.percentile(inputExp.image.array, [0, 25, 50, 75, 100])
        self.log.info(f"Zero level: {zero} P: {imin} {i25} {imedian} {i75} {imax}")

        return pipeBase.Struct(
            outputExp=inputExp,
        )


class CpFringeCombineConnections(cpCombine.CalibCombineByFilterConnections,
                                 dimensions=("instrument", "detector", "physical_filter")):
    outputFringeTable = cT.Output(
        name="cpFringeTable",
        doc="Output fringe samples table.",
        storageClass="ArrowAstropy",  # Should this be an IsrCalib?
        dimensions=["instrument", "detector", "physical_filter"],
        isCalibration=True,
    )

    def __init__(self, *, config=None):
        super().__init__(config=config)


class CpFringeCombineConfig(cpCombine.CalibCombineByFilterConfig,
                            pipelineConnections=CpFringeCombineConnections):
    # This should subtask the fringe task from ip_isr.
    def setDefaults(self):
        self.exposureScaling = "Unity"


class CpFringeCombineTask(cpCombine.CalibCombineTask):
    """Subclass the default cpCombine to add fringe specific options."""

    def run(self, inputExpHandles, inputScales=None, inputDims=None):
        """Add-on to the default run task to measure fringe samples."""
        pass
        # results = super().run(inputExpHandles, inputScales, inputDims)
