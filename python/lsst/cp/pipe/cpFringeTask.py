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
import lsst.cp.pipe.cpFunctions as cpFunctions
import lsst.meas.algorithms as measAlg
import lsst.afw.detection as afwDet


__all__ = ["CpFringeTask", "CpFringeTaskConfig"]


class CpFringeConnections(pipeBase.PipelineTaskConnections,
                          dimensions=("instrument", "detector", "physical_filter", "visit"),
                          defaultTemplates={}):
    inputExp = cT.Input(
        name="cpFringeISR",
        doc="Input pre-processed exposures to combine.",
        storageClass="ExposureF",
        dimensions=("instrument", "visit", "detector"),
        deferLoad=False,
        multiple=False,
    )

    outputExp = cT.Output(
        name="cpFringeProc",
        doc="Output combined proposed calibration.",
        storageClass="ExposureF",
        dimensions=("instrument", "detector", "physical_filter", "visit"),
    )


class CpFringeTaskConfig(pipeBase.PipelineTaskConfig,
                         pipelineConnections=CpFringeConnections):
    stats = pexConfig.ConfigurableField(
        target=cpFunctions.CalibStatsTask,
        doc="Statistics task to use.",
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
        doc="Detection psf gaussian sigma.",
    )

    def setDefaults(self):
        self.detection.reEstimateBackground = False


class CpFringeTask(pipeBase.PipelineTask,
                   pipeBase.CmdLineTask):
    """Combine pre-processed fringe frames into a proposed master calibration.
    """
    ConfigClass = CpFringeTaskConfig
    _DefaultName = "cpFringe"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.makeSubtask("stats")
        self.makeSubtask("subtractBackground")
        self.makeSubtask("detection")

    def run(self, inputExp):
        """Preprocess input exposures prior to FRINGE combination.

        This task scales and renormalizes the input frame based on the
        image background, and then masks all pixels above the
        detection threshold.

        Parameters
        ----------
        inputExp : `lsst.afw.image.Exposure`
            Pre-processed fringe frame data to combine.

        Returns
        -------
        outputExp : `lsst.afw.image.Exposure`
            Fringe pre-processed frame.

        """
        bg = self.stats.run(inputExp)
        self.subtractBackground.run(inputExp)
        mi = inputExp.getMaskedImage()
        mi /= bg

        fpSets = self.detection.detectFootprints(inputExp, sigma=self.config.detectSigma)
        mask = mi.getMask()
        detected = 1 << mask.addMaskPlane("DETECTED")
        for fpSet in (fpSets.positive, fpSets.negative):
            if fpSet is not None:
                afwDet.setMaskFromFootprintList(mask, fpSet.getFootprints(), detected)

        # Return
        return pipeBase.Struct(
            outputExp=inputExp,
        )
