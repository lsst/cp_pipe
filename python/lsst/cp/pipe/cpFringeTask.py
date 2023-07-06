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
import lsst.afw.image as afwImage

from .cpCombine import CalibCombineByFilterTask, CalibCombineByFilterTaskConfig

__all__ = ["CpFringeTask", "CpFringeTaskConfig",
           "CpFringeCombineTask", "CpFringeCombineTaskConfig"]


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
    stats = pexConfig.ConfigurableField(
        target=cpCombine.CalibStatsTask,
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


class CpFringeTask(pipeBase.PipelineTask):
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
        results : `lsst.pipe.base.Struct`
            The results struct containing:

            ``outputExp``
                Fringe pre-processed frame (`lsst.afw.image.Exposure`).
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

        return pipeBase.Struct(
            outputExp=inputExp,
        )


class CpFringeCombineConnections(CalibCombineByFilterConnections):
    # Use the parent version.  Do we need a special storageClass for
    # the output for multiple images?
    pass


class CpFringeCombineTaskConfig(CalibCombineByFilterTaskConfig):
    nComponent = pexConfig.Field(
        dtype=int,
        default=3,
        doc="Number of PCA components to retain in the output fringe.",
        check=lambda x: x >= 1
    )


class CpFringeCombineTask(CalibCombineByFilterTask):
    """Task to combine input fringe frames into a final set of combined
    fringes.
    """

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
        # combineType = afwMath.stringToStatisticsProperty(self.config.combine)

        subregionSizeArr = self.config.subregionSize
        subregionSize = geom.Extent2I(subregionSizeArr[0], subregionSizeArr[1])
        for subBbox in self._subBBoxIter(target.getBBox(), subregionSize):
            imageSet = afwImage.ImagePcaF()
            for expHandle, expScale in zip(expHandleList, expScaleList):
                inputExp = expHandle.get(parameters={"bbox": subBbox})
                self.applyScale(inputExp, subBbox, expScale)
                imageSet.addImage(inputExp.image, 1.0)  # Should this use the scale instead?
            imageSet.analyse()

            # This doesn't guarantee that eigenImage_0 is always the same, does it?
            for eigenValue, eigenImage in zip(imageSet.getEigenImages(), imageSet.getEigenValues()):
                target.image.assign(eigenImage, subBox)  # this is wrong as well.
