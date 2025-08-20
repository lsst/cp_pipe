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
# import numpy as np

# import lsst.afw.cameraGeom
import lsst.pipe.base as pipeBase
import lsst.pex.config as pexConfig

__all__ = [
    "CpMonochromaticQEScanBinTask",
    "CpMonochromaticQEScanBinConfig",
    "CpMonochromaticQEScanFitTask",
    "CpMonochromaticQEScanFitConfig",
]


class CpMonochromaticQEScanBinConnections(
    pipeBase.PipelineTaskConnections,
    dimensions=("instrument", "exposure"),
):
    camera = pipeBase.connectionTypes.PrerequisiteInput(
        name="camera",
        doc="Camera Geometry definition.",
        storageClass="Camera",
        dimensions=("instrument", ),
        isCalibration=True,
    )
    input_exposure_handles = pipeBase.connectionTypes.Input(
        name="cpMonochromaticQEScanIsrExp",
        doc="Input monochromatic no-filter exposures.",
        storageClass="Exposure",
        dimensions=("instrument", "exposure", "detector"),
        multiple=True,
        deferLoad=True,
    )
    input_photodiode_data = pipeBase.connectionTypes.Input(
        name="photodiode",
        doc="Photodiode readings data.",
        storageClass="IsrCalib",
        dimensions=("instrument", "exposure"),
        multiple=True,
        deferLoad=True,
    )
    output_binned = pipeBase.connectionTypes.Output(
        name="cpMonochromaticQEScanBinned",
        doc="Binned table with full focal-plane data.",
        storageClass="ArrowAstropy",
        dimensions=("instrument", "exposure"),
    )


class CpMonochromaticQEScanBinConfig(
    pipeBase.PipelineTaskConfig,
    pipelineConnections=CpMonochromaticQEScanBinConnections,
):
    bin_factor = pexConfig.Field(
        dtype=int,
        doc="Binning factor for flats going into the focal plane.",
        default=128,
    )


class CpMonochromaticQEScanBinTask(pipeBase.PipelineTask):
    """Task to stack + bin monochromatic flats."""

    ConfigClass = CpMonochromaticQEScanBinConfig
    _DefaultName = "cpMonochromaticQEScanBin"

    def run(self, *, camera, input_exposure_handles):
        """Run CpMonochromaticQEScanBinTask.

        """
        pass


class CpMonochromaticQEScanFitConnections(
    pipeBase.PipelineTaskConnections,
    dimensions=("instrument"),
):
    camera = pipeBase.connectionTypes.PrerequisiteInput(
        name="camera",
        doc="Camera Geometry definition.",
        storageClass="Camera",
        dimensions=("instrument", ),
        isCalibration=True,
    )
    input_binned_handles = pipeBase.connectionTypes.Input(
        name="cpMonochromaticQEScanBinned",
        doc="Binned tables with full focal-plane data.",
        storageClass="ArrowAstropy",
        dimensions=("instrument", "exposure"),
        multiple=True,
        deferLoad=True,
    )
    # Make an output connection with a name.


class CpMonochromaticQEScanFitConfig(
    pipeBase.PipelineTaskConfig,
    pipelineConnections=CpMonochromaticQEScanFitConnections,
):
    # This will require the flat gradient fitter code.
    pass


class CpMonochromaticQEScanFitTask(pipeBase.PipelineTask):
    """Task to fit QE from monochromatic flats."""

    ConfigClass = CpMonochromaticQEScanFitConfig
    _DefaultName = "cpMonochromaticQEScanFit"

    def run(self, *, camera, input_binned_handles):
        pass
