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

import lsst.pipe.base
# import lsst.pex.config


__all__ = [
    "CpMonochromaticQEScanFitTask",
    "CpMonochromaticQEScanFitConfig",
]


class CpMonochromaticQEScanFitConnections(
    lsst.pipe.base.PipelineTaskConnections,
    dimensions=("instrument",),
):
    camera = lsst.pipe.base.connectionTypes.PrerequisiteInput(
        name="camera",
        doc="Camera Geometry definition.",
        storageClass="Camera",
        dimensions=("instrument", ),
        isCalibration=True,
    )
    input_binned_handles = lsst.pipe.base.connectionTypes.Input(
        name="cpMonochromaticQEScanBinned",
        doc="Binned tables with full focal-plane data.",
        storageClass="ArrowAstropy",
        dimensions=("instrument", "exposure"),
        multiple=True,
        deferLoad=True,
    )
    # Make an output connection with a name.


class CpMonochromaticQEScanFitConfig(
    lsst.pipe.base.PipelineTaskConfig,
    pipelineConnections=CpMonochromaticQEScanFitConnections,
):
    # This will require the flat gradient fitter code.
    pass


class CpMonochromaticQEScanFitTask(lsst.pipe.base.PipelineTask):
    """Task to fit QE from monochromatic flats."""

    ConfigClass = CpMonochromaticQEScanFitConfig
    _DefaultName = "cpMonochromaticQEScanFit"

    def run(self, *, camera, input_binned_handles):
        pass
