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
from collections import defaultdict
import logging

# import lsst.afw.cameraGeom
import lsst.pipe.base as pipeBase
import lsst.pex.config as pexConfig

from .utilsEfd import CpEfdClient

__all__ = [
    "CpMonochromaticFlatBinTask",
    "CpMonochromaticFlatBinConfig",
    "CpMonochromaticQEScanFitTask",
    "CpMonochromaticQEScanFitConfig",
]


class CpMonochromaticFlatBinConnections(
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
        name="cpMonochromaticFlatIsrExp",
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
        name="cpMonochromaticFlatBinned",
        doc="Binned table with full focal-plane data.",
        storageClass="ArrowAstropy",
        dimensions=("instrument", "exposure"),
    )

    def adjust_all_quanta(self, adjuster):
        _LOG = logging.getLogger(__name__)

        # Build a dict keyed by exposure.
        # Each entry is a dict of {detector: quantumId}
        # And everything will be sorted by exposure and detector.
        quantum_id_dict = defaultdict(dict)
        for quantum_id in sorted(adjuster.iter_data_ids(), key=lambda d: (d["exposure"], d["detector"])):
            exposure = quantum_id["exposure"]
            quantum_id_dict[exposure][quantum_id["detector"]] = quantum_id

        # Retrieve the wavelength for each exposure.
        import IPython
        IPython.embed()


class CpMonochromaticFlatBinConfig(
    pipeBase.PipelineTaskConfig,
    pipelineConnections=CpMonochromaticFlatBinConnections,
):
    bin_factor = pexConfig.Field(
        dtype=int,
        doc="Binning factor for flats going into the focal plane.",
        default=128,
    )
    use_efd_wavelength = pexConfig.Field(
        dtype=bool,
        doc="Use EFD to get monochromatic laser wavelengths?",
        default=True,
    )


class CpMonochromaticFlatBinTask(pipeBase.PipelineTask):
    """Task to stack + bin monochromatic flats."""

    ConfigClass = CpMonochromaticFlatBinConfig
    _DefaultName = "cpMonochromaticFlatBin"

    def run(self, *, camera, input_exposure_handles):
        """Run CpMonochromaticFlatBinTask.

        """
        pass


class CpMonochromaticQEScanFitConnections(
    pipeBase.PipelineTaskConnections,
    dimensions=("instrument",),
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
