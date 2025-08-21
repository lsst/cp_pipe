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
from astropy.time import TimeDelta
import esutil
import logging
import numpy as np

import lsst.pipe.base
import lsst.pex.config
from lsst.ts.xml.enums.TunableLaser import LaserDetailedState

from .utilsEfd import CpEfdClient

__all__ = [
    "CpMonochromaticFlatBinTask",
    "CpMonochromaticFlatBinConfig",
]


class CpMonochromaticFlatBinConnections(
    lsst.pipe.base.PipelineTaskConnections,
    dimensions=("instrument", "exposure"),
):
    camera = lsst.pipe.base.connectionTypes.PrerequisiteInput(
        name="camera",
        doc="Camera Geometry definition.",
        storageClass="Camera",
        dimensions=("instrument", ),
        isCalibration=True,
    )
    input_exposure_handles = lsst.pipe.base.connectionTypes.Input(
        name="cpMonochromaticFlatIsrExp",
        doc="Input monochromatic no-filter exposures.",
        storageClass="Exposure",
        dimensions=("instrument", "exposure", "detector"),
        multiple=True,
        deferLoad=True,
    )
    input_photodiode_data = lsst.pipe.base.connectionTypes.Input(
        name="photodiode",
        doc="Photodiode readings data.",
        storageClass="IsrCalib",
        dimensions=("instrument", "exposure"),
        multiple=True,
        deferLoad=True,
        minimum=0,
    )
    output_binned = lsst.pipe.base.connectionTypes.Output(
        name="cpMonochromaticFlatBinned",
        doc="Binned table with full focal-plane data.",
        storageClass="ArrowAstropy",
        dimensions=("instrument", "exposure"),
    )

    def adjust_all_quanta(self, adjuster):
        _LOG = logging.getLogger(__name__)

        def _add_inputs(source_exposure, target_exposure, remove=True):
            """Add inputs to a quantum.

            Parameters
            ----------
            source_exposure : `int`
                Take the quanta from this source exposure.
            target_exposure : `int`
                And add them to the quanta for this target exposure.
            remove : `bool`, optional
                Remove the source_exposure from the quantum dict?
            """
            inputs = adjuster.get_inputs(quantum_id_dict[source_exposure])

            for handle in inputs["input_exposure_handles"]:
                adjuster.add_input(
                    quantum_id_dict[target_exposure],
                    "input_exposure_handles",
                    handle,
                )

            # There is the possibility that off-laser exposures may
            # not have the photodiode.
            if len(inputs["input_photodiode_data"]):
                adjuster.add_input(
                    quantum_id_dict[target_exposure],
                    "input_photodiode_data",
                    inputs["input_photodiode_data"][0],
                )

            if remove:
                adjuster.remove_quantum(quantum_id_dict[source_exposure])

        # Build a dict keyed by exposure.
        # Everything will be sorted by exposure.
        quantum_id_dict = {}
        for quantum_id in sorted(adjuster.iter_data_ids(), key=lambda d: d["exposure"]):
            exposure = quantum_id["exposure"]
            quantum_id_dict[exposure] = quantum_id

        # Get the wavelength for each exposure.
        _LOG.info("Retrieving wavelength information for each exposure.")
        wavelength_dict = {}
        if self.config.use_efd_wavelength:
            client = CpEfdClient()

            # We do one query because it's much faster.
            # This does assume data was taken in one run.
            expanded = adjuster.expand_quantum_data_id(quantum_id_dict[list(quantum_id_dict.keys())[0]])
            date_start = expanded.exposure.timespan.begin
            date_end = expanded.exposure.timespan.end

            for quantum_id in quantum_id_dict.values():
                expanded = adjuster.expand_quantum_data_id(quantum_id)
                if (start := expanded.exposure.timespan.begin) < date_start:
                    date_start = start
                if (end := expanded.exposure.timespan.end) > date_end:
                    date_end = end

            _LOG.info("Querying EFD for wavelength data.")
            wavelength_series = client.selectTimeSeries(
                "lsst.sal.TunableLaser.wavelength",
                fields=["wavelength"],
                startDate=date_start,
                endDate=date_end,
            )

            # Check for laser detailed state events within the
            # last 24 hours. If this returns nothing then these
            # were not taken with the laser and this will crash,
            # but the task won't work (and neither will the query
            # above for the wavelength).
            _LOG.info("Querying EFD for laser state.")
            state_series = client.selectTimeSeries(
                "lsst.sal.TunableLaser.logevent_detailedState",
                fields=["detailedState"],
                startDate=date_start - TimeDelta(1, format="jd"),
                endDate=date_end,
            )

            for exposure, quantum_id in quantum_id_dict.items():
                expanded = adjuster.expand_quantum_data_id(quantum_id)

                use = (
                    (wavelength_series["time"] >= expanded.exposure.timespan.begin)
                    & (wavelength_series["time"] <= expanded.exposure.timespan.end)
                )
                if use.sum() == 0:
                    # Laser must be off.
                    wavelength_dict[exposure] = 0.0
                else:
                    wavelength_dict[exposure] = float(wavelength_series["wavelength"][use][0])

                use2 = (state_series["time"] < expanded.exposure.timespan.begin)
                last_state = state_series["detailedState"][use2][-1]
                if last_state in (
                    LaserDetailedState.NONPROPAGATING_BURST_MODE,
                    LaserDetailedState.NONPROPAGATING_CONTINUOUS_MODE,
                ):
                    wavelength_dict[exposure] = 0.0
                elif last_state not in (
                    LaserDetailedState.PROPAGATING_BURST_MODE,
                    LaserDetailedState.PROPAGATING_CONTINUOUS_MODE,
                ):
                    raise RuntimeError(
                        "Unknown laser state for exposure %d: %d",
                        exposure,
                        last_state,
                    )
        else:
            raise RuntimeError("Only EFD selection for wavelength is supported.")

        exposures = np.asarray(list(wavelength_dict.keys()))
        wavelengths = np.asarray(list(wavelength_dict.values()))

        h, rev = esutil.stat.histogram(wavelengths, rev=True)

        for ind in np.where(h > 0)[0]:
            i1a = rev[rev[ind]: rev[ind + 1]]

            # If there is more than one, consolidate and remove the
            # other quanta.
            if len(i1a) > 1:
                for exposure in exposures[i1a[1:]]:
                    _add_inputs(exposure, exposures[i1a[0]], remove=True)


class CpMonochromaticFlatBinConfig(
    lsst.pipe.base.PipelineTaskConfig,
    pipelineConnections=CpMonochromaticFlatBinConnections,
):
    bin_factor = lsst.pex.config.Field(
        dtype=int,
        doc="Binning factor for flats going into the focal plane.",
        default=128,
    )
    use_efd_wavelength = lsst.pex.config.Field(
        dtype=bool,
        doc="Use EFD to get monochromatic laser wavelengths?",
        default=True,
    )


class CpMonochromaticFlatBinTask(lsst.pipe.base.PipelineTask):
    """Task to stack + bin monochromatic flats."""

    ConfigClass = CpMonochromaticFlatBinConfig
    _DefaultName = "cpMonochromaticFlatBin"

    def run(self, *, camera, input_exposure_handles):
        """Run CpMonochromaticFlatBinTask.

        """
        pass
