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
#
import numpy as np

import lsst.pipe.base
import lsst.pex.config

from ..utils import FlatGradientFitter


__all__ = [
    "PhotonTransferCurveAdjustGainRatiosConfig",
    "PhotonTransferCurveAdjustGainRatiosTask",
]


class PhotonTransferCurveAdjustGainRatiosConnections(
    lsst.pipe.base.PipelineTaskConnections,
    dimensions=("instrument", "detector"),
):
    exposures = lsst.pipe.base.connectionTypes.Input(
        name="cpPtcIsrExp",
        doc="Input exposures (from PTC ISR) for gain ratio adjustment.",
        storageClass="Exposure",
        dimensions=("instrument", "exposure", "detector"),
        multiple=True,
        deferLoad=True,
    )
    input_ptc = lsst.pipe.base.connectionTypes.Input(
        name="ptcUnadjusted",
        doc="Input PTC to have gain ratios adjusted.",
        storageClass="PhotonTransferCurveDataset",
        dimensions=("instrument", "detector"),
        isCalibration=True,
    )
    output_ptc = lsst.pipe.base.connectionTypes.Output(
        name="ptc",
        doc="Output PTC after gain ratio adjustment.",
        storageClass="PhotonTransferCurveDataset",
        dimensions=("instrument", "detector"),
        isCalibration=True,
    )


class PhotonTransferCurveAdjustGainRatiosConfig(
    lsst.pipe.base.PipelineTaskConfig,
    pipelineConnections=PhotonTransferCurveAdjustGainRatiosConnections,
):
    do_remove_radial_gradient = lsst.pex.config.Field(
        dtype=bool,
        doc="Remove radial gradient before fitting amp gain ratios?",
        default=True,
    )
    radial_gradient_n_spline_nodes = lsst.pex.config.Field(
        dtype=int,
        doc="Number of radial spline nodes for radial gradient.",
        default=20,
    )
    chebyshev_gradient_order = lsst.pex.config.Field(
        dtype=int,
        doc="Order of chebyshev x/y polynomials to remove additional gradients.",
        default=1,
    )
    min_adu = lsst.pex.config.Field(
        dtype=float,
        doc="Minimum number of adu for an exposure to use in gain ratio calculation.",
        default=1000.0,
    )
    max_adu = lsst.pex.config.Field(
        dtype=float,
        doc="Maximum number of adu for an exposure to use in gain ratio calculation.",
        default=20000.0,
    )
    n_flat = lsst.pex.config.Field(
        dtype=int,
        doc="Number of flats (from min_adu to max_adu) to use in gain ratio calculation.",
        default=50,
    )
    random_seed = lsst.pex.config.Field(
        dtype=int,
        doc="Random seed to use for down-sampling input flats.",
        default=12345,
    )


class PhotonTransferCurveAdjustGainRatiosTask(lsst.pipe.base.PipelineTask):
    """Task to remove gradients to fit amp ratio gain adjustments.
    """
    ConfigClass = PhotonTransferCurveAdjustGainRatiosConfig
    _DefaultName = "cpPhotonTransferCurveAdjustGainRatios"

    def run(self, *, exposures, input_ptc):
        """Run the gain adjustment task.

        Parameters
        ----------
        exposures : `list` [`lsst.pipe.base.connections.DeferredDatasetRef`]
            Handles for input exposures.
        input_ptc : `lsst.ip.isr.PhotonTransferCurveDataset`
            Input PTC to adjust.

        Returns
        -------
        results : `lsst.pipe.base.Struct`
            The output struct contains:

            ``output_ptc``
                The output modified PTC.
        """
        pass
