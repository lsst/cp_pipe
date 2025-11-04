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
import numpy as np

import lsst.pipe.base
from lsst.ip.isr import GainCorrection
import lsst.pex.config
from lsst.utils.plotting import make_figure

from .ptc.cpPtcAdjustGainRatios import _choose_reference_amplifier, _compute_gain_ratios
from .utils import bin_flat

__all__ = ["CpMeasureGainCorrectionTask", "CpMeasureGainCorrectionConfig"]


class CpMeasureGainCorrectionConnections(
    lsst.pipe.base.PipelineTaskConnections,
    dimensions=("instrument", "detector"),
):
    input_reference_flat = lsst.pipe.base.connectionTypes.PrerequisiteInput(
        name="flat",
        doc="Input reference flat (certified).",
        storageClass="Exposure",
        dimensions=("instrument", "detector", "physical_filter"),
        isCalibration=True,
    )
    input_reference_ptc = lsst.pipe.base.connectionTypes.PrerequisiteInput(
        name="ptc",
        doc="Input reference ptc (certified).",
        storageClass="PhotonTransferCurveDataset",
        dimensions=("instrument", "detector"),
        isCalibration=True,
    )
    input_flat = lsst.pipe.base.connectionTypes.Input(
        name="flat_gain_correction",
        doc="Input flat to derive gain correction.",
        storageClass="Exposure",
        dimensions=("instrument", "detector", "physical_filter"),
        isCalibration=True,
    )
    output_gain_correction = lsst.pipe.base.connectionTypes.Output(
        name="gain_correction",
        doc="Output gain correction calibration.",
        storageClass="IsrCalib",
        dimensions=("instrument", "detector"),
        isCalibration=True,
    )
    output_flat_ratio_plot = lsst.pipe.base.connectionTypes.Output(
        name="gain_correction_flat_ratio_plot",
        doc="Gain correction flat ratio plot.",
        storageClass="Plot",
        dimensions=("instrument", "detector"),
    )

    def adjustQuantum(self, inputs, outputs, label, dataId):
        # We need to remove reference flats that do not match
        # the physical filter of the input flat.
        input_flat_ref = inputs["input_flat"][1][0]
        physical_filter = input_flat_ref.dataId["physical_filter"]

        input_reference_flat_refs = inputs["input_reference_flat"][1]
        input_reference_flat_ref = None
        for ref in input_reference_flat_refs:
            if ref.dataId["physical_filter"] == physical_filter:
                input_reference_flat_ref = ref
                break

        if input_reference_flat_ref is None:
            raise lsst.pipe.base.NoWorkFound(
                f"No matched input flat with physical filter {physical_filter}.",
            )

        inputs["input_reference_flat"] = (inputs["input_reference_flat"][0], (input_reference_flat_ref,))

        return inputs, outputs


class CpMeasureGainCorrectionConfig(
    lsst.pipe.base.PipelineTaskConfig,
    pipelineConnections=CpMeasureGainCorrectionConnections,
):
    max_noise_reference = lsst.pex.config.Field(
        dtype=float,
        doc="Maximum read noise (e-) in the PTC for an amp to be considered as a reference",
        default=12.0,
    )
    turnoff_percentile_reference = lsst.pex.config.Field(
        dtype=float,
        doc="Percentile threshold for sorting PTC turnoff for an amp to be considered as a reference",
        default=25.0,
    )
    do_remove_radial_gradient = lsst.pex.config.Field(
        dtype=bool,
        doc="Remove radial gradient before measureting amp gain ratios?",
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
    chebyshev_gradient_nsig_clip = lsst.pex.config.Field(
        dtype=float,
        doc="Number of sigma to clip gain ratios when fitting the Chebyshev gradient.",
        default=5.0,
    )
    bin_factor = lsst.pex.config.Field(
        dtype=int,
        doc="Binning factor to compute gradients/gain ratios (pixels).",
        default=8,
    )
    amp_boundary = lsst.pex.config.Field(
        dtype=int,
        doc="Amplifier boundary to ignore when computing gradients/gain ratios (pixels).",
        default=20,
    )
    max_fractional_gain_ratio = lsst.pex.config.Field(
        dtype=float,
        doc="Maximum fractional gain ratio to consider. Any amps with larger "
            "offset will be excluded from the measurement and will have no corrections "
            "computed.",
        default=0.05,
    )


class CpMeasureGainCorrectionTask(lsst.pipe.base.PipelineTask):
    """Task to measure gain corrections."""

    ConfigClass = CpMeasureGainCorrectionConfig
    _DefaultName = "cpMeasureGainCorrection"

    def run(self, *, input_reference_flat, input_reference_ptc, input_flat):
        """

        Parameters
        ----------
        input_reference_flat : `lsst.afw.image.Exposure`
            Input reference flat (typically a certified calibration).
        input_reference_ptc : `lsst.ip.isr.PhotonTransferCurveDataset`
            Input reference PTC (typically a certified calibration).
        input_flat : `lsst.afw.image.Exposure`
            Input flat to derive gain correction (relative to reference flat).

        Returns
        -------
        result : `lsst.pipe.base.Struct`
            Result struct containing:
                ``output_gain_correction`` : `lsst.ip.isr.GainCorrection`
                ``output_flat_ratio_plot`` : `matplotlib.Figure`
        """
        gain_correction = GainCorrection(
            ampNames=input_reference_ptc.ampNames,
            gainAdjustments=np.ones(len(input_reference_ptc.ampNames)),
        )

        fixed_amp_index = _choose_reference_amplifier(
            input_reference_ptc,
            self.config.max_noise_reference,
            self.config.turnoff_percentile_reference,
        )

        if fixed_amp_index < 0:
            return lsst.pipe.base.Struct(output_gain_correction=gain_correction)

        ratio = input_flat.clone()
        ratio.image /= input_reference_flat.image

        binned_ref = bin_flat(input_reference_ptc, input_reference_flat, apply_gains=False)
        binned_flat = bin_flat(input_reference_ptc, input_flat, apply_gains=False)
        binned_ratio = bin_flat(input_reference_ptc, ratio, apply_gains=False)

        # Clip out non-finite and extreme values from both the input flat
        # and the reference flat.
        lo_ref, hi_ref = np.nanpercentile(binned_ref["value"], [5.0, 95.0])
        lo_ref *= 0.8
        hi_ref *= 1.2
        lo_flat, hi_flat = np.nanpercentile(binned_flat["value"], [5.0, 95.0])
        lo_flat *= 0.8
        hi_flat *= 1.2
        use = (
            np.isfinite(binned_ratio["value"])
            & (binned_ref["value"] >= lo_ref)
            & (binned_ref["value"] <= hi_ref)
            & (binned_flat["value"] >= lo_flat)
            & (binned_flat["value"] <= hi_flat)
        )
        binned = binned_ratio[use]

        gain_ratios = _compute_gain_ratios(
            input_flat.getDetector(),
            binned,
            fixed_amp_index,
            do_remove_radial_gradient=self.config.do_remove_radial_gradient,
            radial_gradient_n_spline_nodes=self.config.radial_gradient_n_spline_nodes,
            chebyshev_gradient_order=self.config.chebyshev_gradient_order,
            max_fractional_gain_ratio=self.config.max_fractional_gain_ratio,
            nsig_clip=self.config.chebyshev_gradient_nsig_clip,
            log=self.log,
        )

        gain_correction.gainAdjustments[:] = gain_ratios

        value2 = binned["value"].copy()
        for ind in range(len(input_reference_ptc.ampNames)):
            value2[binned["amp_index"] == ind] /= gain_ratios[ind]

        vmin, vmax = np.nanpercentile(binned["value"], [10, 90])

        fig = make_figure(figsize=(16, 6))

        ax1 = fig.add_subplot(121)

        im1 = ax1.hexbin(binned["xd"], binned["yd"], C=binned["value"], vmin=vmin, vmax=vmax)

        ax1.set_xlabel("Detector x (pix)")
        ax1.set_ylabel("Detector y (pix)")
        ax1.set_aspect("equal")
        ax1.set_title("Flat Ratio (uncorrected)")

        fig.colorbar(im1, ax=ax1)

        ax2 = fig.add_subplot(122)

        im2 = ax2.hexbin(binned["xd"], binned["yd"], C=value2, vmin=vmin, vmax=vmax)

        ax2.set_xlabel("Detector x (pix)")
        ax2.set_ylabel("Detector y (pix)")
        ax2.set_aspect("equal")
        ax2.set_title("Flat Ratio (corrected)")

        fig.colorbar(im2, ax=ax2)

        return lsst.pipe.base.Struct(
            output_gain_correction=gain_correction,
            output_flat_ratio_plot=fig,
        )
