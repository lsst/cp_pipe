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
import copy
import esutil
import numpy as np
import warnings
from astropy.table import Table

import lsst.pipe.base
import lsst.pex.config
import lsst.afw.cameraGeom

from ..utils import FlatGradientFitter, FlatGainRatioFitter


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
    output_adjust_summary = lsst.pipe.base.connectionTypes.Output(
        name="ptc_adjustment_summary",
        doc="Summary of gain adjustments.",
        storageClass="ArrowAstropy",
        dimensions=("instrument", "detector"),
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
        # Choose the exposures that will be used.
        # For simplicity, we always use the first of a PTC pair.
        rng = np.random.RandomState(seed=self.config.random_seed)

        exp_ids_first = np.asarray(input_ptc.inputExpIdPairs[input_ptc.ampNames[0]])[:, 0]

        avg = np.zeros(len(exp_ids_first))
        n_amp = 0
        for amp_name in input_ptc.ampNames:
            if amp_name in input_ptc.badAmps:
                continue

            avg[:] += input_ptc.finalMeans[amp_name]
            n_amp += 1

        avg /= n_amp

        exp_ids = exp_ids_first[((avg > self.config.min_adu) & (avg < self.config.max_adu))]
        if len(exp_ids) > self.config.n_flat:
            exp_ids = rng.choice(exp_ids, size=self.config.n_flat, replace=False)
        exp_ids = np.sort(exp_ids)

        # Figure out the reference amplifier.
        gain_array = np.zeros(len(input_ptc.ampNames))
        for i, amp_name in enumerate(input_ptc.ampNames):
            gain_array[i] = input_ptc.gain[amp_name]
        good, = np.where(np.isfinite(gain_array))

        if len(good) <= 1:
            self.log.warning("Insufficient good amplifiers for PTC gain adjustment.")
            return lsst.pipe.base.Struct(output_ptc=input_ptc)

        st = np.argsort(gain_array[good])
        fixed_amp_num = good[st[int(0.5*len(good))]]
        self.log.info(
            "Using amplifier %d (%s) as fixed reference amplifier.",
            fixed_amp_num,
            input_ptc.ampNames[fixed_amp_num],
        )

        gain_ratio_array = np.zeros((len(exp_ids), len(input_ptc.ampNames)))

        for i, exp_id in enumerate(exp_ids):
            for handle in exposures:
                if exp_id == handle.dataId["exposure"]:
                    exposure = handle.get()
                    break

            self.log.info("Fitting gain ratios on exposure %d.", exp_id)
            gain_ratio_array[i, :] = self._compute_gain_ratios(input_ptc, exposure, fixed_amp_num)

        output_ptc = copy.copy(input_ptc)

        # Compute the summary table.
        summary = Table()
        summary.meta["fixed_amp_num"] = fixed_amp_num
        summary.meta["fixed_amp_name"] = input_ptc.ampNames[fixed_amp_num]

        summary["exp_id"] = exp_ids

        summary["mean_adu"] = np.zeros(len(exp_ids))

        a, b = esutil.numpy_util.match(exp_ids_first, exp_ids)
        summary["mean_adu"][b] = avg[a]

        corrections = np.zeros(len(input_ptc.ampNames))
        for i, amp_name in enumerate(input_ptc.ampNames):
            summary[f"{amp_name}_gain_ratio"] = gain_ratio_array[:, i]

            corrections[i] = np.median(gain_ratio_array[:, i])

        # Adjust the median correction to 1.0 so we do not change the
        # gain of the detector on average.
        # This is needed in case the reference amplifier is skewed in
        # terms of offsets even though it has the median gain.
        med_correction = np.median(corrections)

        for i, amp_name in enumerate(output_ptc.ampNames):
            correction = corrections[i] / med_correction
            new_gain = output_ptc.gainUnadjusted[amp_name] / correction
            self.log.info(
                "Adjusting gain from amplifier %s by factor of %.5f (from %.5f to %.5f)",
                amp_name,
                correction,
                output_ptc.gain[amp_name],
                new_gain,
            )
            output_ptc.gain[amp_name] = new_gain

        return lsst.pipe.base.Struct(output_ptc=output_ptc, gain_adjust_summary=summary)

    def _compute_gain_ratios(self, ptc, exposure, fixed_amp_num):
        """Compute the gain ratios from a given non-gain-corrected exposure.

        Parameters
        ----------
        ptc : `lsst.ip.isr.PhotonTransferCurveDataset`
            PTC to correct exposure.
        exposure : `lsst.afw.image.Exposure`
            Exposure to measure gain ratios.
        fixed_amp_num : `int`
            Use this amp number as the fixed point (gain ratio == 1.0).

        Returns
        -------
        amp_gain_ratios : `np.ndarray`
            Amp gain ratios, relative to fixed amp.
        """
        if not exposure.metadata.get("LSST ISR BOOTSTRAP", True):
            raise RuntimeError(
                "PhotonTransferCurveAdjustGainRatiosTask can only be run on bootstrap exposures.",
            )

        # Gain-correct the exposure.
        detector = exposure.getDetector()

        for amp_name in ptc.ampNames:
            if amp_name in ptc.badAmps:
                exposure[detector[amp_name].getBBox()].image.array[:, :] = np.nan
                continue

            exposure[detector[amp_name].getBBox()].image.array[:, :] *= ptc.gainUnadjusted[amp_name]

        # Next we bin the detector, avoiding amp edges.
        xd_arrays = []
        yd_arrays = []
        value_arrays = []
        amp_arrays = []

        amp_boundary = self.config.amp_boundary
        bin_factor = self.config.bin_factor

        for i, amp in enumerate(detector):
            arr = exposure[amp.getBBox()].image.array

            n_step_y = (arr.shape[0] - (2 * amp_boundary)) // bin_factor
            y_min = amp_boundary
            y_max = bin_factor * n_step_y + y_min
            n_step_x = (arr.shape[1] - (2 * amp_boundary)) // bin_factor
            x_min = amp_boundary
            x_max = bin_factor * n_step_x + x_min

            arr = arr[y_min: y_max, x_min: x_max]
            binned = arr.reshape((n_step_y, bin_factor, n_step_x, bin_factor))
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", r"Mean of empty")
                binned = np.nanmean(binned, axis=1)
                binned = np.nanmean(binned, axis=2)

            xx = np.arange(binned.shape[1]) * bin_factor + bin_factor / 2. + x_min
            yy = np.arange(binned.shape[0]) * bin_factor + bin_factor / 2. + y_min
            x, y = np.meshgrid(xx, yy)
            x = x.ravel() + amp.getBBox().getBeginX()
            y = y.ravel() + amp.getBBox().getBeginY()
            value = binned.ravel()

            xd_arrays.append(x)
            yd_arrays.append(y)
            value_arrays.append(value)
            amp_arrays.append(np.full(len(x), i))

        xd = np.concatenate(xd_arrays)
        yd = np.concatenate(yd_arrays)
        value = np.concatenate(value_arrays)
        amp_num = np.concatenate(amp_arrays)

        # Clip out non-finite and extreme values.
        # We need to be careful about unmasked defects, so we take
        # a tighter percentile and expand.
        lo, hi = np.nanpercentile(value, [5.0, 95.0])
        lo *= 0.8
        hi *= 1.2
        use = (np.isfinite(value) & (value >= lo) & (value <= hi))
        xd = xd[use]
        yd = yd[use]
        value = value[use]
        amp_num = amp_num[use]

        # If configured, fit the radial gradient.
        if self.config.do_remove_radial_gradient:
            transform = detector.getTransform(
                lsst.afw.cameraGeom.PIXELS,
                lsst.afw.cameraGeom.FOCAL_PLANE,
            )
            xy = np.vstack((xd, yd))
            xf, yf = np.vsplit(transform.getMapping().applyForward(xy), 2)
            xf = xf.ravel()
            yf = yf.ravel()
            radius = np.sqrt(xf**2. + yf**2.)

            nodes = np.linspace(np.min(radius), np.max(radius), self.config.radial_gradient_n_spline_nodes)

            # Put in a normalization for fitting.
            norm = np.nanpercentile(value, 95.0)

            fitter = FlatGradientFitter(
                nodes,
                xf,
                yf,
                value/norm,
                np.array([]),
                constrain_zero=False,
                fit_centroid=False,
                fit_gradient=False,
                fit_outer_gradient=False,
                fp_centroid_x=0.0,
                fp_centroid_y=0.0,
            )
            p0 = fitter.compute_p0()
            pars = fitter.fit(p0)

            value /= fitter.compute_model(pars)

        # Fit gain ratios.
        fitter = FlatGainRatioFitter(
            exposure.getBBox(),
            self.config.chebyshev_gradient_order,
            xd,
            yd,
            amp_num,
            value,
            fixed_amp_num,
        )
        pars = fitter.fit()

        return pars[fitter.indices["amp_pars"]]
