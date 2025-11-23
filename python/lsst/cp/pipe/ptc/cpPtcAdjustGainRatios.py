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
import logging
import numpy as np
from astropy.table import Table

import lsst.pipe.base
import lsst.pex.config
import lsst.afw.cameraGeom

from ..utils import FlatGradientFitter, FlatGainRatioFitter, bin_flat


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
    max_fractional_gain_ratio = lsst.pex.config.Field(
        dtype=float,
        doc="Maximum fractional gain ratio to consider per-exposure. Any amps with larger "
            "offset will be excluded from the gradient fit and will have no corrections "
            "applied.",
        default=0.05,
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
            ``output_adjust_summary``
                The summary of adjustments.
        """
        # Choose the exposures that will be used.
        # For simplicity, we always use the first of a PTC pair.
        rng = np.random.RandomState(seed=self.config.random_seed)

        exp_ids_first = np.asarray(input_ptc.inputExpIdPairs[input_ptc.ampNames[0]])[:, 0]

        avg = np.zeros(len(exp_ids_first))
        n_amp = np.zeros(len(exp_ids_first), dtype=np.int64)
        for amp_name in input_ptc.ampNames:
            if amp_name in input_ptc.badAmps:
                continue

            finite = np.isfinite(input_ptc.finalMeans[amp_name])
            avg[finite] += input_ptc.finalMeans[amp_name][finite]
            n_amp[finite] += 1

        avg /= n_amp

        exp_ids = exp_ids_first[((avg > self.config.min_adu) & (avg < self.config.max_adu))]
        if len(exp_ids) > self.config.n_flat:
            exp_ids = rng.choice(exp_ids, size=self.config.n_flat, replace=False)
        exp_ids = np.sort(exp_ids)

        fixed_amp_index = _choose_reference_amplifier(
            input_ptc,
            self.config.max_noise_reference,
            self.config.turnoff_percentile_reference,
        )

        if fixed_amp_index < 0:
            return lsst.pipe.base.Struct(output_ptc=input_ptc, output_adjust_summary=Table())

        self.log.info(
            "Using amplifier %d (%s) as fixed reference amplifier.",
            fixed_amp_index,
            input_ptc.ampNames[fixed_amp_index],
        )

        gain_ratio_array = np.zeros((len(exp_ids), len(input_ptc.ampNames)))

        for i, exp_id in enumerate(exp_ids):
            for handle in exposures:
                if exp_id == handle.dataId["exposure"]:
                    exposure = handle.get()
                    break

            self.log.info("Fitting gain ratios on exposure %d.", exp_id)
            if not exposure.metadata.get("LSST ISR BOOTSTRAP", True):
                raise RuntimeError(
                    "PhotonTransferCurveAdjustGainRatiosTask can only be run on bootstrap exposures.",
                )

            binned = bin_flat(
                input_ptc,
                exposure,
                bin_factor=self.config.bin_factor,
                amp_boundary=self.config.amp_boundary,
                apply_gains=True,
            )

            # Clip out non-finite and extreme values.
            # We need to be careful about unmasked defects, so we take
            # a tighter percentile and expand.
            lo, hi = np.nanpercentile(binned["value"], [5.0, 95.0])
            lo *= 0.8
            hi *= 1.2
            use = (np.isfinite(binned["value"]) & (binned["value"] >= lo) & (binned["value"] <= hi))
            binned = binned[use]

            gain_ratio_array[i, :] = _compute_gain_ratios(
                exposure.getDetector(),
                binned,
                fixed_amp_index,
                do_remove_radial_gradient=self.config.do_remove_radial_gradient,
                radial_gradient_n_spline_nodes=self.config.radial_gradient_n_spline_nodes,
                chebyshev_gradient_order=self.config.chebyshev_gradient_order,
                max_fractional_gain_ratio=self.config.max_fractional_gain_ratio,
                log=self.log,
            )

        output_ptc = copy.copy(input_ptc)

        # Compute the summary table.
        summary = Table()
        summary.meta["fixed_amp_index"] = fixed_amp_index
        summary.meta["fixed_amp_name"] = input_ptc.ampNames[fixed_amp_index]

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

        summary.meta["median_correction"] = med_correction

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

        return lsst.pipe.base.Struct(output_ptc=output_ptc, output_adjust_summary=summary)


def _choose_reference_amplifier(
    ptc,
    max_noise_reference,
    turnoff_percentile_reference,
    log=None,
):
    """Choose a reference amplifier from a PTC.

    Parameters
    ----------
    ptc : `lsst.ip.isr.PhotonTransferCurveDataset`
        Input PTC.
    max_noise_reference : `float`
        Maximum read noise (e-) in the PTC for an amp to be considered.
    turnoff_percentile_reference : `float`
        Percentile threshold for sorting PTC turnoff for an amp to be
        considered.
    log : `logging.logger`, optional
        Log object.

    Returns
    -------
    reference_amp_index : `int`
        Index of the reference amplifier.
    """
    log = log if log else logging.getLogger(__name__)

    gain_array = np.zeros(len(ptc.ampNames))
    turnoff_array = np.zeros(len(ptc.ampNames))
    for i, amp_name in enumerate(ptc.ampNames):
        gain_array[i] = ptc.gain[amp_name]
        turnoff_array[i] = ptc.ptcTurnoff[amp_name]
        # Do not consider any amplifier with high noise
        # as a reference amplifier.
        if ptc.noise[amp_name] > max_noise_reference:
            log.info(
                "Excluding amplifier %s as a possible reference amplifier due to high noise (%.2f)",
                amp_name,
                ptc.noise[amp_name],
            )
            gain_array[i] = np.nan

    turnoff_threshold = np.nanpercentile(turnoff_array, turnoff_percentile_reference)
    gain_array[turnoff_array < turnoff_threshold] = np.nan
    good, = np.where(np.isfinite(gain_array))

    if len(good) <= 1:
        log.warning("Insufficient good amplifiers for PTC gain adjustment.")
        return -1

    st = np.argsort(np.nan_to_num(gain_array[good]))
    ref_amp_index = good[st[int(0.5*len(good))]]

    return ref_amp_index


def _compute_gain_ratios(
    detector,
    binned,
    fixed_amp_index,
    do_remove_radial_gradient=True,
    radial_gradient_n_spline_nodes=20,
    chebyshev_gradient_order=1,
    max_fractional_gain_ratio=0.05,
    nsig_clip=5.0,
    log=None,
):
    """Compute the gain ratios from a given non-gain-corrected exposure.

    Parameters
    ----------
    detector : `lsst.afw.cameraGeom.Detector`
        Detector object.
    binned : `astropy.table.Table`
        Table of binned values. Will contain ``xd``, ``yd`` (detector
        bin positions); ``value`` (binned value); ``amp_index``
        (index of the amplifiers).
    fixed_amp_index : `int`
        Use this amp index as the fixed point (gain ratio == 1.0).
    do_remove_radial_gradient : `bool`, optional
        Remove radial gradient before fitting amp gain ratios?
    radial_gradient_n_spline_nodes : `int`, optional
        Number of radial spline nodes for radial gradient.
    chebyshev_gradient_order : `int`, optional
        Order of chebyshev x/y polynomials to remove additional gradients.
    max_fractional_gain_ratio : `int`, optional
        Maximum fractional gain ratio to consider per-exposure. Any
        amps with larger offset will be excluded from the gradient fit
        and will have no corrections computed.
    nsig_clip : `float`, optional
        Number of sigma on distribution of gain ratios to clip when fitting
        out Chebyshev gradient.
    log : `logging.logger`, optional
        Log object.

    Returns
    -------
    amp_gain_ratios : `np.ndarray`
        Amp gain ratios, relative to fixed amp.
    """
    log = log if log else logging.getLogger(__name__)

    n_amp = len(detector)
    amp_names = [amp.getName() for amp in detector]

    # We iterate up to 8x to look for any bad amps (that have a gain offset
    # fit greater than max_fractional_gain_ratio) and reject them
    # from the fits.
    bad_amps_converged = False
    n_iter = 0
    value_uncorrected = binned["value"].copy()
    while not bad_amps_converged and n_iter < n_amp // 2:
        # If configured, fit the radial gradient.
        if do_remove_radial_gradient:
            transform = detector.getTransform(
                lsst.afw.cameraGeom.PIXELS,
                lsst.afw.cameraGeom.FOCAL_PLANE,
            )
            xy = np.vstack((binned["xd"], binned["yd"]))
            xf, yf = np.vsplit(transform.getMapping().applyForward(xy), 2)
            xf = xf.ravel()
            yf = yf.ravel()
            radius = np.sqrt(xf**2. + yf**2.)

            nodes = np.linspace(
                np.min(radius),
                np.max(radius),
                radial_gradient_n_spline_nodes,
            )

            # Put in a normalization for fitting.
            norm = np.nanpercentile(binned["value"], 95.0)

            fitter = FlatGradientFitter(
                nodes,
                xf,
                yf,
                binned["value"]/norm,
                np.array([]),
                constrain_zero=False,
                fit_centroid=False,
                fit_gradient=False,
                fp_centroid_x=0.0,
                fp_centroid_y=0.0,
            )
            p0 = fitter.compute_p0()
            pars = fitter.fit(p0)

            binned["value"] /= fitter.compute_model(pars)

        # Fit gain ratios.
        fitter = FlatGainRatioFitter(
            detector.getBBox(),
            chebyshev_gradient_order,
            binned["xd"],
            binned["yd"],
            binned["amp_index"],
            binned["value"],
            np.arange(n_amp),
            fixed_amp_index,
        )
        pars = fitter.fit(nsig_clip=nsig_clip)
        amp_pars = pars[fitter.indices["amp_pars"]]

        fractional_gain_ratio = np.abs(amp_pars - 1.0)
        max_ratio_ind = np.argmax(fractional_gain_ratio)

        if fractional_gain_ratio[max_ratio_ind] > max_fractional_gain_ratio:
            log.warning(
                "Found bad amp %s with offset parameter %.2f",
                amp_names[max_ratio_ind],
                amp_pars[max_ratio_ind],
            )
            good = (binned["amp_index"] != max_ratio_ind)
            binned = binned[good]
            binned["value"] = value_uncorrected[good]

            value_uncorrected = binned["value"].copy()
        else:
            bad_amps_converged = True

        n_iter += 1

    return amp_pars
