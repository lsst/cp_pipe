#
# LSST Data Management System
#
# Copyright 2008-2017  AURA/LSST.
#
# This product includes software developed by the
# LSST Project (http://www.lsst.org/).
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
# You should have received a copy of the LSST License Statement and
# the GNU General Public License along with this program.  If not,
# see <https://www.lsstcorp.org/LegalNotices/>.
#
"""Test cases for cp_pipe PTC gain adjustment code."""

import copy
import logging
import unittest
import numpy as np
from scipy.interpolate import Akima1DInterpolator

import lsst.utils.tests

import lsst.afw.cameraGeom
from lsst.afw.image import ExposureF
from lsst.cp.pipe.ptc import PhotonTransferCurveAdjustGainRatiosTask
from lsst.cp.pipe.ptc.cpPtcAdjustGainRatios import _compute_gain_ratios
from lsst.cp.pipe.utils import bin_flat
from lsst.ip.isr import IsrMockLSST, PhotonTransferCurveDataset
from lsst.pipe.base import InMemoryDatasetHandle


class PtcAdjustGainRatiosTestCase(lsst.utils.tests.TestCase):
    def setUp(self):
        mock = IsrMockLSST()
        self.camera = mock.getCamera()
        self.detector = self.camera[20]
        self.amp_names = [amp.getName() for amp in self.detector]

        rng = np.random.RandomState(seed=12345)

        self.gain_true = {}
        for amp_name in self.amp_names:
            self.gain_true[amp_name] = float(rng.normal(loc=1.5, scale=0.05, size=1)[0])

    def _gain_correct_flat(self, flat, ptc, adjust_ratios):
        """Gain correct a flat to a biased and debiased version.

        Parameters
        ----------
        flat : `lsst.afw.image.Exposure`
            Flat to correct.
        ptc : `lsst.ip.isr.PhotonTransferCurveDataset`
            PTC dataset with unadjusted gains.
        adjust_ratios : `np.ndarray`
            Array of adjustment ratios.

        Returns
        -------
        flat_biased : `lsst.afw.image.Exposure`
            Flat with the original unadjusted gains.
        flat_debiased : `lsst.afw.image.Exposure`
            Flat with adjusted gains.
        """
        flat_biased = flat.clone()
        flat_debiased = flat.clone()
        for i, amp_name in enumerate(self.amp_names):
            bbox = self.detector[amp_name].getBBox()
            if amp_name in ptc.badAmps:
                flat_biased[bbox].image.array[:, :] = np.nan
                flat_debiased[bbox].image.array[:, :] = np.nan
                continue

            flat_biased[bbox].image.array[:, :] *= ptc.gainUnadjusted[amp_name]
            flat_debiased[bbox].image.array[:, :] *= (ptc.gainUnadjusted[amp_name] / adjust_ratios[i])

        return flat_biased, flat_debiased

    def test_compute_gain_ratios_small_gradient(self):
        fixed_amp_num = 2

        flat, radial_gradient, planar_gradient = _get_adjusted_flat(
            self.detector,
            self.gain_true,
            10000.0,
            0.98,
            -0.00005,
            0.00001,
        )

        ptc = PhotonTransferCurveDataset()
        ptc.ampNames = self.amp_names
        ptc.badAmps = []

        gain_biased = copy.copy(self.gain_true)
        gain_biased[self.amp_names[0]] *= 1.002
        gain_biased[self.amp_names[4]] *= 0.998

        # These are the gains recorded in the PTC.
        for amp_name in self.amp_names:
            ptc.gain[amp_name] = gain_biased[amp_name]
            ptc.gainUnadjusted[amp_name] = gain_biased[amp_name]
            ptc.noise[amp_name] = 10.0
            ptc.ptcTurnoff[amp_name] = 50000.0

        with self.assertNoLogs(level=logging.WARNING):
            binned = bin_flat(ptc, flat.clone(), bin_factor=2, amp_boundary=0, apply_gains=True)
            lo, hi = np.nanpercentile(binned["value"], [5.0, 95.0])
            lo *= 0.8
            hi *= 1.2
            use = (np.isfinite(binned["value"]) & (binned["value"] >= lo) & (binned["value"] <= hi))
            binned = binned[use]

            adjust_ratios = _compute_gain_ratios(
                flat.getDetector(),
                binned,
                fixed_amp_num,
                radial_gradient_n_spline_nodes=3,
            )

        # Make two flats with the old adjustment and the new adjustment.
        flat_biased, flat_debiased = self._gain_correct_flat(flat, ptc, adjust_ratios)

        # Remove the gradients...
        flat_biased.image.array[:, :] /= radial_gradient
        flat_biased.image.array[:, :] /= planar_gradient
        flat_debiased.image.array[:, :] /= radial_gradient
        flat_debiased.image.array[:, :] /= planar_gradient

        # Insist that the standard deviation decreases after
        # removing gradients.
        self.assertLess(
            np.std(flat_debiased.image.array.ravel()),
            np.std(flat_biased.image.array.ravel()),
        )

    def test_compute_gain_ratios_large_gradient(self):
        fixed_amp_num = 2

        flat, radial_gradient, planar_gradient = _get_adjusted_flat(
            self.detector,
            self.gain_true,
            10000.0,
            0.6,
            -0.00005,
            0.00001,
        )

        ptc = PhotonTransferCurveDataset()
        ptc.ampNames = self.amp_names
        ptc.badAmps = []

        gain_biased = copy.copy(self.gain_true)
        gain_biased[self.amp_names[0]] *= 1.002
        gain_biased[self.amp_names[4]] *= 0.998

        # These are the gains recorded in the PTC.
        for amp_name in self.amp_names:
            ptc.gain[amp_name] = gain_biased[amp_name]
            ptc.gainUnadjusted[amp_name] = gain_biased[amp_name]
            ptc.noise[amp_name] = 10.0
            ptc.ptcTurnoff[amp_name] = 50000.0

        with self.assertNoLogs(level=logging.WARNING):
            binned = bin_flat(ptc, flat.clone(), bin_factor=2, amp_boundary=0, apply_gains=True)
            lo, hi = np.nanpercentile(binned["value"], [5.0, 95.0])
            lo *= 0.8
            hi *= 1.2
            use = (np.isfinite(binned["value"]) & (binned["value"] >= lo) & (binned["value"] <= hi))
            binned = binned[use]

            adjust_ratios = _compute_gain_ratios(
                flat.getDetector(),
                binned,
                fixed_amp_num,
                radial_gradient_n_spline_nodes=3,
            )

        # Make two flats with the old adjustment and the new adjustment.
        flat_biased, flat_debiased = self._gain_correct_flat(flat, ptc, adjust_ratios)

        # Remove the gradients...
        flat_biased.image.array[:, :] /= radial_gradient
        flat_biased.image.array[:, :] /= planar_gradient
        flat_debiased.image.array[:, :] /= radial_gradient
        flat_debiased.image.array[:, :] /= planar_gradient

        # Insist that the standard deviation decreases after
        # removing gradients.
        self.assertLess(
            np.std(flat_debiased.image.array.ravel()),
            np.std(flat_biased.image.array.ravel()),
        )

    def test_compute_gain_ratios_bad_amp(self):
        fixed_amp_num = 2

        flat, radial_gradient, planar_gradient = _get_adjusted_flat(
            self.detector,
            self.gain_true,
            10000.0,
            0.6,
            -0.00005,
            0.00001,
            bad_amps=[self.amp_names[1]],
        )

        ptc = PhotonTransferCurveDataset()
        ptc.ampNames = self.amp_names
        ptc.badAmps = [self.amp_names[1]]

        gain_biased = copy.copy(self.gain_true)
        gain_biased[self.amp_names[0]] *= 1.002
        gain_biased[self.amp_names[4]] *= 0.998

        # These are the gains recorded in the PTC.
        for amp_name in self.amp_names:
            ptc.gain[amp_name] = gain_biased[amp_name]
            ptc.gainUnadjusted[amp_name] = gain_biased[amp_name]
            ptc.noise[amp_name] = 10.0
            ptc.ptcTurnoff[amp_name] = 50000.0

        with self.assertNoLogs(level=logging.WARNING):
            binned = bin_flat(ptc, flat.clone(), bin_factor=2, amp_boundary=0, apply_gains=True)
            lo, hi = np.nanpercentile(binned["value"], [5.0, 95.0])
            lo *= 0.8
            hi *= 1.2
            use = (np.isfinite(binned["value"]) & (binned["value"] >= lo) & (binned["value"] <= hi))
            binned = binned[use]

            adjust_ratios = _compute_gain_ratios(
                flat.getDetector(),
                binned,
                fixed_amp_num,
                radial_gradient_n_spline_nodes=3,
            )

        # Make two flats with the old adjustment and the new adjustment.
        flat_biased, flat_debiased = self._gain_correct_flat(flat, ptc, adjust_ratios)

        # Remove the gradients...
        flat_biased.image.array[:, :] /= radial_gradient
        flat_biased.image.array[:, :] /= planar_gradient
        flat_debiased.image.array[:, :] /= radial_gradient
        flat_debiased.image.array[:, :] /= planar_gradient

        # Insist that the standard deviation decreases after
        # removing gradients.
        # Note that this contains nans which must be filtered.
        self.assertLess(
            np.nanstd(flat_debiased.image.array.ravel()),
            np.nanstd(flat_biased.image.array.ravel()),
        )

    def test_compute_gain_ratios_weird_amp(self):
        fixed_amp_num = 2

        # The weird amps have a large offset but are not known
        # as bad.
        flat, radial_gradient, planar_gradient = _get_adjusted_flat(
            self.detector,
            self.gain_true,
            10000.0,
            0.6,
            -0.00005,
            0.00001,
            bad_amps=[self.amp_names[1]],
        )

        ptc = PhotonTransferCurveDataset()
        ptc.ampNames = self.amp_names

        gain_biased = copy.copy(self.gain_true)
        gain_biased[self.amp_names[0]] *= 1.002
        gain_biased[self.amp_names[4]] *= 0.998

        # These are the gains recorded in the PTC.
        for amp_name in self.amp_names:
            ptc.gain[amp_name] = gain_biased[amp_name]
            ptc.gainUnadjusted[amp_name] = gain_biased[amp_name]
            ptc.noise[amp_name] = 10.0
            ptc.ptcTurnoff[amp_name] = 50000.0

        max_fractional_gain_ratio = 0.05
        with self.assertLogs(level=logging.WARNING) as cm:
            binned = bin_flat(ptc, flat.clone(), bin_factor=2, amp_boundary=0, apply_gains=True)
            lo, hi = np.nanpercentile(binned["value"], [5.0, 95.0])
            lo *= 0.8
            hi *= 1.2
            use = (np.isfinite(binned["value"]) & (binned["value"] >= lo) & (binned["value"] <= hi))
            binned = binned[use]

            adjust_ratios = _compute_gain_ratios(
                flat.getDetector(),
                binned,
                fixed_amp_num,
                radial_gradient_n_spline_nodes=3,
                max_fractional_gain_ratio=max_fractional_gain_ratio,
            )
        self.assertIn("Found bad amp", cm.output[0])

        np.testing.assert_array_less(np.abs(adjust_ratios - 1.0), max_fractional_gain_ratio)

        # Make two flats with the old adjustment and the new adjustment.
        flat_biased, flat_debiased = self._gain_correct_flat(flat, ptc, adjust_ratios)

        # Remove the gradients...
        flat_biased.image.array[:, :] /= radial_gradient
        flat_biased.image.array[:, :] /= planar_gradient
        flat_debiased.image.array[:, :] /= radial_gradient
        flat_debiased.image.array[:, :] /= planar_gradient

        # Insist that the standard deviation decreases after
        # removing gradients.
        # Note that this contains nans which must be filtered.
        self.assertLess(
            np.nanstd(flat_debiased.image.array.ravel()),
            np.nanstd(flat_biased.image.array.ravel()),
        )

    def test_task_run(self):
        levels = np.linspace(500.0, 25000.0, 40)

        flat_handles = []
        exp_id_pairs = []
        for i, level in enumerate(levels):
            exp_id_1 = 100 + 2*i
            exp_id_2 = exp_id_1 + 1

            data_id_1 = {
                "detector": self.detector.getId(),
                "exposure": exp_id_1,
            }
            data_id_2 = {
                "detector": self.detector.getId(),
                "exposure": exp_id_2,
            }

            flat, radial_gradient, planar_gradient = _get_adjusted_flat(
                self.detector,
                self.gain_true,
                levels[i],
                0.6,
                -0.00005,
                0.00001,
            )
            flat_handles.append(InMemoryDatasetHandle(flat.clone(), dataId=data_id_1))
            flat_handles.append(InMemoryDatasetHandle(flat.clone(), dataId=data_id_2))

            exp_id_pairs.append([exp_id_1, exp_id_2])

        ptc = PhotonTransferCurveDataset()
        ptc.ampNames = self.amp_names
        ptc.badAmps = []

        gain_biased = copy.copy(self.gain_true)
        gain_biased[self.amp_names[0]] *= 1.002
        gain_biased[self.amp_names[4]] *= 0.998

        # These are the gains recorded in the PTC.
        for amp_name in self.amp_names:
            ptc.gain[amp_name] = gain_biased[amp_name]
            ptc.gainUnadjusted[amp_name] = gain_biased[amp_name]
            ptc.noise[amp_name] = 10.0
            ptc.ptcTurnoff[amp_name] = 50000.0

            ptc.rawMeans[amp_name] = levels
            ptc.finalMeans[amp_name] = levels

            ptc.inputExpIdPairs[amp_name] = exp_id_pairs

        # Add in one noisy amp which will not be considered
        # to be the reference amp.
        ptc.noise[self.amp_names[-1]] = 15.0
        # And add in one low turnoff amp.
        ptc.ptcTurnoff[self.amp_names[-2]] = 10000.0

        config = PhotonTransferCurveAdjustGainRatiosTask.ConfigClass()
        config.bin_factor = 2
        config.n_flat = 10
        config.amp_boundary = 0
        config.radial_gradient_n_spline_nodes = 3
        config.n_flat = 10

        task = PhotonTransferCurveAdjustGainRatiosTask(config=config)
        struct = task.run(exposures=flat_handles, input_ptc=ptc)

        output_ptc = struct.output_ptc
        summary = struct.gain_adjust_summary

        self.assertGreater(summary["mean_adu"].min(), config.min_adu)
        self.assertLess(summary["mean_adu"].max(), config.max_adu)

        self.assertIn("fixed_amp_index", summary.meta.keys())
        self.assertIn("fixed_amp_name", summary.meta.keys())
        self.assertIn("median_correction", summary.meta.keys())

        # In this idealized case, the summary values should be the same
        # for every flux level.
        adjust_ratios = np.zeros(len(self.amp_names))
        for i, amp_name in enumerate(self.amp_names):
            np.testing.assert_array_equal(
                summary[f"{amp_name}_gain_ratio"][0],
                summary[f"{amp_name}_gain_ratio"],
            )
            adjust_ratios[i] = output_ptc.gainUnadjusted[amp_name] / output_ptc.gain[amp_name]
            self.assertFloatsAlmostEqual(
                adjust_ratios[i],
                summary[f"{amp_name}_gain_ratio"] / summary.meta["median_correction"],
            )

        # Confirm this actually works.
        flat = flat_handles[0].get()
        flat_biased, flat_debiased = self._gain_correct_flat(flat, ptc, adjust_ratios)

        # Remove the gradients...
        flat_biased.image.array[:, :] /= radial_gradient
        flat_biased.image.array[:, :] /= planar_gradient
        flat_debiased.image.array[:, :] /= radial_gradient
        flat_debiased.image.array[:, :] /= planar_gradient

        # Insist that the standard deviation decreases after
        # removing gradients.
        self.assertLess(
            np.std(flat_debiased.image.array.ravel()),
            np.std(flat_biased.image.array.ravel()),
        )

    def test_task_run_bad_amp(self):
        levels = np.linspace(500.0, 25000.0, 40)

        bad_amps = [self.amp_names[1]]

        flat_handles = []
        exp_id_pairs = []
        for i, level in enumerate(levels):
            exp_id_1 = 100 + 2*i
            exp_id_2 = exp_id_1 + 1

            data_id_1 = {
                "detector": self.detector.getId(),
                "exposure": exp_id_1,
            }
            data_id_2 = {
                "detector": self.detector.getId(),
                "exposure": exp_id_2,
            }

            flat, radial_gradient, planar_gradient = _get_adjusted_flat(
                self.detector,
                self.gain_true,
                levels[i],
                0.6,
                -0.00005,
                0.00001,
                bad_amps=bad_amps,
            )
            flat_handles.append(InMemoryDatasetHandle(flat.clone(), dataId=data_id_1))
            flat_handles.append(InMemoryDatasetHandle(flat.clone(), dataId=data_id_2))

            exp_id_pairs.append([exp_id_1, exp_id_2])

        ptc = PhotonTransferCurveDataset()
        ptc.ampNames = self.amp_names
        ptc.badAmps = bad_amps

        gain_biased = copy.copy(self.gain_true)
        gain_biased[self.amp_names[0]] *= 1.002
        gain_biased[self.amp_names[4]] *= 0.998

        # These are the gains recorded in the PTC.
        for amp_name in self.amp_names:
            ptc.gain[amp_name] = gain_biased[amp_name]
            ptc.gainUnadjusted[amp_name] = gain_biased[amp_name]
            ptc.noise[amp_name] = 10.0
            ptc.ptcTurnoff[amp_name] = 50000.0

            ptc.rawMeans[amp_name] = levels
            ptc.finalMeans[amp_name] = levels

            ptc.inputExpIdPairs[amp_name] = exp_id_pairs

        config = PhotonTransferCurveAdjustGainRatiosTask.ConfigClass()
        config.bin_factor = 2
        config.n_flat = 10
        config.amp_boundary = 0
        config.radial_gradient_n_spline_nodes = 3
        # Make sure the code works when we have fewer flats.
        config.n_flat = 100

        task = PhotonTransferCurveAdjustGainRatiosTask(config=config)
        struct = task.run(exposures=flat_handles, input_ptc=ptc)

        output_ptc = struct.output_ptc
        summary = struct.gain_adjust_summary

        self.assertGreater(summary["mean_adu"].min(), config.min_adu)
        self.assertLess(summary["mean_adu"].max(), config.max_adu)

        self.assertIn("fixed_amp_index", summary.meta.keys())
        self.assertIn("fixed_amp_name", summary.meta.keys())
        self.assertIn("median_correction", summary.meta.keys())

        # In this idealized case, the summary values should be the same
        # for every flux level.
        adjust_ratios = np.zeros(len(self.amp_names))
        for i, amp_name in enumerate(self.amp_names):
            np.testing.assert_array_equal(
                summary[f"{amp_name}_gain_ratio"][0],
                summary[f"{amp_name}_gain_ratio"],
            )
            adjust_ratios[i] = output_ptc.gainUnadjusted[amp_name] / output_ptc.gain[amp_name]
            self.assertFloatsAlmostEqual(
                adjust_ratios[i],
                summary[f"{amp_name}_gain_ratio"] / summary.meta["median_correction"],
            )

        # Confirm this actually works.
        flat = flat_handles[0].get()
        flat_biased, flat_debiased = self._gain_correct_flat(flat, ptc, adjust_ratios)

        # Remove the gradients...
        flat_biased.image.array[:, :] /= radial_gradient
        flat_biased.image.array[:, :] /= planar_gradient
        flat_debiased.image.array[:, :] /= radial_gradient
        flat_debiased.image.array[:, :] /= planar_gradient

        # Insist that the standard deviation decreases after
        # removing gradients.
        self.assertLess(
            np.nanstd(flat_debiased.image.array.ravel()),
            np.nanstd(flat_biased.image.array.ravel()),
        )


def _get_adjusted_flat(
    detector,
    gain_dict,
    normalization,
    radial_gradient_outer_value,
    planar_gradient_x,
    planar_gradient_y,
    bad_amps=[],
    weird_amps=[],
    n_radial_nodes=10,
):
    """Get an adjusted flat.

    Parameters
    ----------
    detector : `lsst.afw.cameraGeom.Detector`
        Detector object.
    gain_dict : `dict` [`float`]
        Dictionary of amp names to gains.
    normalization : `float`
        Normalization value.
    radial_gradient_outer_value : `float`
        Value at max radius (should be < 1.0).
    planar_gradient_x : `float`
        Planar gradient x slope.
    planar_gradient_y : `float`
        Planar gradient y slope.
    bad_amps : `list` [`str`], optional
        List of bad amp names.
    n_radial_nodes : `int`, optional
        Number of radial nodes.

    Returns
    -------
    flat : `lsst.afw.image.Exposure`
        Flat image generated.
    radial_gradient : `np.ndarray`
        Radial gradient image applied.
    planar_gradient : `np.ndarray`
        Planar gradient image applied.
    """
    flat = ExposureF(detector.getBBox())
    flat.setDetector(detector)

    xx = np.arange(flat.image.array.shape[1], dtype=np.float64)
    yy = np.arange(flat.image.array.shape[0], dtype=np.float64)
    x, y = np.meshgrid(xx, yy)
    x = x.ravel()
    y = y.ravel()

    transform = detector.getTransform(lsst.afw.cameraGeom.PIXELS, lsst.afw.cameraGeom.FOCAL_PLANE)
    xy = np.vstack((x, y))
    xf, yf = np.vsplit(transform.getMapping().applyForward(xy), 2)
    xf = xf.ravel()
    yf = yf.ravel()

    radius = np.sqrt(xf**2. + yf**2.)

    nodes = np.linspace(radius.min(), radius.max(), n_radial_nodes)
    spline_values = np.linspace(1.0, 0.98, n_radial_nodes)

    spl = Akima1DInterpolator(nodes, spline_values, method="akima")
    radial_gradient = spl(radius).reshape(flat.image.array.shape)
    flat.image.array[:, :] = radial_gradient.copy()

    # This will be a parameter
    normalization = 10000.0

    flat.image.array[:, :] *= normalization

    # Add a bit of a planar gradient; approx +/- 1%.
    planar_gradient = (
        1
        - 0.00005 * (x - detector.getBBox().getCenter().getX())
        + 0.00001 * (y - detector.getBBox().getCenter().getY())
    ).reshape(flat.image.array.shape)

    flat.image.array[:, :] *= planar_gradient

    for amp in detector:
        flat.image[amp.getBBox()].array[:, :] /= gain_dict[amp.getName()]

    # Make the bad amps goofy.
    for bad_amp in bad_amps:
        flat.image[detector[bad_amp].getBBox()].array[:, :] *= 0.1

    return flat, radial_gradient, planar_gradient


class TestMemory(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
