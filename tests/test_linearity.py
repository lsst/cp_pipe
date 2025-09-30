#!/usr/bin/env python

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
"""Test cases for cp_pipe linearity code."""

from astropy.table import Table
import logging
import unittest
import numpy as np
import copy
from scipy.interpolate import Akima1DInterpolator

import lsst.utils
import lsst.utils.tests

from lsst.ip.isr import PhotonTransferCurveDataset, IsrMockLSST

import lsst.afw.image
import lsst.afw.math
from lsst.cp.pipe import LinearitySolveTask, LinearityNormalizeTask, LinearityDoubleSplineSolveTask
from lsst.cp.pipe.cpLinearitySolve import _computeTurnoffAndMax, _noderator
from lsst.cp.pipe.ptc import PhotonTransferCurveSolveTask, PhotonTransferCurveExtractPairTask
from lsst.cp.pipe.utils import funcPolynomial, funcAstier
from lsst.daf.base import DateTime
from lsst.ip.isr.isrMock import FlatMock, IsrMock
from lsst.pipe.base import InMemoryDatasetHandle


class FakeCamera(list):
    def getName(self):
        return "FakeCam"


class LinearityTaskTestCase(lsst.utils.tests.TestCase):
    """Test case for the linearity tasks."""

    def setUp(self):
        mock_image_config = IsrMock.ConfigClass()
        mock_image_config.flatDrop = 0.99999
        mock_image_config.isTrimmed = True

        self.dummy_exposure = FlatMock(config=mock_image_config).run()
        self.detector = self.dummy_exposure.getDetector()
        self.input_dims = {"detector": 0}

        self.camera = FakeCamera([self.detector])

        self.amp_names = []
        for amp in self.detector:
            self.amp_names.append(amp.getName())

        self.sequencerMetadata = {
            "SEQNAME": "a_sequencer",
            "SEQFILE": "a_sequencer_file",
            "SEQCKSUM": "deadbeef",
        }

    def _create_ptc(
        self,
        amp_names,
        exp_times,
        means,
        ccobcurr=None,
        photo_charges=None,
        temperatures=None,
        ptc_turnoff=None,
    ):
        """
        Create a PTC with values for linearity tests.

        Parameters
        ----------
        amp_names : `list` [`str`]
            Names of amps.
        exp_times : `np.ndarray`
            Array of exposure times.
        means : `np.ndarray`
            Array of means.
        ccobcurr : `np.ndarray`, optional
            Array of CCOBCURR to put into auxiliary values.
        photo_charges : `np.ndarray`, optional
            Array of photoCharges to put into ptc.
        temperatures : `np.ndarray`, optional
            Array of temperatures (TEMP6) to put into ptc.
        ptc_turnoff : `float`, optional
            Turnoff value to set (by hand) for testing.

        Returns
        -------
        ptc : `lsst.ip.isr.PhotonTransferCurveDataset`
            PTC filled with relevant values.
        """
        exp_id_pairs = np.arange(len(exp_times)*2).reshape((len(exp_times), 2)).tolist()

        if photo_charges is None:
            photo_charges = np.full(len(exp_times), np.nan)

        datasets = []
        for i in range(len(exp_times)):
            partial = PhotonTransferCurveDataset(amp_names, ptcFitType="PARTIAL", covMatrixSide=1)
            for amp_name in amp_names:
                # For the first amp, we add a few bad points.
                if amp_name == amp_names[0] and i >= 5 and i < 7:
                    exp_id_mask = False
                    raw_mean = np.nan
                else:
                    exp_id_mask = True
                    raw_mean = means[i]

                partial.setAmpValuesPartialDataset(
                    amp_name,
                    inputExpIdPair=exp_id_pairs[i],
                    rawExpTime=exp_times[i],
                    rawMean=raw_mean,
                    rawVar=raw_mean,
                    kspValue=1.0,
                    expIdMask=exp_id_mask,
                    photoCharge=photo_charges[i],
                    overscanMedianLevel=100.0,
                )

            aux_dict = {}
            if ccobcurr is not None:
                aux_dict["CCOBCURR"] = ccobcurr[i]
            if temperatures is not None:
                aux_dict["TEMP6"] = temperatures[i]

            if aux_dict:
                partial.setAuxValuesPartialDataset(aux_dict)

            datasets.append(partial)

            datasets.append(PhotonTransferCurveDataset(amp_names, ptcFitType="DUMMY"))

        config = PhotonTransferCurveSolveTask.ConfigClass()
        config.maximumRangeCovariancesAstier = 1
        config.maxDeltaInitialPtcOutlierFit = 100_000.0
        solve_task = PhotonTransferCurveSolveTask(config=config)
        # Suppress logging here.
        with self.assertNoLogs(level=logging.CRITICAL):
            ptc = solve_task.run(datasets).outputPtcDataset

        # Make the last amp a bad amp.
        ptc.badAmps = [amp_names[-1]]

        if ptc_turnoff is not None:
            for amp_name in amp_names:
                if amp_name in ptc.badAmps:
                    ptc.ptcTurnoff[amp_name] = np.nan
                    ptc.finalMeans[amp_name][:] = np.nan
                else:
                    ptc.ptcTurnoff[amp_name] = ptc_turnoff
                    high = (ptc.rawMeans[amp_name] > ptc_turnoff)
                    ptc.expIdMask[amp_name][high] = False
                    ptc.finalMeans[amp_name][high] = np.nan

        ptc.updateMetadata(**self.sequencerMetadata, setCalibInfo=True)
        ptc.updateMetadata(camera=self.camera, detector=self.detector)

        return ptc

    def _check_linearity(self, linearity_type, min_adu=0.0, max_adu=100000.0):
        """Run and check linearity.

        Parameters
        ----------
        linearity_type : `str`
            Must be ``Polynomial``, ``Squared``, or ``LookupTable``.
        min_adu : `float`, optional
            Minimum cut on ADU for fit.
        max_adu : `float`, optional
            Maximum cut on ADU for fit.
        """
        flux = 1000.
        time_vec = np.arange(1., 101., 5)
        k2_non_linearity = -5e-6
        coeff = k2_non_linearity/(flux**2.)

        mu_vec = flux * time_vec + k2_non_linearity * time_vec**2.

        ptc = self._create_ptc(self.amp_names, time_vec, mu_vec)

        config = LinearitySolveTask.ConfigClass()
        config.linearityType = linearity_type
        config.minLinearAdu = min_adu
        config.maxLinearAdu = max_adu

        task = LinearitySolveTask(config=config)
        linearizer = task.run(ptc, [self.dummy_exposure], self.camera, self.input_dims).outputLinearizer

        if linearity_type == "LookupTable":
            t_max = config.maxLookupTableAdu / flux
            time_range = np.linspace(0.0, t_max, config.maxLookupTableAdu)
            signal_ideal = time_range * flux
            signal_uncorrected = funcPolynomial(np.array([0.0, flux, k2_non_linearity]), time_range)
            linearizer_table_row = signal_ideal - signal_uncorrected

        # Skip the last amp which is marked bad.
        for i, amp_name in enumerate(ptc.ampNames[:-1]):
            if linearity_type in ["Squared", "Polynomial"]:
                self.assertFloatsAlmostEqual(linearizer.fitParams[amp_name][0], 0.0, atol=1e-2)
                self.assertFloatsAlmostEqual(linearizer.fitParams[amp_name][1], 1.0, rtol=1e-5)
                self.assertFloatsAlmostEqual(linearizer.fitParams[amp_name][2], coeff, rtol=1e-6)

                if linearity_type == "Polynomial":
                    self.assertFloatsAlmostEqual(linearizer.fitParams[amp_name][3], 0.0)

                if linearity_type == "Squared":
                    self.assertEqual(len(linearizer.linearityCoeffs[amp_name]), 1)
                    self.assertFloatsAlmostEqual(linearizer.linearityCoeffs[amp_name][0], -coeff, rtol=1e-6)
                else:
                    self.assertEqual(len(linearizer.linearityCoeffs[amp_name]), 2)
                    self.assertFloatsAlmostEqual(linearizer.linearityCoeffs[amp_name][0], -coeff, rtol=1e-6)
                    self.assertFloatsAlmostEqual(linearizer.linearityCoeffs[amp_name][1], 0.0)

            else:
                index = linearizer.linearityCoeffs[amp_name][0]
                self.assertEqual(index, i)
                self.assertEqual(len(linearizer.tableData[index, :]), len(linearizer_table_row))
                self.assertFloatsAlmostEqual(linearizer.tableData[index, :], linearizer_table_row, rtol=1e-4)

            lin_mask = np.isfinite(linearizer.fitResiduals[amp_name])
            lin_mask_expected = (mu_vec > min_adu) & (mu_vec < max_adu) & ptc.expIdMask[amp_name]

            self.assertListEqual(lin_mask.tolist(), lin_mask_expected.tolist())
            self.assertFloatsAlmostEqual(linearizer.fitResiduals[amp_name][lin_mask], 0.0, atol=1e-2)

            # If we apply the linearity correction, we should get the true
            # linear values out.
            image = lsst.afw.image.ImageF(len(mu_vec), 1)
            image.array[:, :] = mu_vec
            lin_func = linearizer.getLinearityTypeByName(linearizer.linearityType[amp_name])
            lin_func()(
                image,
                coeffs=linearizer.linearityCoeffs[amp_name],
                table=linearizer.tableData,
                log=None,
            )

            linear_signal = flux * time_vec
            self.assertFloatsAlmostEqual(image.array[0, :] / linear_signal, 1.0, rtol=1e-6)

        self._check_linearizer_lengths(linearizer)

    def _check_linearizer_lengths(self, linearizer):
        # Check that the lengths of all the fields match.
        lenCoeffs = -1
        lenParams = -1
        lenParamsErr = -1
        lenResiduals = -1
        lenFit = -1
        for ampName in linearizer.ampNames:
            if lenCoeffs < 0:
                lenCoeffs = len(linearizer.linearityCoeffs[ampName])
                lenParams = len(linearizer.fitParams[ampName])
                lenParamsErr = len(linearizer.fitParamsErr[ampName])
                lenResiduals = len(linearizer.fitResiduals[ampName])
                lenFit = len(linearizer.linearFit[ampName])
            else:
                self.assertEqual(
                    len(linearizer.linearityCoeffs[ampName]),
                    lenCoeffs,
                    msg=f"amp {ampName} linearityCoeffs length mismatch",
                )
                self.assertEqual(
                    len(linearizer.fitParams[ampName]),
                    lenParams,
                    msg=f"amp {ampName} fitParams length mismatch",
                )
                self.assertEqual(
                    len(linearizer.fitParamsErr[ampName]),
                    lenParamsErr,
                    msg=f"amp {ampName} fitParamsErr length mismatch",
                )
                self.assertEqual(
                    len(linearizer.fitResiduals[ampName]),
                    lenResiduals,
                    msg=f"amp {ampName} fitResiduals length mismatch",
                )
                self.assertEqual(
                    len(linearizer.linearFit[ampName]),
                    lenFit,
                    msg=f"amp {ampName} linearFit length mismatch",
                )

    def test_linearity_polynomial(self):
        """Test linearity with polynomial fit."""
        self._check_linearity("Polynomial")

    def test_linearity_squared(self):
        """Test linearity with a single order squared solution."""
        self._check_linearity("Squared")

    def test_linearity_table(self):
        """Test linearity with a lookup table solution."""
        self._check_linearity("LookupTable")

    def test_linearity_polynomial_aducuts(self):
        """Test linearity with polynomial and ADU cuts."""
        self._check_linearity("Polynomial", min_adu=10000.0, max_adu=90000.0)

    def _check_linearity_spline(
        self,
        do_pd_offsets=False,
        n_points=200,
        do_mu_offset=False,
        do_weight_fit=False,
        do_temperature_fit=False,
    ):
        """Check linearity with a spline solution.

        Parameters
        ----------
        do_pd_offsets : `bool`, optional
            Apply offsets to the photodiode data.
        do_mu_offset : `bool`, optional
            Apply constant offset to mu data.
        do_weight_fit : `bool`, optional
            Fit the weight parameters?
        do_temperature_fit : `bool`, optional
            Apply a temperature dependence and fit it?
        """
        np.random.seed(12345)

        # Create a test dataset representative of real data.
        pd_values = np.linspace(1e-8, 2e-5, n_points)
        time_values = pd_values * 1000000.
        linear_ratio = 5e9
        mu_linear = linear_ratio * pd_values

        # Test spline parameters are taken from a test fit to LSSTCam
        # data, run 7193D, detector 22, amp C00. The exact fit is not
        # important, but this is only meant to be representative of
        # the shape of the non-linearity that we see.

        n_nodes = 10

        non_lin_spline_nodes = np.linspace(0, mu_linear.max(), n_nodes)
        non_lin_spline_values = np.array(
            [0.0, -8.87, 1.46, 1.69, -6.92, -68.23, -78.01, -11.56, 80.26, 185.01]
        )

        spl = lsst.afw.math.makeInterpolate(
            non_lin_spline_nodes,
            non_lin_spline_values,
            lsst.afw.math.stringToInterpStyle("AKIMA_SPLINE"),
        )

        mu_values = mu_linear + spl.interpolate(mu_linear)

        # Add a temperature dependence if necessary.
        if do_temperature_fit:
            temp_coeff = 0.0006
            temperatures = np.random.normal(scale=0.5, size=len(mu_values)) - 100.0

            # We use a negative sign here because we are doing the
            # opposite of the correction.
            mu_values *= (1 - temp_coeff*(temperatures - (-100.0)))
        else:
            temperatures = None

        # Add a constant offset if necessary.
        if do_mu_offset:
            offset_value = 2.0
            mu_values += offset_value
        else:
            offset_value = 0.0

        # Add some noise.
        mu_values += np.random.normal(scale=mu_values, size=len(mu_values)) / 10000.

        # Add some outlier values.
        if n_points >= 200:
            outlier_indices = np.arange(5) + 170
        else:
            outlier_indices = []
        mu_values[outlier_indices] += 200.0

        # Add some small offsets to the pd_values if requested.
        pd_values_offset = pd_values.copy()
        ccobcurr = None
        if do_pd_offsets:
            ccobcurr = np.zeros(pd_values.size)
            n_points_group = n_points//4
            group0 = np.arange(n_points_group)
            group1 = np.arange(n_points_group) + n_points_group
            group2 = np.arange(n_points_group) + 2*n_points_group
            group3 = np.arange(n_points_group) + 3*n_points_group
            ccobcurr[group0] = 0.01
            ccobcurr[group1] = 0.02
            ccobcurr[group2] = 0.03
            ccobcurr[group3] = 0.04

            pd_offset_factors = [0.995, 1.0, 1.005, 1.002]
            pd_values_offset[group0] *= pd_offset_factors[0]
            pd_values_offset[group2] *= pd_offset_factors[2]
            pd_values_offset[group3] *= pd_offset_factors[3]

        # Add one bad photodiode value, but don't put it at the very
        # end because that would change the spline node positions
        # and make comparisons to the "truth" here in the tests
        # more difficult.
        pd_values_offset[-2] = np.nan

        ptc = self._create_ptc(
            self.amp_names,
            time_values,
            mu_values,
            ccobcurr=ccobcurr,
            photo_charges=pd_values_offset,
            temperatures=temperatures,
        )

        config = LinearitySolveTask.ConfigClass()
        config.linearityType = "Spline"
        config.usePhotodiode = True
        config.minLinearAdu = 0.0
        config.splineKnots = n_nodes
        config.splineGroupingMinPoints = 101
        config.doSplineFitOffset = do_mu_offset
        config.doSplineFitWeights = do_weight_fit
        config.splineFitWeightParsStart = [7.2e-5, 1e-4]
        config.doSplineFitTemperature = do_temperature_fit
        config.maxFracLinearityDeviation = 0.05

        if do_pd_offsets:
            config.splineGroupingColumn = "CCOBCURR"

        if do_temperature_fit:
            config.splineFitTemperatureColumn = "TEMP6"

        task = LinearitySolveTask(config=config)
        linearizer = task.run(
            ptc,
            [self.dummy_exposure],
            self.camera,
            self.input_dims,
        ).outputLinearizer

        for key, value in self.sequencerMetadata.items():
            self.assertEqual(linearizer.metadata[key], value)

        for key in ["INSTRUME", "DETECTOR", "DET_NAME", "DET_SER"]:
            self.assertEqual(linearizer.metadata[key], ptc.metadata[key])

        if do_weight_fit:
            # These checks currently fail, and weight fitting is not
            # recommended.
            return

        # Skip the last amp which is marked bad.
        for amp_name in ptc.ampNames[:-1]:
            # This test data doesn't have a real turnoff.
            self.assertEqual(linearizer.linearityTurnoff[amp_name], np.nanmax(ptc.rawMeans[amp_name]))
            self.assertEqual(linearizer.linearityMaxSignal[amp_name], np.nanmax(ptc.rawMeans[amp_name]))

            lin_mask = np.isfinite(linearizer.fitResiduals[amp_name])

            # Make sure that non-finite initial values in range are also
            # masked.
            check, = np.where(~np.isfinite(ptc.rawMeans[amp_name]))
            if len(check) > 0:
                np.testing.assert_array_equal(lin_mask[check], False)

            # Make sure the outliers are masked.
            np.testing.assert_array_equal(lin_mask[outlier_indices], False)

            # Check the turnoff and max values.
            resid_atol = 4e-4
            self.assertFloatsAlmostEqual(
                linearizer.fitResiduals[amp_name][lin_mask] / mu_linear[lin_mask],
                0.0,
                atol=resid_atol,
            )

            # Loose check on the chi-squared.
            self.assertLess(linearizer.fitChiSq[amp_name], 2.0)

            # Check the residual sigma_mad.
            self.assertLess(linearizer.fitResidualsSigmaMad[amp_name], 1.2e-4)

            # If we apply the linearity correction, we should get the true
            # linear values out.
            image = lsst.afw.image.ImageF(len(mu_values), 1)
            image.array[:, :] = mu_values
            lin_func = linearizer.getLinearityTypeByName(linearizer.linearityType[amp_name])
            lin_func()(
                image,
                coeffs=linearizer.linearityCoeffs[amp_name],
                log=None,
            )

            # We scale by the median because of ambiguity in the overall
            # gain parameter which is not part of the non-linearity.
            ratio = image.array[0, lin_mask]/mu_linear[lin_mask]
            # When we have an offset, this test gets a bit confused
            # mixing truth and offset values.
            ratio_rtol = 5e-2 if do_mu_offset else 5e-4
            self.assertFloatsAlmostEqual(
                ratio / np.median(ratio),
                1.0,
                rtol=ratio_rtol,
            )

            # Check that the spline parameters recovered are consistent,
            # with input to some low-grade precision.
            # The first element should be identically zero.
            self.assertFloatsEqual(linearizer.linearityCoeffs[amp_name][0], 0.0)

            # We have two different comparisons here; for the terms that are
            # |value| < 20 (offset) or |value| > 20 (ratio), to avoid
            # divide-by-small-number problems. In all cases these are
            # approximate, and the real test is in the residuals.
            small = (np.abs(non_lin_spline_values) < 20)

            spline_atol = 6.0 if do_pd_offsets else 2.0
            spline_rtol = 0.14 if do_pd_offsets else 0.05

            self.assertFloatsAlmostEqual(
                linearizer.linearityCoeffs[amp_name][n_nodes:][small],
                non_lin_spline_values[small],
                atol=spline_atol,
            )
            self.assertFloatsAlmostEqual(
                linearizer.linearityCoeffs[amp_name][n_nodes:][~small],
                non_lin_spline_values[~small],
                rtol=spline_rtol,
            )

            # And check the offsets if they were included.
            if do_pd_offsets:
                # The relative scaling is to group 1.
                fit_offset_factors = linearizer.fitParams[amp_name][1] / linearizer.fitParams[amp_name]
                extra_pars = 0
                if do_mu_offset:
                    extra_pars += 1
                if do_temperature_fit:
                    extra_pars += 1

                if extra_pars > 0:
                    fit_offset_factors = fit_offset_factors[:-extra_pars]

                self.assertFloatsAlmostEqual(fit_offset_factors, np.array(pd_offset_factors), rtol=6e-4)

            # And check if the offset is fit well.
            fit_offset = None
            fit_temp_coeff = None
            if do_mu_offset and do_temperature_fit:
                fit_offset = linearizer.fitParams[amp_name][-2]
                fit_temp_coeff = linearizer.fitParams[amp_name][-1]
            elif do_mu_offset:
                fit_offset = linearizer.fitParams[amp_name][-1]
            elif do_temperature_fit:
                fit_temp_coeff = linearizer.fitParams[amp_name][-1]

            if fit_offset is not None:
                self.assertFloatsAlmostEqual(fit_offset, offset_value, rtol=6e-3)

            if fit_temp_coeff is not None:
                self.assertFloatsAlmostEqual(fit_temp_coeff, temp_coeff, rtol=2e-2)

        self._check_linearizer_lengths(linearizer)

    def test_linearity_spline(self):
        self._check_linearity_spline(do_pd_offsets=False, do_mu_offset=False)

    def test_linearity_spline_offsets(self):
        self._check_linearity_spline(do_pd_offsets=True, do_mu_offset=False)

    def test_linearity_spline_mu_offset(self):
        self._check_linearity_spline(do_pd_offsets=True, do_mu_offset=True)

    def test_linearity_spline_fit_weights(self):
        self._check_linearity_spline(do_pd_offsets=True, do_mu_offset=True, do_weight_fit=True)

    def test_linearity_spline_fit_temperature(self):
        self._check_linearity_spline(do_pd_offsets=True, do_mu_offset=True, do_temperature_fit=True)

    def test_linearity_spline_offsets_too_few_points(self):
        with self.assertRaisesRegex(RuntimeError, "too few points"):
            self._check_linearity_spline(do_pd_offsets=True, n_points=100)

    def test_linearity_turnoff(self):
        # Use some real LSSTComCam linearity data to measure the turnoff.
        abscissa, ordinate, ptc_mask = self._comcam_raw_linearity_data()

        config = LinearitySolveTask.ConfigClass()
        task = LinearitySolveTask(config=config)

        with self.assertNoLogs(level=logging.WARNING):
            turnoff_index, turnoff, max_signal, _ = _computeTurnoffAndMax(
                abscissa,
                ordinate,
                ptc_mask,
                np.zeros(len(abscissa)),
            )

        # This was visually inspected such that these are reasonable.
        self.assertEqual(turnoff_index, 90)
        np.testing.assert_almost_equal(turnoff, 99756.30512572)
        np.testing.assert_almost_equal(max_signal, 108730.32842316)

        # Do the linearity fit with these data.
        ptc = self._create_ptc(
            self.amp_names,
            abscissa * 1000000000,
            ordinate,
            photo_charges=abscissa,
            ptc_turnoff=np.max(ordinate[ptc_mask]),
        )
        config = LinearitySolveTask.ConfigClass()
        config.linearityType = "Spline"
        config.usePhotodiode = True
        config.minLinearAdu = 30.0
        config.splineKnots = 10
        config.doSplineFitOffset = False
        config.doSplineFitWeights = False
        config.splineFitWeightParsStart = [7.2e-5, 1e-4]
        config.doSplineFitTemperature = False

        task = LinearitySolveTask(config=config)
        linearizer = task.run(
            ptc,
            [self.dummy_exposure],
            self.camera,
            self.input_dims,
        ).outputLinearizer

        # Confirm that the turnoff for the good amps is the same.
        for amp_name in self.amp_names:
            if amp_name in ptc.badAmps:
                self.assertTrue(np.isnan(linearizer.linearityTurnoff[amp_name]))
                self.assertTrue(np.isnan(linearizer.linearityMaxSignal[amp_name]))
            else:
                self.assertEqual(linearizer.linearityTurnoff[amp_name], turnoff)
                self.assertEqual(linearizer.linearityMaxSignal[amp_name], max_signal)

                # Check that the linearizer gives reasonable values over the
                # range up to the ptc turnoff.
                nodes, values = np.split(linearizer.linearityCoeffs[amp_name], 2)
                self.assertEqual(values[0], 0.0)
                to_test = (nodes > 0.0) & (nodes < ptc.ptcTurnoff[amp_name])
                np.testing.assert_array_less(np.abs(values[to_test]/nodes[to_test]), 0.002)

                # Check the residuals are reasonable up to the linearity
                # turnoff.
                to_test = ((ptc.rawMeans[amp_name] <= linearizer.linearityTurnoff[amp_name])
                           & np.isfinite(ptc.rawMeans[amp_name]))
                residuals_scaled = linearizer.fitResiduals[amp_name][to_test]/ptc.rawMeans[amp_name][to_test]
                np.testing.assert_array_less(np.abs(residuals_scaled), 0.0015)

        # Try again after cutting it off, make sure it warns.
        cutoff = (ordinate < turnoff)

        with self.assertLogs(level=logging.INFO) as cm:
            turnoff_index2, turnoff2, max_signal2, _ = _computeTurnoffAndMax(
                abscissa[cutoff],
                ordinate[cutoff],
                ptc_mask[cutoff],
                np.zeros(len(abscissa))[cutoff],
            )
        self.assertIn("No linearity turnoff", cm.output[0])
        self.assertEqual(turnoff_index2, len(ptc_mask[cutoff]) - 1)

    def test_linearity_turnoff_lsstcam(self):
        # Use some real LSSTCam linearity data to measure the turnoff.
        exp_times, photo_charges, raw_means, ptc_mask = self._lsstcam_raw_linearity_data()

        ptc = PhotonTransferCurveDataset()
        ptc.ampNames = ["Amp1", "Amp2"]
        for amp_name in ptc.ampNames:
            ptc.inputExpIdPairs[amp_name] = np.zeros((len(exp_times), 2), dtype=np.int64).tolist()
            ptc.rawExpTimes[amp_name] = exp_times
            ptc.photoCharges[amp_name] = photo_charges
            ptc.rawMeans[amp_name] = raw_means
            ptc.expIdMask[amp_name] = ptc_mask
            ptc.gain[amp_name] = 1.0

        # Construct an auxiliary array by hand since we know these data.
        aux_arr = np.zeros(len(exp_times))
        grouping_values_truth = np.zeros(len(exp_times), dtype=np.int64)
        ratio = raw_means / exp_times
        low, = np.where(ratio < 100)
        grouping_values_truth[low] = 0
        aux_arr[low] = 0.15
        med, = np.where((ratio > 800) & (800 < 1000))
        grouping_values_truth[med] = 1
        aux_arr[med] = 0.5
        high, = np.where((ratio > 1200) | (exp_times > 70.0))
        grouping_values_truth[high] = 2
        aux_arr[high] = 0.8

        ptc.auxValues["AUX"] = aux_arr

        def _compare_grouping_values(arr_a, arr_b):
            self.assertEqual(len(arr_a), len(arr_b))
            u_a, ind_a = np.unique(arr_a, return_index=True)
            u_a = arr_a[np.sort(ind_a)]
            u_b, ind_b = np.unique(arr_b, return_index=True)
            u_b = arr_b[np.sort(ind_b)]
            self.assertEqual(len(u_a), len(u_b))
            for value_a, value_b in zip(u_a, u_b):
                np.testing.assert_array_equal(arr_a == value_a, arr_b == value_b)

        # First test: do grouping by aux value.
        config = LinearitySolveTask.ConfigClass()
        config.splineGroupingColumn = "AUX"
        task = LinearitySolveTask(config=config)

        grouping_values = task._determineInputGroups(ptc)
        _compare_grouping_values(grouping_values, grouping_values_truth)

        # Second test: do grouping automatically, with exp time.
        config = LinearitySolveTask.ConfigClass()
        config.doAutoGrouping = True
        task = LinearitySolveTask(config=config)

        grouping_values = task._determineInputGroups(ptc)
        _compare_grouping_values(grouping_values, grouping_values_truth)

        # Third test: do grouping automatically, with photodiode.
        config = LinearitySolveTask.ConfigClass()
        config.usePhotodiode = True
        config.doAutoGrouping = True
        config.autoGroupingUseExptime = False
        config.autoGroupingThreshold = 0.008
        config.minPhotodiodeCurrent = 1e-10
        task = LinearitySolveTask(config=config)

        # We have to modify here to allow the lowest point to be "ungrouped"
        # because it has an illegally low photocharge.
        grouping_values = task._determineInputGroups(ptc)
        grouping_values_truth_mod = grouping_values_truth.copy()
        grouping_values_truth_mod[0] = -1
        _compare_grouping_values(grouping_values, grouping_values_truth_mod)

    def test_linearity_fit_lsstcam(self):
        """
        Check that fitting the linearity works reasonably well for
        both the photodiode and exposure time.
        """
        exp_times, photo_charges, raw_means, ptc_mask = self._lsstcam_raw_linearity_data()

        # Cut off the fake data at the start and end.
        exp_times = exp_times[1: -2]
        photo_charges = photo_charges[1: -2]
        raw_means = raw_means[1: -2]
        ptc_mask = ptc_mask[1: -2]

        ptc = PhotonTransferCurveDataset()
        ptc.ampNames = self.amp_names
        for amp_name in ptc.ampNames:
            ptc.inputExpIdPairs[amp_name] = np.zeros((len(exp_times), 2), dtype=np.int64).tolist()
            ptc.rawExpTimes[amp_name] = exp_times
            ptc.photoCharges[amp_name] = photo_charges
            ptc.rawMeans[amp_name] = raw_means
            ptc.expIdMask[amp_name] = ptc_mask
            ptc.gain[amp_name] = 1.0
            ptc.ptcTurnoff[amp_name] = 75565.8
            ptc.inputExpPairMjdStartList[amp_name] = np.cumsum(exp_times)

        # Use the photodiode to solve for linearity.
        config = LinearitySolveTask.ConfigClass()
        config.usePhotodiode = True
        config.doAutoGrouping = True
        config.splineKnots = 5
        config.trimmedState = False
        config.linearityType = "Spline"
        config.doSplineFitWeights = False
        config.doSplineFitTemperature = False
        config.doSplineFitOffset = True
        config.splineFitMaxIter = 40
        config.splineGroupingMinPoints = 50
        task = LinearitySolveTask(config=config)

        linearizer_pd = task.run(ptc, [self.dummy_exposure], self.camera, self.input_dims).outputLinearizer

        # Use the exposure time to solve for linearity.
        config = LinearitySolveTask.ConfigClass()
        config.usePhotodiode = False
        config.doAutoGrouping = True
        config.splineKnots = 5
        config.trimmedState = False
        config.linearityType = "Spline"
        config.doSplineFitWeights = False
        config.doSplineFitTemperature = False
        config.doSplineFitOffset = True
        config.splineFitMaxIter = 40
        config.splineGroupingMinPoints = 50
        task = LinearitySolveTask(config=config)

        ptc2 = copy.deepcopy(ptc)
        for amp_name in self.amp_names:
            ptc2.photoCharges[amp_name][:] = 0.0

        linearizer_et = task.run(ptc2, [self.dummy_exposure], self.camera, self.input_dims).outputLinearizer

        # Check that the linearizer with the photodiode seems reasonable.
        amp_name = self.amp_names[0]
        nodes_pd, values_pd = np.split(linearizer_pd.linearityCoeffs[amp_name], 2)
        np.testing.assert_array_less(np.abs(values_pd), 100.0)
        in_range_pd = np.isfinite(linearizer_pd.fitResiduals[amp_name])
        np.testing.assert_array_less(
            np.abs(linearizer_pd.fitResiduals[amp_name][in_range_pd] / ptc.rawMeans[amp_name][in_range_pd]),
            0.004,
        )

        # Check that the linearizer with the exposure time seems reasonable.
        nodes_et, values_et = np.split(linearizer_et.linearityCoeffs[amp_name], 2)
        np.testing.assert_array_less(np.abs(values_et), 100.0)
        in_range_et = np.isfinite(linearizer_et.fitResiduals[amp_name])
        np.testing.assert_array_less(
            np.abs(linearizer_et.fitResiduals[amp_name][in_range_et] / ptc.rawMeans[amp_name][in_range_et]),
            0.004,
        )

        # Check consistency. This is very rough
        self.assertTrue(np.allclose(nodes_et, nodes_pd))
        self.assertTrue(np.allclose(values_et[1:] / nodes_et[1:], values_pd[1:] / nodes_pd[1:], atol=5e-4))

    def test_linearity_renormalization_lsstcam(self):
        # In this test we take the sample data, and set a bunch
        # of amps with the same values. After renormalization and
        # refitting there should be no residuals. (This is not a
        # recommended way of using the renormalization code).
        exp_times, photo_charges, raw_means, ptc_mask = self._lsstcam_raw_linearity_data()

        # Cut off the fake data at the start and end.
        exp_times = exp_times[1: -2]
        photo_charges = photo_charges[1: -2]
        raw_means = raw_means[1: -2]
        ptc_mask = ptc_mask[1: -2]

        ptc = PhotonTransferCurveDataset()
        ptc.ampNames = self.amp_names
        exp_ids = np.arange(len(exp_times)*2).reshape((len(exp_times), 2))
        for amp_name in ptc.ampNames:
            ptc.inputExpIdPairs[amp_name] = exp_ids.tolist()
            ptc.rawExpTimes[amp_name] = exp_times
            ptc.photoCharges[amp_name] = photo_charges
            ptc.rawMeans[amp_name] = raw_means
            ptc.expIdMask[amp_name] = ptc_mask
            ptc.gain[amp_name] = 1.0
            ptc.ptcTurnoff[amp_name] = 75565.8
            ptc.inputExpPairMjdStartList[amp_name] = np.cumsum(exp_times)

        # Use the exposure time to solve for linearity.
        config = LinearitySolveTask.ConfigClass()
        config.usePhotodiode = False
        config.doAutoGrouping = True
        config.splineKnots = 5
        config.trimmedState = False
        config.linearityType = "Spline"
        config.doSplineFitWeights = False
        config.doSplineFitTemperature = False
        config.doSplineFitOffset = True
        config.splineFitMaxIter = 40
        config.splineGroupingMinPoints = 50
        task = LinearitySolveTask(config=config)

        linearizer = task.run(ptc, [self.dummy_exposure], self.camera, self.input_dims).outputLinearizer

        # Now run the normalization task.
        nconfig = LinearityNormalizeTask.ConfigClass()
        nconfig.normalizeDetectors = [0]
        nconfig.referenceDetector = 0
        nconfig.doNormalizeAbsoluteLinearizer = False
        ntask = LinearityNormalizeTask(config=nconfig)

        ptc_handles = [
            InMemoryDatasetHandle(
                ptc,
                dataId={"detector": 0},
            ),
        ]
        linearizer_handles = [
            InMemoryDatasetHandle(
                linearizer,
                dataId={"detector": 0},
            ),
        ]

        normalization = ntask.run(
            camera=self.camera,
            inputPtcHandles=ptc_handles,
            inputLinearizerHandles=linearizer_handles,
        ).outputNormalization

        # And re-solve the linearity.
        config.useFocalPlaneNormalization = True
        task2 = LinearitySolveTask(config=config)

        linearizer2 = task2.run(
            ptc,
            [self.dummy_exposure],
            self.camera,
            self.input_dims,
            inputNormalization=normalization,
        ).outputLinearizer

        # Check for zero residuals.
        resids = linearizer2.fitResiduals[self.amp_names[0]]/linearizer2.fitResidualsModel[self.amp_names[0]]
        self.assertFloatsAlmostEqual(np.nan_to_num(resids), 0.0, atol=5e-6)

    def _comcam_raw_linearity_data(self):
        # These are LSSTComCam measurements taken from a calibration
        # run as part of DM-46357.
        photo_charges = np.array(
            [
                3.22636000e-10, 3.38780400e-10, 3.54841300e-10, 3.87126000e-10,
                4.03360000e-10, 4.35709800e-10, 4.51693200e-10, 4.83922500e-10,
                5.16297600e-10, 5.48525400e-10, 5.80748400e-10, 6.13082500e-10,
                6.45148000e-10, 6.77922000e-10, 7.25721750e-10, 7.74712800e-10,
                8.22586650e-10, 8.70876900e-10, 9.19831800e-10, 9.68568000e-10,
                1.03278400e-09, 1.08122255e-09, 1.14522645e-09, 1.22590280e-09,
                1.29087600e-09, 1.37134750e-09, 1.45214100e-09, 1.53323350e-09,
                1.62903910e-09, 1.72676065e-09, 1.83921900e-09, 1.93578600e-09,
                2.06456960e-09, 2.17876500e-09, 2.30696180e-09, 2.45230720e-09,
                2.59714735e-09, 2.74327300e-09, 2.91997345e-09, 3.08153670e-09,
                3.27340599e-09, 3.46962700e-09, 3.67778820e-09, 3.88937850e-09,
                4.12976640e-09, 4.37225980e-09, 4.62984095e-09, 4.90212160e-09,
                5.19730540e-09, 5.50027885e-09, 5.83860750e-09, 6.17903475e-09,
                6.54889207e-09, 6.93710400e-09, 7.35386640e-09, 7.79136960e-09,
                8.25815040e-09, 8.74769030e-09, 9.27673375e-09, 9.82420530e-09,
                1.04087197e-08, 1.10388366e-08, 1.17030225e-08, 1.23862272e-08,
                1.31366576e-08, 1.39122921e-08, 1.47334462e-08, 1.56154856e-08,
                1.65528171e-08, 1.75371688e-08, 1.85675291e-08, 1.96824430e-08,
                2.08605509e-08, 2.21063200e-08, 2.34186546e-08, 2.48260115e-08,
                2.62975235e-08, 2.78649723e-08, 2.95408665e-08, 3.12853772e-08,
                3.31612268e-08, 3.51149011e-08, 3.72152551e-08, 3.94439604e-08,
                4.17678940e-08, 4.42723820e-08, 4.69153456e-08, 4.96932949e-08,
                5.27096702e-08, 5.58084160e-08, 5.91300138e-08, 6.26865948e-08,
                6.64351212e-08, 7.03429300e-08, 7.45695391e-08, 7.89900791e-08,
                8.37413792e-08, 9.40025100e-08, 9.95942560e-08, 8.87289232e-08,
            ],
        )
        raw_means = np.array(
            [
                545.61773431, 572.92351724, 599.73600889, 654.80522687,
                681.89236561, 736.36468948, 763.35233159, 818.3558025,
                872.28757051, 926.97891772, 981.93722274, 1036.0641088,
                1091.15536336, 1145.34900099, 1227.09504139, 1308.5247877,
                1390.45828326, 1472.05043362, 1553.69250543, 1636.30579999,
                1744.80644944, 1826.53524631, 1936.27610568, 2072.63450198,
                2181.56004372, 2316.47487095, 2453.76647885, 2589.54660646,
                2754.19117479, 2916.91523806, 3107.95392897, 3272.65202763,
                3490.14569333, 3680.430583, 3896.17664859, 4146.03684447,
                4391.418256, 4633.96031196, 4935.13618128, 5208.9852533,
                5533.58336929, 5860.25295412, 6218.8609866, 6570.97901761,
                6977.63082612, 7386.3804372, 7823.50205095, 8291.10153869,
                8780.51405056, 9300.4324222, 9870.51463262, 10446.11494902,
                11069.51440385, 11733.55548663, 12437.83099445, 13179.9215084,
                13972.28308964, 14789.63243003, 15692.2466057, 16622.22691148,
                17592.14597643, 18670.98290396, 19778.68937623, 20960.06668402,
                22216.68121994, 23518.23906809, 24907.61374144, 26424.82837902,
                28002.38769651, 29681.8198974, 31422.27519355, 33285.31574829,
                35303.14661517, 37381.69611718, 39633.10759127, 41958.90250067,
                44458.19627488, 47100.85288959, 49902.33562771, 52837.68322699,
                55976.05887806, 59291.58601354, 62822.08965674, 66519.7625026,
                70480.78728045, 74721.82649122, 79220.00783343, 83928.39988665,
                88959.09479122, 94289.83758648, 99756.30512572, 104244.9446884,
                106938.47971957, 108134.21868844, 108602.08685706, 108702.48845118,
                108730.32842316, 108799.56402709, 108803.45364906, 108851.90458027,
            ],
        )

        ptc_mask = np.array(
            [
                True, True, True, True, True, True, True, True, True,
                True, True, True, True, True, True, True, True, True,
                True, True, True, True, True, True, True, True, True,
                True, True, True, True, True, True, True, True, True,
                True, True, True, True, True, True, True, True, True,
                True, True, True, True, True, True, True, True, True,
                True, True, True, True, True, True, True, True, True,
                True, True, True, True, True, True, True, True, True,
                True, True, True, True, True, True, True, True, True,
                True, True, True, True, True, True, True, False, False,
                False, False, False, False, False, False, False, False, False,
                False,
            ],
        )

        return photo_charges, raw_means, ptc_mask

    def _lsstcam_raw_linearity_data(self):
        # These are LSSTCam in-dome measurements taken from a calibration
        # run as part of DM-51470. Additional fake measurements have been
        # added that are below the photodiode limit or above the linearity
        # turnoff to test those aspects of the code.

        exp_times = np.array(
            [
                1.0,
                1.30007792, 2.60008502, 2.90007734, 3.20008755, 3.50007701,
                3.80058908, 4.10007262, 4.70008993, 6.00008655, 6.30008602,
                7.19907308, 7.50007725, 8.79909039, 9.10008216, 9.40107679,
                9.70009685, 10.9000566, 11.30008197, 11.90007639, 11.6015799,
                12.20007157, 12.80010366, 13.10108137, 13.70007062, 1.16707611,
                15.00008154, 15.30057025, 16.20007229, 1.3335712, 17.20008039,
                1.50008011, 18.4000895, 19.00007415, 21.20008397, 22.10008192,
                1.00008583, 1.00108552, 1.00058579, 24.00007939, 2.16806793,
                2.33307958, 1.32606268, 2.50008583, 2.66807985, 1.76707816,
                3.33358383, 3.50056648, 1.98808122, 3.83307695, 2.21007681,
                4.00059676, 4.16754985, 2.42907929, 4.66706753, 2.65107918,
                4.83408904, 5.00008345, 5.3330698, 5.50058341, 5.66706872,
                5.83259177, 3.31405592, 6.16708112, 3.53410554, 6.66809487,
                6.83407307, 3.97509313, 7.16708589, 4.19707441, 7.66709065,
                4.41806412, 8.50006843, 5.30106926, 9.83357477, 5.52107573,
                5.74307823, 10.33309817, 10.83407974, 11.50107312, 6.40508437,
                11.66708422, 11.83308768, 12.16760278, 6.84805465, 12.667593,
                7.06858993, 12.83307147, 7.50908542, 13.50007677, 13.83408427,
                14.00057459, 14.16807103, 7.95108128, 15.00008368, 15.00008249,
                15.00109196, 15.00109196, 15.00007319, 15.00058889, 15.00008273,
                15.00009346, 15.00007486, 15.00108337, 15.00109553, 15.00105524,
                15.00056362, 15.16708612, 15.33409524, 8.83507133, 9.05507565,
                16.33209324, 16.50058484, 16.66708279, 16.8340795, 17.33309412,
                9.71907902, 17.50007749, 17.6670754, 18.00008154, 18.16807771,
                10.38105488, 10.60207963, 19.16808653, 19.66707206, 20.1670773,
                20.66808057, 20.8330586, 11.7070713, 21.16708922, 21.33307409,
                11.92708254, 21.83359718, 22.00107551, 22.33308864, 12.59008217,
                22.83308482, 23.00007963, 12.8110795, 23.16708589, 23.33407331,
                13.03108048, 23.50059509, 13.47310114, 24.33407569, 13.69507217,
                24.66806412, 13.91508937, 25.00106311, 25.16805458, 25.33307934,
                25.50108433, 14.35706139, 26.167588, 14.57808304, 26.33309245,
                26.50007701, 26.8335743, 27.00106406, 15.24107933, 27.50059843,
                27.83407807, 28.001091, 28.16757345, 28.33307838, 28.50007868,
                15.90408254, 29.00007534, 16.124089, 29.16807365, 29.33310032,
                16.34508967, 29.50060129, 16.56608367, 30.00009513, 16.78757954,
                17.00807858, 17.45009923, 17.89208889, 18.11209965, 18.55409193,
                18.77606678, 19.21658659, 19.66007686, 20.54309058, 20.76208639,
                21.86608076, 22.30808902, 23.41357875, 23.63407302, 24.07609272,
                24.29708791, 24.51807809, 24.95957422, 25.18107915, 25.40207815,
                25.84308672, 26.0640893, 26.50607157, 26.72708106, 26.94707298,
                27.39009333, 27.83209324, 29.37858009, 30.04057312, 30.48257208,
                30.9245832, 31.14458227, 31.36508775, 31.58659363, 31.80709291,
                32.47107959, 32.91309094, 33.35208154, 34.01608729, 34.68008041,
                35.56258774, 36.4460969, 36.66707397, 36.88811016, 37.10908842,
                37.99308038, 38.65557122, 38.87708688, 39.75906563, 40.42207551,
                40.8640964, 41.30507493, 41.52558637, 42.18909407, 42.41057515,
                42.63108134, 43.07208395, 43.29357743, 43.51308179, 43.73508239,
                45.2810781, 45.50158954, 45.72306991, 46.1655736, 46.3850894,
                46.60510993, 47.04907584, 47.27006769, 47.48958015, 47.93157744,
                48.37308311, 48.59308457, 48.81608295, 49.47805619, 49.69907236,
                50.1410749, 50.5805645, 51.02406359, 51.68808389, 52.56959724,
                53.01307774, 53.67507911, 54.77957559, 55.00056076, 80.0, 90.0])

        photo_charges = np.array(
            [
                1e-11,
                1.82817260e-09, 3.53148649e-09, 4.06277898e-09, 4.35869544e-09,
                4.78330872e-09, 5.22915865e-09, 5.78226268e-09, 6.48815270e-09,
                8.25712380e-09, 8.51615745e-09, 1.02197526e-08, 1.05099569e-08,
                1.22837309e-08, 1.28114263e-08, 1.30927807e-08, 1.33760842e-08,
                1.53687893e-08, 1.56255057e-08, 1.61675487e-08, 1.64177802e-08,
                1.68976483e-08, 1.75646793e-08, 1.81330449e-08, 1.87132138e-08,
                2.91637877e-08, 2.07739826e-08, 2.18182545e-08, 2.19715436e-08,
                3.32051817e-08, 2.36684166e-08, 3.72810404e-08, 2.53565939e-08,
                2.59972840e-08, 2.96734901e-08, 3.02007963e-08, 4.55297856e-08,
                4.55746585e-08, 4.56246682e-08, 3.37802654e-08, 5.42455205e-08,
                5.81404431e-08, 6.04291412e-08, 6.24474643e-08, 6.64574931e-08,
                8.04680210e-08, 8.31232983e-08, 8.72382377e-08, 9.05033796e-08,
                9.55119486e-08, 1.00627749e-07, 9.97876553e-08, 1.04031649e-07,
                1.10573123e-07, 1.16371623e-07, 1.20965846e-07, 1.20569281e-07,
                1.24911511e-07, 1.32978879e-07, 1.37649727e-07, 1.41153092e-07,
                1.45213041e-07, 1.50986279e-07, 1.54133358e-07, 1.60948534e-07,
                1.66481769e-07, 1.70432685e-07, 1.81069871e-07, 1.78186815e-07,
                1.91063717e-07, 1.91086417e-07, 2.01389150e-07, 2.11345064e-07,
                2.41348459e-07, 2.45181656e-07, 2.51154663e-07, 2.61397147e-07,
                2.56946642e-07, 2.69402724e-07, 2.86605066e-07, 2.91289314e-07,
                2.90809466e-07, 2.95092192e-07, 3.03471554e-07, 3.11797168e-07,
                3.16873891e-07, 3.22205728e-07, 3.19714411e-07, 3.41500913e-07,
                3.36233190e-07, 3.44508625e-07, 3.49027053e-07, 3.53546367e-07,
                3.62263658e-07, 3.73349576e-07, 3.73605323e-07, 3.73355323e-07,
                3.73424493e-07, 3.73690267e-07, 3.74063761e-07, 3.73328827e-07,
                3.73513232e-07, 3.74292450e-07, 3.74058846e-07, 3.74199516e-07,
                3.73805125e-07, 3.74801019e-07, 3.77549794e-07, 3.82109671e-07,
                4.02813124e-07, 4.12015519e-07, 4.06590291e-07, 4.11174216e-07,
                4.15620699e-07, 4.20346548e-07, 4.31620648e-07, 4.42191673e-07,
                4.36834459e-07, 4.40553709e-07, 4.48188921e-07, 4.52683265e-07,
                4.72329423e-07, 4.82356891e-07, 4.77541263e-07, 4.90793892e-07,
                5.04259245e-07, 5.15785320e-07, 5.20004090e-07, 5.32583692e-07,
                5.27353277e-07, 5.31060399e-07, 5.43193778e-07, 5.44255280e-07,
                5.47986412e-07, 5.56257861e-07, 5.73223115e-07, 5.69229732e-07,
                5.73483192e-07, 5.84122304e-07, 5.78339239e-07, 5.80922371e-07,
                5.93536614e-07, 5.85343924e-07, 6.13633727e-07, 6.07300226e-07,
                6.23672682e-07, 6.14832531e-07, 6.33686470e-07, 6.24144708e-07,
                6.26834243e-07, 6.31470139e-07, 6.35002684e-07, 6.54419975e-07,
                6.52114981e-07, 6.63381949e-07, 6.56440208e-07, 6.63513673e-07,
                6.69092777e-07, 6.73678035e-07, 6.95448734e-07, 6.86136092e-07,
                6.94088711e-07, 6.97076014e-07, 7.02235128e-07, 7.06743404e-07,
                7.12541252e-07, 7.23700469e-07, 7.22175554e-07, 7.34407902e-07,
                7.26219718e-07, 7.30123718e-07, 7.43971514e-07, 7.36034974e-07,
                7.54263790e-07, 7.48980267e-07, 7.65759441e-07, 7.75726865e-07,
                7.94317280e-07, 8.14849394e-07, 8.24539049e-07, 8.44726672e-07,
                8.54993598e-07, 8.74385137e-07, 8.95097268e-07, 9.34897394e-07,
                9.44723938e-07, 9.96059623e-07, 1.01658226e-06, 1.06624643e-06,
                1.07704428e-06, 1.09541320e-06, 1.10813448e-06, 1.11542569e-06,
                1.13651858e-06, 1.14503830e-06, 1.15640469e-06, 1.17643071e-06,
                1.18651568e-06, 1.20725029e-06, 1.21732323e-06, 1.22769579e-06,
                1.24728449e-06, 1.26572090e-06, 1.33844128e-06, 1.36705783e-06,
                1.38812903e-06, 1.40766463e-06, 1.41726953e-06, 1.42920091e-06,
                1.43961190e-06, 1.44785313e-06, 1.47873846e-06, 1.49792958e-06,
                1.51710029e-06, 1.55136912e-06, 1.57902098e-06, 1.62106226e-06,
                1.65860885e-06, 1.66904934e-06, 1.68223145e-06, 1.68977232e-06,
                1.72980750e-06, 1.75960946e-06, 1.76905651e-06, 1.80953552e-06,
                1.83924183e-06, 1.86149564e-06, 1.88067160e-06, 1.88919825e-06,
                1.92234685e-06, 1.93096053e-06, 1.94121685e-06, 1.95972805e-06,
                1.97182165e-06, 1.98003698e-06, 1.99059056e-06, 2.06132937e-06,
                2.07290637e-06, 2.08140970e-06, 2.10187884e-06, 2.11122988e-06,
                2.12235208e-06, 2.14279142e-06, 2.15065476e-06, 2.16372892e-06,
                2.18410909e-06, 2.20113980e-06, 2.21079153e-06, 2.22210751e-06,
                2.25089289e-06, 2.26316028e-06, 2.28389811e-06, 2.30118685e-06,
                2.32201405e-06, 2.35181662e-06, 2.39407598e-06, 2.41469194e-06,
                2.44405239e-06, 2.49260152e-06, 2.50544044e-06, 3.64427700e-06,
                4.09981162e-06,
            ],
        )

        raw_means = np.array(
            [
                70.0,
                92.86846201, 182.20115781, 210.17843191, 224.75373652,
                249.41990014, 270.3750848, 298.13356004, 335.2491762,
                419.85418654, 440.20870837, 523.35053806, 533.95229561,
                624.33520762, 654.19741753, 663.41872989, 691.00929737,
                781.82840105, 797.53228322, 829.48856791, 829.77401764,
                869.06926893, 910.33500907, 924.56998291, 961.33369162,
                1028.74632532, 1080.23191006, 1115.72725114, 1145.29298889,
                1171.88485425, 1215.27614848, 1316.63175163, 1319.90794786,
                1329.20039219, 1513.53524709, 1544.58715074, 1577.72509245,
                1580.42422803, 1582.08035824, 1730.68368453, 1909.06875316,
                2051.60147106, 2095.65486231, 2200.50081135, 2346.93082746,
                2790.61021016, 2933.16186159, 3077.82799132, 3136.72492514,
                3371.89652643, 3491.51091289, 3520.01335135, 3670.82553208,
                3838.56300847, 4102.6067899, 4195.75368619, 4247.1381654,
                4403.42219284, 4690.74436914, 4847.29352829, 4976.48877486,
                5120.97366735, 5240.94946909, 5433.09472747, 5590.18821641,
                5873.91979779, 6013.31482841, 6283.71507483, 6297.6863956,
                6631.55638028, 6747.12360371, 6992.05043094, 7473.70409509,
                8383.09997091, 8651.84396759, 8724.92350749, 9081.20376784,
                9088.49929752, 9526.30681728, 10113.24561528, 10125.09736262,
                10275.02009773, 10413.30226, 10716.08347878, 10831.62417085,
                11174.97345771, 11187.10433017, 11297.94913483, 11874.65293352,
                11877.8316926, 12174.65186236, 12321.831863, 12476.93862506,
                12595.30191395, 13183.70211442, 13186.68422581, 13187.2122269,
                13191.52699678, 13196.79290796, 13204.41871041, 13205.11971342,
                13210.91337662, 13213.64928761, 13215.36094497, 13215.79616817,
                13217.20001814, 13228.6272711, 13341.71183545, 13492.87662206,
                13983.88641078, 14320.43230657, 14365.05001829, 14521.47003941,
                14676.59292711, 14836.45065264, 15256.72039757, 15383.54413924,
                15429.46336546, 15552.75261379, 15829.4470338, 15980.36094398,
                16422.11284161, 16773.36075251, 16873.76417919, 17347.93050569,
                17780.51617957, 18213.68136006, 18370.11017442, 18520.97641651,
                18626.94414553, 18774.53387747, 18872.50742297, 19220.11365054,
                19371.40480846, 19669.37342574, 19918.48776394, 20119.78165825,
                20260.25387413, 20297.01090622, 20420.94501903, 20540.65397603,
                20647.36826825, 20695.08978438, 21326.45152597, 21450.48862807,
                21693.06777509, 21746.54361219, 22045.21577418, 22057.42176734,
                22169.88708478, 22322.21832224, 22461.40321731, 22752.13471875,
                23049.46907456, 23089.19566699, 23176.810312, 23435.02471521,
                23659.65784493, 23804.24440484, 24177.15459574, 24239.82590366,
                24538.08527224, 24653.42322741, 24834.63478483, 24968.23972901,
                25166.10351596, 25177.0618337, 25535.14004216, 25536.46094104,
                25672.96722423, 25819.18373405, 25881.30357988, 26032.17886968,
                26236.72558439, 26474.25109961, 26624.66379018, 26976.051577,
                27641.68356156, 28354.41545864, 28684.89234527, 29401.04773183,
                29749.42931579, 30424.52582892, 31124.41787207, 32531.15694883,
                32876.38317561, 34657.47503525, 35358.55517438, 37065.5660748,
                37453.44032424, 38121.45997741, 38516.02714121, 38806.56101841,
                39499.85472998, 39834.64346431, 40243.25907892, 40942.40514966,
                41279.92463902, 41977.41767446, 42332.91963243, 42700.34604663,
                43372.00824829, 44037.57319312, 46538.15651649, 47533.00991698,
                48252.54686708, 48926.33644579, 49278.174449, 49662.66006463,
                49986.82486074, 50323.45507686, 51397.80697509, 52070.1823656,
                52704.70512386, 53869.17336706, 54765.71986416, 56274.31527709,
                57552.6631826, 57973.25092965, 58357.98782968, 58660.00904752,
                60045.31679453, 61092.97983081, 61429.86712469, 62836.09931594,
                63824.97166359, 64603.41083149, 65266.46358084, 65563.45761106,
                66707.66379327, 66974.76300745, 67353.24759166, 68038.21337638,
                68400.62102102, 68744.27455369, 69097.6239373, 71540.01858064,
                71916.28448469, 72279.98815642, 72939.45375775, 73289.46074577,
                73692.17736103, 74419.50733063, 74659.92625471, 75082.30090212,
                75797.60997538, 76437.89738215, 76797.10336102, 77171.44260012,
                78130.37592659, 78568.4843529, 79279.27081724, 79917.26494941,
                80621.12449606, 81643.09849749, 83131.19278591, 83798.73997235,
                84860.43286242, 86483.42983764, 86958.02382474, 100000.0,
                100000.0,
            ],
        )

        ptc_mask = np.array(
            [
                False,
                False, True, True, True, True, True, True, True, True,
                True, True, True, True, True, True, True, True, True,
                True, True, True, True, True, True, True, True, True,
                True, True, True, True, True, True, True, True, True,
                True, True, True, True, True, True, True, True, True,
                True, True, True, True, True, True, True, True, True,
                True, True, True, True, True, True, True, True, True,
                True, True, True, True, True, True, True, True, True,
                True, True, True, True, True, True, True, True, True,
                True, True, True, True, True, True, True, True, True,
                True, True, True, True, True, True, True, True, True,
                True, True, True, True, True, True, True, True, True,
                True, True, True, True, True, True, True, True, True,
                True, True, True, True, True, True, True, True, True,
                True, True, True, True, True, True, True, True, True,
                True, True, True, True, True, True, True, True, True,
                True, True, True, True, True, True, True, True, True,
                True, True, True, True, True, True, True, True, True,
                True, True, True, True, True, True, True, True, True,
                True, True, True, True, True, True, True, True, True,
                True, True, True, True, True, True, True, True, True,
                True, True, True, True, True, True, True, True, True,
                True, True, True, True, True, True, True, True, True,
                True, True, True, True, True, True, True, True, True,
                True, True, True, True, True, True, True, True, True,
                True, True, True, True, True, True, True, True, True,
                True, True, False, False, False, False, False, False, False,
                False, False, False, False, False, False, False, False, False,
                False, False, False, False, False, False, False, False, False,
            ],
        )

        return exp_times, photo_charges, raw_means, ptc_mask


class DoubleSplineLinearityTestCase(lsst.utils.tests.TestCase):
    def setUp(self):
        mock = IsrMockLSST()
        self.camera = mock.getCamera()
        self.detector = self.camera[20]
        self.detector_id = self.detector.getId()

    def _compute_logistic_nonlinearity(self, xvals, midpoint, amplitude, transition=3000.0):
        """Compute a simple non-linearity with a logistic curve.

        Parameters
        ----------
        xvals : `np.ndarray`
            Input count values.
        midpoint : `float`
            Transition point for logistic curve.
        amplitude : `float`
            Fractional amplitude of logistic curve.
        transition : `float`, optional
            Transition value (sharpness).

        Returns
        -------
        offsets : `np.ndarray`
            Offset count values.
        """
        frac_offset = amplitude / 2. - (amplitude / (1. + np.exp(-(1./transition)*(xvals - midpoint))))

        return xvals * frac_offset

    def test_linearity_doublespline(self):
        n_pair = 100
        pair_sigma = 0.005  # Fractional variation.

        rng = np.random.RandomState(seed=12345)

        amp_names = [amp.getName() for amp in self.detector]
        n_amps = len(amp_names)

        rel_amplitudes = rng.uniform(low=0.001, high=0.003, size=n_amps)
        rel_midpoints = rng.uniform(low=20000.0, high=50000.0, size=n_amps)
        linearity_turnoffs = rng.uniform(low=90000.0, high=100000.0, size=n_amps)

        noises = rng.uniform(low=5.0, high=10.0, size=n_amps)
        gains = rng.uniform(low=1.4, high=1.6, size=n_amps)
        a00s = rng.uniform(low=-2e-6, high=-4e-6, size=n_amps)
        ptc_turnoffs = rng.uniform(low=70000.0, high=80000.0, size=n_amps)

        ref_amp_index = np.argmax(linearity_turnoffs)
        ref_amp_name = amp_names[ref_amp_index]

        # Make sure this one is significantly higher for consistency.
        linearity_turnoffs[ref_amp_index] *= 1.1

        # Reset the relative linearity for the reference amp.
        rel_amplitudes[ref_amp_index] = 0.0

        # Create an absolute linearizer.
        abs_amplitude = -0.001
        abs_midpoint = 10000.0

        range_e = np.asarray([50.0, linearity_turnoffs[ref_amp_index]]) * gains[ref_amp_index]

        pair_levels_e = np.linspace(range_e[0], range_e[1], n_pair)

        ptc_extract_config = PhotonTransferCurveExtractPairTask.ConfigClass()
        ptc_extract_config.doOutputBinnedImages = True
        ptc_extract_config.maximumRangeCovariancesAstier = 1
        ptc_extract_config.minNumberGoodPixelsForCovariance = 100
        ptc_extract_config.numEdgeSuspect = 4
        ptc_extract_config.edgeMaskLevel = "AMP"
        ptc_extract_config.doGain = False

        ptc_extract_task = PhotonTransferCurveExtractPairTask(config=ptc_extract_config)

        pair_handles = []
        binned_handles = []

        normalization_exposures = []
        normalization_values = []

        for i in range(n_pair):
            # We generate two images with the same level for
            # simpler testing.
            # The level is done in electrons then we apply gain,
            # and compute the variance from the Astier function.

            levels_e = pair_levels_e[i] + rng.normal(loc=0.0, scale=pair_sigma * pair_levels_e[i], size=2)
            # levels_e = pair_levels_e[i] + np.array([0.0, 0.0])
            exptime = np.mean(levels_e) / 10.0

            # normalization_exposures.extend([i * 2

            flat_pair = []
            for j in range(2):
                flat = lsst.afw.image.ExposureF(self.detector.getBBox())
                flat.setDetector(self.detector)
                visit_info = lsst.afw.image.VisitInfo(
                    exposureTime=exptime,
                    date=DateTime("2025-09-25T00:00:00", DateTime.TAI),
                )
                flat.info.id = i * 2 + j
                flat.info.setVisitInfo(visit_info)
                flat_pair.append(flat)

                normalization_exposures.append(i * 2 + j)
                normalization_values.append((levels_e[j] / 10.) / exptime)

            for j in range(2):
                flat = flat_pair[j]
                # Compute the variance for all the amps at this level.
                var_adu = funcAstier([a00s, gains, noises**2.], levels_e[j] / gains)

                # Adjust the variance for those above the ptc turnoff
                var_adu[(levels_e[j] / gains) > ptc_turnoffs] *= 0.5

                # Build the flat.
                for k, amp in enumerate(self.detector):

                    # Offset things above the linearizer turnoff.
                    # I don't know if this will actually work ...
                    level_e = levels_e[j]
                    if level_e > linearity_turnoffs[k] * gains[k]:
                        level_e *= 0.8
                        var_adu[k] *= 0.8

                    noise_key = f"LSST ISR OVERSCAN RESIDUAL SERIAL STDEV {amp.getName()}"
                    flat.metadata[noise_key] = noises[k] / gains[k]
                    median_key = f"LSST ISR OVERSCAN SERIAL MEDIAN {amp.getName()}"
                    flat.metadata[median_key] = 0.0
                    pedestal_key = f"LSST ISR AMPOFFSET PEDESTAL {amp.getName()}"
                    flat.metadata[pedestal_key] = 0.0

                    bbox = amp.getBBox()
                    flat[bbox].image.array[:, :] = rng.normal(
                        loc=level_e,
                        scale=np.sqrt(var_adu[k]) * gains[k],
                        size=flat[bbox].image.array.shape,
                    ) / gains[k]

                    # Apply the non-linearity... first the absolute.
                    abs_offset = self._compute_logistic_nonlinearity(
                        flat[bbox].image.array,
                        abs_midpoint,
                        abs_amplitude,
                    )
                    flat[bbox].image.array[:, :] += abs_offset

                    # And then the relative.
                    rel_offset = self._compute_logistic_nonlinearity(
                        flat[bbox].image.array,
                        rel_midpoints[k],
                        rel_amplitudes[k],
                    )
                    flat[bbox].image.array[:, :] += rel_offset

            # Run the PTC extraction on the pair and store the rebinned images.
            # How long does this take?  Memory?

            handles = [
                InMemoryDatasetHandle(
                    flat,
                    dataId={"exposure": flat.info.id, "detector": flat.getDetector().getId()},
                )
                for flat in flat_pair
            ]
            inputDims = [flat.info.id for flat in flat_pair]
            results = ptc_extract_task.run(inputExp=handles, inputDims=inputDims)

            data_id = {"exposure": flat_pair[0].info.id, "detector": flat_pair[0].getDetector().getId()}
            pair_handles.append(
                InMemoryDatasetHandle(
                    results.outputCovariance,
                    dataId=data_id,
                )
            )
            binned_handles.append(
                InMemoryDatasetHandle(
                    results.outputBinnedImages,
                    dataId=data_id,
                )
            )

        normalization = Table({"exposure": normalization_exposures, "normalization": normalization_values})

        # Build the linearizer PTC.
        ptc_solve_config = PhotonTransferCurveSolveTask.ConfigClass()
        ptc_solve_config.ptcFitType = "EXPAPPROXIMATION"
        ptc_solve_config.maximumRangeCovariancesAstier = 1
        ptc_solve_config.maximumRangeCovariancesAstierFullCovFit = 1
        ptc_solve_config.ksTestMinPvalue = 0.0
        # This is a large number because these small amplifiers are noisy.
        ptc_solve_config.sigmaCutPtcOutliers = 20.0

        ptc_solve_task = PhotonTransferCurveSolveTask(config=ptc_solve_config)
        input_covariances = [handle.get() for handle in pair_handles]
        results = ptc_solve_task.run(input_covariances, camera=self.camera, detId=self.detector_id)
        linearizer_ptc = results.outputPtcDataset

        # Now we run the linearize solving code.

        linearity_solve_config = LinearityDoubleSplineSolveTask.ConfigClass()
        linearity_solve_config.relativeSplineMinimumSignalNode = 0.0
        linearity_solve_config.relativeSplineLowThreshold = 0.0
        linearity_solve_config.relativeSplineMidNodeSize = 10000.0
        linearity_solve_config.relativeSplineHighNodeSize = 10000.0
        linearity_solve_config.absoluteSplineNodeSize = 10000.0
        linearity_solve_config.useFocalPlaneNormalization = True
        linearity_solve_task = LinearityDoubleSplineSolveTask(config=linearity_solve_config)

        results = linearity_solve_task.run(
            inputPtc=linearizer_ptc,
            camera=self.camera,
            inputBinnedImagesHandles=binned_handles,
            inputNormalization=normalization,
        )

        linearizer = results.outputLinearizer

        # Check that we chose the correct reference amplifier.
        self.assertEqual(linearizer.absoluteReferenceAmplifier, ref_amp_name)

        # Check the relative residuals.
        for amp_name in linearizer.ampNames:
            if amp_name == ref_amp_name:
                continue

            frac_resid = linearizer.fitResiduals[amp_name] / linearizer.inputOrdinate[amp_name]

            self.assertFloatsAlmostEqual(np.nan_to_num(frac_resid), 0.0, atol=7e-4)

        # Check the absolute residuals.
        abs_frac_resid = linearizer.fitResiduals[ref_amp_name] / linearizer.inputOrdinate[ref_amp_name]
        self.assertFloatsAlmostEqual(np.nan_to_num(abs_frac_resid), 0.0, atol=5e-4)

        def _check_spline_midpoint(coeffs, expected_midpoint, turnoff, tolerance=6000.0):
            centers, values = np.split(coeffs, 2)

            xvals = np.linspace(1.0, centers[-1], 1000)
            spl = Akima1DInterpolator(centers, values, method="akima")
            yvals = spl(xvals) / xvals

            grad = np.gradient(yvals, xvals)
            # This is the point of maximum gradient, which should match
            # the input midpoints.
            max_grad = xvals[np.argmax(np.abs(grad))]

            self.assertFloatsAlmostEqual(max_grad, expected_midpoint, atol=tolerance)

            self.assertFloatsAlmostEqual(centers[-1], turnoff, atol=1000.0)

        # Check that the linearizers are consistent with expectations.
        coeffs = linearizer.linearityCoeffs[ref_amp_name]
        n_nodes1 = int(coeffs[0])
        n_nodes2 = int(coeffs[1])
        self.assertEqual(n_nodes1, 0)
        abs_coeff = coeffs[2 + 2 * n_nodes1: 2 + 2 * n_nodes1 + 2 * n_nodes2]

        _check_spline_midpoint(abs_coeff, abs_midpoint, linearity_turnoffs[ref_amp_index], tolerance=4000.0)

        for i, amp_name in enumerate(linearizer.ampNames):
            if amp_name == ref_amp_name:
                continue

            coeffs = linearizer.linearityCoeffs[amp_name]

            n_nodes1 = int(coeffs[0])
            n_nodes2 = int(coeffs[1])

            spline_coeff1 = coeffs[2: 2 + 2 * n_nodes1]

            _check_spline_midpoint(spline_coeff1, rel_midpoints[i], linearity_turnoffs[i])

            # Confirm that the absolute parameters are matched.
            spline_coeff2 = coeffs[2 + 2 * n_nodes1: 2 + 2 * n_nodes1 + 2 * n_nodes2]
            self.assertFloatsAlmostEqual(spline_coeff2, abs_coeff)

    def test_noderator(self):
        # Test "regular" usage.
        low = 5000.0
        mid = 70000.0
        high = 100000.0
        nodes = _noderator(low, mid, high, 50.0, 750.0, 5000.0, 2000.0)

        self.assertEqual(nodes[0], 0.0)
        self.assertEqual(nodes[1], 50.0)
        self.assertEqual(nodes[-1], high)
        self.assertTrue(np.any(nodes == low))
        self.assertTrue(np.any(nodes == mid))
        self.assertTrue(np.all(np.arange(len(nodes)) == np.argsort(nodes)))
        self.assertGreater(nodes[2] - nodes[1], 750.0)
        self.assertGreater(nodes[10] - nodes[9], 5000.0)
        self.assertGreater(nodes[-1] - nodes[-2], 2000.0)

        # Test no "low turnoff"
        low = 0.0
        nodes = _noderator(low, mid, high, 0.0, 750.0, 5000.0, 2000.0)
        self.assertEqual(nodes[0], 0.0)
        self.assertEqual(nodes[-1], high)
        self.assertTrue(np.any(nodes == mid))
        self.assertGreater(nodes[1] - nodes[0], 5000.0)
        self.assertGreater(nodes[-1] - nodes[-2], 2000.0)

        # Test no "high turnoff"
        low = 5000.0
        high = mid - 1.0
        nodes = _noderator(low, mid, high, 50.0, 750.0, 5000.0, 2000.0)
        self.assertEqual(nodes[0], 0.0)
        self.assertEqual(nodes[1], 50.0)
        self.assertEqual(nodes[-1], mid)
        self.assertTrue(np.any(nodes == low))
        self.assertTrue(np.any(nodes == mid))
        self.assertTrue(np.all(np.arange(len(nodes)) == np.argsort(nodes)))
        self.assertGreater(nodes[2] - nodes[1], 750.0)
        self.assertGreater(nodes[10] - nodes[9], 5000.0)
        self.assertGreater(nodes[-1] - nodes[-2], 5000.0)


class TestMemory(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
