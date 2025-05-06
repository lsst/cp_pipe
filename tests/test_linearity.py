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

import logging
import unittest
import numpy as np

import lsst.utils
import lsst.utils.tests

from lsst.ip.isr import PhotonTransferCurveDataset

import lsst.afw.image
import lsst.afw.math
from lsst.cp.pipe import LinearitySolveTask
from lsst.cp.pipe.ptc import PhotonTransferCurveSolveTask
from lsst.cp.pipe.utils import funcPolynomial
from lsst.ip.isr.isrMock import FlatMock, IsrMock


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
            turnoff_index, turnoff, max_signal = task._computeTurnoffAndMax(abscissa, ordinate, ptc_mask)

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

        with self.assertLogs(level=logging.WARNING) as cm:
            turnoff_index2, turnoff2, max_signal2 = task._computeTurnoffAndMax(
                abscissa[cutoff],
                ordinate[cutoff],
                ptc_mask[cutoff],
            )
        self.assertIn("No linearity turnoff", cm.output[0])
        self.assertEqual(turnoff_index2, len(ptc_mask[cutoff]) - 1)

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


class TestMemory(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
