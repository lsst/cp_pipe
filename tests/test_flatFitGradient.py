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
"""Test cases for cp_pipe flat gradient code."""

import unittest
import numpy as np
from scipy.interpolate import Akima1DInterpolator

import lsst.utils.tests

import lsst.afw.cameraGeom
from lsst.afw.image import ExposureF
from lsst.cp.pipe import CpFlatFitGradientsTask, CpFlatApplyGradientsTask
from lsst.ip.isr import IsrMockLSST, Defects, FlatGradient
from lsst.pipe.base import InMemoryDatasetHandle


class FlatFitGradientTestCase(lsst.utils.tests.TestCase):
    def setUp(self):
        mock = IsrMockLSST()
        initial_camera = mock.getCamera()
        camera_builder = initial_camera.rebuild()
        for counter, detector in enumerate(camera_builder):
            detector.setType(lsst.afw.cameraGeom.DetectorType.SCIENCE)
            if counter < 2:
                detector.setPhysicalType("pseudoITL")
            else:
                detector.setPhysicalType("pseudoE2V")
        self.camera = camera_builder.finish()

        self.filter_label = lsst.afw.image.FilterLabel(band="i", physical="i_0")

    def _get_flat_handle_dict(
        self,
        radial_nodes,
        radial_values,
        normalization,
        itl_ratio=1.0,
        delta_x=0.0,
        delta_y=0.0,
        gradient_x=0.0,
        gradient_y=0.0,
        outer_gradient_x=0.0,
        outer_gradient_y=0.0,
        outer_gradient_radius=np.inf,
    ):
        spl = Akima1DInterpolator(radial_nodes, radial_values, method="akima")

        flat_handle_dict = {}
        for detector in self.camera:
            flat = ExposureF(detector.getBBox())

            flat.setDetector(detector)
            flat.setFilter(self.filter_label)

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

            centroid_x = 0.0 + delta_x
            centroid_y = 0.0 + delta_y
            radius = np.sqrt((xf - centroid_x)**2. + (yf - centroid_y)**2.)
            value = spl(np.clip(radius, radial_nodes[0], radial_nodes[-1]))

            gradient = 1 + gradient_x*(xf - 0.0) + gradient_y*(yf - 0.0)
            value /= gradient

            if np.isfinite(outer_gradient_radius):
                fp_radius = np.sqrt(xf**2. + yf**2.)
                outer = (fp_radius > outer_gradient_radius)
                outer_gradient = 1 + outer_gradient_x*(xf - 0.0) + outer_gradient_y*(yf - 0.0)
                value[outer] /= outer_gradient[outer]

            flat.image.array[:, :] = value.reshape(flat.image.array.shape) * normalization

            if "ITL" in detector.getPhysicalType():
                flat.image.array[:, :] *= itl_ratio

            data_id = {
                "detector": detector.getId(),
                "physical_filter": None,
            }
            flat_handle_dict[detector.getId()] = InMemoryDatasetHandle(flat, dataId=data_id)

        return flat_handle_dict

    def _get_defect_handle_dict(self):
        defect_handle_dict = {}
        for detector in self.camera:
            defects = Defects()
            defects.append(lsst.geom.Box2I(lsst.geom.Point2I(0, 0), lsst.geom.Point2I(7, 7)))

            data_id = {
                "detector": detector.getId(),
            }
            defect_handle_dict[detector.getId()] = InMemoryDatasetHandle(defects, dataId=data_id)

        return defect_handle_dict

    def test_radial_only(self):
        radial_nodes = np.array([0, 1, 2, 3, 4, 4.5, 5.2], dtype=np.float64)
        radial_values = np.array([1.0, 1.0, 1.0, 0.9, 0.5, 0.3, 0.0], dtype=np.float64)
        normalization = 1.1

        flat_handle_dict = self._get_flat_handle_dict(radial_nodes, radial_values, normalization)
        defect_handle_dict = self._get_defect_handle_dict()

        config = CpFlatFitGradientsTask.ConfigClass()
        config.bin_factor = 4  # Small detectors for the test.
        config.normalize_center_radius = 1.0
        config.outer_gradient_radius = 4.5
        config.radial_spline_nodes = radial_nodes.tolist()
        config.detector_boundary = 5
        config.do_normalize_center = True
        config.do_fit_centroid = False
        config.do_fit_gradient = False
        config.do_fit_outer_gradient = False
        config.do_normalize_center = True

        task = CpFlatFitGradientsTask(config=config)
        gradient = task.run(
            camera=self.camera,
            input_flat_handle_dict=flat_handle_dict,
            input_defect_handle_dict=defect_handle_dict,
        ).output_gradient

        self.assertFloatsAlmostEqual(gradient.normalizationFactor, normalization, rtol=1e-7)
        self.assertFloatsAlmostEqual(gradient.radialSplineNodes, radial_nodes)
        self.assertFloatsAlmostEqual(gradient.radialSplineValues, radial_values, atol=1e-3)

    def test_radial_only_nozero(self):
        radial_nodes = np.array([0, 1, 2, 3, 4, 4.5, 5.2], dtype=np.float64)
        radial_values = np.array([1.0, 1.0, 1.0, 0.9, 0.5, 0.3, 0.0], dtype=np.float64)
        normalization = 1.1

        flat_handle_dict = self._get_flat_handle_dict(radial_nodes, radial_values, normalization)
        defect_handle_dict = self._get_defect_handle_dict()

        config = CpFlatFitGradientsTask.ConfigClass()
        config.bin_factor = 4  # Small detectors for the test.
        config.normalize_center_radius = 1.0
        config.outer_gradient_radius = 4.5
        config.radial_spline_nodes = radial_nodes.tolist()
        config.detector_boundary = 5
        config.do_constrain_zero = False
        config.do_normalize_center = True
        config.do_fit_centroid = False
        config.do_fit_gradient = False
        config.do_fit_outer_gradient = False
        config.do_normalize_center = True

        task = CpFlatFitGradientsTask(config=config)
        gradient = task.run(
            camera=self.camera,
            input_flat_handle_dict=flat_handle_dict,
            input_defect_handle_dict=defect_handle_dict,
        ).output_gradient

        self.assertFloatsAlmostEqual(gradient.normalizationFactor, normalization, rtol=1e-7)
        self.assertFloatsAlmostEqual(gradient.radialSplineNodes, radial_nodes)
        self.assertFloatsAlmostEqual(gradient.radialSplineValues[: -1], radial_values[: -1], atol=1e-3)
        self.assertFloatsAlmostEqual(gradient.radialSplineValues[-1], radial_values[-1], atol=0.20)

    def test_radial_centroid(self):
        radial_nodes = np.array([0, 1, 2, 3, 4, 4.5, 5.2], dtype=np.float64)
        radial_values = np.array([1.0, 1.0, 1.0, 0.9, 0.5, 0.3, 0.0], dtype=np.float64)
        normalization = 1.1

        delta_x = 0.01
        delta_y = -0.01

        flat_handle_dict = self._get_flat_handle_dict(
            radial_nodes,
            radial_values,
            normalization,
            delta_x=delta_x,
            delta_y=delta_y,
        )
        defect_handle_dict = self._get_defect_handle_dict()

        config = CpFlatFitGradientsTask.ConfigClass()
        config.bin_factor = 4  # Small detectors for the test.
        config.normalize_center_radius = 1.0
        config.outer_gradient_radius = 4.5
        config.radial_spline_nodes = radial_nodes.tolist()
        config.detector_boundary = 5
        config.do_normalize_center = True
        config.do_fit_centroid = True
        config.do_fit_gradient = False
        config.do_fit_outer_gradient = False
        config.do_normalize_center = True

        task = CpFlatFitGradientsTask(config=config)
        gradient = task.run(
            camera=self.camera,
            input_flat_handle_dict=flat_handle_dict,
            input_defect_handle_dict=defect_handle_dict,
        ).output_gradient

        self.assertFloatsAlmostEqual(gradient.normalizationFactor, normalization, rtol=1e-7)
        self.assertFloatsAlmostEqual(gradient.radialSplineNodes, radial_nodes)
        self.assertFloatsAlmostEqual(gradient.radialSplineValues, radial_values, atol=1e-3)
        self.assertFloatsAlmostEqual(gradient.centroidDeltaX, delta_x, atol=6e-3)
        self.assertFloatsAlmostEqual(gradient.centroidDeltaY, delta_y, atol=6e-3)

    def test_radial_plane(self):
        radial_nodes = np.array([0, 1, 2, 3, 4, 4.5, 5.2], dtype=np.float64)
        radial_values = np.array([1.0, 1.0, 1.0, 0.9, 0.5, 0.3, 0.0], dtype=np.float64)
        normalization = 1.1

        gradient_x = 0.01
        gradient_y = -0.01

        flat_handle_dict = self._get_flat_handle_dict(
            radial_nodes,
            radial_values,
            normalization,
            gradient_x=gradient_x,
            gradient_y=gradient_y,
        )
        defect_handle_dict = self._get_defect_handle_dict()

        config = CpFlatFitGradientsTask.ConfigClass()
        config.bin_factor = 4  # Small detectors for the test.
        config.normalize_center_radius = 1.0
        config.outer_gradient_radius = 4.5
        config.radial_spline_nodes = radial_nodes.tolist()
        config.detector_boundary = 5
        config.do_normalize_center = True
        config.do_fit_centroid = False
        config.do_fit_gradient = True
        config.do_fit_outer_gradient = False
        config.do_normalize_center = True

        task = CpFlatFitGradientsTask(config=config)
        gradient = task.run(
            camera=self.camera,
            input_flat_handle_dict=flat_handle_dict,
            input_defect_handle_dict=defect_handle_dict,
        ).output_gradient

        self.assertFloatsAlmostEqual(gradient.normalizationFactor, normalization, rtol=1e-2)
        self.assertFloatsAlmostEqual(gradient.radialSplineNodes, radial_nodes)
        self.assertFloatsAlmostEqual(gradient.radialSplineValues, radial_values, atol=2e-3)
        self.assertFloatsAlmostEqual(gradient.gradientX, gradient_x, atol=1e-4)
        self.assertFloatsAlmostEqual(gradient.gradientY, gradient_y, atol=1e-4)

    def test_radial_planes(self):
        radial_nodes = np.array([0, 1, 2, 3, 4, 4.5, 5.2], dtype=np.float64)
        radial_values = np.array([1.0, 1.0, 1.0, 0.9, 0.5, 0.3, 0.0], dtype=np.float64)
        normalization = 1.1

        gradient_x = 0.01
        gradient_y = -0.01
        outer_gradient_x = -0.005
        outer_gradient_y = 0.005

        flat_handle_dict = self._get_flat_handle_dict(
            radial_nodes,
            radial_values,
            normalization,
            gradient_x=gradient_x,
            gradient_y=gradient_y,
            outer_gradient_x=outer_gradient_x,
            outer_gradient_y=outer_gradient_y,
            outer_gradient_radius=4.5,
        )
        defect_handle_dict = self._get_defect_handle_dict()

        config = CpFlatFitGradientsTask.ConfigClass()
        config.bin_factor = 4  # Small detectors for the test.
        config.normalize_center_radius = 1.0
        config.outer_gradient_radius = 4.5
        config.radial_spline_nodes = radial_nodes.tolist()
        config.detector_boundary = 5
        config.do_normalize_center = True
        config.do_fit_centroid = False
        config.do_fit_gradient = True
        config.do_fit_outer_gradient = True
        config.do_normalize_center = True

        task = CpFlatFitGradientsTask(config=config)
        gradient = task.run(
            camera=self.camera,
            input_flat_handle_dict=flat_handle_dict,
            input_defect_handle_dict=defect_handle_dict,
        ).output_gradient

        self.assertFloatsAlmostEqual(gradient.normalizationFactor, normalization, rtol=1e-2)
        self.assertFloatsAlmostEqual(gradient.radialSplineNodes, radial_nodes)
        self.assertFloatsAlmostEqual(gradient.radialSplineValues, radial_values, atol=5e-3)
        self.assertFloatsAlmostEqual(gradient.gradientX, gradient_x, atol=1e-4)
        self.assertFloatsAlmostEqual(gradient.gradientY, gradient_y, atol=1e-4)
        self.assertFloatsAlmostEqual(gradient.outerGradientX, outer_gradient_x, atol=5e-3)
        self.assertFloatsAlmostEqual(gradient.outerGradientY, outer_gradient_y, atol=5e-3)

    def test_radial_centroid_planes(self):
        radial_nodes = np.array([0, 1, 2, 3, 4, 4.5, 5.2], dtype=np.float64)
        radial_values = np.array([1.0, 1.0, 1.0, 0.9, 0.5, 0.3, 0.0], dtype=np.float64)
        normalization = 1.1

        itl_ratio = 0.9
        gradient_x = 0.01
        gradient_y = -0.01
        outer_gradient_x = -0.005
        outer_gradient_y = 0.005
        delta_x = 0.01
        delta_y = -0.01

        flat_handle_dict = self._get_flat_handle_dict(
            radial_nodes,
            radial_values,
            normalization,
            itl_ratio=itl_ratio,
            delta_x=delta_x,
            delta_y=delta_y,
            gradient_x=gradient_x,
            gradient_y=gradient_y,
            outer_gradient_x=outer_gradient_x,
            outer_gradient_y=outer_gradient_y,
            outer_gradient_radius=4.5,
        )
        defect_handle_dict = self._get_defect_handle_dict()

        config = CpFlatFitGradientsTask.ConfigClass()
        config.bin_factor = 4  # Small detectors for the test.
        config.normalize_center_radius = 1.0
        config.outer_gradient_radius = 4.5
        config.radial_spline_nodes = radial_nodes.tolist()
        config.detector_boundary = 5
        config.do_normalize_center = True
        config.do_fit_centroid = True
        config.do_fit_gradient = True
        config.do_fit_outer_gradient = True
        config.do_normalize_center = True

        task = CpFlatFitGradientsTask(config=config)
        gradient = task.run(
            camera=self.camera,
            input_flat_handle_dict=flat_handle_dict,
            input_defect_handle_dict=defect_handle_dict,
        ).output_gradient

        self.assertFloatsAlmostEqual(gradient.normalizationFactor, normalization, rtol=1e-2)
        self.assertFloatsAlmostEqual(gradient.radialSplineNodes, radial_nodes)
        self.assertFloatsAlmostEqual(gradient.radialSplineValues, radial_values, atol=2e-3)
        self.assertFloatsAlmostEqual(gradient.itlRatio, itl_ratio, atol=1e-4)
        self.assertFloatsAlmostEqual(gradient.centroidDeltaX, delta_x, atol=6e-3)
        self.assertFloatsAlmostEqual(gradient.centroidDeltaY, delta_y, atol=6e-3)
        self.assertFloatsAlmostEqual(gradient.gradientX, gradient_x, atol=1e-4)
        self.assertFloatsAlmostEqual(gradient.gradientY, gradient_y, atol=1e-4)
        self.assertFloatsAlmostEqual(gradient.outerGradientX, outer_gradient_x, atol=5e-3)
        self.assertFloatsAlmostEqual(gradient.outerGradientY, outer_gradient_y, atol=5e-3)

    def test_apply(self):
        # This will create source and target; no fitting.

        # Create the "sky flat" target vignetting.
        radial_nodes_sky = np.array([0, 1, 2, 3, 4, 4.5, 5.2], dtype=np.float64)
        radial_values_sky = np.array([1.0, 1.0, 1.0, 1.0, 0.8, 0.5, 0.0], dtype=np.float64)
        normalization_sky = 1.1

        sky_gradient = FlatGradient()
        sky_gradient.setParameters(
            radialSplineNodes=radial_nodes_sky,
            radialSplineValues=radial_values_sky,
            normalizationFactor=normalization_sky,
        )

        # Create the "dome flat" source vignetting.
        radial_nodes_dome = np.array([0, 1, 2, 3, 4, 4.5, 5.2], dtype=np.float64)
        radial_values_dome = np.array([1.0, 1.0, 1.0, 0.9, 0.5, 0.3, 0.0], dtype=np.float64)
        normalization_dome = 1.15
        itl_ratio = 0.9
        gradient_x = 0.01
        gradient_y = -0.01
        outer_gradient_x = -0.005
        outer_gradient_y = 0.005
        delta_x = 0.01
        delta_y = -0.01

        dome_gradient = FlatGradient()
        dome_gradient.setParameters(
            radialSplineNodes=radial_nodes_dome,
            radialSplineValues=radial_values_dome,
            normalizationFactor=normalization_dome,
            itlRatio=itl_ratio,
            gradientX=gradient_x,
            gradientY=gradient_y,
            outerGradientX=outer_gradient_x,
            outerGradientY=outer_gradient_y,
            outerGradientRadius=4.5,
            centroidDeltaX=delta_x,
            centroidDeltaY=delta_y,
        )

        dome_flat_handles = self._get_flat_handle_dict(
            radial_nodes_dome,
            radial_values_dome,
            normalization_dome,
            itl_ratio=itl_ratio,
            delta_x=delta_x,
            delta_y=delta_y,
            gradient_x=gradient_x,
            gradient_y=gradient_y,
            outer_gradient_x=outer_gradient_x,
            outer_gradient_y=outer_gradient_y,
            outer_gradient_radius=4.5,
        )

        config = CpFlatApplyGradientsTask.ConfigClass()
        task = CpFlatApplyGradientsTask(config=config)

        corrected_dome_flat_handles = {}
        for key, handle in dome_flat_handles.items():
            struct = task.run(
                camera=self.camera,
                input_flat=handle.get(),
                reference_gradient=sky_gradient,
                gradient=dome_gradient,
            )
            corrected_dome_flat_handles[key] = InMemoryDatasetHandle(struct.output_flat, dataId=handle.dataId)

        # Now if we fit the corrected handles they should have the
        # same radial structure as the target, with no centroid or
        # gradient, and normalization 1.0.
        defect_handle_dict = self._get_defect_handle_dict()

        config = CpFlatFitGradientsTask.ConfigClass()
        config.bin_factor = 4  # Small detectors for the test.
        config.normalize_center_radius = 1.0
        config.outer_gradient_radius = 4.5
        config.radial_spline_nodes = radial_nodes_sky.tolist()
        config.detector_boundary = 5
        config.do_normalize_center = True
        config.do_fit_centroid = True
        config.do_fit_gradient = True
        config.do_fit_outer_gradient = True
        config.do_normalize_center = True

        task = CpFlatFitGradientsTask(config=config)
        gradient = task.run(
            camera=self.camera,
            input_flat_handle_dict=corrected_dome_flat_handles,
            input_defect_handle_dict=defect_handle_dict,
        ).output_gradient

        self.assertFloatsAlmostEqual(gradient.normalizationFactor, 1.0, rtol=1e-3)
        self.assertFloatsAlmostEqual(gradient.radialSplineNodes, radial_nodes_sky)
        self.assertFloatsAlmostEqual(gradient.radialSplineValues, radial_values_sky, atol=5e-4)
        self.assertFloatsAlmostEqual(gradient.itlRatio, itl_ratio, atol=1e-4)
        self.assertFloatsAlmostEqual(gradient.centroidDeltaX, 0.0, atol=6e-3)
        self.assertFloatsAlmostEqual(gradient.centroidDeltaY, 0.0, atol=6e-3)
        self.assertFloatsAlmostEqual(gradient.gradientX, 0.0, atol=1e-7)
        self.assertFloatsAlmostEqual(gradient.gradientY, 0.0, atol=1e-7)
        self.assertFloatsAlmostEqual(gradient.outerGradientX, 0.0, atol=5e-3)
        self.assertFloatsAlmostEqual(gradient.outerGradientY, 0.0, atol=5e-3)


class TestMemory(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
