#!/usr/bin/env python

#
# LSST Data Management System
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
"""Test cases for cp_pipe pipelines."""

import glob
import os
import unittest

from lsst.pipe.base import Pipeline, PipelineGraph
import lsst.utils

try:
    import lsst.obs.lsst
    has_obs_lsst = True
except ImportError:
    has_obs_lsst = False

try:
    import lsst.obs.subaru
    has_obs_subaru = True
except ImportError:
    has_obs_subaru = False

try:
    import lsst.obs.decam
    has_obs_decam = True
except ImportError:
    has_obs_decam = False


class CalibrationPipelinesTestCase(lsst.utils.tests.TestCase):
    """Test case for building the pipelines."""

    def setUp(self):
        self.pipeline_path = os.path.join(lsst.utils.getPackageDir("cp_pipe"), "pipelines")

    def _get_pipelines(self, exclude=[]):
        pipelines = {
            "cpBfk.yaml",
            "cpBias.yaml",
            "cpCrosstalk.yaml",
            "cpCti.yaml",
            "cpDarkForDefects.yaml",
            "cpDark.yaml",
            "cpDefectsIndividual.yaml",
            "cpDefects.yaml",
            "cpFilterScan.yaml",
            "cpFlatSingleChip.yaml",
            "cpFlat.yaml",
            "cpFringe.yaml",
            "cpLinearizer.yaml",
            "cpMonochromatorScan.yaml",
            "cpPlotPtc.yaml",
            "cpPtc.yaml",
            "cpSky.yaml",
        }

        for ex in exclude:
            pipelines.remove(ex)

        return pipelines

    def _check_pipeline(self, pipeline_file):
        # Confirm that the file is there.
        self.assertTrue(os.path.isfile(pipeline_file), msg=f"Could not find {pipeline_file}")

        # The following loads the pipeline and confirms that it can parse all
        # the configs.
        try:
            pipeline = Pipeline.fromFile(pipeline_file)
            graph = pipeline.to_graph()
        except Exception as e:
            raise RuntimeError(f"Could not process {pipeline_file}") from e

        self.assertIsInstance(graph, PipelineGraph)

    def test_ingredients(self):
        """Check that all pipelines in pipelines/_ingredients are tested."""
        glob_str = os.path.join(self.pipeline_path, "_ingredients", "*.yaml")
        ingredients = set(
            [os.path.basename(pipeline) for pipeline in glob.glob(glob_str)]
        )
        expected = self._get_pipelines()
        self.assertEqual(ingredients, expected)

    def test_cameras(self):
        """Check that all the cameras in pipelines are tested."""
        glob_str = os.path.join(self.pipeline_path, "*")
        paths = set(
            [os.path.basename(path) for path in glob.glob(glob_str)]
        )
        expected = {
            "DECam",
            "HSC",
            "_ingredients",
            "LATISS",
            "LSSTCam",
            "LSSTCam-imSim",
            "LSSTComCam",
            "LSSTComCamSim",
            "LSST-TS8",
            "README.md",
        }
        self.assertEqual(paths, expected)

    @unittest.skipIf(not has_obs_lsst, reason="Cannot test LATISS pipelines without obs_lsst")
    def test_latiss_pipelines(self):
        for pipeline in self._get_pipelines(exclude=["cpMonochromatorScan.yaml"]):
            self._check_pipeline(os.path.join(self.pipeline_path, "LATISS", pipeline))

    @unittest.skipIf(not has_obs_lsst, reason="Cannot test LSSTCam pipelines without obs_lsst")
    def test_lsstcam_pipelines(self):
        for pipeline in self._get_pipelines(exclude=[
                "cpDarkForDefects.yaml",
                "cpFilterScan.yaml",
                "cpMonochromatorScan.yaml"
        ]):
            self._check_pipeline(os.path.join(self.pipeline_path, "LSSTCam", pipeline))

    @unittest.skipIf(not has_obs_lsst, reason="Cannot test LSSTCam-imSim pipelines without obs_lsst")
    def test_lsstcam_imsim_pipelines(self):
        for pipeline in self._get_pipelines(exclude=[
                "cpDarkForDefects.yaml",
                "cpFilterScan.yaml",
                "cpMonochromatorScan.yaml"
        ]):
            self._check_pipeline(os.path.join(self.pipeline_path, "LSSTCam-imSim", pipeline))

    @unittest.skipIf(not has_obs_lsst, reason="Cannot test LSSTComCam pipelines without obs_lsst")
    def test_lsstcomcam_pipelines(self):
        for pipeline in self._get_pipelines(exclude=[
                "cpDarkForDefects.yaml",
                "cpFilterScan.yaml",
                "cpMonochromatorScan.yaml"
        ]):
            self._check_pipeline(os.path.join(self.pipeline_path, "LSSTComCam", pipeline))

    @unittest.skipIf(not has_obs_lsst, reason="Cannot test LSSTComCamSim pipelines without obs_lsst")
    def test_lsstcomcamsim_pipelines(self):
        for pipeline in self._get_pipelines(exclude=[
                "cpDarkForDefects.yaml",
                "cpFilterScan.yaml",
                "cpMonochromatorScan.yaml"
        ]):
            self._check_pipeline(os.path.join(self.pipeline_path, "LSSTComCamSim", pipeline))

    @unittest.skipIf(not has_obs_lsst, reason="Cannot test LSST-TS8 pipelines without obs_lsst")
    def test_lsst_ts8_pipelines(self):
        for pipeline in self._get_pipelines(exclude=[
                "cpDarkForDefects.yaml",
                "cpFilterScan.yaml",
                "cpMonochromatorScan.yaml"
        ]):
            self._check_pipeline(os.path.join(self.pipeline_path, "LSST-TS8", pipeline))

    @unittest.skipIf(not has_obs_decam, reason="Cannot test DECam pipelines without obs_decam")
    def test_decam_pipelines(self):
        for pipeline in self._get_pipelines(exclude=[
                "cpDarkForDefects.yaml",
                "cpFilterScan.yaml",
                "cpMonochromatorScan.yaml"
        ]):
            self._check_pipeline(os.path.join(self.pipeline_path, "DECam", pipeline))

    @unittest.skipIf(not has_obs_subaru, reason="Cannot test HSC pipelines without obs_subaru")
    def test_hsc_pipelines(self):
        for pipeline in self._get_pipelines(exclude=[
                "cpDarkForDefects.yaml",
                "cpFilterScan.yaml",
                "cpMonochromatorScan.yaml"
        ]):
            self._check_pipeline(os.path.join(self.pipeline_path, "HSC", pipeline))


class TestMemory(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
