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

import unittest
import numpy as np
# import copy
# import logging

import lsst.utils
import lsst.utils.tests

from lsst.ip.isr import PhotonTransferCurveDataset

from lsst.cp.pipe import LinearitySolveTask
from lsst.ip.isr.isrMock import FlatMock, IsrMock


class FakeCamera(list):
    def getName(self):
        return "FakeCam"


class LinearityTaskTestCase(lsst.utils.tests.TestCase):
    """Test case for the linearity tasks."""

    def test_linearity_polynomial(self):
        """Test linearity with polynomial fit."""

        mock_image_config = IsrMock.ConfigClass()
        mock_image_config.flatDrop = 0.99999
        mock_image_config.isTrimmed = True

        dummy_exposure = FlatMock(config=mock_image_config).run()
        detector = dummy_exposure.getDetector()
        input_dims = {"detector": 0}

        camera = FakeCamera([detector])

        ampNames = []
        for amp in detector:
            ampNames.append(amp.getName())

        ptc = PhotonTransferCurveDataset(ampNames=ampNames)

        flux = 1000.
        time_vec = np.arange(1., 101., 5)
        k2_non_linearity = -5e-6
        mu_vec = flux * time_vec + k2_non_linearity * time_vec**2.

        for ampName in ptc.ampNames:
            ptc.expIdMask[ampName] = np.ones(len(time_vec), dtype=bool)
            ptc.inputExpIdPairs[ampName] = np.arange(len(time_vec)*2).reshape((len(time_vec), 2)).tolist()
            ptc.rawExpTimes[ampName] = time_vec
            ptc.rawMeans[ampName] = mu_vec

        config = LinearitySolveTask.ConfigClass()
        config.linearityType = "Polynomial"

        task = LinearitySolveTask(config=config)
        linearizer = task.run(ptc, [dummy_exposure], camera, input_dims).outputLinearizer

        coeff = k2_non_linearity/(flux**2.)

        for ampName in ptc.ampNames:
            # ampName = amp.getName()
            self.assertAlmostEqual(0.0, linearizer.fitParams[ampName][0], places=3)
            self.assertAlmostEqual(1.0, linearizer.fitParams[ampName][1], places=5)
            self.assertAlmostEqual(coeff, linearizer.fitParams[ampName][2], places=6)

            self.assertAlmostEqual(0.0, linearizer.linearityCoeffs[ampName][0], places=3)
            self.assertAlmostEqual(-1.0, linearizer.linearityCoeffs[ampName][1], places=5)
            self.assertAlmostEqual(-coeff, linearizer.linearityCoeffs[ampName][2], places=6)


class TestMemory(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
