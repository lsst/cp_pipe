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
"""Test cases for lsst.cp.pipe.BrighterFatterKernelSolveTask.
"""

import unittest
import numpy as np

import lsst.utils
import lsst.utils.tests

import lsst.ip.isr as ipIsr
import lsst.cp.pipe as cpPipe
import lsst.afw.cameraGeom as cameraGeom


class BfkSolveTaskTestCase(lsst.utils.tests.TestCase):
    """A test case for the brighter fatter kernel solver.
    """

    def setUp(self):
        """Set up a plausible PTC dataset, with 1% of the expected variance
        shifted into covariance terms.
        """
        cameraBuilder = cameraGeom.Camera.Builder('fake camera')
        detectorWrapper = cameraGeom.testUtils.DetectorWrapper(numAmps=1, cameraBuilder=cameraBuilder)
        self.detector = detectorWrapper.detector
        self.camera = cameraBuilder.finish()

        self.defaultConfig = cpPipe.BrighterFatterKernelSolveConfig()
        self.ptc = ipIsr.PhotonTransferCurveDataset(ampNames=['amp 1'], ptcFitType='FULLCOVARIANCE',
                                                    covMatrixSide=3)
        self.ptc.expIdMask['amp 1'] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        self.ptc.finalMeans['amp 1'] = [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]
        self.ptc.rawMeans['amp 1'] = self.ptc.finalMeans['amp 1']
        self.ptc.finalVars['amp 1'] = 0.99 * np.array(self.ptc.finalMeans['amp 1'], dtype=float)
        self.ptc.covariances['amp 1'] = []
        for mean, variance in zip(self.ptc.finalMeans['amp 1'], self.ptc.finalVars['amp 1']):
            residual = mean - variance
            covariance = [[variance, 0.5 * residual, 0.1 * residual],
                          [0.2 * residual, 0.1 * residual, 0.05 * residual],
                          [0.025 * residual, 0.015 * residual, 0.01 * residual]]
            self.ptc.covariances['amp 1'].append(covariance)

        self.ptc.gain['amp 1'] = 1.0
        self.ptc.noise['amp 1'] = 5.0

        # This is empirically determined from the above parameters.
        self.ptc.aMatrix['amp 1'] = np.array([[2.14329806e-06, -4.28659612e-07, -5.35824515e-08],
                                              [-1.07164903e-06, -2.14329806e-07, -3.21494709e-08],
                                              [-2.14329806e-07, -1.07164903e-07, -2.14329806e-08]])

        # This is empirically determined from the above parameters.
        self.expectation = np.array([[4.88348887e-08, 1.01136877e-07, 1.51784114e-07,
                                      1.77570668e-07, 1.51784114e-07, 1.01136877e-07, 4.88348887e-08],
                                     [9.42026776e-08, 2.03928507e-07, 3.28428909e-07,
                                      4.06714446e-07, 3.28428909e-07, 2.03928507e-07, 9.42026776e-08],
                                     [1.24047315e-07, 2.70512582e-07, 4.44123665e-07,
                                      5.78099493e-07, 4.44123665e-07, 2.70512582e-07, 1.24047315e-07],
                                     [1.31474000e-07, 2.77801372e-07, 3.85123870e-07,
                                      -5.42128333e-08, 3.85123870e-07, 2.77801372e-07, 1.31474000e-07],
                                     [1.24047315e-07, 2.70512582e-07, 4.44123665e-07,
                                      5.78099493e-07, 4.44123665e-07, 2.70512582e-07, 1.24047315e-07],
                                     [9.42026776e-08, 2.03928507e-07, 3.28428909e-07,
                                      4.06714446e-07, 3.28428909e-07, 2.03928507e-07, 9.42026776e-08],
                                     [4.88348887e-08, 1.01136877e-07, 1.51784114e-07,
                                      1.77570668e-07, 1.51784114e-07, 1.01136877e-07, 4.88348887e-08]])

    def test_averaged(self):
        """Test "averaged" brighter-fatter kernel.
        """
        task = cpPipe.BrighterFatterKernelSolveTask()

        results = task.run(self.ptc, ['this is a dummy exposure'], self.camera, {'detector': 1})
        self.assertFloatsAlmostEqual(results.outputBFK.ampKernels['amp 1'], self.expectation, atol=1e-5)

    def test_aMatrix(self):
        """Test solution from Astier et al. 2019 "A" matrix
        """
        config = cpPipe.BrighterFatterKernelSolveConfig()
        config.useAmatrix = True
        task = cpPipe.BrighterFatterKernelSolveTask(config=config)

        results = task.run(self.ptc, ['this is a dummy exposure'], self.camera, {'detector': 1})
        self.assertFloatsAlmostEqual(results.outputBFK.ampKernels['amp 1'], self.expectation, atol=1e-5)

    def test_quadratic(self):
        """Test quadratic correlation solver.

        This requires a different model for the variance, so cannot
        use the one generated by setUp.  This model is not entirely
        physical, but will ensure that accidental code changes are
        detected.
        """
        config = cpPipe.BrighterFatterKernelSolveConfig()
        config.correlationQuadraticFit = True
        config.forceZeroSum = True
        task = cpPipe.BrighterFatterKernelSolveTask(config=config)

        ptc = ipIsr.PhotonTransferCurveDataset(ampNames=['amp 1'], ptcFitType='FULLCOVARIANCE',
                                               covMatrixSide=3)
        ptc.expIdMask['amp 1'] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        ptc.finalMeans['amp 1'] = [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]
        ptc.rawMeans['amp 1'] = ptc.finalMeans['amp 1']
        ptc.finalVars['amp 1'] = 9e-5 * np.square(np.array(ptc.finalMeans['amp 1'], dtype=float))
        ptc.covariances['amp 1'] = []
        for mean, variance in zip(ptc.finalMeans['amp 1'], ptc.finalVars['amp 1']):
            residual = variance
            covariance = [[variance, 0.5 * residual, 0.1 * residual],
                          [0.2 * residual, 0.1 * residual, 0.05 * residual],
                          [0.025 * residual, 0.015 * residual, 0.01 * residual]]
            ptc.covariances['amp 1'].append(covariance)

        ptc.gain['amp 1'] = 1.0
        ptc.noise['amp 1'] = 5.0

        results = task.run(ptc, ['this is a dummy exposure'], self.camera, {'detector': 1})

        expectation = np.array([[4.05330882e-08, 2.26654412e-07, 5.66636029e-07, 7.56066176e-07,
                                 5.66636029e-07, 2.26654412e-07, 4.05330882e-08],
                                [-6.45220588e-08, 2.99448529e-07, 1.28382353e-06, 1.89099265e-06,
                                 1.28382353e-06, 2.99448529e-07, -6.45220588e-08],
                                [-5.98069853e-07, -1.14816176e-06, -2.12178309e-06, -4.75974265e-06,
                                 -2.12178309e-06, -1.14816176e-06, -5.98069853e-07],
                                [-1.17959559e-06, -3.52224265e-06, -1.28630515e-05, -6.16863971e-05,
                                 -1.28630515e-05, -3.52224265e-06, -1.17959559e-06],
                                [-5.98069853e-07, -1.14816176e-06, -2.12178309e-06, -4.75974265e-06,
                                 -2.12178309e-06, -1.14816176e-06, -5.98069853e-07],
                                [-6.45220588e-08, 2.99448529e-07, 1.28382353e-06, 1.89099265e-06,
                                 1.28382353e-06, 2.99448529e-07, -6.45220588e-08],
                                [4.05330882e-08, 2.26654412e-07, 5.66636029e-07, 7.56066176e-07,
                                 5.66636029e-07, 2.26654412e-07, 4.05330882e-08]])
        self.assertFloatsAlmostEqual(results.outputBFK.ampKernels['amp 1'], expectation, atol=1e-5)


class TestMemory(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
