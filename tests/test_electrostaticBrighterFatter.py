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
# see:
#   https://www.lsstcorp.org/LegalNotices/
#
"""Test cases for lsst.cp.pipe.ElectrostaticBrighterFatterSolveTask.
"""

import unittest
import numpy as np

import lsst.utils
import lsst.utils.tests

import lsst.ip.isr as ipIsr
import lsst.cp.pipe as cpPipe
import lsst.afw.cameraGeom as cameraGeom


class ElectrostaticBrighterFatterSolveTaskTestCase(lsst.utils.tests.TestCase):
    """ A test case for the electrostatic brighter-fatter solver.
    """

    def setUp(self):
        """ Set up an empirical PTC dataset.
        """
        cameraBuilder = cameraGeom.Camera.Builder('fake camera')
        detectorWrapper = cameraGeom.testUtils.DetectorWrapper(
            numAmps=16, cameraBuilder=cameraBuilder,
        )

        self.detector = detectorWrapper.detector

        self.camera = cameraBuilder.finish()

        self.defaultConfig = cpPipe.BrighterFatterKernelSolveConfig()
        ampNames = [f"amp {i}" for i in range(16)]

        self.ptc = ipIsr.PhotonTransferCurveDataset(
            ampNames=ampNames, ptcFitType='FULLCOVARIANCE',
            covMatrixSide=5,
        )

        self.ptc.badAmps = [ampNames[3]]  # Randomly set one amp as bad

        # These comes from the empirical average a matrix from
        # LSSTCam/calib/DM-51897 for detector R22_S11
        self.aMatrixMean = np.array([
            [-3.29036958e-06, 4.16003921e-07, 6.19957768e-08,
             1.73248403e-08, 1.15894578e-08],
            [1.75050697e-07, 1.25755064e-07, 4.21671079e-08,
             1.70534397e-08, 6.97588077e-09],
            [5.90596810e-08, 3.85312617e-08, 2.25833070e-08,
             9.42166010e-09, 8.42694139e-09],
            [1.95655870e-08, 1.74141350e-08, 1.09739771e-08,
             8.40992104e-09, 3.83679509e-09],
            [1.10096895e-08, 7.17968238e-09, 7.99943349e-09,
             2.64435974e-09, 4.21998095e-09],
        ])

        self.aMatrixSigma = np.array([
            [43.61315413, 5.25852337, 3.49826154, 3.25020952,
             3.70877012],
            [6.24446168, 2.80054151, 2.36052852, 3.03517635,
             2.75386129],
            [4.52060910, 2.87046950, 2.81298062, 3.23891077,
             3.48839977],
            [4.78699906, 3.56665023, 2.49834086, 2.93183221,
             2.58406590],
            [3.08975509, 2.39008599, 2.76165328, 2.83526835,
             3.14183071],
        ]) * 1e-9

        rawMeans = np.array([
            1000, 2000, 3000, 4000, 5000,
            6000, 7000, 8000, 9000, 10000
        ], dtype=np.float64)

        rawVars = np.array([
            708.54575502, 1398.11082581, 2081.09149706,
            2757.56675448, 3427.61457982, 4091.31195080,
            4748.73484115, 5399.95822064, 6045.05605499,
            6684.10130596,
        ])

        for i, ampName in enumerate(ampNames):
            rng = np.random.default_rng(i)
            self.ptc.aMatrix[ampName] = rng.normal(
                loc=self.aMatrixMean, scale=self.aMatrixSigma,
            )
            self.ptc.gain[ampName] = 1.0
            self.ptc.rawMeans[ampName] = rawMeans
            self.ptc.rawVars[ampName] = rawVars
            self.ptc.finalMeans[ampName] = rawMeans
            self.ptc.finalVars[ampName] = rawVars
            self.ptc.expIdMask[ampName] = np.ones_like(rawMeans, dtype=bool)
            self.ptc.covariances[ampName] = []
            for mean, variance in zip(rawMeans, rawVars):
                residual = mean - variance
                m = np.array([
                    [1, -2e-2, -3e-3, 3e-3, 0e0],
                    [-5e-3, -2e-03, 1e-3, 0e0, 0e0],
                    [-3e-3, -2e-3, 1e-3, 2e-3, 1e-3],
                    [-2e-3, -1e-3, 2e-3, 3e-3, 0e0],
                    [-1e-3, 1e-3, -2e-3, 0e0, 0e0],
                ])  # 5x5
                covariance = m * np.full_like(m, residual)
                covariance[0][0] = variance

                self.ptc.covariances[ampName].append(covariance)

            self.ptc.covariancesModel[ampName] = (
                self.ptc.covariances[ampName]
            )
            self.ptc.gain[ampName] = 1.0
            self.ptc.noise[ampName] = 5.0
            self.ptc.noiseMatrix[ampName] = np.zeros((5, 5))
            self.ptc.noiseMatrix[ampName][0][0] = (
                self.ptc.noise[ampName]**2
            )

        self.sequencerMetadata = {
            "SEQNAME": "a_sequencer",
            "SEQFILE": "a_sequencer_file",
            "SEQCKSUM": "hedgehog",
        }
        self.ptc.updateMetadata(
            **self.sequencerMetadata,
            setCalibInfo=True,
        )

        # This is empirically determined from the above parameters.
        self.expectationN = np.array([
            [-9.63074863e-07, -1.83533755e-07, -6.82742854e-08],
            [-1.67887612e-07, -1.11711275e-07, -5.49232536e-08],
            [-2.61332932e-08, -4.24786799e-08, -3.24952712e-08],
        ])
        self.expectationS = self.expectationN

        self.expectationE = np.array([
            [-6.78238572e-07, -1.82999671e-07, -2.77419497e-08],
            [-1.66240036e-07, -1.07364730e-07, -4.28616924e-08],
            [-6.54927788e-08, -5.34029295e-08, -3.22502537e-08],
        ])
        self.expectationW = self.expectationE

        self.expectationFitParams = {
            "thickness": 100.0,
            "pixelsize": 10.0,
            "zq": 2.88749,
            "zsh_minus_zq": 0.0,
            "zsh": 2.88749,
            "zsv_minus_zq": 1.08651,
            "zsv": 3.974,
            "a": 0.0005,
            "b": 3.5,
            "alpha": 0.64823,
            "beta": 0.0,
        }

    def test_average(self):
        """Test "averaged" input aMatrix and aMatrixSigma.
        """
        config = cpPipe.ElectrostaticBrighterFatterSolveTask().ConfigClass()
        config.fitRange = 3
        task = cpPipe.ElectrostaticBrighterFatterSolveTask(config=config)

        results = task.run(self.ptc, self.camera, {'detector': 1})

        ebf = results.outputBF

        self.assertFloatsAlmostEqual(
            ebf.aMatrix, self.aMatrixMean,
            atol=3*self.aMatrixSigma,
        )
        self.assertFloatsAlmostEqual(
            ebf.aMatrixSigma, self.aMatrixSigma,
            atol=3*self.aMatrixSigma/np.sqrt(8)
        )

        for key, value in self.sequencerMetadata.items():
            self.assertEqual(ebf.metadata[key], value)

        self.assertEqual(ebf.metadata["INSTRUME"], self.camera.getName())
        self.assertEqual(ebf.metadata["DETECTOR"], self.detector.getId())
        self.assertEqual(ebf.metadata["DET_NAME"], self.detector.getName())
        self.assertEqual(ebf.metadata["DET_SER"], self.detector.getSerial())

    def test_electrostaticSolution(self):
        """Test the full electrostatic solution.
        """
        config = cpPipe.ElectrostaticBrighterFatterSolveTask().ConfigClass()
        config.fitRange = 3
        task = cpPipe.ElectrostaticBrighterFatterSolveTask(config=config)

        results = task.run(self.ptc, self.camera, {'detector': 1})
        ebf = results.outputBF

        for n in ebf.fitParamNames:
            self.assertFloatsAlmostEqual(
                ebf.fitParams[n],
                self.expectationFitParams[n],
                atol=1e-3,
            )

        d = 1
        self.assertFloatsAlmostEqual(
            ebf.aN[:d, :d], self.expectationN[:d, :d],
            rtol=1e-3,
        )
        self.assertFloatsAlmostEqual(
            ebf.aS[:d, :d], self.expectationS[:d, :d],
            rtol=1e-3,
        )
        self.assertFloatsAlmostEqual(
            ebf.aE[:d, :d], self.expectationE[:d, :d],
            rtol=1e-3,
        )
        self.assertFloatsAlmostEqual(
            ebf.aW[:d, :d], self.expectationW[:d, :d],
            rtol=1e-3,
        )


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
