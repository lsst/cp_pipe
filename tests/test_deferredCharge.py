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
"""Test cases for lsst.cp.pipe.deferredCharge."""

import unittest

import lsst.utils
import lsst.utils.tests

import lsst.cp.pipe as cpPipe
from lsst.ip.isr import IsrMock


class CpCtiSolveTaskTestCase(lsst.utils.tests.TestCase):
    """A test case for the deferred charge/CTI solve task."""

    def setUp(self):
        self.camera = IsrMock().getCamera()

        overscanMeansA = [7.18039751e-01, 4.56550479e-01, 4.14261669e-01, 2.88099229e-01, -2.34310962e-02,
                          -4.59854975e-02, -1.14491098e-02, 5.19846082e-02, 2.05635265e-01, 1.25147207e-02,
                          9.00449380e-02, -2.39106059e-01, -1.52413145e-01, 4.63459678e-02, 1.85195580e-01,
                          -1.58051759e-01, -8.76842241e-04, -5.09192124e-02, 2.58496821e-01, 2.54267782e-01,
                          -1.37611866e-01, 3.35322201e-01, 1.04846083e-01, -2.16551736e-01, -8.82746354e-02,
                          -1.00256450e-01, 2.73297966e-01, -4.52805981e-02, 3.40960979e-01, 7.80628920e-02,
                          -2.90697180e-02, -6.99492991e-02, -1.06599867e-01, 6.89002723e-02, 1.46290688e-02,
                          1.19647197e-01, -1.54527843e-01, 9.35689881e-02, -1.06599934e-01, -2.13166289e-02,
                          9.35688764e-02, -1.19286761e-01, 1.18098985e-02, -3.69616691e-03, -6.14914447e-02,
                          -5.81059000e-03, 9.42736641e-02, 3.92978266e-02, -1.55937240e-01, 3.76202404e-01,
                          -1.13648064e-01, 1.71803936e-01, 6.17138995e-03, 8.22918862e-02, -2.84214199e-01,
                          -2.99097435e-03, -1.31973490e-01, -2.84214795e-01, -2.99140741e-03, -3.76546055e-01,
                          5.97376414e-02, -1.91883057e-01, -1.34087920e-01, -3.23684871e-01]
        overscanMeansB = [1.50365152e+01, 4.43178511e+00, 2.66550946e+00, 1.67382801e+00, 1.10997069e+00,
                          8.89361799e-01, 4.66469795e-01, 6.10956728e-01, 6.67343795e-01, 5.22854805e-01,
                          -1.15006611e-01, 2.67710119e-01, 2.05686077e-01, 1.84541523e-01, 8.65717679e-02,
                          5.51738311e-03, 2.35288814e-01, 3.45944524e-01, 7.81139359e-02, 1.52119964e-01,
                          2.02162191e-01, 3.44150960e-02, -2.86277920e-01, 1.43662184e-01, 3.21276844e-01,
                          -6.21452965e-02, 8.58670697e-02, -1.63320359e-02, -1.07958235e-01, -1.60820082e-01,
                          -2.19705645e-02, -1.55181482e-01, -2.39055425e-01, -2.75705636e-01, 6.33126274e-02,
                          -5.50971478e-02, -2.42579415e-01, -9.87957790e-02, 1.08421087e-01, -1.12892322e-01,
                          1.89090632e-02, -1.53086300e-03, -2.18615308e-01, -2.19320312e-01, 9.22102109e-02,
                          -4.87535410e-02, -1.81964979e-01, -4.17055413e-02, -4.24422681e-01, -1.96061105e-01,
                          -1.35127297e-02, -1.77031055e-01, -2.30597332e-01, -4.01868790e-01, -4.18784261e-01,
                          -3.75085384e-01, -3.49007100e-01, -1.77735761e-01, -7.41272718e-02, -1.92537069e-01,
                          2.46565759e-01, -3.44777972e-01, -2.85573214e-01, -2.34121397e-01]
        overscanMeansC = [0.212578110, 0.107817403, -0.122200218, -0.0089812368, -0.067990060, 0.040077099,
                          -0.021402006, 0.090923088, -0.099587158, 0.274797124, -0.016930788, 0.045007070,
                          -0.00379911056, -0.16088248, 0.055911896, 0.0601755001, -0.046872945, 0.210018355,
                          0.081641635, -0.046147249, -0.0059020276, 0.108368757, -0.033966731, -0.0058644798,
                          -0.075746922, -0.203826510, 0.12620401, -0.0156685544, -0.09631182, 0.089754454,
                          0.03789926, 0.0304515115, -0.082173715, -0.061332140, -0.24894494, -0.155137551,
                          -0.073825312, 0.24538413, -0.069597074, 0.192338801, -0.0539746876, -0.184556000,
                          -0.173069382, -0.209975778, 0.086679191, 0.016299034, -0.0094125706, -0.100099911,
                          0.061981365, 0.086250364, 0.209128404, -0.0067993622, 0.171072270, -0.29266333,
                          0.075172274, -0.29375612, -0.13377650, 0.0125964781, -0.124991264, 0.226516831,
                          0.128244484, -0.05019844, -0.149249925, -0.1557398]
        overscanMeansD = [4.0867248, 1.43194193, 0.95319573, 0.43219185, 0.53112239, 0.28648, 0.323903486,
                          0.27622156, 0.26031138, 0.144442975, 0.0149878587, 0.062969929, 0.018541051,
                          -0.237687056, 0.22804558, 0.0600504708, 0.140250022, -0.137477808, 0.119911710,
                          0.03770870, -0.20021377, 0.187175400, 0.0168790129, -0.110724371, 0.099311580,
                          0.0079969534, -0.157593577, -0.178876067, -0.214948580, -0.11354382, 0.148154530,
                          -0.056012520, 0.11851939, 0.067902033, 0.18970736, -0.181487703, -0.0101017127,
                          0.100998570, -0.0309096733, -0.034450136, -0.066357072, -0.058662959, 0.146185921,
                          -0.218474021, -0.173691633, 0.055349625, -0.178158524, -0.012917378, -0.166576555,
                          -0.063862754, 0.113169933, -0.33518338, -0.074239500, 0.22262230, -0.066653975,
                          -0.200271016, -0.013275277, 0.100596499, -0.092528954, 0.0339541714, 0.113119135,
                          -0.150720824, 0.038237873, 0.17603852613429813]

        self.inputMeasurements = [
            {'CTI': {'C:0,0': {'FIRST_MEAN': 117.810165, 'LAST_MEAN': 1.09791130e+02,
                               'IMAGE_MEAN': 117.810165, 'OVERSCAN_VALUES': overscanMeansA,
                               'OVERSCAN_COLUMNS': [x for x in range(0, 64)]},
                     'C:0,1': {'FIRST_MEAN': 117.810165, 'LAST_MEAN': 1.09791130e+02,
                               'IMAGE_MEAN': 117.810165, 'OVERSCAN_VALUES': overscanMeansA,
                               'OVERSCAN_COLUMNS': [x for x in range(0, 64)]},
                     'C:0,2': {'FIRST_MEAN': 117.810165, 'LAST_MEAN': 1.09791130e+02,
                               'IMAGE_MEAN': 117.810165, 'OVERSCAN_VALUES': overscanMeansA,
                               'OVERSCAN_COLUMNS': [x for x in range(0, 64)]},
                     'C:0,3': {'FIRST_MEAN': 117.810165, 'LAST_MEAN': 1.09791130e+02,
                               'IMAGE_MEAN': 117.810165, 'OVERSCAN_VALUES': overscanMeansA,
                               'OVERSCAN_COLUMNS': [x for x in range(0, 64)]},
                     'C:1,0': {'FIRST_MEAN': 117.810165, 'LAST_MEAN': 1.09791130e+02,
                               'IMAGE_MEAN': 117.810165, 'OVERSCAN_VALUES': overscanMeansA,
                               'OVERSCAN_COLUMNS': [x for x in range(0, 64)]},
                     'C:1,1': {'FIRST_MEAN': 117.810165, 'LAST_MEAN': 1.09791130e+02,
                               'IMAGE_MEAN': 117.810165, 'OVERSCAN_VALUES': overscanMeansA,
                               'OVERSCAN_COLUMNS': [x for x in range(0, 64)]},
                     'C:1,2': {'FIRST_MEAN': 117.810165, 'LAST_MEAN': 1.09791130e+02,
                               'IMAGE_MEAN': 117.810165, 'OVERSCAN_VALUES': overscanMeansA,
                               'OVERSCAN_COLUMNS': [x for x in range(0, 64)]},
                     'C:1,3': {'FIRST_MEAN': 117.810165, 'LAST_MEAN': 1.09791130e+02,
                               'IMAGE_MEAN': 117.810165, 'OVERSCAN_VALUES': overscanMeansA,
                               'OVERSCAN_COLUMNS': [x for x in range(0, 64)]}}},
            {'CTI': {'C:0,0': {'FIRST_MEAN': 36562.082, 'LAST_MEAN': 3.45901172e+04,
                               'IMAGE_MEAN': 36562.082, 'OVERSCAN_VALUES': overscanMeansB,
                               'OVERSCAN_COLUMNS': [x for x in range(0, 64)]},
                     'C:0,1': {'FIRST_MEAN': 36562.082, 'LAST_MEAN': 3.45901172e+04,
                               'IMAGE_MEAN': 36562.082, 'OVERSCAN_VALUES': overscanMeansB,
                               'OVERSCAN_COLUMNS': [x for x in range(0, 64)]},
                     'C:0,2': {'FIRST_MEAN': 36562.082, 'LAST_MEAN': 3.45901172e+04,
                               'IMAGE_MEAN': 36562.082, 'OVERSCAN_VALUES': overscanMeansB,
                               'OVERSCAN_COLUMNS': [x for x in range(0, 64)]},
                     'C:0,3': {'FIRST_MEAN': 36562.082, 'LAST_MEAN': 3.45901172e+04,
                               'IMAGE_MEAN': 36562.082, 'OVERSCAN_VALUES': overscanMeansB,
                               'OVERSCAN_COLUMNS': [x for x in range(0, 64)]},
                     'C:1,0': {'FIRST_MEAN': 36562.082, 'LAST_MEAN': 3.45901172e+04,
                               'IMAGE_MEAN': 36562.082, 'OVERSCAN_VALUES': overscanMeansB,
                               'OVERSCAN_COLUMNS': [x for x in range(0, 64)]},
                     'C:1,1': {'FIRST_MEAN': 36562.082, 'LAST_MEAN': 3.45901172e+04,
                               'IMAGE_MEAN': 36562.082, 'OVERSCAN_VALUES': overscanMeansB,
                               'OVERSCAN_COLUMNS': [x for x in range(0, 64)]},
                     'C:1,2': {'FIRST_MEAN': 36562.082, 'LAST_MEAN': 3.45901172e+04,
                               'IMAGE_MEAN': 36562.082, 'OVERSCAN_VALUES': overscanMeansB,
                               'OVERSCAN_COLUMNS': [x for x in range(0, 64)]},
                     'C:1,3': {'FIRST_MEAN': 36562.082, 'LAST_MEAN': 3.45901172e+04,
                               'IMAGE_MEAN': 36562.082, 'OVERSCAN_VALUES': overscanMeansB,
                               'OVERSCAN_COLUMNS': [x for x in range(0, 64)]}}},

            {'CTI': {'C:0,0': {'FIRST_MEAN': 994.811, 'LAST_MEAN': 936.415,
                               'IMAGE_MEAN': 994.811, 'OVERSCAN_VALUES': overscanMeansC,
                               'OVERSCAN_COLUMNS': [x for x in range(0, 64)]},
                     'C:0,1': {'FIRST_MEAN': 994.811, 'LAST_MEAN': 936.415,
                               'IMAGE_MEAN': 994.811, 'OVERSCAN_VALUES': overscanMeansC,
                               'OVERSCAN_COLUMNS': [x for x in range(0, 64)]},
                     'C:0,2': {'FIRST_MEAN': 994.811, 'LAST_MEAN': 936.415,
                               'IMAGE_MEAN': 994.811, 'OVERSCAN_VALUES': overscanMeansC,
                               'OVERSCAN_COLUMNS': [x for x in range(0, 64)]},
                     'C:0,3': {'FIRST_MEAN': 994.811, 'LAST_MEAN': 936.415,
                               'IMAGE_MEAN': 994.811, 'OVERSCAN_VALUES': overscanMeansC,
                               'OVERSCAN_COLUMNS': [x for x in range(0, 64)]},
                     'C:1,0': {'FIRST_MEAN': 994.811, 'LAST_MEAN': 936.415,
                               'IMAGE_MEAN': 994.811, 'OVERSCAN_VALUES': overscanMeansC,
                               'OVERSCAN_COLUMNS': [x for x in range(0, 64)]},
                     'C:1,1': {'FIRST_MEAN': 994.811, 'LAST_MEAN': 936.415,
                               'IMAGE_MEAN': 994.811, 'OVERSCAN_VALUES': overscanMeansC,
                               'OVERSCAN_COLUMNS': [x for x in range(0, 64)]},
                     'C:1,2': {'FIRST_MEAN': 994.811, 'LAST_MEAN': 936.415,
                               'IMAGE_MEAN': 994.811, 'OVERSCAN_VALUES': overscanMeansC,
                               'OVERSCAN_COLUMNS': [x for x in range(0, 64)]},
                     'C:1,3': {'FIRST_MEAN': 994.811, 'LAST_MEAN': 936.415,
                               'IMAGE_MEAN': 994.811, 'OVERSCAN_VALUES': overscanMeansC,
                               'OVERSCAN_COLUMNS': [x for x in range(0, 64)]}}},
            {'CTI': {'C:0,0': {'FIRST_MEAN': 12215.778, 'LAST_MEAN': 11536.875,
                               'IMAGE_MEAN': 12215.778, 'OVERSCAN_VALUES': overscanMeansD,
                               'OVERSCAN_COLUMNS': [x for x in range(0, 64)]},
                     'C:0,1': {'FIRST_MEAN': 12215.778, 'LAST_MEAN': 11536.875,
                               'IMAGE_MEAN': 12215.778, 'OVERSCAN_VALUES': overscanMeansD,
                               'OVERSCAN_COLUMNS': [x for x in range(0, 64)]},
                     'C:0,2': {'FIRST_MEAN': 12215.778, 'LAST_MEAN': 11536.875,
                               'IMAGE_MEAN': 12215.778, 'OVERSCAN_VALUES': overscanMeansD,
                               'OVERSCAN_COLUMNS': [x for x in range(0, 64)]},
                     'C:0,3': {'FIRST_MEAN': 12215.778, 'LAST_MEAN': 11536.875,
                               'IMAGE_MEAN': 12215.778, 'OVERSCAN_VALUES': overscanMeansD,
                               'OVERSCAN_COLUMNS': [x for x in range(0, 64)]},
                     'C:1,0': {'FIRST_MEAN': 12215.778, 'LAST_MEAN': 11536.875,
                               'IMAGE_MEAN': 12215.778, 'OVERSCAN_VALUES': overscanMeansD,
                               'OVERSCAN_COLUMNS': [x for x in range(0, 64)]},
                     'C:1,1': {'FIRST_MEAN': 12215.778, 'LAST_MEAN': 11536.875,
                               'IMAGE_MEAN': 12215.778, 'OVERSCAN_VALUES': overscanMeansD,
                               'OVERSCAN_COLUMNS': [x for x in range(0, 64)]},
                     'C:1,2': {'FIRST_MEAN': 12215.778, 'LAST_MEAN': 11536.875,
                               'IMAGE_MEAN': 12215.778, 'OVERSCAN_VALUES': overscanMeansD,
                               'OVERSCAN_COLUMNS': [x for x in range(0, 64)]},
                     'C:1,3': {'FIRST_MEAN': 12215.778, 'LAST_MEAN': 11536.875,
                               'IMAGE_MEAN': 12215.778, 'OVERSCAN_VALUES': overscanMeansD,
                               'OVERSCAN_COLUMNS': [x for x in range(0, 64)]}}}]
        self.inputDims = [{'detector': 20, 'instrument': 'IsrMock', 'exposure': 2019101200433},
                          {'detector': 20, 'instrument': 'IsrMock', 'exposure': 2019101300154},
                          {'detector': 20, 'instrument': 'IsrMock', 'exposure': 2019101300004},
                          {'detector': 20, 'instrument': 'IsrMock', 'exposure': 2019101200333}]

        self.task = cpPipe.CpCtiSolveTask()

    def test_task(self):
        """A test for the main CpCtiSolveTask.

        This should excercise most of the new code.
        """
        results = self.task.run(self.inputMeasurements, self.camera, self.inputDims)

        calib = results.outputCalib
        # Check that the result matches expectation.
        self.assertAlmostEqual(calib.globalCti['C:0,0'], 1.0e-7, 4)
        self.assertAlmostEqual(calib.driftScale['C:0,0'], 1.8105e-4, 4)
        self.assertAlmostEqual(calib.decayTime['C:0,0'], 3.08095, 4)

        # Check that all amps are equal.
        for ampName in calib.globalCti.keys():
            self.assertEqual(calib.globalCti['C:0,0'], calib.globalCti[ampName])
            self.assertEqual(calib.driftScale['C:0,0'], calib.driftScale[ampName])
            self.assertEqual(calib.decayTime['C:0,0'], calib.decayTime[ampName])


class TestMemory(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
