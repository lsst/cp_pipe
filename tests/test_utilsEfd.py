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

import unittest

import lsst.utils.tests

from lsst.cp.pipe.utilsEfd import CpEfdClient


class UtilsEfdTestCase(lsst.utils.tests.TestCase):
    """Unit test for EFD access code."""

    def setUp(self):
        # Initialize a client.
        try:
            self.client = CpEfdClient()
        except Exception as e:
            self.client = None
            raise unittest.SkipTest(f"Could not initialize EFD client: {e}")

    def test_monochromator(self):
        data = self.client.getEfdMonochromatorData(
            dateMin="2023-12-19T00:00:00",
            dateMax="2023-12-19T23:59:59"
        )

        indexDate, wavelength = self.client.parseMonochromatorStatus(
            data,
            "2023-12-19T14:37:19.498"
        )
        self.assertEqual(wavelength, 550.0)
        self.assertEqual(indexDate, "2023-12-19T14:37:17.799")

    def test_electrometer(self):
        data = self.client.getEfdElectrometerData(
            dateMin="2024-05-30T00:00:00",
            dateMax="2024-05-30T05:00:00",
        )

        # Test single lookups:
        for (iDate, rDate), (iVal, rVal) in zip([("2024-05-30T04:21:48.6", "2024-05-30T04:22:08"),
                                                 ("2024-05-30T04:21:50", "2024-05-30T04:22:10"),
                                                 ("2024-05-30T04:21:53", "2024-05-30T04:22:13"),
                                                 ("2024-05-30T04:21:58", "2024-05-30T04:22:18"),
                                                 ("2024-05-30T04:22:17", "2024-05-30T04:22:37")],
                                                [(-1.5269e-07, -1.5244e-07),
                                                 (-1.5168e-07, -1.5137e-07),
                                                 (-1.5165e-07, -1.5205e-07),
                                                 (-1.5223e-07, -1.5147e-07),
                                                 (-1.5226e-07, -1.5558e-07)]):
            indexDate, intensity, _ = self.client.parseElectrometerStatus(
                data,
                iDate
            )
            self.assertFloatsAlmostEqual(intensity, iVal, atol=1e-10)

            indexDateReference, intensityReference, _ = self.client.parseElectrometerStatus(
                data,
                rDate
            )
            self.assertFloatsAlmostEqual(intensityReference, rVal, atol=1e-10)

        # Test integrated lookups:
        iDate = "2024-05-30T04:21:48.6"
        iDateEnd = "2024-05-30T04:22:18"
        rDate = "2024-05-30T04:22:08"
        rDateEnd = "2024-05-30T04:22:38"
        indexDate, intensity, endDate = self.client.parseElectrometerStatus(
            data,
            iDate,
            dateEnd=iDateEnd,
            doIntegrateSamples=True,
            index=201
        )
        self.assertFloatsAlmostEqual(intensity, -1.52297e-7, atol=1e-10)
        indexDateReference, intensityReference, endDateReference = self.client.parseElectrometerStatus(
            data,
            rDate,
            dateEnd=rDateEnd,
            doIntegrateSamples=True,
            index=201
        )
        self.assertFloatsAlmostEqual(intensityReference, -1.532977e-07, atol=1e-10)

    def test_electrometer_alternate(self):
        # This should raise if no dates are passed:
        with self.assertRaises(RuntimeError):
            data = self.client.getEfdElectrometerData(
                dataSeries='lsst.sal.Electrometer.logevent_logMessage')

        data = self.client.getEfdElectrometerData(
            dataSeries='lsst.sal.Electrometer.logevent_logMessage',
            dateMin='2024-07-26T16:00:00',
            dateMax='2024-07-26T20:00:00')

        # Test single lookups.  These should not be integrated.
        for iDate, iVal in zip(["2024-07-26T16:38:32.228",
                                "2024-07-26T16:38:54.581",
                                "2024-07-26T16:40:56.579",
                                "2024-07-26T16:42:58.553"],
                               [-2.24234e-07,
                                -2.24388e-07,
                                -2.24105e-07,
                                -2.23784e-07]):
            indexDate, intensity, _ = self.client.parseElectrometerStatus(
                data,
                iDate
            )
            self.assertFloatsAlmostEqual(intensity, iVal, atol=1e-10)


class MemoryTester(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
