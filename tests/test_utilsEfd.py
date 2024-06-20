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
            self.log.warning(f"Could not initialize EFD client: {e}")
            self.client = None

    def test_monochromator(self):
        if self.client is None:
            self.log.warning("No EFD client: skipping monochromator test.")
            return

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
        if self.client is None:
            self.log.warning("No EFD client: skipping electrometer test.")
            return
        data = self.client.getEfdElectrometerData(
            # dateMin="2024-05-30T13:00:00",
            # dateMax="2024-05-30T15:00:00"
        )

        for iDate, rDate in [("2024-05-30T04:21:48", "2024-05-30T04:22:08"),
                             ("2024-05-30T04:21:50", "2024-05-30T04:22:10"),
                             ("2024-05-30T04:21:53", "2024-05-30T04:22:13"),
                             ("2024-05-30T04:21:58", "2024-05-30T04:22:18"),
                             ("2024-05-30T04:22:17", "2024-05-30T04:22:37")]:
            indexDate, intensity = self.client.parseElectrometerStatus(
                data,
                iDate
            )

            indexDateReference, intensityReference = self.client.parseElectrometerStatus(
                data,
                rDate
            )
            print(iDate, indexDate, intensity, rDate, indexDateReference, intensityReference)


class MemoryTester(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
