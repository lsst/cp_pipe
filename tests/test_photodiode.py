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
"""Test cases for lsst.cp.pipe.FindDefectsTask."""

import unittest
import tempfile

import lsst.utils
import lsst.utils.tests

from lsst.cp.pipe.ptc.photodiode import PhotodiodeData


class PhotodiodeDataTestCase(lsst.utils.tests.TestCase):
    """A test case dealing with photodiode data."""

    def test_getBOTphotodiodeData(self):
        # Need a fake dataRef to even check that this raises or returns None
        pass

    def test_photodiodeClass(self):
        with tempfile.NamedTemporaryFile(mode='w+t') as f:
            # nonuniform time steps, negative currents
            f.write('0. -1E-10\n')
            f.write('1. 2E-10\n')
            f.write('4. 3E-10\n')
            f.write('6. 2E-10\n')

            f.file.flush()

            diodeData = PhotodiodeData(f.name)
            charge = diodeData.getCharge()
            self.assertEqual(charge, 1.3e-9)


class TestMemory(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
