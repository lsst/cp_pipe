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
"""Test cases for cp_pipe."""

from __future__ import absolute_import, division, print_function
import unittest

import lsst.utils
import lsst.utils.tests

noEotestMsg = ""
noEotest = False
try:
    import lsst.eotest
except ImportError:
    noEotestMsg = "No eotest setup, so skipping unit test"
    noEotest = True


class CpPipeTestCase(lsst.utils.tests.TestCase):
    """A test case for cp_pipe."""

    def testExample(self):
        pass

    @unittest.skipIf(noEotest, noEotestMsg)
    def testImport(self):
        import lsst.cp.pipe as cpPipe  # noqa: F401

    @unittest.skipIf(noEotest, noEotestMsg)
    def testClassInstantiation(self):
        from lsst.cp.pipe import CpTask
        cpConfig = CpTask.ConfigClass()
        cpConfig.eotestOutputPath = '/some/test/path'  # must not be empty for validate() to pass
        cpTask = CpTask(config=cpConfig)  # noqa: F841


class TestMemory(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
