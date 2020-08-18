#!/usr/bin/env python
#
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
#
"""Mark cp_pipe generated calibrations as valid, and register them with
the butler with an appropriate use date range.
"""
from lsst.cp.pipe.cpCertify import CertifyCalibration

import argparse
import logging

import lsst.log
from lsst.log import Log

from lsst.daf.butler import Butler

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("root", help="Path to butler to use")
    parser.add_argument("inputCollection", help="Input collection to pull from.")
    parser.add_argument("outputCollection", help="Output collection to add to.")
    parser.add_argument("datasetTypeName", help="Dataset type to bless.")

    parser.add_argument("-v", "--verbose", action="store_const", dest="logLevel",
                        default=Log.INFO, const=Log.DEBUG,
                        help="Set the log level to DEBUG.")
    parser.add_argument("-b", "--beginDate",
                        help="Start date for using the calibration")
    parser.add_argument("-e", "--endDate",
                        help="End date for using the calibration")
    parser.add_argument("-s", "--skipCalibrationLabel", action="store_true",
                        default=False, dest="skipCL",
                        help="Do not attempt to register the calibration label.")

    args = parser.parse_args()
    log = Log.getLogger("lsst.daf.butler")
    log.setLevel(args.logLevel)

    # DM-22527: Clean up syntax/log handling in gen3 repo scripts.
    lgr = logging.getLogger("lsst.daf.butler")
    lgr.setLevel(logging.INFO if args.logLevel == Log.INFO else logging.DEBUG)
    lgr.addHandler(lsst.log.LogHandler())

    butler = Butler(args.root, run=args.inputCollection)

    # Do the thing.
    certify = CertifyCalibration(butler=butler,
                                 inputCollection=args.inputCollection,
                                 outputCollection=args.outputCollection)

    certify.findInputs(args.datasetTypeName)
    if not args.skipCL:
        certify.addCalibrationLabel(beginDate=args.beginDate, endDate=args.endDate)
    certify.registerCalibrations(args.datasetTypeName)
