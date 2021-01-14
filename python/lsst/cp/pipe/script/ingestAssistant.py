#!/usr/bin/env python
# This file is part of cp_pipe.
#
# Developed for the LSST Data Management System.
# This product includes software developed by the LSST Project
# (http://www.lsst.org).
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
# You should have received a copy of the LSST License Statement and
# the GNU General Public License along with this program.  If not,
# see <http://www.lsstcorp.org/LegalNotices/>.
#

__all__ = ('main',)

import argparse
import datetime
import os
import sys

from lsst.ip.isr import CrosstalkCalib, Defects, Linearizer


def build_argparser():
    """Construct an argument parser for the ``ingestAssistant.py'' script.

    Returns
    -------
    argparser : `argparse.ArgumentParser`
        The argument parser that defines the ``translate_header.py``
        command-line interface.
    """

    parser = argparse.ArgumentParser(description="Rewrite stack generated calibs for ingestCuratedCalibs.py")
    parser.add_argument('filename', type=str, help="Filename of existing calibration.")
    parser.add_argument('repo', type=str, help="Gen2 repository root.")
    parser.add_argument('--date', type=str, help="Date to assign to this calibration.",
                        default=None)

    return parser


def read_file(filename):
    """Read a file on disk, and return the appropriate calibration.

    Parameters
    ----------
    filename : `str`
        Path to read.

    Returns
    -------
    calib : `lsst.ip.isr.CrosstalkCalib`, `lsst.ip.isr.Defects`,
             or `lsst.ip.isr.Linearizer`
        The python representation of the calibration.

    Raises
    ------
    RuntimeError :
        Raised if a calibration could not be read.
    """
    try:
        calib = CrosstalkCalib().readFits(filename)
        return calib
    except Exception:
        pass

    try:
        calib = Defects().readFits(filename)
        return calib
    except Exception:
        pass

    try:
        calib = Linearizer().readFits(filename)
        return calib
    except Exception:
        pass

    raise RuntimeError(f"Could not identify file {filename}")


def construct_filename(calib, repo, date=None):
    """Construct the new output filename.

    Parameters
    ----------
    calib : `lsst.ip.isr.CrosstalkCalib`, `lsst.ip.isr.Defects`,
            or `lsst.ip.isr.Linearizer`
        The calibration.
    repo : `str`
        Path to the gen2 repo.
    date : `str`, optional
        Date to use as validStart.

    Returns
    -------
    filename : `str`
        Filename to write the calibration to.
    """
    calibType = calib._OBSTYPE.lower()
    instrument = calib.getMetadata().get('INSTRUME', None)
    detName = calib.getMetadata().get('DET_NAME', None)

    if detName is None or instrument is None:
        raise RuntimeError("Could not determine instrument/detector.")

    instrument = instrument.lower()
    detName = detName.lower()

    if date is None:
        date = calib.getMetadata().get('CALIBDATE', None)
        if date is None:
            raise RuntimeError("Could not determine a date.")
    dateTime = datetime.datetime.fromisoformat(date)
    date = dateTime.strftime("%Y%m%dT%H%M%S")

    pathname = os.path.abspath(os.path.join(repo, instrument, calibType, detName))
    filename = f"{date}.yaml"
    print(pathname, filename)
    return pathname, filename


def ingestAssistant(filename, repo, date=None):
    """Rewrite calibration locations for gen2.

    Parameters
    ----------
    filename : `str`
        Calibration file to read.
    repo : `str`
        Path to destination repository.
    date : `str`, optional
        Date to use as the validStart.
    """
    calib = read_file(filename)
    outPath, outFile = construct_filename(calib, repo, date)
    os.makedirs(outPath, exist_ok=True)
    calib.writeText(os.path.join(outPath, outFile))


def main():
    args = build_argparser().parse_args()

    try:
        ingestAssistant(args.filename, args.repo, date=args.date)
    except Exception as e:
        print(f"{e}", file=sys.stderr)
        return 1
    return 0
