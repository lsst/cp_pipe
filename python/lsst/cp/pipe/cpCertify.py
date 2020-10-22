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
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
import argparse
import astropy.time
import sys

import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase
from lsst.daf.butler import Butler, CollectionType, Timespan


class CertifyCalibration(pipeBase.Task):
    """Create a way to bless existing calibration products.

    The inputs are assumed to have been constructed via cp_pipe, and
    already exist in the butler.

    Parameters
    ----------
    registry : `lsst.daf.butler.Registry`
        Registry pointing at the butler repository to operate on.
    inputCollection : `str`
        Data collection to pull calibrations from.  Usually an existing
        `~CollectionType.RUN` or `~CollectionType.CHAINED` collection, and may
        _not_ be a `~CollectionType.CALIBRATION` collection or a nonexistent
        collection.
    outputCollection : `str`
        Data collection to store final calibrations.  If it already exists, it
        must be a `~CollectionType.CALIBRATION` collection.  If not, a new
        `~CollectionType.CALIBRATION` collection with this name will be
        registered.
    lastRunOnly : `bool`, optional
        If `True` (default) and ``inputCollection`` is a
        `~CollectionType.CHAINED` collection, only search its first child
        collection (which usually corresponds to the last processing run),
        instead of all child collections in the chain.  This behavior ensures
        that datasets in a collection used as input to that processing run
        are never included in the certification.
    **kwargs :
        Additional arguments forwarded to `lsst.pipe.base.Task.__init__`.
    """
    _DefaultName = 'CertifyCalibration'
    ConfigClass = pexConfig.Config

    def __init__(self, *, butlerRoot, inputCollection, outputCollection, lastRunOnly=True, **kwargs):
        super().__init__(**kwargs)

        butler = Butler(butlerRoot, run=inputCollection)
        self.registry = butler.registry

        if lastRunOnly:
            try:
                inputCollection, _ = next(iter(self.registry.getCollectionChain(inputCollection)))
            except TypeError:
                # Not a CHAINED collection; do nothing.
                pass
        self.inputCollection = inputCollection
        self.outputCollection = outputCollection

    def run(self, datasetTypeName, beginDate=None, endDate=None):
        """Certify all of the datasets of the given type in the input
        collection.

        Parameters
        ----------
        datasetTypeName : `str`
            Name of the dataset type to certify.
        beginDate : `str`
            Date this calibration should start being used.
        endDate : `str`
            Date this calibration should stop being used.
        """
        timespan = Timespan(
            begin=astropy.time.Time(beginDate) if beginDate is not None else None,
            end=astropy.time.Time(endDate) if endDate is not None else None,
        )

        refs = set(self.registry.queryDatasets(datasetTypeName, collections=[self.inputCollection]))
        if not refs:
            raise RuntimeError(f"No inputs found for dataset {datasetTypeName} in {self.inputCollection}.")
        self.registry.registerCollection(self.outputCollection, type=CollectionType.CALIBRATION)
        self.registry.certify(self.outputCollection, refs, timespan)


def build_parser():
    """Construct the argument parser for ``certifyCalibrations.py``.

    Returns
    -------
    parser : `argparse.ArgumentParser`
        The constructed argument parser.
    """

    parser = argparse.ArgumentParser(description="""
    Certify a cp_pipe generated calibration as valid for a given date
    range, and register them with the butler.
    """)

    parser.add_argument("root", help="Path to butler to use")
    parser.add_argument("inputCollection", help="Input collection to pull from.")
    parser.add_argument("outputCollection", help="Output collection to add to.")
    parser.add_argument("datasetTypeName", help="Dataset type to bless.")

    parser.add_argument("-b", "--beginDate",
                        help="Start date for using the calibration")
    parser.add_argument("-e", "--endDate",
                        help="End date for using the calibration")
    parser.add_argument(
        "--search-all-inputs",
        dest="lastRunOnly",
        action="store_false",
        default=True,
        help=(
            "Search all children of the given input collection if it is a "
            "CHAINED collection, instead of just the most recent one."
        )
    )

    return parser


def main():
    args = build_parser().parse_args()

    try:
        certify = CertifyCalibration(butlerRoot=args.root,
                                     inputCollection=args.inputCollection,
                                     outputCollection=args.outputCollection,
                                     lastRunOnly=args.lastRunOnly)
        certify.run(args.datasetTypeName, beginDate=args.beginDate, endDate=args.endDate)
    except Exception as e:
        print(f"{e}", file=sys.stderr)
        return 1
    return 0
