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
__all__ = ("main", )


import argparse
import logging
import os
from lsst.daf.butler import Butler, CollectionType


def build_argparser():
    """Construct the argument parser for exporting calibrations.

    Returns
    -------
    argparser : `argparse.ArgumentParser`
        The argument parser that defines the ``exportCalibs.py``
        interface.
    """
    parser = argparse.ArgumentParser(description="Export all calibrations in the given collections.")
    parser.add_argument("butler", help="Directory or butler.yaml to export from")
    parser.add_argument("output", default="calibs", help="Directory to export to")
    parser.add_argument("--collections", '-c', action='append', help="Collections to save.")

    return parser


def main():
    args = build_argparser().parse_args()

    log = logging.getLogger(__name__)

    # Instantiate source butler.
    butler = Butler(args.butler)

    # Loop over this set of calibration types.
    calibTypes = ['bias', 'dark', 'flat', 'defects', 'linearity', 'bfk', 'ptc', 'crosstalk']

    # Make export destination if it doesn't already exist.
    os.makedirs(args.output, exist_ok=True)

    # Begin export
    with butler.export(directory=args.output, format="yaml", transfer="auto") as export:
        for collection in args.collections:
            log.info("Checking collection: ", collection)

            # Get collection information.
            collectionsToExport = []
            runCollectionFound = False
            collectionType = butler.registry.getCollectionType(collection)
            if collectionType == CollectionType.CHAINED:
                # Iterate over the chain:
                collectionRecords = butler.registry.getCollectionChain(collection)

                for child in collectionRecords:
                    childType = butler.registry.getCollectionType(child)
                    if childType == CollectionType.CALIBRATION:
                        if not runCollectionFound:
                            # If we've found a RUN collection, skip
                            # any other CALIBRATION collections we
                            # find.
                            collectionsToExport.append(child)
                            pass
                    elif childType == CollectionType.RUN:
                        if not runCollectionFound:
                            # Only include the first RUN collection we
                            # find.  That is the true collection the
                            # calibration was constructed in.
                            collectionsToExport.append(child)
                            runCollectionFound = True
                    else:
                        log.warn("Skipping collection %s of type %s.", child, childType)
                if not runCollectionFound:
                    collectionsToExport.append(collection)

            for exportable in collectionsToExport:
                try:
                    log.info("Saving collection %s.", exportable)
                    export.saveCollection(exportable)
                except Exception as e:
                    log.warn("Did not save collection %s due to %s.", exportable, e)

            # Get datasets.
            if runCollectionFound:
                # This should contain the calibration with it's
                # association information.
                items = []
                for calib in calibTypes:
                    try:
                        found = set(butler.registry.queryDatasets(calib, collections=[collection],
                                                                  instrument='LATISS', detector=0))
                        items.extend(found)
                    except Exception as e:
                        log.warn("Did not find %s in collection %s due to %s.", calib, collection, e)

                if len(items) > 0:
                    try:
                        log.info("Saving dataset(s) %s", items)
                        export.saveDatasets(items)
                    except Exception as e:
                        log.warn("Did not save datasets %s due to %s.", items, e)
