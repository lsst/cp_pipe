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


def parseCalibrationCollection(registry, collection, datasetTypes):
    """Search a calibration collection for calibration datasets.

    Parameters
    ----------
    registry : `lsst.daf.butler.Registry`
        Butler registry to use.
    collection : `str`
        Collection to search.  This should be a CALIBRATION
        collection.

    Returns
    -------
    exportCollections : `list` [`str`]
        List of collections to save on export.
    exportDatasets : `list` [`lsst.daf.butler.queries.DatasetQueryResults`]
        Datasets to save on export.
    """
    exportCollections = []
    exportDatasets = []
    for calibType in datasetTypes:
        associations = registry.queryDatasetAssociations(calibType, collections=[collection])
        for result in associations:
            exportDatasets.append(result.ref)
            exportCollections.append(result.ref.run)
    return exportCollections, exportDatasets


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
        collectionsToExport = []
        datasetsToExport = []

        for collection in args.collections:
            log.info("Checking collection: ", collection)

            # Get collection information.
            collectionsToExport.append(collection)
            collectionType = butler.registry.getCollectionType(collection)
            if collectionType == CollectionType.CHAINED:
                # Iterate over the chain:
                collectionsToExport.append(collection)
                collectionRecords = butler.registry.getCollectionChain(collection)

                for child in collectionRecords:
                    childType = butler.registry.getCollectionType(child)
                    collectionsToExport.append(child)
                    if childType == CollectionType.CALIBRATION:
                        exportCollections, exportDatasets = parseCalibrationCollection(butler.registry,
                                                                                       child,
                                                                                       calibTypes)
                        collectionsToExport.extend(exportCollections)
                        datasetsToExport.extend(exportDatasets)
            elif collectionType == CollectionType.CALIBRATION:
                exportCollections, exportDatasets = parseCalibrationCollection(butler.registry,
                                                                               collection,
                                                                               calibTypes)
                collectionsToExport.append(collection)
                collectionsToExport.extend(exportCollections)
                datasetsToExport.extend(exportDatasets)
            else:
                log.warn("Not checking collection %s of type %s.", collection, collectionType)

        collectionsToExport = list(set(collectionsToExport))
        datasetsToExport = list(set(datasetsToExport))

        for exportable in collectionsToExport:
            try:
                log.info("Saving collection %s.", exportable)
                export.saveCollection(exportable)
            except Exception as e:
                log.warn("Did not save collection %s due to %s.", exportable, e)

        try:
            log.info("Saving dataset(s)")
            export.saveDatasets(datasetsToExport)
        except Exception as e:
            log.warn("Did not save datasets %s due to %s.", exportable, e)
