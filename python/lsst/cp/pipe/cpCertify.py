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

import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase
from lsst.daf.butler import CollectionType


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

    def __init__(self, *, registry, inputCollection, outputCollection, lastRunOnly=True, **kwargs):
        super().__init__(**kwargs)
        self.registry = registry
        if lastRunOnly:
            try:
                inputCollection, _ = next(iter(self.registry.getCollectionChain(inputCollection)))
            except TypeError:
                # Not a CHAINED collection; do nothing.
                pass
        self.inputCollection = inputCollection
        self.outputCollection = outputCollection

    def run(self, datasetTypeName, timespan):
        """Certify all of the datasets of the given type in the input
        collection.

        Parameters
        ----------
        datasetTypeName : `str`
            Name of the dataset type to certify.
        timespan : `lsst.daf.butler.Timespan`
            Timespan for the validity range.
        """
        refs = set(self.registry.queryDatasets(datasetTypeName, collections=[self.inputCollection]))
        if not refs:
            raise RuntimeError(f"No inputs found for dataset {datasetTypeName} in {self.inputCollection}.")
        self.registry.registerCollection(self.outputCollection, type=CollectionType.CALIBRATION)
        self.registry.certify(self.outputCollection, refs, timespan)
