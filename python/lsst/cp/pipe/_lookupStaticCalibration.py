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

from lsst.obs.base import Instrument

__all__ = ["lookupStaticCalibration"]


def lookupStaticCalibration(datasetType, registry, quantumDataId, collections):
    """A lookup function override for QuantumGraph generation that allows a
    PipelineTask to have an input dataset (usually a camera) that is formally a
    calibration with a validity range, without having a temporal data ID for
    the lookup, by asserting that there is in fact only dataset for all time.

    Parameters
    ----------
    datasetType : `lsst.daf.butler.DatasetType`
        Dataset type to look up.
    registry : `lsst.daf.butler.Registry`
        Registry for the data repository being searched.
    quantumDataId : `lsst.daf.butler.DataCoordinate`
        Data ID for the quantum of the task this dataset will be passed to.
        This must include an "instrument" key, and should also include any
        keys that are present in ``datasetType.dimensions``.  If it has an
        ``exposure`` or ``visit`` key, that's a sign that this function is
        not actually needed, as those come with the temporal information that
        would allow a real validity-range lookup.
    collections : `lsst.daf.butler.registry.CollectionSearch`
        Collections passed by the user when generating a QuantumGraph.  Ignored
        by this function (see notes below).

    Returns
    -------
    refs : `list` [ `DatasetRef` ]
        A zero- or single-element list containing the matching dataset, if one
        was found.

    Notes
    -----
    This works by looking in the `~CollectionType.RUN` collection
    that `lsst.obs.base.Instrument.writeCuratedCalibrations` (currently!) uses,
    instead of the collections passed into it.  This may be considered
    surprising by users (but will usually go unnoticed because the dataset
    returned _is_ actually in those given input colllections, too).  It may
    stop working entirely once we have data repositories with multiple
    calibration collections; a better workaround or a more principled change
    to the PipelineTasks that use this function (which are by definition asking
    for something ill-defined) will ultimately be needed.
    """
    instrument = Instrument.fromName(quantumDataId["instrument"], registry)
    unboundedCollection = instrument.makeUnboundedCalibrationRunName()
    ref = registry.findDataset(datasetType, dataId=quantumDataId, collections=[unboundedCollection])
    if ref is None:
        return []
    else:
        return [ref]
