# This file is part of obs_lsst.
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
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

__all__ = ('PhotodiodeData', 'getBOTphotodiodeData')

import os
import numpy as np

import lsst.log as lsstLog


def getBOTphotodiodeData(dataRef, dataPath='/project/shared/BOT/_parent/raw/photodiode_data/',
                         logger=None):
    """Get the photodiode data associated with a BOT dataRef.

    This is a temporary Gen2-only interface to the photodiode readings from
    the SLAC Run3 datasets onwards.

    Parameters
    ----------
    dataRef : `lsst.daf.persistence.ButlerDataRef`
        dataRef of the of the detector/visit to load the data for.

    Returns
    -------
    photodiodeData : `lsst.cp.pipe.photodiode.PhotodiodeData` or `None`
        The full time-series of the photodiode readings, with methods to
        integrate the photocharge, or None if the expected file isn't found.

    """
    if logger is None:
        logger = lsstLog.Log.getDefaultLogger()

    def getKeyFromDataId(dataRef, key):
        if key in dataRef.dataId:
            return dataRef.dataId[key]
        else:
            result = dataRef.getButler().queryMetadata('raw', key, dataRef.dataId)
            assert len(result) == 1, f"Failed to find unique value for {key} with {dataRef.dataId}"
            return result[0]

    dayObs = getKeyFromDataId(dataRef, 'dayObs')
    seqNum = getKeyFromDataId(dataRef, 'seqNum')

    filePattern = 'Photodiode_Readings_%s_%06d.txt'

    dayObsAsNumber = dayObs.replace('-', '')
    diodeFilename = os.path.join(dataPath, filePattern%(dayObsAsNumber, seqNum))

    if not os.path.exists(diodeFilename):
        logger.warn(f"Failed to find the photodiode data at {diodeFilename}")
        return None

    try:
        photodiodeData = PhotodiodeData(diodeFilename)
        return photodiodeData
    except Exception:
        logger.warn(f"Photodiode data found at {diodeFilename} but failed to load.")
        return None


class PhotodiodeData():
    def __init__(self, filename):
        self.times, self.values = np.loadtxt(filename, unpack=True)
        return

    def getCharge(self, subtractBaseline=False):
        if subtractBaseline:
            raise NotImplementedError

        charge = 0
        for i, (time, current) in enumerate(zip(self.times[:-1], self.values[:-1])):
            timestep = self.times[i+1] - time
            averageCurrent = (self.values[i+1] + current) / 2.
            charge += averageCurrent * timestep
        return charge
