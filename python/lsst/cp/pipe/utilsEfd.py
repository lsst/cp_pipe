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

__all__ = ['CpEfdClient']

import logging
import asyncio
import nest_asyncio

from astropy.time import Time

haveEfd = True

try:
    from lsst_efd_client import EfdClient
except ImportError:
    haveEfd = False
    pass


class CpEfdClient():
    """An EFD client to retrieve calibration results.

    Parameters
    ----------
    efdInstance : `str`, optional
        EFD instance name to connect to.
    log : `logging.Logger`, optional
        Log to write messages to.
    """

    def __init__(self, efdInstance="usdf_efd", log=None):
        self.log = log if log else logging.getLogger(__name__)

        nest_asyncio.apply()
        if haveEfd:
            self.client = EfdClient(efdInstance)
        else:
            self.client = None
            self._emitWarning()

    def _emitWarning(self):
        if not haveEfd:
            self.log.warning("EFD client not available.")
            return None

    def close(self):
        """Delete self, which will close all open connections."""
        del self

    def getEfdMonochromatorData(self, dataSeries=None, dateMin=None, dateMax=None):
        """Retrieve Electrometer data from the EFD.

        Parameters
        ----------
        dataSeries : `str`, optional
            Data series to request from the EFD.
        dateMin : `str`, optional
            Minimum date to retrieve from EFD.
        dateMax : `str`, optional
            Maximum date to retrieve from EFD.

        Returns
        -------
        results : `pandas.DataFrame`
            The table of results returned from the EFD.
        """
        self._emitWarning()
        # This is currently the only monochromator available.
        dataSeries = dataSeries if dataSeries else "lsst.sal.ATMonochromator.logevent_wavelength"

        if dateMin:
            start = Time(dateMin, format='isot', scale='utc')
        else:
            start = Time("1970-01-01T00:00:00", format='isot', scale='utc')
        if dateMax:
            stop = Time(dateMax, format='isot', scale='utc')
        else:
            stop = Time("2199-01-01T00:00:00", format='isot', scale='utc')

        loop = asyncio.get_event_loop()
        results = loop.run_until_complete(self.client.select_time_series(dataSeries, ['wavelength'],
                                                                         start.utc, stop.utc))
        results.sort_index(inplace=True)
        return results

    def parseMonochromatorStatus(self, data, dateStr):
        """Determine monochromator status for a specific date.

        Parameters
        ----------
        data : `pandas.DataFrame`
            The dataframe of monochromator results from the EFD.
        dateStr : `str`
            The date to look up in the status for.

        Returns
        -------
        indexDate : `str`
            Date string indicating the monochromator state change.
        wavelength : `float`
            Monochromator commanded peak.
        """
        dateValue = Time(dateStr, format='isot', scale='utc')
        # I have words here.
        if data.index[0] > dateValue or data.index[-1] < dateValue:
            raise RuntimeError("Requested date is outside of data range.")

        # I have more words.
        low = 0
        high = len(data)
        idx = (high + low) // 2
        found = False
        iteration = 0
        while not found:
            if idx < 0 or idx > len(data) or iteration > 10:
                raise RuntimeError("Search for date failed?")
            self.log.debug("parse search %d %d %d %d %s %s",
                           low, high, idx, found, data.index[idx], dateValue)

            if data.index[idx] <= dateValue:
                low = idx
            elif data.index[idx] > dateValue:
                high = idx

            idx = (high + low) // 2
            iteration += 1
            if high - low == 1:
                found = True
        # End binary search.

        return data.index[idx], data['wavelength'].iloc[idx]
