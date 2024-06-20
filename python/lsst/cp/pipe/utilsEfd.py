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
import numpy as np
import requests

from astropy.table import Table
from astropy.time import Time
from urllib.parse import urljoin


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

        authDict = self._getAuth(efdInstance)
        self._auth = (authDict["username"], authDict["password"])
        self._databaseName = "efd"
        self._databaseUrl = urljoin(f"https://{authDict['host']}", authDict["path"])

    def _getAuth(self, instanceAlias):
        """Get authorization credentials.

        Parameters
        ----------
        instanceAlias : `str`
            EFD instance to get credentials for.

        Returns
        -------
        credentials : `dict` [`str`, `str`]
            A dictionary of authorization credentials, including at
            least these key/value pairs:

            ``"username"``
                Login username.
            ``"password"``
                Login passwords.
            ``"host"``
                Host to connect to.
            ``"path"``
                Directory path for EFD instance.

        Raises
        ------
        RuntimeError :
            Raised if the HTTPS request fails.
        """
        serviceEndpoint = "https://roundtable.lsst.codes/segwarides/"
        url = urljoin(serviceEndpoint, f"creds/{instanceAlias}")
        response = requests.get(url)

        if response.status_code == 200:
            return response.json()
        else:
            raise RuntimeError(f"Could not connect to {url}")

    def getSchemaDtype(self, topicName):
        """Get datatypes for a topic.

        Parameters
        ----------
        topicName : `str`
            Topic to get datatypes for

        Returns
        -------
        datatypes : `list` [`tuple` [`str`, `str`]]
            List of tuples of field names and data types.
        """
        query = f"SHOW FIELD KEYS FROM \"{topicName}\""
        data = self.query(query)

        values = data["results"][0]["series"][0]["values"]

        dtype = [("time", "str")]
        for (fieldName, fieldType) in values:
            if fieldType == "float":
                fieldDtype = np.float64
            elif fieldType == "integer":
                fieldDtype = np.int64
            elif fieldType == "string":
                fieldDtype = "str"
            dtype.append((fieldName, fieldDtype))
        return dtype

    def query(self, query):
        """Execute an EFD query.

        Parameters
        ----------
        query : `str`
            Query to run.

        Returns
        -------
        results : `dict`
            Dictionary of results returned.

        Raises
        ------
        RuntimeError :
            Raised if the the database could not be read from.
        """
        params = {
            "db": self._databaseName,
            "q": query,
        }

        try:
            response = requests.get(f"{self._databaseUrl}/query", params=params, auth=self._auth)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Could not read data from database with query: {query}") from e

    def selectTimeSeries(self, topicName, fields=[], startDate=None, endDate=None):
        """Query a topic for a time series.

        Parameters
        ----------
        topicName : `str`
            Database "topic" to query.
        fields : `list`, optional
            List of fields to return.  If empty, all fields are
            returned.
        startDate : `astropy.time.Time`, optional
            Start date (in UTC) to limit the results returned.
        endDate : `astropy.time.Time`, optional
            End date (in UTC) to limit the results returned.

        Returns
        -------
        table : `astropy.table.Table`
            A table containing the fields requested, with each row
            corresponding to one date (available in the ``"time"``
            column).
        """
        query = "SELECT "

        if not fields:
            query += "*"
        else:
            query += ",".join(fields)

        query += f" FROM \"{topicName}\""

        if startDate is not None or endDate is not None:
            query += " WHERE"

        if startDate is not None:
            query += f" time >= '{startDate.utc.isot}Z'"
            if endDate is not None:
                query += " AND"
        if endDate is not None:
            query += f" time <= '{endDate.utc.isot}Z'"

        data = self.query(query)

        # data is a dictionary with one key, "results"
        results = data["results"][0
]
        if "series" in results:
            series = results["series"][0]
        else:
            raise RuntimeError(f"No results found for query: {query}")

        schemaDtype = self.getSchemaDtype(topicName)
        tableDtype = []
        for dtype in schemaDtype:
            if dtype[0] in series["columns"]:
                tableDtype.append(dtype[1])

        table = Table(rows=series["values"], names=series["columns"], dtype=tableDtype)
        table["time"] = Time(table["time"], scale="utc")
        table.sort("time")
        if 'private_sndStamp' in table.columns:
            table['private_sndStamp'] = Time(table['private_sndStamp'], format='unix_tai')
            table.sort("private_sndStamp")

        return table

    def searchResults(self, data, dateStr):
        """Determine the entry for a specific date.

        Parameters
        ----------
        data : `astropy.table.Table`
            The table of results from the EFD.
        dateStr : `str`
            The date to look up in the status for.

        Returns
        -------
        result = `astropy.table.Row`
            The row of the data table corresponding to ``dateStr``.
        """
        dateValue = Time(dateStr, format='isot', scale='tai')
        # Table is now sorted on "time", which is in UTC.

        # Check that the date we want to consider is contained in the
        # EFD data.
        if data["time"][0] > dateValue or data["time"][-1] < dateValue:
            raise RuntimeError("Requested date is outside of data range.")

        # Binary search through the EFD entries in date, until the
        # most recent monochromator state update prior to the spectrum
        # in question is found.
        low = 0
        high = len(data)
        idx = (high + low) // 2
        found = False
        iteration = 0
        while not found:
            if idx < 0 or idx > len(data) or iteration > 20:
                raise RuntimeError("Search for date failed?")

            myTime = data["private_sndStamp"][idx]
            if myTime <= dateValue:
                low = idx
            elif myTime > dateValue:
                high = idx

            idx = (high + low) // 2
            iteration += 1
            if high - low == 1:
                found = True
            self.log.info("parse search %d %d %d %d %s %s",
                           low, high, idx, found, myTime, dateValue)

        # End binary search.
        return data[idx], idx

    def getEfdMonochromatorData(self, dataSeries=None, dateMin=None, dateMax=None):
        """Retrieve Monochromator data from the EFD.

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
        results : `astropy.table.Table`
            The table of results returned from the EFD.
        """
        # This is currently the only monochromator available.
        dataSeries = dataSeries if dataSeries else "lsst.sal.ATMonochromator.logevent_wavelength"

        if dateMin:
            startDate = Time(dateMin, format='isot', scale='tai')
        else:
            startDate = None
        if dateMax:
            stopDate = Time(dateMax, format='isot', scale='tai')
        else:
            stopDate = None

        results = self.selectTimeSeries(dataSeries, ['wavelength', 'private_sndStamp'],
                                        startDate, stopDate)
        return results

    def parseMonochromatorStatus(self, data, dateStr):
        """Determine monochromator status for a specific date.

        Parameters
        ----------
        data : `astropy.table.Table`
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
        result, _ = self.searchResults(data, dateStr)
        myTime = result["private_sndStamp"]
        return myTime.strftime("%Y-%m-%dT%H:%M:%S.%f"), result['wavelength']

    def getEfdElectrometerData(self, dataSeries=None, dateMin=None, dateMax=None):
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
        results : `astropy.table.Table`
            The table of results returned from the EFD.
        """
        # This is currently the only monochromator available.
        dataSeries = dataSeries if dataSeries else "lsst.sal.Electrometer.logevent_intensity"
        # "lsst.sal.Electrometer.logevent_intensity"

        if dateMin:
            startDate = Time(dateMin, format='isot', scale='tai')
        else:
            startDate = None
        if dateMax:
            stopDate = Time(dateMax, format='isot', scale='tai')
        else:
            stopDate = None

        results = self.selectTimeSeries(dataSeries, [],
                                        # ['intensity', 'private_sndStamp'],
                                        startDate, stopDate)

        return results

    def parseElectrometerStatus(self, data, dateStr, index=201):
        """Determine monochromator status for a specific date.

        Parameters
        ----------
        data : `astropy.table.Table`
            The dataframe of monochromator results from the EFD.
        dateStr : `str`
            The date to look up in the status for.
        index : `int`
            The salIndex of the device we want to read.  For LATISS,
            this should be 201.

        Returns
        -------
        indexDate : `str`
            Date string indicating the monochromator state change.
        wavelength : `float`
            Monochromator commanded peak.

        """
        if index is not None:
            mask = (data['salIndex'] == index)
            data = data[mask]

        result, idx = self.searchResults(data, dateStr)
        myTime = result["private_sndStamp"]

        # import pdb; pdb.set_trace()
        return myTime.strftime("%Y-%m-%dT%H:%M:%S.%f"), result['intensity']
