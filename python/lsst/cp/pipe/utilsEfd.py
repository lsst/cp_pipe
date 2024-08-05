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

__all__ = ["CpEfdClient"]

import logging
import numpy as np
import re
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

    def __init__(self, efdInstance="usdf_efd", dieOnSearch=False, log=None):
        self.log = log if log else logging.getLogger(__name__)
        self.dieOnSearch = dieOnSearch

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
            Start date to limit the results returned.
        endDate : `astropy.time.Time`, optional
            End date to limit the results returned.

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
        results = data["results"][0]
        if "series" in results:
            series = results["series"][0]
        else:
            raise RuntimeError(f"No results found for query: {query}")

        schemaDtype = self.getSchemaDtype(topicName)
        tableDtype = []
        for dtype in schemaDtype:
            if dtype[0] in series["columns"]:
                tableDtype.append(dtype[1])

        # The value stored in "time" may not be consistent and
        # monotonic (and is in UTC).  "private_sndStamp" comes from
        # the device itself, and is therefore preferred.
        table = Table(rows=series["values"], names=series["columns"], dtype=tableDtype)
        table["time"] = Time(table["time"], scale="utc").tai
        table.sort("time")
        if "private_sndStamp" in table.columns:
            table["private_sndStamp"] = Time(table["private_sndStamp"], format="unix_tai")
            table.sort("private_sndStamp")

        return table

    def searchResults(self, data, dateStr):
        """Find the row entry in ``data`` immediately preceding the specified
        date.

        Parameters
        ----------
        data : `astropy.table.Table`
            The table of results from the EFD.
        dateStr : `str`
            The date (in TAI) to look up in the status for.

        Returns
        -------
        result = `astropy.table.Row`
            The row of the data table corresponding to ``dateStr``.
        """
        dateValue = Time(dateStr, scale='tai', format="isot")
        # Table is now sorted on "time", which is in TAI.

        # Check that the date we want to consider is contained in the
        # EFD data.
        if data["time"][0] > dateValue or data["time"][-1] < dateValue:
            msg = f"Requested date {dateStr} outside of data range {data['time'][0]}-{data['time'][-1]}"
            if self.dieOnSearch:
                raise RuntimeError(msg)
            else:
                # Return the start, as we're more likely to have errors in that direction.
                self.log.warning(msg)
                return data[0], np.nan

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
                raise RuntimeError(f"Search for date failed: {dateValue} {idx} {iteration}.")

            myTime = data["private_sndStamp"][idx]
            if myTime <= dateValue:
                low = idx
            elif myTime > dateValue:
                high = idx

            idx = (high + low) // 2
            iteration += 1
            if high - low == 1:
                found = True
            self.log.debug("parse search %d %d %d %d %s %s",
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
            Minimum date (in TAI) to retrieve from EFD.
        dateMax : `str`, optional
            Maximum date (in TAI) to retrieve from EFD.

        Returns
        -------
        results : `astropy.table.Table`
            The table of results returned from the EFD.
        """
        # This is currently the only monochromator available.
        dataSeries = dataSeries if dataSeries else "lsst.sal.ATMonochromator.logevent_wavelength"

        if dateMin:
            startDate = Time(dateMin, format="isot", scale="tai")
        else:
            startDate = None
        if dateMax:
            stopDate = Time(dateMax, format="isot", scale="tai")
        else:
            stopDate = None

        results = self.selectTimeSeries(dataSeries, ["wavelength", "private_sndStamp"],
                                        startDate, stopDate)
        return results

    def parseMonochromatorStatus(self, data, dateStr):
        """Determine monochromator status for a specific date.

        Parameters
        ----------
        data : `astropy.table.Table`
            The dataframe of monochromator results from the EFD.
        dateStr : `str`
            The date (in TAI) to look up in the status for.

        Returns
        -------
        indexDate : `str`
            Date string (in TAI) indicating the monochromator state change.
        wavelength : `float`
            Monochromator commanded peak.
        """
        result, _ = self.searchResults(data, dateStr)
        myTime = result["private_sndStamp"]
        return myTime.strftime("%Y-%m-%dT%H:%M:%S.%f"), result["wavelength"]

    def getEfdElectrometerData(self, dataSeries=None, dateMin=None, dateMax=None):
        """Retrieve Electrometer data from the EFD.

        Parameters
        ----------
        dataSeries : `str`, optional
            Data series to request from the EFD.
        dateMin : `str`, optional
            Minimum date (in TAI) to retrieve from EFD.
        dateMax : `str`, optional
            Maximum date (in TAI) to retrieve from EFD.

        Returns
        -------
        results : `astropy.table.Table`
            The table of results returned from the EFD.
        """
        # All electrometer data gets written to the same series.
        defaultSeries = "lsst.sal.Electrometer.logevent_intensity"
        alternateSeries = "lsst.sal.Electrometer.logevent_logMessage"
        dataSeries = dataSeries if dataSeries else defaultSeries

        if dateMin:
            startDate = Time(dateMin, format="isot", scale="tai")
        else:
            startDate = None
        if dateMax:
            stopDate = Time(dateMax, format="isot", scale="tai")
        else:
            stopDate = None

        if (dataSeries == alternateSeries and (startDate is None or stopDate is None)):
            raise RuntimeError("Cannot query logevent_logMessage without dates to limit memory issues.")

        results = self.selectTimeSeries(dataSeries, [],
                                        startDate, stopDate)
        if dataSeries == "lsst.sal.Electrometer.logevent_logMessage":
            results = self.rewriteElectrometerStatus(results)
        return results

    def rewriteElectrometerStatus(self, inResults):
        """Rewrite intermediate electrometer data extracted from the EFD
        logEvents.

        Parameters
        ----------
        inResults : `astropy.table.Table`
            The table of results returned from the EFD.

        Returns
        -------
        outResults : `astropy.table.Table`
            The rewritten table containing only electrometer summary
            status events.
        """
        # This is fragile against upstream changes.
        # Ignore all entries that are not the ones we care about.
        outResults = inResults[inResults["functionName"] == "write_fits_file"]
        outResults = outResults[outResults["level"] == 20]

        # These will be new columns
        intensityMean = []
        intensityStdev = []
        intensityTime = []
        intensityFile = []

        for row in outResults:
            # Fallback values in case the regexp fails
            mean = np.nan
            stdev = np.nan
            time_mean = np.nan
            filename = "REGEXP_FAIL"

            # Find the last "filename\.extension" before the first newline;
            #      the last [] grouped values before the second newline;
            #      the last [] grouped values before the end of the string.
            magic = re.findall(r"\b\w+.+?(\w+?\.\w+)\n\b\w+.+\[(.*?)\]\n\b\w+.+\[(.*?)\]$",
                               row["message"])
            if len(magic) != 0:
                # If we matched, split the grouped values, cast them to floats.
                filename, intensity_str, time_str = magic[0]
                mean, median, stdev = intensity_str.split(",")
                time_mean, time_median = time_str.split(",")
                mean = float(mean)
                median = float(median)
                stdev = float(stdev)
                time_mean = float(time_mean)
                time_median = float(time_median)
                # Censor the saturated points so plots look nice.
                if np.abs(mean) > 1e37:
                    mean = np.nan

            intensityMean.append(mean)
            intensityStdev.append(stdev)
            intensityFile.append(filename)
            intensityTime.append(time_mean)

        # Add our new columns at the start of the column list
        outResults.add_column(intensityMean, name="intensity", index=1)
        outResults.add_column(intensityStdev, name="intensityStd", index=2)
        outResults.add_column(intensityTime, name="intensityTimeMean", index=3)
        outResults.add_column(intensityFile, name="expectedLfaFile", index=4)
        return outResults

    def parseElectrometerStatus(self, data, dateStr, dateEnd=None,
                                doIntegrateSamples=False, index=201):
        """Determine electrometer status for a specific date.

        Parameters
        ----------
        data : `astropy.table.Table`
            The dataframe of electrometer results from the EFD.
        dateStr : `str`
            The date (in TAI) to look up in the status for.
        dateEnd : `str`
            The end date (in TAI) to look in the status for.
        doIntegrateSamples: `bool`
            If true, take the average of all samples between
            ``dateStr`` and ``dateEnd``.
        index : `int`
            The salIndex of the device we want to read.  For LATISS,
            this should be 201.  For the main telescope, 101.

        Returns
        -------
        indexDate : `str`
            Date string (in TAI) indicating the electrometer state
            change.
        intensity: `float`
            Average electrometer intensity.
        """
        if index is not None:
            mask = (data["salIndex"] == index)
            data = data[mask]

        # searchResults returns the first entry prior to this date
        result, idx = self.searchResults(data, dateStr)

        myTime = result["private_sndStamp"]
        myIntensity = result["intensity"]
        myEndTime = None

        if doIntegrateSamples:
            myEndResult, myEndIdx = self.searchResults(data, dateEnd)
            if myEndIdx != idx:
                myEndTime = myEndResult["private_sndStamp"]
                myIntensity = np.mean(data[idx+1:myEndIdx]["intensity"])

        return (myTime.strftime("%Y-%m-%dT%H:%M:%S.%f"),
                myIntensity,
                myEndTime.strftime("%Y-%m-%dT%H:%M:%S.%f") if myEndTime else None)
