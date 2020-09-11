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
import datetime
from astropy.time import Time

import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase
from lsst.daf.butler import DatasetType


class CertifyCalibration(pipeBase.Task):
    """Create a way to bless existing calibration products.

    The inputs are assumed to have been constructed via cp_pipe, and
    already exist in the butler.

    Parameters
    ----------
    butler : `lsst.daf.butler.Butler`
        Butler repository to use.
    inputCollection : `str`
        Data collection to pull calibrations from.
    outputCollection : `str`
        Data collection to store final calibrations.
    **kwargs :
        Additional arguments forwarded to `lsst.pipe.base.Task.__init__`.
    """
    _DefaultName = 'CertifyCalibration'
    ConfigClass = pexConfig.Config

    def __init__(self, *, butler, inputCollection, outputCollection,
                 **kwargs):
        super().__init__(**kwargs)
        self.butler = butler
        self.registry = self.butler.registry
        self.inputCollection = inputCollection
        self.outputCollection = outputCollection

        self.calibrationLabel = None
        self.instrument = None

    def findInputs(self, datasetTypeName, inputDatasetTypeName=None):
        """Find and prepare inputs for blessing.

        Parameters
        ----------
        datasetTypeName : `str`
            Dataset that will be blessed.
        inputDatasetTypeName : `str`, optional
            Dataset name for the input datasets.  Default to
            datasetTypeName + "Proposal".

        Raises
        ------
        RuntimeError
            Raised if no input datasets found or if the calibration
            label exists and is not empty.
        """
        if inputDatasetTypeName is None:
            inputDatasetTypeName = datasetTypeName + "Proposal"

        self.inputValues = list(self.registry.queryDatasets(inputDatasetTypeName,
                                                            collections=[self.inputCollection],
                                                            deduplicate=True))
        # THIS IS INELEGANT AT BEST => fixed by passing deduplicate=True above.
        # self.inputValues = list(filter(lambda vv: self.inputCollection in vv.run, self.inputValues))

        if len(self.inputValues) == 0:
            raise RuntimeError(f"No inputs found for dataset {inputDatasetTypeName} "
                               f"in {self.inputCollection}")

        # Construct calibration label and choose instrument to use.
        self.calibrationLabel = f"{datasetTypeName}/{self.inputCollection}"
        self.instrument = self.inputValues[0].dataId['instrument']

        # Prepare combination of new data ids and object data:
        self.newDataIds = [value.dataId for value in self.inputValues]

        self.objects = [self.butler.get(value) for value in self.inputValues]

    def registerCalibrations(self, datasetTypeName):
        """Add blessed inputs to the output collection.

        Parameters
        ----------
        datasetTypeName : `str`
            Dataset type these calibrations will be registered for.
        """
        # Find/make the run we will use for the output
        self.registry.registerRun(self.outputCollection)
        self.butler.run = self.outputCollection
        self.butler.collection = None

        try:
            self.registerDatasetType(datasetTypeName, self.newDataIds[0])
        except Exception as e:
            print(f"Could not registerDatasetType {datasetTypeName}.  Failure {e}?")

        with self.butler.transaction():
            for newId, data in zip(self.newDataIds, self.objects):
                self.butler.put(data, datasetTypeName, dataId=newId,
                                calibration_label=self.calibrationLabel,
                                producer=None)

    def registerDatasetType(self, datasetTypeName, dataId):
        """Ensure registry can handle this dataset type.

        Parameters
        ----------
        datasetTypeName : `str`
            Name of the dataset that will be registered.
        dataId : `lsst.daf.butler.dataId`
            Data ID providing the list of dimensions for the new
            datasetType.
        """
        storageClassMap = {'crosstalk': 'CrosstalkCalib'}
        storageClass = storageClassMap.get(datasetTypeName, 'ExposureF')

        dimensionArray = set(list(dataId.keys()) + ["calibration_label"])
        datasetType = DatasetType(datasetTypeName,
                                  dimensionArray,
                                  storageClass,
                                  universe=self.butler.registry.dimensions)
        self.butler.registry.registerDatasetType(datasetType)

    def addCalibrationLabel(self, name=None, instrument=None,
                            beginDate="1970-01-01", endDate="2038-12-31"):

        """Method to allow tasks to add calibration_label for master calibrations.

        Parameters
        ----------
        name : `str`, optional
            A unique string for the calibration_label key.
        instrument : `str`, optional
            Instrument this calibration is for.
        beginDate : `str`, optional
            An ISO 8601 date string for the beginning of the valid date range.
        endDate : `str`, optional
            An ISO 8601 date string for the end of the valid date range.

        Raises
        ------
        RuntimeError :
            Raised if the instrument or calibration_label name are not set.
        """
        if name is None:
            name = self.calibrationLabel
        if instrument is None:
            instrument = self.instrument
        if name is None and instrument is None:
            raise RuntimeError("Instrument and calibration_label name not set.")

        try:
            existingValues = self.registry.queryDataIds(['calibration_label'],
                                                        instrument=self.instrument,
                                                        calibration_label=name)
            existingValues = [a for a in existingValues]
            print(f"Found {len(existingValues)} Entries for {self.calibrationLabel}")
        except LookupError:
            self.butler.registry.insertDimensionData(
                "calibration_label",
                {
                    "name": name,
                    "instrument": instrument,
                    "datetime_begin": Time(datetime.datetime.fromisoformat(beginDate), scale='utc'),
                    "datetime_end": Time(datetime.datetime.fromisoformat(endDate), scale='utc'),
                }
            )
