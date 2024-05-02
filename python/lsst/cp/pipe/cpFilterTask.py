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
import numpy as np
import copy

from astropy.table import Table

import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase
import lsst.pipe.base.connectionTypes as cT

from . import CalibCombineTask
from lsst.daf.base import PropertyList

__all__ = ["CpFilterScanTask", "CpFilterScanTaskConfig"]


class CpFilterScanConnections(pipeBase.PipelineTaskConnections,
                              dimensions=("instrument", "detector")):
    inputExpHandles = cT.Input(
        name="postISRCCD",
        doc="Input exposures to measure statistics from.",
        storageClass="Exposure",
        dimensions=("instrument", "physical_filter", "exposure", "detector"),
        multiple=True,
        deferLoad=True,
    )
    inputElectrometerHandles = cT.Input(
        name="electrometer",
        doc="Input electrometer data.",
        storageClass="IsrCalib",
        dimensions=("instrument", "exposure"),
        multiple=True,
        deferLoad=True,
    )
    outputData = cT.Output(
        name="cpFilterScanTest",
        doc="Output table to write.",
        storageClass="ArrowAstropy",
        dimensions=("instrument", "detector"),
    )

    def __init__(self, *, config=None):
        super().__init__(config=config)

        if config and not config.useElectrometer:
            self.inputs.discard("inputElectrometerHandles")


class CpFilterScanTaskConfig(pipeBase.PipelineTaskConfig,
                             pipelineConnections=CpFilterScanConnections):
    maskNameList = pexConfig.ListField(
        dtype=str,
        doc="Mask list to exclude from statistics calculations.",
        default=["DETECTED", "BAD", "CR", "NO_DATA", "INTRP"]
    )
    useElectrometer = pexConfig.Field(
        dtype=bool,
        doc="Use electrometer data?",
        default=False,
    )
    referenceFilter = pexConfig.Field(
        dtype=str,
        doc="Filter to use as baseline reference.",
        default="empty~empty",
    )
    combine = pexConfig.ConfigurableField(
        target=CalibCombineTask,
        doc="Combine task to use for header merging.",
    )


class CpFilterScanTask(pipeBase.PipelineTask):
    """Create filter scan from appropriate data.
    """

    ConfigClass = CpFilterScanTaskConfig
    _DefaultName = "cpFilterScan"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.makeSubtask("combine")

    def run(self, inputExpHandles, inputElectrometerHandles=None):
        """Construct filter scan from the header and visit info of processed
        exposures.

        Parameters
        ----------
        inputExpHandles : `list` [`lsst.daf.butler.DeferredDatasetHandle`]
            Input list of exposure handles to combine.
        inputElectrometerHandles : `list` [`lsst.daf.butler.DeferredDatasetHandle`], optional # noqa W505
            Input list of electrometer/photodiode measurement handles
            to combine.

        Returns
        -------
        results : `lsst.pipe.base.Struct`
            The results struct containing:

            ``outputData``
                Final combined filter scan, with a single table
                containing the measured throughput for all input
                filters at the various wavelength values indicated in
                the exposure's observationReason
                (`astropy.table.Table`).
        """
        filterScanResults = {}
        filterSet = set()
        for handle in inputExpHandles:
            visitInfo = handle.get(component="visitInfo")
            metadata = handle.get(component="metadata")

            physical_filter = handle.dataId.physical_filter.name
            exposureId = handle.dataId.exposure
            exposureTime = visitInfo.exposureTime
            observationReason = visitInfo.observationReason

            values = [v for k, v in metadata.items() if "LSST ISR FINAL MEAN" in k]

            scaleFactor = exposureTime
            if self.config.useElectrometer:
                for eHandle in inputElectrometerHandles:
                    if eHandle.dataId.exposure == exposureId:
                        charge = eHandle.get().integrate()
                        if charge > 0.0 and np.isfinite(charge):
                            scaleFactor /= charge
                        break

            scan = {
                'physical_filter': physical_filter,
                'scale': scaleFactor,
                'flux': np.median(values),
            }
            filterSet.add(physical_filter)

            _, key = observationReason.split("_")
            key = float(key)
            if key in filterScanResults:
                filterScanResults[key].append(scan)
            else:
                filterScanResults[key] = [scan]

        filterScan = []
        for wavelength in filterScanResults.keys():
            scans = filterScanResults[wavelength]

            referenceScan = [x for x in scans if x['physical_filter'] == self.config.referenceFilter][0]
            referenceValue = referenceScan['scale'] / referenceScan['flux']

            wavelengthScan = {'wavelength': float(wavelength)}
            for filterName in filterSet:
                wavelengthScan[filterName] = np.nan

            for scan in scans:
                if np.isfinite(wavelengthScan[scan['physical_filter']]):
                    self.log.warn(f"Multiple instances of filter: {scan['physical_filter']}")
                wavelengthScan[scan['physical_filter']] = referenceValue * scan['flux'] / scan['scale']

            filterScan.append(copy.copy(wavelengthScan))

        outMetadata = PropertyList()
        self.combine.combineHeaders(inputExpHandles, calib=None, metadata=outMetadata,
                                    calibType="FilterScan", scales=None)
        filteredMetadata = {k: v for k, v in outMetadata.toDict().items() if v is not None}

        catalog = Table(filterScan)
        catalog.meta = filteredMetadata

        return pipeBase.Struct(outputData=catalog)
