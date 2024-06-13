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

from .cpCombine import CalibCombineTask
from .utilsEfd import CpEfdClient
from lsst.daf.base import PropertyList, DateTime

__all__ = [
    "CpFilterScanTask", "CpFilterScanTaskConfig",
    "CpMonochromatorScanTask", "CpMonochromatorScanConfig"
]


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
        name="cpFilterScan",
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
        default="empty",
    )
    combine = pexConfig.ConfigurableField(
        target=CalibCombineTask,
        doc="Combine task to use for header merging.",
    )
    efdClientInstance = pexConfig.Field(
        dtype=str,
        doc="EFD instance to use for monochromator results.",
        default="usdf_efd",
    )


class CpFilterScanTask(pipeBase.PipelineTask):
    r"""Create filter scan from appropriate data.

    This task constructs a filter scan from a sequence of flat
    exposures taken in the following manner:

    - A monochromator is set to a target wavelength.
    - An optional spectrum may be taken with the fiber spectrograph to
      provide an independent measure of the peak wavelength and
      bandpass.
    - A flat exposure is taken with a "reference filter," usually a
      white-band or empty filter, that provides a baseline source
      brightness at the monochromator's target wavelength.
    - A flat exposure is taken with the filter to be scanned.
    - Optional electrometer/photodiode data may also be taken during
      the two flat exposures to correct for source brightness
      variations.

    From these pairs of exposures, we can determine the filter
    throughput by calculating the flux per second with the filter:
    :math:`F_filter(\lambda0) = median(f_amplifiers) / t_exposure`
    And without:
    :math:`F_reference(\lambda0) = median(f_amplifiers) / t_exposure`
    where the f_amplifiers are the per-amplifier statistics calculated
    by IsrTask. If the illumination source was perfectly stable, the
    filter throughput at that wavelength would simply be:
    :math:`throughput_raw(\lambda0) = F_filter / F_reference`

    We can correct for any illumination changes with the optional
    the electrometer measurements, E, which provide an independent
    measure of the incident flux for the two exposures, such that:
    :math:`throughput(\lambda0) = throughput_raw * E_reference / E_filter`

    Repeating this procedure at multiple monochromator settings builds
    up a catalog of throughput measurements across the filter
    bandpass.  Additional differences can exist between the
    monochromator setting (retrieved here from the EFD) and the actual
    wavelengths of light that are permitted, so a matching
    CpMonochromatorScan can be generated to determine what the actual
    values of :math:`\lambda0` observed were.
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
        efdClient = CpEfdClient(efdInstance=self.config.efdClientInstance)
        monochromatorData = efdClient.getEfdMonochromatorData()

        # Iterate over input exposures:
        for handle in inputExpHandles:
            visitInfo = handle.get(component="visitInfo")
            metadata = handle.get(component="metadata")

            physical_filter = handle.dataId['physical_filter']
            exposureId = handle.dataId['exposure']
            exposureTime = visitInfo.exposureTime

            values = [v for k, v in metadata.items() if "LSST ISR FINAL MEDIAN" in k]

            scaleFactor = exposureTime
            if self.config.useElectrometer:
                for eHandle in inputElectrometerHandles:
                    if eHandle.dataId.exposure == exposureId:
                        charge = eHandle.get().integrate()
                        if charge > 0.0 and np.isfinite(charge):
                            scaleFactor /= charge
                        break

            # Create a scan entry for this exposure.
            scan = {
                'physical_filter': physical_filter,
                'scale': scaleFactor,
                'flux': np.median(values),
            }
            filterSet.add(physical_filter)

            _, wavelengthKey = efdClient.parseMonochromatorStatus(monochromatorData,
                                                                  visitInfo.date.toString(DateTime.TAI))
            wavelengthKey = float(wavelengthKey)
            self.log.debug(f"Scan: {exposureId} {wavelengthKey} {scan}")

            # Append the scan for this exposure to the list of other
            # scans taken at this wavelength setting:
            if wavelengthKey in filterScanResults:
                filterScanResults[wavelengthKey].append(scan)
            else:
                filterScanResults[wavelengthKey] = [scan]

        filterScan = []
        for wavelength in filterScanResults.keys():
            # We expect there to be at least one pair of exposures
            # with a given wavelength setting: One with a filter in
            # the beam, and one with the "reference filter" (which
            # should be the empty/white/no-filter setting).  The ratio
            # of the scaled fluxes in this pair should give the filter
            # transmission at that wavelength for that filter.
            scans = filterScanResults[wavelength]

            referenceScan = [x for x in scans if x['physical_filter'] == self.config.referenceFilter]
            if len(referenceScan) == 0:
                # No reference scan at this wavelength.
                self.log.warning(f"No reference scan at this wavelength: {wavelengthKey}")
                continue
            referenceScan = referenceScan[0]
            referenceValue = referenceScan['scale'] / referenceScan['flux']

            # Create a dictionary of measurements at this wavelength,
            # using the filter names as the keys.  Since we may do
            # multiple filters in a single sequence (iterating through
            # all filters at a certain monochromator setting),
            # initialize all known filters (from our initial exposure
            # scan) to NaN.  Note that the reference filter is a known
            # filter, and should have a throughput of 1.0 relative to
            # itself.
            wavelengthScan = {'wavelength': float(wavelength)}
            for filterName in filterSet:
                wavelengthScan[filterName] = np.nan

            for scan in scans:
                # If the entry for this filter already exists, then we
                # have multiple entries at this wavelength.
                if np.isfinite(wavelengthScan[scan['physical_filter']]):
                    self.log.warning(
                        f"Multiple instances of filter {scan['physical_filter']} at {wavelength}"
                    )

                wavelengthScan[scan['physical_filter']] = referenceValue * scan['flux'] / scan['scale']

            filterScan.append(copy.copy(wavelengthScan))

        outMetadata = PropertyList()
        self.combine.combineHeaders(inputExpHandles, calib=None, metadata=outMetadata,
                                    calibType="FilterScan", scales=None)
        filteredMetadata = {k: v for k, v in outMetadata.toDict().items() if v is not None}

        catalog = Table(filterScan)
        catalog.meta = filteredMetadata

        return pipeBase.Struct(outputData=catalog)


class CpMonochromatorScanConnections(pipeBase.PipelineTaskConnections,
                                     dimensions=("instrument", )):
    inputExpHandles = cT.Input(
        name="rawSpectrum",
        doc="Input spectrograph measurements.",
        storageClass="FiberSpectrum",
        dimensions=("instrument", "exposure", "detector"),
        multiple=True,
        deferLoad=True,
    )
    outputData = cT.Output(
        name="cpMonochromatorScan",
        doc="Output table to write.",
        storageClass="ArrowAstropy",
        dimensions=("instrument", ),
    )


class CpMonochromatorScanConfig(pipeBase.PipelineTaskConfig,
                                pipelineConnections=CpMonochromatorScanConnections):
    efdClientInstance = pexConfig.Field(
        dtype=str,
        doc="EFD instance to use for monochromator results.",
        default="usdf_efd",
    )
    efdUnits = pexConfig.Field(
        dtype=str,
        doc="Units to use for EFD monochromator results.",
        default="nm",
    )
    headerDateKey = pexConfig.Field(
        dtype=str,
        doc="Header keyword to use for observation date.",
        default="DATE-BEG",
    )
    peakBoxSize = pexConfig.Field(
        dtype=int,
        doc="Half-size of the box used for fitting the spectrum peak.",
        default=50,
    )


class CpMonochromatorScanTask(pipeBase.PipelineTask):
    """Compare EFD monochromator results to fiber spectrograph spectra.

    This task provides a complementary measurement to associate with
    the CpFilterScan.  While taking the filter scan exposures used for
    CpFilterScanTask, the attached fiber spectrograph can be used to
    measure the spectrum of the light that passes through the
    monochromator.  This task takes those spectra, fits a Gaussian to
    the peak in each one, and records those fit parameters along with
    the monochromator setting recorded in the EFD.  This information
    can then be used to correct the CpFilterScan measurements from the
    nominal wavelength values to those actually observed.
    """
    ConfigClass = CpMonochromatorScanConfig
    _DefaultName = "cpMonochromatorScan"

    def run(self, inputExpHandles):
        """Match EFD results to spectrograph.

        Parameters
        ----------
        inputExpHandles : `list` [`lsst.daf.butler.DeferredDatasetHandle`]
            Input list of exposure handles to combine.

        Returns
        -------
        results : `lsst.pipe.base.Struct`
            The results struct containing:

            ``outputData``
                Final combined filter scan, with a single table
                containing the commanded monochromator wavelength, as
                well as the Gaussian fit to the peak of the fiber
                spectrograph spectrum (`astropy.table.Table`).
        """
        monochromatorScanResults = []
        efdClient = CpEfdClient(efdInstance=self.config.efdClientInstance)
        monochromatorData = efdClient.getEfdMonochromatorData()

        for handle in inputExpHandles:
            exposureId = handle.dataId['exposure']
            spectrum = handle.get()

            # Fit spectrum with Gaussian model.
            fitMean, fitSigma, peakWavelength, fwhm, fitRange = self._fitPeak(spectrum,
                                                                              self.config.peakBoxSize)

            # Look up the monochromator state for this spectrum:
            date = spectrum.metadata[self.config.headerDateKey]
            monoDate, monoValue = efdClient.parseMonochromatorStatus(monochromatorData, date)
            entry = {
                'exposure': exposureId,
                'fit_mean': fitMean,
                'fit_sigma': fitSigma,
                'peak': peakWavelength,
                'fwhm': fwhm,
                'fit_range': fitRange,
                'exp_date': date,
                'mono_date': monoDate,
                'mono_state': monoValue,
            }
            monochromatorScanResults.append(entry)

        return pipeBase.Struct(
            outputData=Table(monochromatorScanResults)
        )

    @staticmethod
    def _fitPeak(spectrum, peakBoxSize=20):
        """Fit a Gaussian to the peak of the spectrum.

        Parameters
        ----------
        spectrum : `lsst.obs.fiberspectrograph.FiberSpectrum`
            The spectrum to fit.
        peakBoxSize : `int`
            Size of the box to use for fitting.

        Returns
        -------
        mean : `float`
            The fitted mean of the peak.
        sigma : `float`
            The fitted standard deviation of the peak.
        peak : `float`
            The wavelength of the peak.
        fwhm : `float`
            Wavelength quantized estimate of the FWHM.
        range : `float`
            Range over which Gaussian fit was performed.
        """
        maxIdx = np.argmax(spectrum.flux)
        maxF = spectrum.flux[maxIdx]
        med = np.median(spectrum.flux)

        # Normalize spectrum so the peak is at 1.0 and the median
        # value is zero.
        normFlux = (spectrum.flux - med) / (maxF - med)

        highHM = None
        lowHM = None
        high = None
        low = None
        # Search for the 50% points for FWHM estimate.  Search for the
        # 10% points to define the range for Gaussian fit.
        for dx in range(maxIdx, maxIdx + peakBoxSize):
            if not highHM:
                if normFlux[dx] < 0.5:
                    highHM = dx
            if not high:
                if normFlux[dx] < 0.1:
                    high = dx
        for dx in range(maxIdx, maxIdx - peakBoxSize, -1):
            if not lowHM:
                if normFlux[dx] > 0.5:
                    lowHM = dx - 1
            if not low:
                if normFlux[dx] > 0.1:
                    low = dx - 1

        # The above search should be fine, but let's ensure we have
        # reasonable limits.
        if highHM and not high:
            high = highHM
        if lowHM and not low:
            low = lowHM

        # Do Gaussian fit in log-space
        ff = np.polyfit(spectrum.wavelength[low:high].to_value(), np.log(normFlux[low:high]), 2)
        # Convert fit parameters to Gaussian parameters:
        # log(F) = (log(A) - 0.5 * (m/s)**2) + x m/s**2 - 0.5 (x/s)**2
        #                 ff[2]                  ff[1]     ff[0]
        # s**2 = 1 / (-2 * ff[0]) = m / ff[1]

        sigma = np.sqrt(1 / (-2.0 * ff[0]))
        mean = -0.5 * ff[1] / ff[0]

        return (mean, sigma,
                spectrum.wavelength[maxIdx].to_value(),
                spectrum.wavelength[highHM].to_value() - spectrum.wavelength[lowHM].to_value(),
                spectrum.wavelength[high].to_value() - spectrum.wavelength[low].to_value())
