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

import lsst.afw.detection as afwDetect
import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase
import lsst.pipe.base.connectionTypes as cT

from astropy.stats import sigma_clipped_stats
from astropy.table import Table
from scipy.optimize import curve_fit

from lsst.geom import Point2I

__all__ = ["QuadNotchExtractConfig", "QuadNotchExtractTask",
           "QuadNotchMergeConfig", "QuadNotchMergeTask"]


class QuadNotchExtractConnections(pipeBase.PipelineTaskConnections,
                                  dimensions=("instrument", "exposure", "detector")):
    inputExp = cT.Input(
        name="post_isr_quadnotch",
        doc="Input ISR-processed exposures to combine.",
        storageClass="Exposure",
        dimensions=("instrument", "exposure", "detector"),
        multiple=False,
        deferLoad=False,
    )
    camera = cT.PrerequisiteInput(
        name="camera",
        storageClass="Camera",
        doc="Input camera to construct complete exposures.",
        dimensions=["instrument"],
        isCalibration=True,
    )

    outputData = cT.Output(
        name="quadNotchSingle",
        doc="Output quad-notch analysis.",
        storageClass="ArrowAstropy",
        dimensions=("instrument", "exposure", "detector"),
    )


class QuadNotchExtractConfig(pipeBase.PipelineTaskConfig,
                             pipelineConnections=QuadNotchExtractConnections):
    """Configuration for quad-notch processing."""
    nSigma = pexConfig.Field(
        dtype=float,
        default=2.0,
        doc="Significance of detected objects.",
    )
    nPixMin = pexConfig.Field(
        dtype=int,
        default=5,
        doc="Minimum area for a detected object.",
    )
    grow = pexConfig.Field(
        dtype=int,
        default=0,
        doc="CZW",
    )
    xWindow = pexConfig.Field(
        dtype=int,
        default=0,
        doc="CZW",
    )
    yWindow = pexConfig.Field(
        dtype=int,
        default=50,
        doc="CZW",
    )
    xGauge = pexConfig.Field(
        dtype=float,
        default=1.75,
        doc="CZW",
    )
    threshold = pexConfig.Field(
        dtype=float,
        default=1.2e5,
        doc="CZW",
    )
    targetReplacements = pexConfig.DictField(
        keytype=str,
        itemtype=str,
        doc="Dictionary of target names with replacements; Any not specified will not be changed.",
        default={
            "MU-COL": "HD38666"
        }
    )


class QuadNotchExtractTask(pipeBase.PipelineTask):
    """Task to measure quad-notch data."""

    ConfigClass = QuadNotchExtractConfig
    _DefaultName = "quadNotchExtract"

    FLAG_SUCCESS = 0x0
    FLAG_UNKNOWN_ERROR = 0x1
    FLAG_NO_FOOTPRINT_FOUND = 0x2

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def runQuantum(self, butlerQC, inputRefs, outputRefs):
        inputs = butlerQC.get(inputRefs)

        inputs["inputDims"] = dict(inputRefs.inputExp.dataId.required)

        outputs = self.run(**inputs)
        butlerQC.put(outputs, outputRefs)

    def run(self, inputExp, camera, inputDims):
        """Quadnotch extraction task.

        Parameters
        ----------
        inputExp : `lsst.afw.image.Exposure`
            Input exposure to do analysis on.
        camera : `lsst.afw.cameraGeom.Camera`
            Camera geometry to use.
        inputDims: `dict`
            Dictionary of input dimensions.

        Returns
        -------
        outputData : tbd
            Extracted data for this exposure.
        """
        row = {}
        detector = inputExp.getDetector()

        # Get visitInfo based values:
        row['airmass'] = inputExp.visitInfo.boresightAirmass
        row['azimuth'] = inputExp.visitInfo.boresightAzAlt[0].asDegrees()
        row['altitude'] = inputExp.visitInfo.boresightAzAlt[1].asDegrees()
        # Cast to isoformat so we don't try to persist an object
        # in the output table.
        row['date'] = inputExp.visitInfo.date.toPython().isoformat()
        row['hourangle'] = inputExp.visitInfo.boresightHourAngle.asDegrees()
        row['expTime'] = inputExp.visitInfo.exposureTime
        row['target'] = inputExp.visitInfo.object
        if row['target'] in self.config.targetReplacements.keys():
            row['target'] = self.config.targetReplacments[row['target']]

        # These can be gotten from the butler, but are also in the header.
        metadata = inputExp.getMetadata()
        row['day_obs'] = int(metadata.get("DAYOBS", 0))
        row['exposureId'] = inputDims['exposure']

        # This can be retrieved from consDB, but not really, because
        # we're not supposed to access consDB in pipetasks.
        row['mount_jitter'] = np.nan

        # Also from visitInfo, but this is where things start to happen.
        target_amp = 3
        centered = False

        match inputExp.visitInfo.observationReason:
            case "x_offset_50":
                target_amp = 2
                centered = True
            case "x_offset_-50":
                target_amp = 4
                centered = True
            case "x_offset_0":
                target_amp = 3
                centered = True
            case _:
                pass

        row['target_amp'] = target_amp
        row['centered'] = centered
        row['flags'] = self.FLAG_SUCCESS

        bbox = detector[target_amp].getBBox()
        # ampImage = inputExp[bbox]   # unused error
        # I think this gets returned on failure?

        fpSet = self.findObjects(inputExp)
        if len(fpSet.getFootprints()) == 0:
            rowAddendum = self._returnFailure(self.FLAG_NO_FOOTPRINT_FOUND)
        else:
            starCentroid, centerX = self.getCutoutLocation(inputExp, fpSet)

            # If we failed to find a center earlier, find one now.
            # This does not handle the case where the commanded offset
            # resulted in a bad exposure.
            while centered is False:
                if centerX < bbox.minX:
                    target_amp -= 1
                    bbox = detector[target_amp].getBBox()
                    row['target_amp'] = target_amp
                elif centerX > bbox.maxX:
                    target_amp += 1
                    bbox = detector[target_amp].getBBox()
                    row['target_amp'] = target_amp
                else:
                    centered = True

            cutout = inputExp[bbox]

            # Ignoring debug plots.
            rowAddendum = self.getAdvancedFluxes(cutout)

        # Combine base row with updates
        row.update(rowAddendum)
        outputTable = Table([row])

        return pipeBase.Struct(
            outputData=outputTable,
        )

    @staticmethod
    def _returnFailure(flag):
        row = {}
        row['flux'] = []
        row['cdf95'] = []
        row['centroids'] = []
        row['background'] = []
        row['flags'] = flag
        # This doesn't return everything it should
        return row

    def findObjects(self, exposure):
        """Quick detection method.

        Parameters
        ----------
        exposure : `lsst.afw.image.Exposure`
            The exposure to detect objects in.

        Returns
        -------
        footprints : `lsst.afw.detection.FootprintSet`
            Footprints of detections

        Notes
        -----
        This comes from summit_utils/utils.py/detectObjectsInExp.
        """
        exposureCopy = exposure.clone()
        median = np.nanmedian(exposureCopy.image.array)
        exposureCopy.image -= median

        threshold = afwDetect.Threshold(self.config.nSigma, afwDetect.Threshold.STDEV)
        footPrintSet = afwDetect.FootprintSet(exposureCopy.getMaskedImage(),
                                              threshold,
                                              "DETECTED",
                                              self.config.nPixMin)
        if self.config.grow > 0:
            isotropic = True
            footPrintSet = afwDetect.FootprintSet(footPrintSet, self.config.grow, isotropic)

        return footPrintSet

    def getCutoutLocation(self, inputExp, fpSet):
        """TBD
        Parameters
        ----------
        inputExp : `lsst.afw.image.Exposure`
            Exposure to study.
        fpSet : `lsst.afw.detection.FootprintSet`
            List of detected footprints.

        Returns
        -------
        centroid : `tuple`?
        centerX : float?
        """
        median = np.nanmedian(inputExp.image.array)
        centerOfMass_numerator = 0
        centerOfMass_denominator = 0

        fluxes = []
        centroids = []
        for fp in fpSet.getFootprints():
            if fp.getArea() < self.config.nPixMin:
                continue
            centroid = fp.getCentroid()
            height = fp.getBBox().height
            flux = fp.getSpans().flatten(inputExp.image.array - median).sum()
            centerOfMass_numerator += centroid[0]*flux*height
            centerOfMass_denominator += flux*height

            fluxes.append(flux)
            centroids.append(centroid)

        # Take the largest flux centroid as the central star.  This
        # should probably have shape checks.
        starCentroidIndex = np.array(fluxes).argmax()
        starCentroid = centroids[starCentroidIndex]
        centerX = centerOfMass_numerator / centerOfMass_denominator

        return starCentroid, centerX

    def getAdvancedFluxes(self, exp):
        """TBD
        Parameters
        ----------
        exp : `lsst.afw.image.Exposure`
           Single amp cutout to consider.

        Returns
        -------
        row : `dict`
            Addendum to the output row.
        """
        # CZW: No plotting, unless we add debug mode hooks.
        row = {}

        fpSet = self.findObjects(exp)
        if (fpLen := len(fpSet.getFootprints())) < 4:
            if fpLen == 0:
                return self._returnFailure(self.FLAG_NO_FOOTPRINT_FOUND)
            else:
                return self._returnFailure(self.FLAG_UNKNOWN_ERROR)

        centroids = []
        bboxes = []
        for fp in fpSet.getFootprints():
            centroid = fp.getCentroid()
            ampBBox = exp.getBBox()
            cautiousBox = ampBBox.erodedBy(100)   # CZW: This should be configurable
            if cautiousBox.contains(Point2I(centroid)):
                centroids.append(centroid)
                bboxes.append(fp.getBBox())
        if len(bboxes) < 4:
            return self._returnFailure(self.FLAG_UNKNOWN_ERROR)

        center_guess_x = np.array(centroids).T[0] - exp.image.getX0()
        left = np.zeros(4)
        right = np.zeros(4)
        for idx in range(4):
            left[idx] = bboxes[idx].minY - exp.image.getY0()
            right[idx] = bboxes[idx].maxY - exp.image.getY0()

        bin_edges = [int(left[0] - 2*self.config.yWindow),
                     int(left[1] + right[0]) // 2,
                     int(left[2] + right[1]) // 2,
                     int(left[3] + right[2]) // 2,
                     int(right[3] + self.config.yWindow)]

        # now find midpoint in y, estimate midpoint for x
        x_centers = np.zeros(4, dtype=int)
        x_min = np.zeros(4, dtype=int)
        x_max = np.zeros(4, dtype=int)

        alt_centroids = []
        fwhms = np.zeros(4, dtype=int)
        fit_parameters = []

        for idx in range(4):
            mid_y = int((bin_edges[idx + 1] - bin_edges[idx])/2. + bin_edges[idx])
            # CZW: this should be configurable:
            x_data = np.median(exp.image.array[mid_y - 5:mid_y + 5, :], axis=0)
            alt_x_vals = np.arange(len(x_data))

            x_norm = (x_data - x_data.min())/(x_data.max() - x_data.min())
            theta, covariance = curve_fit(self._d_gaussian,
                                          alt_x_vals,
                                          x_norm,
                                          p0=[0.5, center_guess_x[idx], 3, 20],
                                          bounds=(0, [1, 400, 25, 100]),
                                          max_nfev=10_000)
            fit_parameters.append(theta)
            x_centers[idx] = int(theta[1])
            alt_centroids.append([x_centers[idx], mid_y])
            if self.config.xWindow == 0:
                scale = 2*np.sqrt(2*np.log(2))
                fwhm_1 = int(scale * np.abs(theta[2]))
                fwhm_2 = int(scale * np.abs(theta[3]))
                fwhm = int(theta[0]*fwhm_1 + 2*(1-theta[0])*fwhm_2)

                fwhms[idx] = 2*int(self.config.xGauge*fwhm) + 20

        if self.config.xWindow != 0:
            fwhms = self.config.xWindow*np.ones(4, dtype=int)
        max_width = np.max(fwhms)

        x_min = (x_centers - int(max_width/2))
        x_max = (x_centers + int(max_width/2))

        # CZW: Then we do it again?  Just for the plots?
        # for idx in range(4):
        #     mid_y = int((bin_edges[idx + 1] - bin_edges[idx])/2. + bin_edges[idx])  # noqa W505
        #     # CZW: this should be configurable:
        #     x_data = np.median(exp[mid_y - 5:mid_y + 5, :], axis=0)
        #     x_data_alt = np.sum(exp[bin_edges[idx]:bin_edges[idx + 1]], axis=0) # noqa W505
        # Do we not need a :?
        #     alt_x_vals = np.arange(len(x_data))

        #     x_norm = (x_data - x_data.min())/(x_data.max() - x_data.min())
        #     x_norm_alt = (x_data_alt - x_data.min())/(x_data_alt.max() - x_data_alt.min())  # noqa W505
        # typo?

        # Use center to get both data and background boxes:
        flag = self.FLAG_SUCCESS
        flux = []
        background = []
        percentiles = []
        counts_above_threshold = []
        ranks = np.arange(60, 101)

        for idx in range(4):
            # Background first:
            # CZW: These should be configurable.
            left = exp.image.array[bin_edges[idx]:bin_edges[idx + 1],
                                   x_min[idx] - 10:x_min[idx]]
            right = exp.image.array[bin_edges[idx]:bin_edges[idx + 1],
                                    x_max[idx]: x_max[idx] + 10]
            background_box = np.concatenate((left, right), axis=1)
            _, background_vec, _ = sigma_clipped_stats(background_box, axis=1)

            # Define aperture box:
            signal_box = exp.image.array[bin_edges[idx]:bin_edges[idx + 1],
                                         x_min[idx]:x_max[idx]]
            corrected = signal_box - background_vec[:, np.newaxis]

            background.append(np.mean(background_vec))
            flux.append(np.sum(corrected))
            counts_above_threshold.append(np.sum(corrected > self.config.threshold))

            if bin_edges[0] <= 0 or signal_box.size == 0:
                percentiles.append(np.zeros_like(ranks))
                flag = self.FLAG_UNKNOWN_ERROR | self.FLAG_NO_FOOTPRINT_FOUND
            else:
                percentiles.append(np.percentile(corrected, ranks))

        row['flux'] = flux
        row['percentiles'] = percentiles
        row['centroids'] = alt_centroids
        row['background'] = np.mean(np.array(background))  # This is mean of means of bkg_vec.
        row['flag'] = flag
        row['fwhm'] = fwhms
        row['counts_above_threshold'] = counts_above_threshold

        return row

    @staticmethod
    def _d_gaussian(x, a, mean, sigma_1, sigma_2):
        """Double Gaussian function.

        CZW: docstring
        """
        return a*np.exp(-(x-mean)**2/(2*sigma_1**2)) + (1-a)*np.exp(-(x-mean)**2/(2*sigma_2**2))


class QuadNotchMergeConnections(pipeBase.PipelineTaskConnections,
                                dimensions=("instrument", "detector")):
    inputData = cT.Input(
        name="quadNotchSingle",
        doc="Quad-notch measurements from individual exposures.",
        storageClass="ArrowAstropy",
        dimensions=("instrument", "exposure", "detector"),
        multiple=True,
        deferLoad=False,
    )
    outputData = cT.Output(
        name="quadNotchCombined",
        doc="Output combined quad-notch analysis.",
        storageClass="ArrowAstropy",
        dimensions=("instrument", "detector"),
    )


class QuadNotchMergeConfig(pipeBase.PipelineTaskConfig,
                           pipelineConnections=QuadNotchMergeConnections):
    """Configuration for quad-notch processing."""
    nSigma = pexConfig.Field(
        dtype=float,
        default=2.0,
        doc="This is a dummy parameter that isn't used, but I didn't want no config here.",
    )


class QuadNotchMergeTask(pipeBase.PipelineTask):
    """Task to measure quad-notch data."""

    ConfigClass = QuadNotchMergeConfig
    _DefaultName = "quadNotchMerge"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def run(self, inputData, camera):
        """
        """
        rows = []
        for dataset in inputData:
            rows.append(dataset[0])

        outputTable = Table(rows)
        return pipeBase.Struct(
            outputData=outputTable,
        )
