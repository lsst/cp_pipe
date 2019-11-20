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
# along with this program.  If

# Utility tasks/methods copied from pipe_drivers/constructCalibs.py

import datetime
import numpy as np
import time

import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase
import lsst.pipe.base.connectionTypes as cT
import lsst.afw.math as afwMath
import lsst.afw.image as afwImage
import lsst.daf.base as dafBase


class BlessCalibration(pipeBase.Task):
    """Create a way to bless existing calibration products.

    The inputs are assumed to have been constructed via cp_pipe, and
    already exist in the butler.

    Parameters
    ----------
    kwargs : `dict`
        Parameters to initialize the blessing process.  Keys:

        ``"butler"``
            Data butler to use.
        ``"inputCollection"``
            Data collection to pull calibrations from.
        ``"outputCollection"``
            Data collection to store final calibrations.
    """
    def __init__(self, **kwargs):
        self.butler = kwargs['butler']
        self.registry = self.butler.registry

        self.inputCollection = kwargs['inputCollection']
        self.outputCollection = kwargs['outputCollection']

        self.calibrationLabel = ''
        self.instrument = ''

    def findInputs(self, datasetTypeName, inputTypeString=None):
        """Find and prepare inputs for blessing.

        Parameters
        ----------
        datasetTypeName : `str`
            Dataset that will be blessed.
        inputTypeString : `str`
            Dataset name for the input datasets.

        Raises
        ------
        RuntimeError :
            Raised if no input datasets found or if the calibration
            label exists and is not empty.
        """
        if inputTypeString is None:
            inputDatasetTypeName = datasetTypeName + "Proposal"

        self.inputValues = list(self.registry.queryDatasets(inputDatasetTypeName,
                                                            collections=[self.inputCollection]))
        if len(self.inputValues) == 0:
            raise RuntimeError(f"No inputs found for dataset {inputDatasetTypeName} "
                               f"in {self.inputCollection}")

        # Construct calibration label and choose instrument to use.
        self.calibrationLabel = "{}/{}".format(datasetTypeName,
                                               self.inputCollection)
        self.instrument = self.inputValues[0].dataId['instrument']

        # Prepare combination of new data ids and object data:
        self.newDataId = []
        self.objects = []
        for input in self.inputValues:
            self.newDataId.append(input.dataId)
            self.objects.append(self.butler.get(input))

    def registerCalibrations(self, datasetTypeName):
        """Add blessed inputs to the output collection.

        Parameters
        ----------
        datasetTypeName : `str`
            Dataset type these calibrations will be registered for.
        """

        # Find/make the run we will use for the output
        run = self.registry.getRun(collection=self.outputCollection)
        if run is None:
            run = self.registry.makeRun(self.outputCollection)
        self.butler.run = run
        self.butler.collection = None

        for newId, data in zip(self.newDataId, self.objects):
            # Special case known special storageClasses.
            if datasetTypeName in ('bias', 'dark'):
                data = data.getImage()
            elif datasetTypeName in ('flat'):
                data = data.getMaskedImage()

            self.butler.put(data, datasetTypeName, dataId=newId,
                            calibration_label=self.calibrationLabel,
                            producer=None)

    def addCalibrationLabel(self, name=None, instrument=None,
                            beginDate="1970-01-01", endDate="2038-12-31"):
        """Method to allow tasks to add calibration_label for master calibrations.

        Parameters
        ----------
        name : `str`
            A unique string for the calibration_label key.
        instrument : `str`
            Instrument this calibration is for.
        beginDate : `str`
            A hyphen delineated date string for the beginning of the valid date range.
        endDate : `str`
            A hyphen delineated date string for the end of the valid date range.
        """
        if name is None:
            name = self.calibrationLabel
        if instrument is None:
            instrument = self.instrument

        beginY, beginM, beginD = beginDate.split("-")
        endY, endM, endD = endDate.split("-")

        try:
            self.registry.queryDimensions(['calibration_label'],
                                          instrument=self.instrument,
                                          calibration_label=self.calibrationLabel)
        except LookupError:
            self.butler.registry.insertDimensionData(
                "calibration_label",
                {
                    "name": name,
                    "instrument": instrument,
                    "datetime_begin": datetime.datetime(int(beginY), int(beginM), int(beginD), 0, 0, 0),
                    "datetime_end": datetime.datetime(int(endY), int(endM), int(endD), 0, 0, 0)
                }
            )


class CalibStatsConfig(pexConfig.Config):
    """Parameters controlling the measurement of background statistics.
    """
    stat = pexConfig.Field(
        dtype=int,
        default=int(afwMath.MEANCLIP),
        doc="Statistic to use to estimate background (from lsst.afw.math)",
    )
    clip = pexConfig.Field(
        dtype=float,
        default=3.0,
        doc="Clipping threshold for background",
    )
    nIter = pexConfig.Field(
        dtype=int,
        default=3,
        doc="Clipping iterations for background",
    )
    maxVisitsToCalcErrorFromInputVariance = pexConfig.Field(
        dtype=int,
        default=2,
        doc="Maximum number of visits to estimate variance from input variance, not per-pixel spread",
    )
    mask = pexConfig.ListField(
        dtype=str,
        default=["DETECTED", "BAD", "NO_DATA"],
        doc="Mask planes to reject",
    )


class CalibStatsTask(pipeBase.Task):
    """Measure statistics on the background

    This can be useful for scaling the background, e.g., for flats and fringe frames.
    """
    ConfigClass = CalibStatsConfig

    def run(self, exposureOrImage):
        """!Measure a particular statistic on an image (of some sort).

        Parameters
        ----------
        exposureOrImage : `lsst.afw.image.Exposure`, `lsst.afw.image.MaskedImage`, or `lsst.afw.image.Image`
           Exposure or image to calculate statistics on.

        Returns
        -------
        results : `lsst.afw.math.statistics`
           Resulting statistics.
        """
        stats = afwMath.StatisticsControl(self.config.clip, self.config.nIter,
                                          afwImage.Mask.getPlaneBitMask(self.config.mask))
        try:
            image = exposureOrImage.getMaskedImage()
        except Exception:
            try:
                image = exposureOrImage.getImage()
            except Exception:
                image = exposureOrImage

        return afwMath.makeStatistics(image, self.config.stat, stats).getValue()


class CalibCombineConnections(pipeBase.PipelineTaskConnections,
                              dimensions=("instrument", "detector"),
                              defaultTemplates={}):
    inputExps = cT.Input(
        name="cpInputs",
        doc="Input pre-processed exposures to combine.",
        storageClass="ExposureF",
        dimensions=("instrument", "detector", "visit"),
        deferLoad=False,
        multiple=True,
    )

    outputData = cT.Output(
        name="cpProposal",
        doc="Output combined proposed calibration.",
        storageClass="ExposureF",
        dimensions=("instrument", "detector"),  # , "calibration_label"),
    )

    def __init__(self, *, config=None):
        super().__init__(config=config)

        if config and len(config.calibrationDimensions) != 0:
            # CZW: this seems fake, but ok.
            newOutputData = cT.Output(
                name=self.outputData.name,
                doc=self.outputData.doc,
                storageClass=self.outputData.storageClass,
                dimensions=self.allConnections['outputData'].dimensions + tuple(config.calibrationDimensions)
            )
            self.dimensions.update(config.calibrationDimensions)
            self.outputData = newOutputData


class CalibCombineConfig(pipeBase.PipelineTaskConfig,
                         pipelineConnections=CalibCombineConnections):
    """Configuration for combining calib exposures.
    """
    calibrationType = pexConfig.Field(
        dtype=str,
        default="calibration",
        doc="Name of calibration to be generated.",
    )
    calibrationDimensions = pexConfig.ListField(
        dtype=str,
        default=[],
        doc="List of updated dimensions to append to output."
    )

    exposureScaling = pexConfig.ChoiceField(
        dtype=str,
        allowed={
            "None": "No scaling used.",
            "ExposureTime": "Scale inputs by their exposure time.",
            "DarkTime": "Scale inputs by their dark time.",
        },
        default=None,
        doc="Scaling to be applied to each input exposure.",
    )
    outputScaling = pexConfig.Field(
        dtype=float,
        default=float('nan'),
        doc="Scaling value to apply to the output combined product.",
    )

    rows = pexConfig.Field(
        dtype=int,
        default=512,
        doc="Number of rows to read at a time",
    )

    mask = pexConfig.ListField(
        dtype=str,
        default=["SAT", "DETECTED", "INTRP"],
        doc="Mask planes to respect",
    )
    combine = pexConfig.Field(
        dtype=int,
        default=int(afwMath.MEANCLIP),
        doc="Statistic to use for combination (from lsst.afw.math)",
    )
    clip = pexConfig.Field(
        dtype=float,
        default=3.0,
        doc="Clipping threshold for combination",
    )
    nIter = pexConfig.Field(
        dtype=int,
        default=3,
        doc="Clipping iterations for combination",
    )
    stats = pexConfig.ConfigurableField(
        target=CalibStatsTask,
        doc="Background statistics configuration",
    )


class CalibCombineTask(pipeBase.PipelineTask,
                       pipeBase.CmdLineTask):
    """Task to combine calib exposures."""
    ConfigClass = CalibCombineConfig
    _DefaultName = 'cpCombine'

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.makeSubtask("stats")

    def run(self, inputExps=None):
        """!Combine calib exposures for a single detector.

        Parameters
        ----------
        inputExps : `List` [`lsst.afw.image.Exposure`]
            Input list of exposures to combine.

        Returns
        -------
        combinedExp : `lsst.afw.image.Exposure`
            Final combined exposure generated from the inputs.
        """
        width, height = self.getDimensions(inputExps)
        stats = afwMath.StatisticsControl(self.config.clip, self.config.nIter,
                                          afwImage.Mask.getPlaneBitMask(self.config.mask))
        numExps = len(inputExps)
        if numExps < 1:
            raise RuntimeError("No valid input data")
        if numExps < self.config.stats.maxVisitsToCalcErrorFromInputVariance:
            stats.setCalcErrorFromInputVariance(True)

        # Create output exposure for combined data.
        combined = afwImage.MaskedImageF(width, height)
        combinedExp = afwImage.makeExposure(combined)

        # Apply scaling:
        expScales = []
        for index, exp in enumerate(inputExps):
            scale = 1.0
            if exp is None:
                continue

            if self.config.exposureScaling == "ExposureTime":
                scale = exp.getInfo().getVisitInfo().getExposureTime()
            elif self.config.exposureScaling == "DarkTime":
                scale = exp.getInfo().getVisitInfo().getDarkTime()
            elif self.config.exposureScaling == "MeanStats":
                image = exp.getImage()
                scale = afwMath.makeStatistics(image, self.config.stat, stats).getValue()

            expScales.append(scale)
            self.log.info("Scaling input %d by %f" %
                          (index, scale))
            if scale != 1.0 and scale != 0.0 and np.isfinite(scale):
                # Assume cases with bad scales are equivalent to no scaling.
                self.applyScale(exp, scale)

        self.combine(combined, inputExps, stats)

        if np.isfinite(self.config.outputScaling) and self.config.outputScaling != 0.0:
            background = self.stats.run(combined)
            self.log.info("Measured background of stack is %f; adjusting to %f" %
                          (background, self.config.outputScaling))

            self.applyScale(combinedExp, float(background) / float(self.config.outputScaling))
            expScales = [scl * self.config.outputScaling / background for scl in expScales]

        self.interpolateNans(combined)

        # Combine headers
        header = self.combineHeaders(inputExps, calibType=self.config.calibrationType, scales=expScales)
        combinedExp.getMetadata().update(header)

        # Return
        return pipeBase.Struct(
            outputData=combinedExp,
        )

    def getDimensions(self, expList):
        """Get dimensions of the inputs.

        Parameters
        ----------
        expList : `List` [`lsst.afw.image.Exposure`]
            Exps to check the sizes of.

        Returns
        -------
        w, h : `int`
            Unique set of input dimensions.
        """
        dimList = []
        for exp in expList:
            if exp is None:
                continue
            dimList.append(exp.getDimensions())
        return self.getSize(dimList)

    def getSize(self, dimList):
        """Determine a consistent size, given a list of image sizes"""
        dim = set((w, h) for w, h in dimList)
        dim.discard(None)
        if len(dim) != 1:
            raise RuntimeError("Inconsistent dimensions: %s" % dim)
        return dim.pop()

    def applyScale(self, exposure, scale=None):
        """Apply scale to input exposure.

        This implementation applies a flux scaling: the input exposure is
        divided by the provided scale.

        Parameters
        ----------
        exposure : `lsst.afw.image.Exposure`
            Exposure to scale.
        scale : `float`, optional
            Constant scale to divide the exposure by.
        """
        if scale is not None:
            mi = exposure.getMaskedImage()
            mi /= scale

    def combine(self, target, expList, stats):
        """!Combine multiple images.

        Parameters
        ----------
        target : `lsst.afw.image.Exposure`
            Output exposure to construct.
        expList : `List` [`lsst.afw.image.Exposures`]
            Input exposures to combine.
        stats : `lsst.afw.math.statisticsControl`
            Control explaining how to combine the input images.
        """
        images = [img.getMaskedImage() for img in expList if img is not None]

        afwMath.statisticsStack(target, images, afwMath.Property(self.config.combine), stats)

    def combineHeaders(self, expList, calibType="CALIB", scales=None):
        """
        """
        # Header
        header = dafBase.PropertySet()
        header.set("OBSTYPE", calibType)

        # Creation date
        now = time.localtime()
        header.set("CALIB_CREATE_DATE", time.strftime("%Y-%m-%d", now))
        header.set("CALIB_CREATE_TIME", time.strftime("%X %Z", now))

        # Inputs
        for i, exp in enumerate(expList):
            visit = exp.getInfo().getVisitInfo()
            if visit is None:
                continue
            header.set("CPP_INPUT_%d" % (i,), visit.getExposureId())
            # header.set("CPP_INPUT_DATE_%d" % (i,), visit.getDate())
            header.set("CPP_INPUT_EXPT_%d" % (i,), visit.getExposureTime())
            if scales is not None:
                header.set("CPP_INPUT_SCALE_%d" % (i,), scales[i])

        # XYZ?
        return header

    def interpolateNans(self, exp):
        """Interpolate over NANs in the combined image.

        NANs can result from masked areas on the CCD.  We don't want them getting
        into our science images, so we replace them with the median of the image.

        Parameters
        ----------
        exp : `lsst.afw.image.Exposure`
            Exp to check for NaNs.

        Returns
        -------
        median : `float`
            Value used to replace NaN values.
        count : `int`
            Number of NaNs replaced.
        """
        array = exp.getImage().getArray()
        bad = np.isnan(array)

        median = np.median(array[np.logical_not(bad)])
        count = np.sum(np.logical_not(bad))
        array[bad] = median
        if count > 0:
            self.log.warn("Found %s NAN pixels" % (count, ))
