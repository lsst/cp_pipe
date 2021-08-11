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
import numpy as np

import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase
from lsst.cp.pipe.utils import arrangeFlatsByExpTime

from .photodiode import getBOTphotodiodeData

from lsst.pipe.tasks.getRepositoryData import DataRefListRunner
from lsst.cp.pipe.ptc.cpExtractPtcTask import PhotonTransferCurveExtractTask
from lsst.cp.pipe.ptc.cpSolvePtcTask import PhotonTransferCurveSolveTask


__all__ = ['MeasurePhotonTransferCurveTask', 'MeasurePhotonTransferCurveTaskConfig']


class MeasurePhotonTransferCurveTaskConfig(pexConfig.Config):
    extract = pexConfig.ConfigurableField(
        target=PhotonTransferCurveExtractTask,
        doc="Task to measure covariances from flats.",
    )
    solve = pexConfig.ConfigurableField(
        target=PhotonTransferCurveSolveTask,
        doc="Task to fit models to the measured covariances.",
    )
    ccdKey = pexConfig.Field(
        dtype=str,
        doc="The key by which to pull a detector from a dataId, e.g. 'ccd' or 'detector'.",
        default='ccd',
    )
    doPhotodiode = pexConfig.Field(
        dtype=bool,
        doc="Apply a correction based on the photodiode readings if available?",
        default=False,
    )
    photodiodeDataPath = pexConfig.Field(
        dtype=str,
        doc="Gen2 only: path to locate the data photodiode data files.",
        default=""
    )


class MeasurePhotonTransferCurveTask(pipeBase.CmdLineTask):
    """A class to calculate, fit, and plot a PTC from a set of flat pairs.

    The Photon Transfer Curve (var(signal) vs mean(signal)) is a standard
    tool used in astronomical detectors characterization (e.g., Janesick 2001,
    Janesick 2007). If ptcFitType is "EXPAPPROXIMATION" or "POLYNOMIAL",
    this task calculates the PTC from a series of pairs of flat-field images;
    each pair taken at identical exposure times. The difference image of each
    pair is formed to eliminate fixed pattern noise, and then the variance
    of the difference image and the mean of the average image
    are used to produce the PTC. An n-degree polynomial or the approximation
    in Equation 16 of Astier+19 ("The Shape of the Photon Transfer Curve
    of CCD sensors", arXiv:1905.08677) can be fitted to the PTC curve. These
    models include parameters such as the gain (e/DN) and readout noise.

    Linearizers to correct for signal-chain non-linearity are also calculated.
    The `Linearizer` class, in general, can support per-amp linearizers, but
    in this task this is not supported.

    If ptcFitType is "FULLCOVARIANCE", the covariances of the difference
    images are calculated via the DFT methods described in Astier+19 and the
    variances for the PTC are given by the cov[0,0] elements at each signal
    level. The full model in Equation 20 of Astier+19 is fit to the PTC
    to get the gain and the noise.

    Parameters
    ----------
    *args: `list`
        Positional arguments passed to the Task constructor. None used
        at this time.

    **kwargs: `dict`
        Keyword arguments passed on to the Task constructor. None used
        at this time.
    """

    RunnerClass = DataRefListRunner
    ConfigClass = MeasurePhotonTransferCurveTaskConfig
    _DefaultName = "measurePhotonTransferCurve"

    def __init__(self, *args, **kwargs):
        super().__init__(**kwargs)
        self.makeSubtask("extract")
        self.makeSubtask("solve")

    @pipeBase.timeMethod
    def runDataRef(self, dataRefList):
        """Run the Photon Transfer Curve (PTC) measurement task.

        For a dataRef (which is each detector here), and given a list
        of exposure pairs (postISR) at different exposure times,
        measure the PTC.

        Parameters
        ----------
        dataRefList : `list` [`lsst.daf.peristence.ButlerDataRef`]
            Data references for exposures.
        """
        if len(dataRefList) < 2:
            raise RuntimeError("Insufficient inputs to combine.")

        # setup necessary objects
        dataRef = dataRefList[0]
        camera = dataRef.get('camera')

        if len(set([dataRef.dataId[self.config.ccdKey] for dataRef in dataRefList])) > 1:
            raise RuntimeError("Too many detectors supplied")
        # Get exposure list.
        expList = []
        for dataRef in dataRefList:
            try:
                tempFlat = dataRef.get("postISRCCD")
            except RuntimeError:
                self.log.warn("postISR exposure could not be retrieved. Ignoring flat.")
                continue
            expList.append(tempFlat)
        expIds = [exp.getInfo().getVisitInfo().getExposureId() for exp in expList]

        # Create dictionary of exposures, keyed by exposure time
        expDict = arrangeFlatsByExpTime(expList)
        # Call the "extract" (measure flat covariances) and "solve"
        # (fit covariances) subtasks
        resultsExtract = self.extract.run(inputExp=expDict, inputDims=expIds)
        resultsSolve = self.solve.run(resultsExtract.outputCovariances, camera=camera)

        # Fill up the photodiode data, if found, that will be used by
        # linearity task.
        # Get expIdPairs from one of the amps
        expIdsPairsList = []
        ampNames = resultsSolve.outputPtcDataset.ampNames
        for ampName in ampNames:
            tempAmpName = ampName
            if ampName not in resultsSolve.outputPtcDataset.badAmps:
                break
        for pair in resultsSolve.outputPtcDataset.inputExpIdPairs[tempAmpName]:
            first, second = pair[0]
            expIdsPairsList.append((first, second))

        resultsSolve.outputPtcDataset = self._setBOTPhotocharge(dataRef, resultsSolve.outputPtcDataset,
                                                                expIdsPairsList)
        self.log.info("Writing PTC data.")
        dataRef.put(resultsSolve.outputPtcDataset, datasetType="photonTransferCurveDataset")

        return

    def _setBOTPhotocharge(self, dataRef, datasetPtc, expIdList):
        """Set photoCharge attribute in PTC dataset

        Parameters
        ----------
        dataRef : `lsst.daf.peristence.ButlerDataRef`
            Data reference for exposurre for detector to process.

        datasetPtc : `lsst.ip.isr.ptcDataset.PhotonTransferCurveDataset`
            The dataset containing information such as the means, variances
            and exposure times.

        expIdList : `list`
            List with exposure pairs Ids (one pair per list entry).

        Returns
        -------
        datasetPtc : `lsst.ip.isr.ptcDataset.PhotonTransferCurveDataset`
            This is the same dataset as the input parameter, however,
            it has been modified to update the datasetPtc.photoCharge
            attribute.
        """
        if self.config.doPhotodiode:
            for (expId1, expId2) in expIdList:
                charges = [-1, -1]  # necessary to have a not-found value to keep lists in step
                for i, expId in enumerate([expId1, expId2]):
                    # //1000 is a Gen2 only hack, working around the fact an
                    # exposure's ID is not the same as the expId in the
                    # registry. Currently expId is concatenated with the
                    # zero-padded detector ID. This will all go away in Gen3.
                    dataRef.dataId['expId'] = expId//1000
                    if self.config.photodiodeDataPath:
                        photodiodeData = getBOTphotodiodeData(dataRef, self.config.photodiodeDataPath)
                    else:
                        photodiodeData = getBOTphotodiodeData(dataRef)
                    if photodiodeData:  # default path stored in function def to keep task clean
                        charges[i] = photodiodeData.getCharge()
                    else:
                        # full expId (not //1000) here, as that encodes the
                        # the detector number as so is fully qualifying
                        self.log.warn(f"No photodiode data found for {expId}")

                for ampName in datasetPtc.ampNames:
                    datasetPtc.photoCharge[ampName].append((charges[0], charges[1]))
        else:
            # Can't be an empty list, as initialized, because
            # astropy.Table won't allow it when saving as fits
            for ampName in datasetPtc.ampNames:
                datasetPtc.photoCharge[ampName] = np.repeat(np.nan, len(expIdList))

        return datasetPtc
