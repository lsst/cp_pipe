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
import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase
from lsst.cp.pipe.utils import arrangeFlatsByExpTime

from lsst.pipe.tasks.getRepositoryData import DataRefListRunner
from lsst.cp.pipe.ptc.cpExtractPtcTask import PhotonTransferCurveExtractTask
from lsst.cp.pipe.ptc.cpSolvePtcTask import PhotonTransferCurveSolveTask
from lsst.utils.timer import timeMethod


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

    @timeMethod
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
                self.log.warning("postISR exposure could not be retrieved. Ignoring flat.")
                continue
            expList.append(tempFlat)
        expIds = [exp.info.getVisitInfo().id for exp in expList]

        # Create dictionary of exposures, keyed by exposure time
        expDict = arrangeFlatsByExpTime(expList)
        # Call the "extract" (measure flat covariances) and "solve"
        # (fit covariances) subtasks
        resultsExtract = self.extract.run(inputExp=expDict, inputDims=expIds)
        resultsSolve = self.solve.run(resultsExtract.outputCovariances, camera=camera)

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

        self.log.info("Writing PTC data.")
        dataRef.put(resultsSolve.outputPtcDataset, datasetType="photonTransferCurveDataset")

        return
