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
import copy
import numpy as np

from lsst.ip.isr import PhotonTransferCurveDataset
import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase
import lsst.pipe.base.connectionTypes as cT

from deprecated.sphinx import deprecated

from ..utils import ampOffsetGainRatioFixup


__all__ = [
    "PhotonTransferCurveFixupGainRatiosConfig",
    "PhotonTransferCurveFixupGainRatiosTask",
    "PhotonTransferCurveRenameConfig",
    "PhotonTransferCurveRenameTask",
]


# TODO DM-52883: Remove deprecated tasks.
@deprecated(reason="PhotonTransferCurveFixupGainRatiosTask is no longer used. "
                   "This Task will be removed after v30.",
            version="v30.0", category=FutureWarning)
class PhotonTransferCurveFixupGainRatiosConnections(
    pipeBase.PipelineTaskConnections,
    dimensions=("instrument", "detector")
):
    exposureMetadata = cT.Input(
        name="cpPtcFixupGainRatiosIsrExp.metadata",
        doc="Input exposures for gain ratio fixup.",
        storageClass="PropertyList",
        dimensions=("instrument", "exposure", "detector"),
        multiple=True,
    )
    inputPtc = cT.PrerequisiteInput(
        name="ptc",
        doc="Input PTC to modify.",
        storageClass="PhotonTransferCurveDataset",
        dimensions=("instrument", "detector"),
        isCalibration=True,
    )
    outputPtc = cT.Output(
        name="ptcFixed",
        doc="Output modified PTC.",
        storageClass="PhotonTransferCurveDataset",
        dimensions=("instrument", "detector"),
        multiple=False,
        isCalibration=True,
    )


# TODO DM-52883: Remove deprecated tasks.
@deprecated(reason="PhotonTransferCurveFixupGainRatiosTask is no longer used. "
                   "This Task will be removed after v30.",
            version="v30.0", category=FutureWarning)
class PhotonTransferCurveFixupGainRatiosConfig(
    pipeBase.PipelineTaskConfig,
    pipelineConnections=PhotonTransferCurveFixupGainRatiosConnections,
):
    ampOffsetGainRatioMinAdu = pexConfig.Field(
        dtype=float,
        doc="Minimum number of adu to use for amp offset gain ratio fixup.",
        default=1000.0,
    )
    ampOffsetGainRatioMaxAdu = pexConfig.Field(
        dtype=float,
        doc="Maximum number of adu to use for amp offset gain ratio fixup.",
        default=40000.0,
    )


# TODO DM-52883: Remove deprecated tasks.
@deprecated(reason="PhotonTransferCurveFixupGainRatiosTask is no longer used. "
                   "This Task will be removed after v30.",
            version="v30.0", category=FutureWarning)
class PhotonTransferCurveFixupGainRatiosTask(pipeBase.PipelineTask):
    """Task to use on-sky amp ratios to fix up gain ratios in a PTC.

    This uses the ampOffsetGainRatioFixup with on-sky data (preferably
    twilight flats or similar) to update gain ratios.
    """
    ConfigClass = PhotonTransferCurveFixupGainRatiosConfig
    _DefaultName = "cpPhotonTransferCurveFixupGainRatios"

    def runQuantum(self, butlerQC, inputRefs, outputRefs):
        # docstring inherited.
        inputs = butlerQC.get(inputRefs)
        outputs = self.run(inputPtc=inputs["inputPtc"], exposureMetadata=inputs["exposureMetadata"])
        butlerQC.put(outputs, outputRefs)

    def run(self, *, inputPtc, exposureMetadata):
        """Run the gain ratio fixup task.

        Parameters
        ----------
        inputPtc : `lsst.ip.isr.PhotonTransferCurveDataset`
            Input PTC to modify.
        exposureMetadata: `list` [`lsst.daf.base.PropertyList`]
            Input exposure metadata.

        Returns
        -------
        results : `lsst.pipe.base.Struct`
            The output struct contains:

            ``outputPtc``
                The output modified ptc.
        """
        ampNames = inputPtc.ampNames

        # Create a set of fake partial PTC datasets.
        fakePtc = PhotonTransferCurveDataset(
            ampNames=ampNames,
            ptcFitType="FAKEPTC",
            covMatrixSide=1,
            covMatrixSideFullCovFit=1,
        )

        for i, metadata in enumerate(exposureMetadata):
            fakePartialPtc = PhotonTransferCurveDataset(ampNames=ampNames, ptcFitType="PARTIAL")

            for ampName in ampNames:
                fakePartialPtc.setAmpValuesPartialDataset(
                    ampName,
                    inputExpIdPair=(2*i, 2*i + 1),
                    rawExpTime=float(i),
                    rawMean=metadata[f"LSST ISR FINAL MEDIAN {ampName}"],
                    rawVar=metadata[f"LSST ISR FINAL STDEV {ampName}"]**2.,
                    ampOffset=metadata[f"LSST ISR AMPOFFSET PEDESTAL {ampName}"],
                    expIdMask=True,
                    gain=inputPtc.gainUnadjusted[ampName],
                    noise=metadata[f"LSST ISR READNOISE {ampName}"]*inputPtc.gain[ampName],
                    covariance=np.zeros((1, 1)),
                    covSqrtWeights=np.zeros((1, 1)),
                )

            fakePtc.appendPartialPtc(fakePartialPtc)

        detectorMeans = np.zeros(len(exposureMetadata))

        for i in range(len(detectorMeans)):
            arr = np.asarray([fakePtc.rawMeans[ampName][i] for ampName in ampNames])
            detectorMeans[i] = np.nanmean(arr)

        index = np.argsort(detectorMeans)
        fakePtc.sort(index)

        for ampName in ampNames:
            fakePtc.finalMeans[ampName][:] = fakePtc.rawMeans[ampName].copy()
            fakePtc.finalVars[ampName][:] = fakePtc.rawVars[ampName].copy()
            fakePtc.gainUnadjusted[ampName] = inputPtc.gainUnadjusted[ampName]
            fakePtc.gain[ampName] = inputPtc.gainUnadjusted[ampName]

        ampOffsetGainRatioFixup(
            fakePtc,
            self.config.ampOffsetGainRatioMinAdu,
            self.config.ampOffsetGainRatioMaxAdu,
            log=self.log,
        )

        outputPtc = copy.copy(inputPtc)

        # Replace the gain (leave gainUnadjusted alone).
        for ampName in ampNames:
            outputPtc.gain[ampName] = fakePtc.gain[ampName]

        return pipeBase.Struct(
            outputPtc=outputPtc,
        )


# TODO DM-52883: Remove deprecated tasks.
@deprecated(reason="PhotonTransferCurveRenameTask is no longer used. "
                   "This Task will be removed after v30.",
            version="v30.0", category=FutureWarning)
class PhotonTransferCurveRenameConnections(
    pipeBase.PipelineTaskConnections,
    dimensions=("instrument", "detector")
):
    inputPtc = cT.PrerequisiteInput(
        name="ptcFixed",
        doc="Input PTC to rename.",
        storageClass="PhotonTransferCurveDataset",
        dimensions=("instrument", "detector"),
        isCalibration=True,
    )
    outputPtc = cT.Output(
        name="ptc",
        doc="Output PTC that has been renamed.",
        storageClass="PhotonTransferCurveDataset",
        dimensions=("instrument", "detector"),
        multiple=False,
        isCalibration=True,
    )


# TODO DM-52883: Remove deprecated tasks.
@deprecated(reason="PhotonTransferCurveRenameTask is no longer used. "
                   "This Task will be removed after v30.",
            version="v30.0", category=FutureWarning)
class PhotonTransferCurveRenameConfig(
    pipeBase.PipelineTaskConfig,
    pipelineConnections=PhotonTransferCurveRenameConnections,
):
    pass


# TODO DM-52883: Remove deprecated tasks.
@deprecated(reason="PhotonTransferCurveRenameTask is no longer used. "
                   "This Task will be removed after v30.",
            version="v30.0", category=FutureWarning)
class PhotonTransferCurveRenameTask(pipeBase.PipelineTask):
    """Task to rename a ptcFixed into a ptc."""
    ConfigClass = PhotonTransferCurveRenameConfig
    _DefaultName = "cpPhotonTransferCurveRename"

    def runQuantum(self, butlerQC, inputRefs, outputRefs):
        # docstring inherited.
        inputs = butlerQC.get(inputRefs)

        outputs = pipeBase.Struct(outputPtc=inputs["inputPtc"])
        butlerQC.put(outputs, outputRefs)

    def run(self):
        pass
