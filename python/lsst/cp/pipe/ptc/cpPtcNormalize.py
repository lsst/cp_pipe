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
import esutil
import numpy as np
from astropy.table import Table

import lsst.pipe.base as pipeBase
import lsst.pex.config as pexConfig

__all__ = [
    "PhotonTransferCurveNormalizeConfig",
    "PhotonTransferCurveNormalizeTask",
]


class PhotonTransferCurveNormalizeConnnections(
    pipeBase.PipelineTaskConnections,
    dimensions=("instrument",),
):
    # inputCovarianceHandles = pipeBase.connectionTypes.Input(
    #     name="cpLinearizerPtcPartial",
    #     doc="Input covariance pairs.",
    #     storageClass="PhotonTransferCurveDataset",
    #     dimensions=("instrument", "exposure", "detector"),
    #     isCalibration=True,
    #     multiple=True,
    #     deferLoad=True,
    # )
    inputPtcHandles = pipeBase.connectionTypes.Input(
        name="linearizerPtc",
        doc="Input covariances.",
        storageClass="PhotonTransferCurveDataset",
        dimensions=("instrument", "detector"),
        isCalibration=True,
        multiple=True,
        deferLoad=True,
    )

    camera = pipeBase.connectionTypes.PrerequisiteInput(
        name="camera",
        doc="Camera the input data comes from.",
        storageClass="Camera",
        dimensions=("instrument",),
        isCalibration=True,
    )

    outputNormalization = pipeBase.connectionTypes.Output(
        name="cpLinearizerPtcNormalization",
        doc="Normalization table for PTC.",
        storageClass="ArrowAstropy",
        dimensions=("instrument",),
        isCalibration=True,
    )

    def adjustQuantum(self, inputs, outputs, label, dataId):
        ptcRefs = []
        foundRefDetector = False
        for ref in inputs["inputPtcHandles"][1]:
            if ref.dataId["detector"] in self.config.normalizeDetectors:
                ptcRefs.append(ref)
            if ref.dataId["detector"] == self.config.referenceDetector:
                foundRefDetector = True

        if len(ptcRefs) == 0:
            raise pipeBase.NoWorkFound("No input PTCs match the normalization detectors.")
        if not foundRefDetector:
            raise pipeBase.NoWorkFound(
                "PhotonTransferCurveNormalize reference detector not in list of inputs.",
            )
        inputs["inputPtcHandles"] = (inputs["inputPtcHandles"][0], tuple(ptcRefs))

        return inputs, outputs


class PhotonTransferCurveNormalizeConfig(
    pipeBase.PipelineTaskConfig,
    pipelineConnections=PhotonTransferCurveNormalizeConnnections,
):
    normalizeDetectors = pexConfig.ListField(
        dtype=int,
        doc="List of detector numbers to use for normalization.",
        default=None,
        optional=False,
    )
    referenceDetector = pexConfig.Field(
        dtype=int,
        doc="Detector to use as overall reference. Must be in list of "
            "normalizeDetectors.",
        default=None,
        optional=False,
    )
    referenceCounts = pexConfig.Field(
        dtype=float,
        doc="Number of counts (adu) to use as reference target. Exposure "
            "that is closest to this value averaged over normalizeDetectors "
            "will be selected based on this value.",
        default=1000.0,
    )

    def validate(self):
        super().validate()
        if self.referenceDetector not in self.normalizeDetectors:
            raise ValueError("The selected referenceDetector must be in the list of normalizeDetectors.")


class PhotonTransferCurveNormalizeTask(pipeBase.PipelineTask):
    """Class to use data to normalize PTC inputs."""

    ConfigClass = PhotonTransferCurveNormalizeConfig
    _DefaultName = "cpPtcNormalize"

    def run(self, *, camera, inputPtcHandles):
        """Compute the focal-plane normalization.

        Parameters
        ----------
        camera : `lsst.afw.cameraGeom.Camera`
            Input camera.
        inputPtcHandles : `list` [`lsst.daf.butler.DeferredDatasetHandle`]
            Handles for input PTCs to do normalization.

        Returns
        -------
        results : `lsst.pipe.base.Struct`
            The output results structure contains:

            ``outputNormalization``
                Normalization table, `astropy.table.Table`.
        """
        # Get information about the reference detector.
        refDetector = camera[self.config.referenceDetector]
        nAmp = len(refDetector)

        referenceHandle = None
        for handle in inputPtcHandles:
            if handle.dataId["detector"] == self.config.referenceDetector:
                referenceHandle = handle
                break

        if referenceHandle is None:
            raise RuntimeError("Reference detector not in list of input PTCs.")

        referencePtc = referenceHandle.get()

        exposures = np.asarray(referencePtc.inputExpIdPairs[referencePtc.ampNames[0]])[:, 0]

        rawMeans = np.zeros((len(self.config.normalizeDetectors), len(exposures), nAmp))
        rawMeans[:, :, :] = np.nan

        for i, handle in enumerate(inputPtcHandles):
            ptc = handle.get()

            ptcExposures = np.asarray(ptc.inputExpIdPairs[ptc.ampNames[0]])[:, 0]

            if len(ptcExposures) != len(exposures):
                self.log.warning(
                    "PTC for detector %d has %d pairs, fewer than expected %d.",
                    ptc.dataId["detector"],
                    len(ptcExposures),
                    len(exposures),
                )

            a, b = esutil.numpy_util.match(exposures, ptcExposures)
            for j, ampName in enumerate(ptc.ampNames):
                rawMeans[i, a, j] = ptc.rawMeans[ampName][b]

        # Compute the median level over the normalization detectors
        # to find the exposure with counts closest to the target
        # reference counts.
        medianRawMeans = np.nanmedian(rawMeans, axis=[0, 2])
        expRefInd = np.argmin(np.abs(medianRawMeans - self.config.referenceCounts))

        # Compute per-detector, per-amp ratios relative to the reference
        # exposure.
        ratios = rawMeans.copy()
        for i in range(len(self.config.normalizeDetectors)):
            ratios[i, :, :] = rawMeans[i, :, :] / rawMeans[i, expRefInd, :]

        # Compute the median ratio across all the amplifiers.
        medianRatios = np.nanmedian(ratios, axis=[0, 2])

        # These are the per-exposure normalization values.
        table = Table(
            {
                "exposure": np.asarray(exposures),
                "normalization": medianRatios,
            },
        )

        result = pipeBase.Struct(outputNormalization=table)

        return result
