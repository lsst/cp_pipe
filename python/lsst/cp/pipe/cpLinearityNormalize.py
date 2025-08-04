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
import warnings

import lsst.pipe.base as pipeBase
import lsst.pex.config as pexConfig

__all__ = [
    "LinearityNormalizeConfig",
    "LinearityNormalizeTask",
]


class LinearityNormalizeConnnections(
    pipeBase.PipelineTaskConnections,
    dimensions=("instrument",),
):
    inputPtcHandles = pipeBase.connectionTypes.Input(
        name="linearizerPtc",
        doc="Input covariances.",
        storageClass="PhotonTransferCurveDataset",
        dimensions=("instrument", "detector"),
        isCalibration=True,
        multiple=True,
        deferLoad=True,
    )

    inputLinearizerHandles = pipeBase.connectionTypes.Input(
        name="linearizerUnnormalized",
        doc="Unnormalized linearizers.",
        storageClass="Linearizer",
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
        name="cpLinearizerNormalization",
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
                "LinearityNormalize reference detector not in list of PTC inputs.",
            )

        linearizerRefs = []
        foundRefDetector = False
        for ref in inputs["inputLinearizerHandles"][1]:
            if ref.dataId["detector"] in self.config.normalizeDetectors:
                linearizerRefs.append(ref)
            if ref.dataId["detector"] == self.config.referenceDetector:
                foundRefDetector = True

        if len(linearizerRefs) == 0:
            raise pipeBase.NoWorkFound("No input linearizers match the normalization detectors.")
        if not foundRefDetector:
            raise pipeBase.NoWorkFound(
                "LinearityNormalize reference detector not in list of linearizer inputs.",
            )

        inputs["inputPtcHandles"] = (inputs["inputPtcHandles"][0], tuple(ptcRefs))
        inputs["inputLinearizerHandles"] = (inputs["inputLinearizerHandles"][0], tuple(linearizerRefs))

        return inputs, outputs


class LinearityNormalizeConfig(
    pipeBase.PipelineTaskConfig,
    pipelineConnections=LinearityNormalizeConnnections,
):
    normalizeDetectors = pexConfig.ListField(
        dtype=int,
        doc="List of detector numbers to use for normalization.",
        default=None,
        optional=False,
    )
    referenceDetector = pexConfig.Field(
        dtype=int,
        doc="Detector to use as an overall reference for sorting/labeling "
            "exposures. Must be in list of normalizeDetectors.",
        default=None,
        optional=False,
    )
    minValidFraction = pexConfig.Field(
        dtype=float,
        doc="Minimum fraction of normalization amplifiers that must have valid "
            "residuals in order to be used to create a normalization value.",
        default=0.5,
    )

    def validate(self):
        super().validate()
        if self.referenceDetector not in self.normalizeDetectors:
            raise ValueError("The selected referenceDetector must be in the list of normalizeDetectors.")


class LinearityNormalizeTask(pipeBase.PipelineTask):
    """Class to use data to normalize linearity inputs."""

    ConfigClass = LinearityNormalizeConfig
    _DefaultName = "cpPtcNormalize"

    def run(self, *, camera, inputPtcHandles, inputLinearizerHandles):
        """Compute the focal-plane normalization.

        Parameters
        ----------
        camera : `lsst.afw.cameraGeom.Camera`
            Input camera.
        inputPtcHandles : `list` [`lsst.daf.butler.DeferredDatasetHandle`]
            Handles for input PTCs to do normalization.
        inputLinearizerHandles :
            `list` [`lsst.daf.butler.DeferredDatasetHandle`]
            handles for input linearizers to do normalization.

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

        ptcHandleDict = {handle.dataId["detector"]: handle for handle in inputPtcHandles}
        linHandleDict = {handle.dataId["detector"]: handle for handle in inputLinearizerHandles}

        ptcReferenceHandle = ptcHandleDict.get(self.config.referenceDetector, None)
        linReferenceHandle = linHandleDict.get(self.config.referenceDetector, None)

        if ptcReferenceHandle is None or linReferenceHandle is None:
            raise RuntimeError("Reference detector not found in input PTCs or Linearizers.")

        referencePtc = ptcReferenceHandle.get()
        referenceLinearizer = linReferenceHandle.get()

        # Get the exposure numbers and exposure times from the reference PTC.
        # These will be matched for all amps.
        # Note that all amps in a ptc must have the same input exposures.
        exposures = np.asarray(referencePtc.inputExpIdPairs[referencePtc.ampNames[0]])[:, 0]
        exptimes = referencePtc.rawExpTimes[referencePtc.ampNames[0]]

        # Get the input normalization values from the reference linearizer.
        # These will be matched for all amps.
        inputNormalization = referenceLinearizer.inputNormalization[referencePtc.ampNames[0]]

        # The rawMeans and fitResiduals arrays will hold all the exposures
        # and amps for the detectors that go into the global normalization.
        rawMeans = np.zeros((len(self.config.normalizeDetectors), len(exposures), nAmp))
        models = np.zeros_like(rawMeans)
        fitResiduals = np.zeros_like(rawMeans)
        rawMeans[:, :, :] = np.nan
        models[:, :, :] = np.nan
        fitResiduals[:, :, :] = np.nan

        for i, detector in enumerate(self.config.normalizeDetectors):
            ptcHandle = ptcHandleDict.get(detector, None)
            linHandle = linHandleDict.get(detector, None)

            if ptcHandle is None or linHandle is None:
                self.log.warning(
                    f"Detector {detector} configured for normalization, but not found in inputs.",
                )

            ptc = ptcHandle.get()
            lin = linHandle.get()

            ptcExposures = np.asarray(ptc.inputExpIdPairs[ptc.ampNames[0]])[:, 0]

            if len(ptcExposures) != len(exposures):
                self.log.warning(
                    "PTC for detector %d has %d pairs, fewer than expected %d.",
                    ptc.dataId["detector"],
                    len(ptcExposures),
                    len(exposures),
                )

            a, b = esutil.numpy_util.match(exposures, ptcExposures)
            if len(a) == 0:
                self.log.warning(
                    "PTC for detector %d has no exposure matches to the reference!",
                    ptc.dataId["detector"],
                )
                continue

            for j, ampName in enumerate(ptc.ampNames):
                rawMeans[i, a, j] = ptc.rawMeans[ampName][b]
                models[i, a, j] = lin.fitResidualsModel[ampName][b]
                fitResiduals[i, a, j] = (
                    lin.fitResiduals[ampName][b] / lin.fitResidualsModel[ampName][b]
                )

        # Compute the median levels for the normalization detectors.
        medianRawMeans = np.nanmedian(rawMeans, axis=[0, 2])
        medianModel = np.nanmedian(models, axis=[0, 2])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            medianResiduals = np.nanmedian(fitResiduals, axis=[0, 2])
            nValid = np.sum(np.isfinite(fitResiduals), axis=0).sum(axis=1)

        # Only use a normalization value for exposures that have a
        # sufficient number of valid amplifiers
        nNormalizeAmps = fitResiduals.shape[0] * fitResiduals.shape[2]
        bad = ((nValid / nNormalizeAmps) < self.config.minValidFraction) | ~np.isfinite(medianResiduals)
        medianResiduals[bad] = 0.0

        # The residual is computed as:
        #   <r> = (y - m*pe)/y  (1)
        # where <r> is the median residual, y is the model linearized
        # quantity, and m is the slope of the fit of the linearized
        # counts vs the input photocharge or exposure time (pe).
        #
        # We are looking for a normalization factor k which when
        # applied to the photocharge or exposure time (pe) gives
        # zero residual, so we also have the constraint:
        #   0 = (y - m*pe*k)/y  (2)
        #
        # Substituting y = m*pe*k (2) into equation (1) and
        # solving for k we find the slope cancels out and we get:
        #   k = 1 / (1 - <r>)

        # The output normalization is applied on top of the input
        # normalization (which may be all 1s for the first iteration).
        outputNormalization = inputNormalization / (1.0 - medianResiduals)

        # These are the per-exposure normalization values.
        table = Table(
            {
                "exposure": exposures,
                "exptime": exptimes,
                "mean": medianRawMeans,
                "model": medianModel,
                "normalization": outputNormalization,
            },
        )

        result = pipeBase.Struct(outputNormalization=table)

        return result
