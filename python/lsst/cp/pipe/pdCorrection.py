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

import lsst.pipe.base as pipeBase
import lsst.pipe.base.connectionTypes as cT
import lsst.pex.config as pexConfig

from lsst.ip.isr import (PhotodiodeCorrection, IsrProvenance)
from ._lookupStaticCalibration import lookupStaticCalibration

__all__ = ["PhotodiodeCorrectionTask", "PhotodiodeCorrectionConfig"]

# I really want to get all of the ptcs and linearizers, not just for one detector.
# Don't know how to do this.

class PhotodiodeCorrectionConnections(pipeBase.PipelineTaskConnections,
                                      dimensions=("instrument", "exposure")):
    dummy = cT.Input(
        name="raw",
        doc="Dummy exposure.",
        storageClass='Exposure',
        dimensions=("instrument", "exposure", "detector"),
        multiple=True,
        deferLoad=True,
    )

    camera = cT.PrerequisiteInput(
        name="camera",
        doc="Camera Geometry definition.",
        storageClass="Camera",
        dimensions=("instrument", ),
        isCalibration=True,
        lookupFunction=lookupStaticCalibration,
    )
    inputPtc = cT.PrerequisiteInput(
        name="ptc",
        doc="Input PTC dataset.",
        storageClass="PhotonTransferCurveDataset",
        dimensions=("instrument", "detector"),
        multiple=True,
        isCalibration=True,
    )

    inputLinearizer = cT.PrerequisiteInput(
        name="unCorrLinearizer",
        doc="Raw linearizers that have not been corrected.",
        storageClass="Linearizer",
        dimensions=("instrument", "detector"),
        multiple=True,
        isCalibration=True,
    )

    outputPhotodiodeCorrection = cT.Output(
        name="pdCorrection",
        doc="Correction of photodiode systematic error.",
        storageClass="IsrCalib",
        dimensions=("instrument", "exposure"),
        isCalibration=True,
    )


class PhotodiodeCorrectionConfig(pipeBase.PipelineTaskConfig,
                                 pipelineConnections=PhotodiodeCorrectionConnections):
    """Configuration for calculating the photodiode corrections.
    """
    dummyParameter = pexConfig.Field(
        dtype=bool,
        doc="Dummy parameter",
        default=False,
    )


class PhotodiodeCorrectionTask(pipeBase.PipelineTask, pipeBase.CmdLineTask):
    """Calculate the photodiode corrections.
    """

    ConfigClass = PhotodiodeCorrectionConfig
    _DefaultName = 'cpPhotodiodeCorrection'

    def runQuantum(self, butlerQC, inputRefs, outputRefs):
        """Ensure that the input and output dimensions are passed along.

        Parameters
        ----------
        butlerQC : `lsst.daf.butler.butlerQuantumContext.ButlerQuantumContext`
            Butler to operate on.
        inputRefs : `lsst.pipe.base.connections.InputQuantizedConnection`
            Input data refs to load.
        outputRefs : `lsst.pipe.base.connections.OutputQuantizedConnection`
            Output data refs to persist.
        """
        inputs = butlerQC.get(inputRefs)

        # Use the dimensions to set calib/provenance information.
        inputs['inputDims'] = inputRefs.inputPtc[0].dataId.byName()

        # Need to generate a joint list of detectors present in both inputPtc
        # and inputLinearizer.  We do this here because the detector info is
        # not present in inputPtc metadata.  We could move it when that is fixed.
        self.detectorList = []
        for i, lin in enumerate(inputRefs.inputLinearizer):
            linDetector = lin.dataId["detector"]
            for j, ptc in enumerate(inputRefs.inputPtc):
                ptcDetector = ptc.dataId["detector"]
                if ptcDetector == linDetector:
                    self.detectorList.append((linDetector, i, j))
                    break

        outputs = self.run(**inputs)
        butlerQC.put(outputs, outputRefs)

    def run(self, inputPtc, inputLinearizer, dummy, camera, inputDims):
        """Calculate the systematic photodiode correction.

        Parameters
        ----------
        inputPtc : `lsst.ip.isr.PtcDataset`
            Pre-measured PTC dataset.
        inputLinearizer : `lsst.ip.isr.Linearizer`
            Previously measured linearizer.
        dummy : `lsst.afw.image.Exposure`
            The exposure used to select the appropriate PTC dataset.
            In almost all circumstances, one of the input exposures
            used to generate the PTC dataset is the best option.
        camera : `lsst.afw.cameraGeom.Camera`
            Camera geometry.
        inputDims : `lsst.daf.butler.DataCoordinate` or `dict`
            DataIds to use to populate the output calibration.

        Returns
        -------
        results : `lsst.pipe.base.Struct`
            The results struct containing:

            ``outputCorrection``
                Final correction calibration
                (`lsst.ip.isr.PhotodiodeCorrection`).
            ``outputProvenance``
                Provenance data for the new calibration
                (`lsst.ip.isr.IsrProvenance`).

        Notes
        -----

        """

        # Initialize photodiodeCorrection.
        photodiodeCorrection = PhotodiodeCorrection(log=self.log)

        abscissaCorrections = {}
        # Load all of the corrections, keyed by exposure pair
        for (detector, linIndex, ptcIndex) in self.detectorList:
            try:
                thisLinearizer = inputLinearizer[linIndex]
                thisPtc = inputPtc[ptcIndex]
            except (RuntimeError, OSError):
                continue

            for amp in camera[detector].getAmplifiers():
                ampName = amp.getName()
                fluxResidual = thisLinearizer.fitResiduals[ampName]
                linearSlope = thisLinearizer.linearFit[ampName]
                abscissaCorrection = fluxResidual / linearSlope[1]
                for i, pair in enumerate(thisPtc.inputExpIdPairs[ampName]):
                    key = str(pair[0])
                    try:
                        abscissaCorrections[key].append(abscissaCorrection[i])
                    except KeyError:
                        abscissaCorrections[key] = []
                        abscissaCorrections[key].append(abscissaCorrection[i])
        # Now the correction is the median correction
        # across the whole focal plane.
        for key in abscissaCorrections.keys():
            correction = np.nanmedian(abscissaCorrections[key])
            if np.isnan(correction):
                correction = 0.0
            abscissaCorrections[key] = correction

        photodiodeCorrection.abscissaCorrections = abscissaCorrections

        photodiodeCorrection.validate()
        photodiodeCorrection.updateMetadata(camera=camera, filterName='NONE')
        photodiodeCorrection.updateMetadata(setDate=True, setCalibId=True)
        provenance = IsrProvenance(calibType='photodiodeCorrection')

        return pipeBase.Struct(
            outputPhotodiodeCorrection=photodiodeCorrection,
            outputProvenance=provenance,
        )
