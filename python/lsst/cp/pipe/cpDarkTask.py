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
import math
import numpy as np

import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase
import lsst.pipe.base.connectionTypes as cT
from lsst.pipe.tasks.repair import RepairTask
import lsst.meas.algorithms as measAlg
import lsst.afw.geom as afwGeom


__all__ = ["CpDarkTask", "CpDarkTaskConfig"]


class CpDarkConnections(pipeBase.PipelineTaskConnections,
                        dimensions=("instrument", "exposure", "detector")):
    inputExp = cT.Input(
        name="cpDarkISR",
        doc="Input pre-processed exposures to combine.",
        storageClass="Exposure",
        dimensions=("instrument", "exposure", "detector"),
    )

    outputExp = cT.Output(
        name="cpDarkProc",
        doc="Output combined proposed calibration.",
        storageClass="Exposure",
        dimensions=("instrument", "exposure", "detector"),
    )


class CpDarkTaskConfig(pipeBase.PipelineTaskConfig,
                       pipelineConnections=CpDarkConnections):
    psfFwhm = pexConfig.Field(
        dtype=float,
        default=3.0,
        doc="Repair PSF FWHM (pixels).",
    )
    psfSize = pexConfig.Field(
        dtype=int,
        default=21,
        doc="Repair PSF size (pixels).",
    )
    crGrow = pexConfig.Field(
        dtype=int,
        default=2,
        doc="Grow radius for CR (pixels).",
    )
    repair = pexConfig.ConfigurableField(
        target=RepairTask,
        doc="Repair task to use.",
    )


class CpDarkTask(pipeBase.PipelineTask,
                 pipeBase.CmdLineTask):
    """Combine pre-processed dark frames into a proposed master calibration.

    """
    ConfigClass = CpDarkTaskConfig
    _DefaultName = "cpDark"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.makeSubtask("repair")

    def run(self, inputExp):
        """Preprocess input exposures prior to DARK combination.

        This task detects and repairs cosmic rays strikes.

        Parameters
        ----------
        inputExp : `lsst.afw.image.Exposure`
            Pre-processed dark frame data to combine.

        Returns
        -------
        outputExp : `lsst.afw.image.Exposure`
            CR rejected, ISR processed Dark Frame."
        """
        psf = measAlg.SingleGaussianPsf(self.config.psfSize,
                                        self.config.psfSize,
                                        self.config.psfFwhm/(2*math.sqrt(2*math.log(2))))
        inputExp.setPsf(psf)
        scaleExp = inputExp.clone()
        mi = scaleExp.getMaskedImage()

        # DM-23680:
        # Darktime scaling necessary for repair.run() to ID CRs correctly.
        scale = inputExp.getInfo().getVisitInfo().getDarkTime()
        if np.isfinite(scale) and scale != 0.0:
            mi /= scale

        self.repair.run(scaleExp, keepCRs=False)
        if self.config.crGrow > 0:
            crMask = inputExp.getMaskedImage().getMask().getPlaneBitMask("CR")
            spans = afwGeom.SpanSet.fromMask(inputExp.mask, crMask)
            spans = spans.dilated(self.config.crGrow)
            spans.setMask(inputExp.mask, crMask)

        # Undo scaling; as above, DM-23680.
        if np.isfinite(scale) and scale != 0.0:
            mi *= scale

        return pipeBase.Struct(
            outputExp=inputExp,
        )
