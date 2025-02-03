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

import lsst.afw.geom as afwGeom
import lsst.meas.algorithms as measAlg
import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase
import lsst.pipe.base.connectionTypes as cT

from lsst.ip.isr import gainContext, interpolateFromMask
from lsst.pex.exceptions import LengthError
from lsst.pipe.tasks.repair import RepairTask


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
    maskListToInterpolate = pexConfig.ListField(
        dtype=str,
        doc="List of mask planes that should be interpolated.",
        default=['SAT', 'BAD'],
    )
    useLegacyInterp = pexConfig.Field(
        dtype=bool,
        doc="Use the legacy interpolation algorithm. If False use Gaussian Process.",
        default=True,
    )


class CpDarkTask(pipeBase.PipelineTask):
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
        results : `lsst.pipe.base.Struct`
            The results struct containing:

            ``outputExp``
                CR rejected, ISR processed Dark Frame
                (`lsst.afw.image.Exposure`).
        """
        psf = measAlg.SingleGaussianPsf(self.config.psfSize,
                                        self.config.psfSize,
                                        self.config.psfFwhm/(2*math.sqrt(2*math.log(2))))
        inputExp.setPsf(psf)

        # Get the gain used to set the variance plane from the
        # exposure, if possible.  Otherwise, use the cameraGeom value.
        gains = self._get_gains(inputExp)

        # Is this gainContext still required?  TODO DM-48754:
        # Investigate cosmic ray rejection during dark construction
        with gainContext(inputExp, inputExp.getVariance(), apply=True, gains=gains):
            # Scale the variance to match the image plane.  A similar
            # scaling happens during flat-field correction for science
            # images.
            self.log.debug("Median image and variance: %f %f",
                           np.median(inputExp.image.array), np.median(inputExp.variance.array))
            crImage = inputExp.clone()
            # Interpolate the crImage, so the CR code can ignore
            # defects (which will now be interpolated).
            interpolateFromMask(
                maskedImage=crImage.getMaskedImage(),
                fwhm=self.config.psdfFwhm,
                growSaturatedFootprints=self.config.crGrow,
                maskNameList=list(self.config.maskListToInterpolate),
                useLegacyInterp=self.config.useLegacyInterp,
            )

            try:
                self.repair.run(crImage, keepCRs=False)
            except LengthError:
                self.log.warning("CR rejection failed!")

            # Copy results to input frame.
            crBit = crImage.mask.getPlaneBitMask('CR')
            crPixels = np.bitwise_and(crImage.mask.array, crBit)
            inputExp.mask.array[crPixels] |= crBit
            self.log.info("Number of CR pixels: %d",
                          np.count_nonzero(crPixels))

        if self.config.crGrow > 0:
            crMask = inputExp.getMaskedImage().getMask().getPlaneBitMask("CR")
            spans = afwGeom.SpanSet.fromMask(inputExp.mask, crMask)
            spans = spans.dilated(self.config.crGrow)
            spans = spans.clippedTo(inputExp.getBBox())
            spans.setMask(inputExp.mask, crMask)

        # Clear the defect (BAD) mask plane, if it exists.
        planes = inputExp.mask.getMaskPlaneDict()
        if "BAD" in planes:
            inputExp.mask.clearMaskPlane(planes["BAD"])

        return pipeBase.Struct(
            outputExp=inputExp,
        )

    @staticmethod
    def _get_gains(exposure):
        """Get the per-amplifier gains used for this exposure.

        Parameters
        ----------
        exposure : `lsst.afw.image.Exposure`
            The exposure to find gains for.

        Returns
        -------
        gains : `dict` [`str` `float`]
            Dictionary of gain values, keyed by amplifier name.
        """
        det = exposure.getDetector()
        metadata = exposure.getMetadata()
        gains = {}
        for amp in det:
            ampName = amp.getName()
            # The GAIN key may be the new LSST ISR GAIN or the old
            # LSST GAIN.
            if (key1 := f"LSST ISR GAIN {ampName}") in metadata:
                gains[ampName] = metadata[key1]
            elif (key2 := f"LSST GAIN {ampName}") in metadata:
                gains[ampName] = metadata[key2]
            else:
                gains[ampName] = amp.getGain()
        return gains
