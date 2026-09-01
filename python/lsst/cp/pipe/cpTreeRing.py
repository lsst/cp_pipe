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
import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase
import lsst.pipe.base.connectionTypes as cT

import numpy as np

from astropy.table import Table
from scipy.ndimage import gaussian_filter


__all__ = ["CpTreeRingTask", "CpTreeRingTaskConfig"]


class CpTreeRingConnections(pipeBase.PipelineTaskConnections,
                            dimensions=("instrument", "detector", "physical_filter")):
    inputFlat = cT.Input(
        name="flat",
        doc="Input flat to measure the tree rings on.",
        storageClass="Exposure",
        dimensions=("instrument", "exposure", "detector", "physical_filter"),
    )

    # TODO: This will require a new curated dataset type.
    # treeRingCenters = cT.PrerequisiteInput(
    #     name="tree_ring_centers",
    #     doc="Table containing the centers for all the tree rings.",
    #     storageClass="ArrowAstropy",
    #     dimensions=("instrument", ),
    # )

    treeRingImage = cT.Output(
        name="tree_ring_image",
        doc="Measured tree ring image.",
        storageClass="Exposure",
        dimensions=("instrument", "detector", "physical_filter"),
    )
    treeRingCorrectedFlat = cT.Output(
        name="tree_ring_flat",
        doc="The input flat divided by the tree ring solution.",
        storageClass="Exposure",
        dimensions=("instrument", "detector", "physical_filter"),
    )


class CpTreeRingTaskConfig(pipeBase.PipelineTaskConfig,
                           pipelineConnections=CpTreeRingConnections):
    overrideCenters = pexConfig.Field(
        dtype=bool,
        default=True,
        doc="Ignore center table for config based centers if true?",
    )
    overrideX0 = pexConfig.Field(
        dtype=float,
        default=0.0,
        doc="Tree ring center x0 in detector pixels.",
    )
    overrideY0 = pexConfig.Field(
        dtype=float,
        default=0.0,
        doc="Tree ring center y0 in detector pixels.",
    )

    convolveSigma = pexConfig.Field(
        dtype=float,
        default=0.0,
        doc="Gaussian signal to convolve image prior to tree ring measurement.  Disabled if 0.0.",
    )


class CpTreeRingTask(pipeBase.PipelineTask):
    """Measure and divide the tree ring signal from a flat, using the
    specified center.
    """

    ConfigClass = CpTreeRingTaskConfig
    _DefaultName = "cpTreeRing"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def runQuantum(self, butlerQC, inputRefs, outputRefs):
        inputs = butlerQC.get(inputRefs)
        dimensions = dict(inputs["inputFlat"])

        inputs["inputDims"] = dimensions

        outputs = self.run(**inputs)
        butlerQC.put(outputs, outputRefs)

    def run(self, inputFlat, treeRingCenters=None, inputDims=None):
        """Run tree ring fitting on the flat, and divide out that solution.

        Parameters
        ----------
        inputFlat : `lsst.afw.image.Exposure`
            Combined flat to measure the tree ring signals on.  A
            single-LED flat is probably preferred.
        treeRingCenters : `astropy.table.Table`, optional
            Table of per-detector center solutions.

        Returns
        -------
        results : `lsst.pipe.base.Struct`
            The results struct containing:

            ``treeRingImage``
                The measured tree ring signal, in image format
                (`lsst.afw.image.Exposure`).
            ``treeRingCorrectedFlat``
                The input flat, divided by the tree ring image
                (`lsst.afw.image.Exposure`).
        """
        # Get the centers
        x0 = None
        y0 = None
        if treeRingCenters is None and self.config.overrideCenters is not True:
            raise RuntimeError("No center table supplied, and overrideCenters=False.")
        if treeRingCenters:
            if inputDims is None:
                raise RuntimeError("No dimensions supplied; cannot look up centers in table.")
            centerRow = treeRingCenters[treeRingCenters["detector"] == inputDims["detector"]]
            if len(centerRow) != 1:
                raise RuntimeError("Could not find detector in center table.")
            else:
                x0 = centerRow["x0"]
                y0 = centerRow["y0"]
        if self.config.overrideCenters:
            x0 = self.config.overrideX0
            y0 = self.config.overrideY0

        if x0 is None or y0 is None:
            raise RuntimeError("Did not determine the center for the tree rings.  Whoops.")

        measureImage = inputFlat.clone()
        if self.config.convolveSigma != 0.0:
            measureImage.image.array = gaussian_filter(measureImage.image.array, self.config.convolveSigma)

        tree_ring_solution = self.tree_ring_fitter(measureImage, x0, y0)

        # These image products inherit the header from the flat.  Is
        # that a problem?
        tree_ring_image = self.tree_ring_realization(inputFlat, tree_ring_solution, x0, y0)

        tree_ring_corrected_flat = inputFlat.clone()
        tree_ring_corrected_flat.image /= tree_ring_image.image

        # This could return the tree_ring_solution as a table, if we need that.
        return pipeBase.Struct(
            treeRingImage=tree_ring_image,
            treeRingModel=tree_ring_solution.tree_ring_model,
            treeRingCorrectedFlat=tree_ring_corrected_flat,
        )

    @staticmethod
    def tree_ring_fitter(exp, x0, y0):
        """Measure tree rings.

        Parameters
        ----------
        exp : `lsst.afw.image.Exposure`
            Image to measure the tree rings on.
        x0 : `float`
            Center point x of the tree rings.
        y0 : `float`
            Center point y of the tree rings.

        Returns
        -------
        results : `lsst.pipe.base.Struct`
            Results struct containing:

            ``tree_ring_model``
                Table containing the model, sigma, and Npts as a
                function of radius.
            ``tree_ring_dict``
                A dictionary containing the model as a function of
                radius for evaluating the image.
        """
        V = {}

        ydim, xdim = exp.image.array.shape
        for yy in range(ydim):
            for xx in range(xdim):
                R = np.linalg.norm((xx - x0, yy - y0))

                # This is the simplest solution.  A scan of the table
                # of centers suggests that we're likely always
                # starting at a radius of ~1000 pixels, so I don't
                # think we need to do an elaborate transformation.
                RR = int(np.round(R))

                if RR not in V.keys():
                    V[RR] = []

                # Upgrade to configurable mask planes?  If we only
                # measure on flats, we have a more limited set of
                # mask planes on the image.
                if exp.mask.array[yy, xx] == 0:
                    V[RR].append(exp.image.array[yy, xx])

        tree_ring_data = {
            'r': [],
            'value': [],
            'sigma': [],
            'Npts': []
        }

        tree_ring_d = {}
        for RR, FF in sorted(V.items()):
            tree_ring_data["r"].append(RR)
            mF = np.nanmedian(FF)
            sF = np.nanstd(FF)
            tree_ring_data["value"].append(mF)
            tree_ring_data["sigma"].append(sF)
            tree_ring_data["Npts"].append(len(FF))
            tree_ring_d[RR] = mF

        return pipeBase.Struct(
            tree_ring_model=Table(tree_ring_data),
            tree_ring_dict=tree_ring_d,
        )

    @staticmethod
    def tree_ring_realization(exp, TRS, x0, y0):
        """Generate a realization of the tree ring model as an image.

        Parameters
        ----------
        exp : `lsst.afw.image.Exposure`
            Image to create the realization for.
        TRS : `lsst.pipe.base.Struct`
            The results struct from ``tree_ring_fitter``.

        Returns
        -------
        tree_ring_image : `lsst.afw.image.Exposure`
            The image realization.
        """
        tree_ring_image = exp.clone()
        tree_ring_image.image.array[:, :] = 0.0

        ydim, xdim = exp.image.array.shape

        for yy in range(ydim):
            for xx in range(xdim):
                # This should match the algorithm in self.tree_ring_fitter.
                R = np.linalg.norm((xx - x0, yy - y0))
                RR = int(np.round(R))

                tree_ring_image.image.array[yy, xx] = TRS.tree_ring_dict[RR]

        return tree_ring_image
