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
import lsst.geom as geom
import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase
import lsst.pipe.base.connectionTypes as cT
import lsst.cp.pipe.cpCombine as cpCombine
import lsst.meas.algorithms as measAlg
import lsst.afw.detection as afwDet
import lsst.afw.image as afwImage
import lsst.afw.geom as afwGeom

from lsst.meas.algorithms import ImagePlane, MultiImage
from .cpCombine import CalibCombineByFilterTask, CalibCombineByFilterConfig

import numpy as np

__all__ = ["CpFringeTask", "CpFringeTaskConfig",
           "CpFringeCombineTask", "CpFringeCombineTaskConfig"]


class CpFringeConnections(pipeBase.PipelineTaskConnections,
                          dimensions=("instrument", "exposure", "detector")):
    inputExp = cT.Input(
        name="cpFringeISR",
        doc="Input pre-processed exposures to combine.",
        storageClass="Exposure",
        dimensions=("instrument", "exposure", "detector"),
    )

    outputExp = cT.Output(
        name="cpFringeProc",
        doc="Output combined proposed calibration.",
        storageClass="Exposure",
        dimensions=("instrument", "exposure", "detector"),
    )


class CpFringeTaskConfig(pipeBase.PipelineTaskConfig,
                         pipelineConnections=CpFringeConnections):
    stats = pexConfig.ConfigurableField(
        target=cpCombine.CalibStatsTask,
        doc="Statistics task to use.",
    )
    subtractBackground = pexConfig.ConfigurableField(
        target=measAlg.SubtractBackgroundTask,
        doc="Background configuration",
    )
    detection = pexConfig.ConfigurableField(
        target=measAlg.SourceDetectionTask,
        doc="Detection configuration",
    )
    detectSigma = pexConfig.Field(
        dtype=float,
        default=2.5,
        doc="Detection psf gaussian sigma.",
    )
    badMaskPlanes = pexConfig.ListField(
        dtype=str,
        default=["NO_DATA", "DETECTED", "BAD", "EDGE", "SAT", "CR"],
        doc="Mask planes to censor.",
    )

    def setDefaults(self):
        self.detection.reEstimateBackground = False
        self.detection.thresholdValue = self.detectSigma
        self.stats.mask = self.badMaskPlanes


class CpFringeTask(pipeBase.PipelineTask):
    """Combine pre-processed fringe frames into a proposed master calibration.
    """

    ConfigClass = CpFringeTaskConfig
    _DefaultName = "cpFringe"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.makeSubtask("stats")
        self.makeSubtask("subtractBackground")
        self.makeSubtask("detection")

    def run(self, inputExp):
        """Preprocess input exposures prior to FRINGE combination.

        This task scales and renormalizes the input frame based on the
        image background, and then masks all pixels above the
        detection threshold.

        Parameters
        ----------
        inputExp : `lsst.afw.image.Exposure`
            Pre-processed fringe frame data to combine.

        Returns
        -------
        results : `lsst.pipe.base.Struct`
            The results struct containing:

            ``outputExp``
                Fringe pre-processed frame (`lsst.afw.image.Exposure`).
        """
        self.subtractBackground.run(inputExp)
        mi = inputExp.getMaskedImage()

        fpSets = self.detection.detectFootprints(inputExp, sigma=self.config.detectSigma)
        mask = mi.getMask()
        detected = 1 << mask.addMaskPlane("DETECTED")
        for fpSet in (fpSets.positive, fpSets.negative):
            if fpSet is not None:
                afwDet.setMaskFromFootprintList(mask, fpSet.getFootprints(), detected)

        # Grow detected masks?
        spans = afwGeom.SpanSet.fromMask(mask, mask.getPlaneBitMask(["DETECTED"]))
        spans = spans.dilated(1)
        spans = spans.clippedTo(inputExp.getBBox())
        spans.setMask(mask, mask.getPlaneBitMask("DETECTED"))

        # Censor and rescale image.
        badMask = mask.getPlaneBitMask(self.config.badMaskPlanes)
        isBad = (mask.getArray() & badMask) != 0
        mi.image.array[isBad] = np.nan

        # Measure background
        bg = self.stats.run(inputExp)
        print(bg)
        mi -= bg

        # self.subtractBackground.run(inputExp)
        # Measure scale
        # scale = np.nanmedian(mi.image.array)
        # mi /= scale
        mi.image.array[isBad] = 0.0

        return pipeBase.Struct(
            outputExp=inputExp,
        )


class CpFringeCombineConnections(pipeBase.PipelineTaskConnections,
                                 dimensions=("instrument", "detector", "physical_filter")):
    # Use the grand-parent version.  Do we need a special storageClass for
    # the output for multiple images?
    inputExpHandles = cT.Input(
        name="cpInputs",
        doc="Input pre-processed exposures to combine.",
        storageClass="Exposure",
        dimensions=("instrument", "detector", "exposure"),
        multiple=True,
        deferLoad=True,
    )
    inputScales = cT.Input(
        name="cpScales",
        doc="Input scale factors to use.",
        storageClass="StructuredDataDict",
        dimensions=("instrument", ),
        multiple=False,
    )

    outputData = cT.Output(
        name="cpFringeProposal",
        doc="Output combined fringe.",
        storageClass="StampsBase",
        dimensions=("instrument", "detector", "physical_filter"),
        isCalibration=True,
    )

    def __init__(self, *, config=None):
        super().__init__(config=config)

        if config and config.exposureScaling != "InputList":
            self.inputs.discard("inputScales")


class CpFringeCombineTaskConfig(CalibCombineByFilterConfig,
                                pipelineConnections=CpFringeCombineConnections):
    nComponent = pexConfig.Field(
        dtype=int,
        default=3,
        doc="Number of PCA components to retain in the output fringe.",
        check=lambda x: x >= 1
    )
    badMaskList = pexConfig.ListField(
        dtype=str,
        default=["BAD", "SAT", "NO_DATA", "DETECTED"],  # "EDGE" is included in the PCP example.
        doc="List of mask planes to exclude from the fringe analysis.",
    )
    minInputsForPca = pexConfig.Field(
        dtype=int,
        default=100,
        doc="Number of input exposures needed to use PCA algorithm.",
        check=lambda x: x > 1,
    )
    pcaMaxIterations = pexConfig.Field(
        dtype=int,
        default=500,
        doc="Maximum iterations for PCA combine step.",
    )
    pcaConvergenceLimit = pexConfig.Field(
        dtype=float,
        default=1e-6,
        doc="Maximum Euclidean norm of the PCA update for convergence.",
    )
    pcaVerbose = pexConfig.Field(
        dtype=bool,
        default=True,
        doc="Increase PCA verbosity.  Probably to be removed.",
    )


class CpFringeCombineTask(CalibCombineByFilterTask):
    """Task to combine input fringe frames into a final set of combined
    fringes.
    """

    ConfigClass = CpFringeCombineTaskConfig
    _DefaultName = "cpFringeCombine"

    def combine(self, target, expHandleList, expScaleList, stats):
        """Combine multiple images.

        Parameters
        ----------
        target : `lsst.afw.image.Exposure`
            Output exposure to construct.
        expHandleList : `list` [`lsst.daf.butler.DeferredDatasetHandle`]
            Input exposure handles to combine.
        expScaleList : `list` [`float`]
            List of scales to apply to each input image.
        stats : `lsst.afw.math.StatisticsControl`
            Control explaining how to combine the input images.
        """
        if len(expHandleList) <= self.config.minInputsForPca:
            # If we have a small number of inputs, the PCA solution
            # will likely not yield a useful set of fringes.  In that
            # case, it's safer to simply combine the pre-processed
            # frames to form a single fringe.
            return super().combine(target, expHandleList, expScaleList, stats)

        subregionSizeArr = self.config.subregionSize
        subregionSize = geom.Extent2I(subregionSizeArr[0], subregionSizeArr[1])
        # Hard code a size to skip needing to guarantee subregions
        # have the same eigenvalue.
        subregionSize = geom.Extent2I(4200, 4200)

        bitMask = target.mask.getPlaneBitMask(self.config.badMaskList)

        detector = None
        eigenList = []
        metadata = target.getMetadata()
        for subBbox in self._subBBoxIter(target.getBBox(), subregionSize):
            imageSet = []
            # The principle component pursuit operates on numpy
            # arrays, so we need to extract those from our exposures.
            for expHandle, expScale in zip(expHandleList, expScaleList):
                inputExp = expHandle.get(parameters={"bbox": subBbox})

                if detector is None:
                    # Since we will have raveled, let's make sure we
                    # have a detector so we know the target shape.
                    detector = inputExp.getDetector()

                imageArray = inputExp.image.array / expScale
                maskArray = inputExp.mask.array
                isBad = (maskArray & bitMask) > 0
                imageArray[isBad] = np.nan
                imageSet.append(imageArray.ravel())  # Drop image to 1d array?
            imageSet = np.array(imageSet)  # raveledCcdArr

            # Now we need to replace nans with plausible values, so we
            # take column means.
            means = np.nanmean(imageSet, axis=0)
            isBad = np.where(np.isnan(means))
            means[isBad] = 0
            for imageCounter in range(len(imageSet)):
                isBad = np.where(np.isnan(imageSet[imageCounter]))
                imageSet[imageCounter][isBad] = means[imageCounter]  # This is means[isBad] in source.  Typo?

            # Run robust PCA:
            L, S, (u, s, v) = pcp(imageSet,
                                  maxiter=self.config.pcaMaxIterations,
                                  delta=self.config.pcaConvergenceLimit,
                                  verbose=self.config.pcaVerbose)
            # Notes for me:
            # L -> low-rank decomposition == PCA components.  shape=(nComponent, imageX*imageY)  # noQa W505
            # S -> sparse "difference"                        shape=(nComponent, imageX*imageY)  # noQa W505
            # (u, s, v) -> SVD of L0.  shapes=( (nComponent, nComponent),
            #                                   (nComponent),
            #                                   (nComponent, imageX*imageY) )
            # This doesn't guarantee that eigenImage_0
            # is always the same, does it?
            # for Lk in L:
            #    eigenImage = afwImage.ExposureF(Lk.reshape()
            # for eigenCounter, (eigenValue, eigenImage) in enumerate(zip(imageSet.getEigenValues(),  # noQa W505
            #                                                             imageSet.getEigenImages())): # noQa W505
            #     eigenList.append(ImagePlane.factory(eigenImage,
            #                                         metadata, eigenCounter, archive_element=None)) # noQa W505
            #     metadata[f"FRINGE_EIGEN_VALUE_{eigenCounter}"] = eigenValue
            # target.image.assign(eigenImage, subBox) # this is
            # wrong as well.
        eigenList.append(ImagePlane.factory(afwImage.ImageF(u), None, 1, archive_element=None))
        output = MultiImage(eigenList, metadata=metadata, use_mask=False, use_variance=False)
        return output


# Document these, they come from Yusra, and are based on Candes et
# al. 2009, https://doi.org/10.1145/1970392.1970395 /
# https://arxiv.org/abs/0912.3599
def shrink(M, tau):
    sgn = np.sign(M)
    S = np.abs(M) - tau
    S[S < 0.0] = 0.0
    return sgn * S


def pcp(M, delta=1e-6, mu=None, maxiter=500, verbose=False, missing_data=True, **svd_args):
    shape = M.shape
    if missing_data:
        col_mean = np.nanmean(M, axis=1)
        missing = np.where(~(np.isfinite(M)))
        M[missing] = np.take(col_mean, missing[0])
        missing = ~(np.isfinite(M))
        # if np.any(missing):
        #    M = np.array(M)
        #    M[missing] = 0.0
    else:
        missing = np.zeros_like(M, dtype=bool)
        if not np.all(np.isfinite(M)):
            print("The matrix has non-finite entries")

    # Initialize the tuning parameters.
    lam = 1.0 / np.sqrt(np.max(shape))
    if mu is None:
        mu = 0.25 * np.prod(shape) / np.sum(np.abs(M))
        if verbose:
            print("mu = {0}".format(mu))

    # Convergence criterion.
    norm = np.sum(M ** 2)

    # Iterate.
    i = 0
    rank = np.min(shape)
    S = np.zeros(shape)
    Y = np.zeros(shape)
    while i < max(maxiter, 1):
        u, s, v = np.linalg.svd(M - S + Y / mu, full_matrices=False, **svd_args)
        s = shrink(s, 1./mu)
        rank = np.sum(s > 0.0)
        u, s, v = u[:, :rank], s[:rank], v[:rank, :]
        L = np.dot(u, np.dot(np.diag(s), v))

        # Shrinkage step.
        S = shrink(M - L + Y / mu, lam / mu)

        # Lagrange step.
        step = M - L - S
        step[missing] = 0.0
        # step[missing] = np.take(col_mean, missing[0])
        Y += mu * step

        # Check for convergence.
        err = np.sqrt(np.sum(step ** 2) / norm)
        if verbose:
            print("Iteration %s: error=%s, rank=%s, nnz=%s"%(i, err, np.sum(s > 0), np.sum(S > 0)))
        if err < delta:
            break
        i += 1

    if i >= maxiter:
        print("convergence not reached")
    return L, S, (u, s, v)
