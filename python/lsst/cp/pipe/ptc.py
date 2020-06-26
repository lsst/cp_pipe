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

__all__ = ['MeasurePhotonTransferCurveTask',
           'MeasurePhotonTransferCurveTaskConfig',
           'PhotonTransferCurveDataset']

import numpy as np
import matplotlib.pyplot as plt
from sqlite3 import OperationalError
from collections import Counter
from dataclasses import dataclass

import lsst.afw.math as afwMath
import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase
from .utils import (NonexistentDatasetTaskDataIdContainer, PairedVisitListTaskRunner,
                    checkExpLengthEqual, fitLeastSq, fitBootstrap, funcPolynomial, funcAstier)
from scipy.optimize import least_squares

from lsst.ip.isr.linearize import Linearizer
import datetime

from .astierCovPtcUtils import (fftSize, CovFft, computeCovDirect, fitData)


class MeasurePhotonTransferCurveTaskConfig(pexConfig.Config):
    """Config class for photon transfer curve measurement task"""
    ccdKey = pexConfig.Field(
        dtype=str,
        doc="The key by which to pull a detector from a dataId, e.g. 'ccd' or 'detector'.",
        default='ccd',
    )
    ptcFitType = pexConfig.ChoiceField(
        dtype=str,
        doc="Fit PTC to approximation in Astier+19 (Equation 16) or to a polynomial.",
        default="POLYNOMIAL",
        allowed={
            "POLYNOMIAL": "n-degree polynomial (use 'polynomialFitDegree' to set 'n').",
            "ASTIERAPPROXIMATION": "Approximation in Astier+19 (Eq. 16).",
            "ASTIERFULL": "Full covariances model in Astier+19 (Eq. 20)"
        }
    )
    maximumRangeCovariancesAstier = pexConfig.Field(
        dtype=int,
        doc="Maximum range of covariances as in Astier+19",
        default=8,
    )
    covAstierRealSpace = pexConfig.Field(
        dtype=bool,
        doc="Calculate covariances in real space or via FFT? (see appendix A of Astier+19).",
        default=False,
    )
    polynomialFitDegree = pexConfig.Field(
        dtype=int,
        doc="Degree of polynomial to fit the PTC, when 'ptcFitType'=POLYNOMIAL.",
        default=3,
    )
    doCreateLinearizer = pexConfig.Field(
        dtype=bool,
        doc="Calculate non-linearity and persist linearizer?",
        default=False,
    )
    polynomialFitDegreeNonLinearity = pexConfig.Field(
        dtype=int,
        doc="If doCreateLinearizer, degree of polynomial to fit the meanSignal vs exposureTime" +
            " curve to produce the table for LinearizeLookupTable.",
        default=3,
    )
    binSize = pexConfig.Field(
        dtype=int,
        doc="Bin the image by this factor in both dimensions.",
        default=1,
    )
    minMeanSignal = pexConfig.Field(
        dtype=float,
        doc="Minimum value (inclusive) of mean signal (in DN) above which to consider.",
        default=0,
    )
    maxMeanSignal = pexConfig.Field(
        dtype=float,
        doc="Maximum value (inclusive) of mean signal (in DN) below which to consider.",
        default=9e6,
    )
    initialNonLinearityExclusionThresholdPositive = pexConfig.RangeField(
        dtype=float,
        doc="Initially exclude data points with a variance that are more than a factor of this from being"
            " linear in the positive direction, from the PTC fit. Note that these points will also be"
            " excluded from the non-linearity fit. This is done before the iterative outlier rejection,"
            " to allow an accurate determination of the sigmas for said iterative fit.",
        default=0.12,
        min=0.0,
        max=1.0,
    )
    initialNonLinearityExclusionThresholdNegative = pexConfig.RangeField(
        dtype=float,
        doc="Initially exclude data points with a variance that are more than a factor of this from being"
            " linear in the negative direction, from the PTC fit. Note that these points will also be"
            " excluded from the non-linearity fit. This is done before the iterative outlier rejection,"
            " to allow an accurate determination of the sigmas for said iterative fit.",
        default=0.25,
        min=0.0,
        max=1.0,
    )
    sigmaCutPtcOutliers = pexConfig.Field(
        dtype=float,
        doc="Sigma cut for outlier rejection in PTC.",
        default=5.0,
    )
    nSigmaClipPtc = pexConfig.Field(
        dtype=float,
        doc="Sigma cut for afwMath.StatisticsControl()",
        default=5.5,
    )
    nIterSigmaClipPtc = pexConfig.Field(
        dtype=int,
        doc="Number of sigma-clipping iterations for afwMath.StatisticsControl()",
        default=1,
    )
    maxIterationsPtcOutliers = pexConfig.Field(
        dtype=int,
        doc="Maximum number of iterations for outlier rejection in PTC.",
        default=2,
    )
    doFitBootstrap = pexConfig.Field(
        dtype=bool,
        doc="Use bootstrap for the PTC fit parameters and errors?.",
        default=False,
    )
    maxAduForLookupTableLinearizer = pexConfig.Field(
        dtype=int,
        doc="Maximum DN value for the LookupTable linearizer.",
        default=2**18,
    )
    instrumentName = pexConfig.Field(
        dtype=str,
        doc="Instrument name.",
        default='',
    )


@dataclass
class LinearityResidualsAndLinearizersDataset:
    """A simple class to hold the output from the
       `calculateLinearityResidualAndLinearizers` function.
    """
    # Normalized coefficients for polynomial NL correction
    polynomialLinearizerCoefficients: list
    # Normalized coefficient for quadratic polynomial NL correction (c0)
    quadraticPolynomialLinearizerCoefficient: float
    # LUT array row for the amplifier at hand
    linearizerTableRow: list
    meanSignalVsTimePolyFitPars: list
    meanSignalVsTimePolyFitParsErr: list
    fractionalNonLinearityResidual: list
    meanSignalVsTimePolyFitReducedChiSq: float


class PhotonTransferCurveDataset:
    """A simple class to hold the output data from the PTC task.

    The dataset is made up of a dictionary for each item, keyed by the
    amplifiers' names, which much be supplied at construction time.

    New items cannot be added to the class to save accidentally saving to the
    wrong property, and the class can be frozen if desired.

    inputVisitPairs records the visits used to produce the data.
    When fitPtc() or fitCovariancesAstier() is run, a mask is built up, which is by definition
    always the same length as inputVisitPairs, rawExpTimes, rawMeans
    and rawVars, and is a list of bools, which are incrementally set to False
    as points are discarded from the fits.

    PTC fit parameters for polynomials are stored in a list in ascending order
    of polynomial term, i.e. par[0]*x^0 + par[1]*x + par[2]*x^2 etc
    with the length of the list corresponding to the order of the polynomial
    plus one.
    """
    def __init__(self, ampNames, ptcFitType):
        # add items to __dict__ directly because __setattr__ is overridden

        # Common
        # instance variables
        self.__dict__["ptcFitType"] = ptcFitType
        self.__dict__["ampNames"] = ampNames
        self.__dict__["badAmps"] = []

        # raw data variables
        self.__dict__["inputVisitPairs"] = {ampName: [] for ampName in ampNames}
        self.__dict__["visitMask"] = {ampName: [] for ampName in ampNames}
        self.__dict__["rawExpTimes"] = {ampName: [] for ampName in ampNames}
        self.__dict__["rawMeans"] = {ampName: [] for ampName in ampNames}
        self.__dict__["rawVars"] = {ampName: [] for ampName in ampNames}

        # Gain and noise
        self.__dict__["gain"] = {ampName: -1. for ampName in ampNames}
        self.__dict__["gainErr"] = {ampName: -1. for ampName in ampNames}
        self.__dict__["noise"] = {ampName: -1. for ampName in ampNames}
        self.__dict__["noiseErr"] = {ampName: -1. for ampName in ampNames}

        # For Standard PTC calculation
        # fit information
        self.__dict__["ptcFitPars"] = {ampName: [] for ampName in ampNames}
        self.__dict__["ptcFitParsError"] = {ampName: [] for ampName in ampNames}
        self.__dict__["ptcFitReducedChiSquared"] = {ampName: [] for ampName in ampNames}

        # For nonlinearity
        self.__dict__["nonLinearity"] = {ampName: [] for ampName in ampNames}
        self.__dict__["nonLinearityError"] = {ampName: [] for ampName in ampNames}
        self.__dict__["fractionalNonLinearityResiduals"] = {ampName: [] for ampName in ampNames}
        self.__dict__["nonLinearityReducedChiSquared"] = {ampName: [] for ampName in ampNames}
        self.__dict__["coefficientsLinearizePolynomial"] = {ampName: [] for ampName in ampNames}
        self.__dict__["coefficientLinearizeSquared"] = {ampName: [] for ampName in ampNames}

        # For full Astier+19 covariances

        self.__dict__["covariancesTuple"] = {ampName: [] for ampName in ampNames}
        self.__dict__["covariancesFitsWithNoB"] = {ampName: [] for ampName in ampNames}
        self.__dict__["covariancesFits"] = {ampName: [] for ampName in ampNames}

        self.__dict__["finalVars"] = {ampName: [] for ampName in ampNames}
        self.__dict__["finalModelVars"] = {ampName: [] for ampName in ampNames}
        self.__dict__["finalMeans"] = {ampName: [] for ampName in ampNames}
        self.__dict__["aMatrix"] = {ampName: [] for ampName in ampNames}
        self.__dict__["bMatrix"] = {ampName: [] for ampName in ampNames}

    def __setattr__(self, attribute, value):
        """Protect class attributes"""
        if attribute not in self.__dict__:
            raise AttributeError(f"{attribute} is not already a member of PhotonTransferCurveDataset, which"
                                 " does not support setting of new attributes.")
        else:
            self.__dict__[attribute] = value

    def getVisitsUsed(self, ampName):
        """Get the visits used, i.e. not discarded, for a given amp.

        If no mask has been created yet, all visits are returned.
        """
        if len(self.visitMask[ampName]) == 0:
            return self.inputVisitPairs[ampName]

        # if the mask exists it had better be the same length as the visitPairs
        assert len(self.visitMask[ampName]) == len(self.inputVisitPairs[ampName])

        pairs = self.inputVisitPairs[ampName]
        mask = self.visitMask[ampName]
        # cast to bool required because numpy
        return [(v1, v2) for ((v1, v2), m) in zip(pairs, mask) if bool(m) is True]

    def getGoodAmps(self):
        return [amp for amp in self.ampNames if amp not in self.badAmps]


class MeasurePhotonTransferCurveTask(pipeBase.CmdLineTask):
    """A class to calculate, fit, and plot a PTC from a set of flat pairs.

    The Photon Transfer Curve (var(signal) vs mean(signal)) is a standard tool
    used in astronomical detectors characterization (e.g., Janesick 2001,
    Janesick 2007). If ptcFitType is "ASTIERAPPROXIMATION" or "POLYNOMIAL",  this task calculates the
    PTC from a series of pairs of flat-field images; each pair taken at identical exposure
    times. The difference image of each pair is formed to eliminate fixed pattern noise,
    and then the variance of the difference image and the mean of the average image
    are used to produce the PTC. An n-degree polynomial or the approximation in Equation
    16 of Astier+19 ("The Shape of the Photon Transfer Curve of CCD sensors",
    arXiv:1905.08677) can be fitted to the PTC curve. These models include
    parameters such as the gain (e/DN) and readout noise.

    Linearizers to correct for signal-chain non-linearity are also calculated.
    The `Linearizer` class, in general, can support per-amp linearizers, but in this
    task this is not supported.

    If ptcFitType is "ASTIERFULL", the covariances of the difference images are calculated via the
    DFT methods described in Astier+19 and the variances for the PTC are given by the cov[0,0] elements
    at each signal level. The full model in Equation 20 of Astier+19 is fit to the PTC to get the gain
    and the noise.

    Parameters
    ----------

    *args: `list`
        Positional arguments passed to the Task constructor. None used at this
        time.
    **kwargs: `dict`
        Keyword arguments passed on to the Task constructor. None used at this
        time.

    """

    RunnerClass = PairedVisitListTaskRunner
    ConfigClass = MeasurePhotonTransferCurveTaskConfig
    _DefaultName = "measurePhotonTransferCurve"

    def __init__(self, *args, **kwargs):
        pipeBase.CmdLineTask.__init__(self, *args, **kwargs)
        plt.interactive(False)  # stop windows popping up when plotting. When headless, use 'agg' backend too
        self.config.validate()
        self.config.freeze()

    @classmethod
    def _makeArgumentParser(cls):
        """Augment argument parser for the MeasurePhotonTransferCurveTask."""
        parser = pipeBase.ArgumentParser(name=cls._DefaultName)
        parser.add_argument("--visit-pairs", dest="visitPairs", nargs="*",
                            help="Visit pairs to use. Each pair must be of the form INT,INT e.g. 123,456")
        parser.add_id_argument("--id", datasetType="photonTransferCurveDataset",
                               ContainerClass=NonexistentDatasetTaskDataIdContainer,
                               help="The ccds to use, e.g. --id ccd=0..100")
        return parser

    @pipeBase.timeMethod
    def runDataRef(self, dataRef, visitPairs):
        """Run the Photon Transfer Curve (PTC) measurement task.

        For a dataRef (which is each detector here),
        and given a list of visit pairs (postISR) at different exposure times,
        measure the PTC.

        Parameters
        ----------
        dataRef : list of lsst.daf.persistence.ButlerDataRef
            dataRef for the detector for the visits to be fit.

        visitPairs : `iterable` of `tuple` of `int`
            Pairs of visit numbers to be processed together
        """

        # setup necessary objects
        detNum = dataRef.dataId[self.config.ccdKey]
        detector = dataRef.get('camera')[dataRef.dataId[self.config.ccdKey]]
        # expand some missing fields that we need for lsstCam.  This is a work-around
        # for Gen2 problems that I (RHL) don't feel like solving.  The calibs pipelines
        # (which inherit from CalibTask) use addMissingKeys() to do basically the same thing
        #
        # Basically, the butler's trying to look up the fields in `raw_visit` which won't work
        for name in dataRef.getButler().getKeys('bias'):
            if name not in dataRef.dataId:
                try:
                    dataRef.dataId[name] = \
                        dataRef.getButler().queryMetadata('raw', [name], detector=detNum)[0]
                except OperationalError:
                    pass

        amps = detector.getAmplifiers()
        ampNames = [amp.getName() for amp in amps]
        datasetPtc = PhotonTransferCurveDataset(ampNames, self.config.ptcFitType)
        self.log.info('Measuring PTC using %s visits for detector %s' % (visitPairs, detector.getId()))

        tupleRecords = []
        allTags = []
        for (v1, v2) in visitPairs:
            # Get postISR exposures.
            dataRef.dataId['expId'] = v1
            exp1 = dataRef.get("postISRCCD", immediate=True)
            dataRef.dataId['expId'] = v2
            exp2 = dataRef.get("postISRCCD", immediate=True)
            del dataRef.dataId['expId']

            checkExpLengthEqual(exp1, exp2, v1, v2, raiseWithMessage=True)
            expTime = exp1.getInfo().getVisitInfo().getExposureTime()

            tupleRows = []
            for ampNumber, amp in enumerate(detector):
                ampName = amp.getName()
                # covAstier: (i, j, var (cov[0,0]), cov, npix)
                doRealSpace = self.config.covAstierRealSpace
                muDiff, varDiff, covAstier = self.measureMeanVarCov(exp1, exp2, region=amp.getBBox(),
                                                                    covAstierRealSpace=doRealSpace)

                datasetPtc.rawExpTimes[ampName].append(expTime)
                datasetPtc.rawMeans[ampName].append(muDiff)
                datasetPtc.rawVars[ampName].append(varDiff)
                datasetPtc.inputVisitPairs[ampName].append((v1, v2))

                tupleRows += [(muDiff, ) + covRow + (ampNumber, expTime, ampName) for covRow in covAstier]
                tags = ['mu', 'i', 'j', 'var', 'cov', 'npix', 'ext', 'expTime', 'ampName']
            allTags += tags
            tupleRecords += tupleRows
        covariancesWithTags = np.core.records.fromrecords(tupleRecords, names=allTags)

        if self.config.ptcFitType in ["ASTIERFULL", ]:
            # Calculate covariances and fit them, including the PTC, to Astier+19 full model (Eq. 20)
            datasetPtc = self.fitCovariancesAstier(datasetPtc, covariancesWithTags)
        elif self.config.ptcFitType in ["ASTIERAPPROXIMATION", "POLYNOMIAL"]:
            # Fit the PTC to a polynomial or to Astier+19 approximation (Eq. 16)
            # Fill up PhotonTransferCurveDataset object.
            datasetPtc = self.fitPtc(datasetPtc, self.config.ptcFitType)

        # Fit a poynomial to calculate non-linearity and persist linearizer.
        if self.config.doCreateLinearizer:
            numberAmps = len(amps)
            numberAduValues = self.config.maxAduForLookupTableLinearizer
            lookupTableArray = np.zeros((numberAmps, numberAduValues), dtype=np.float32)

            # Fit (non)linearity of signal vs time curve.
            # Fill up PhotonTransferCurveDataset object.
            # Fill up array for LUT linearizer (tableArray).
            # Produce coefficients for Polynomial ans Squared linearizers.
            # Build linearizer objects.
            datasetPtc, linsArray = self.fitNonLinearityAndBuildLinearizers(datasetPtc, detector,
                                                                            tableArray=lookupTableArray,
                                                                            log=self.log)

            butler = dataRef.getButler()
            self.log.info("Writing linearizers: \n "
                          "lookup table (linear component of polynomial fit), \n "
                          "polynomial (coefficients for a polynomial correction), \n "
                          "and squared linearizer (quadratic coefficient from polynomial)")

            for (linearizer, dataType) in linsArray:
                detName = detector.getName()
                now = datetime.datetime.utcnow()
                calibDate = now.strftime("%Y-%m-%d")

                butler.put(linearizer, datasetType=dataType, dataId={'detector': detNum,
                           'detectorName': detName, 'calibDate': calibDate})

        self.log.info(f"Writing PTC data to {dataRef.getUri(write=True)}")
        dataRef.put(datasetPtc, datasetType="photonTransferCurveDataset")

        return pipeBase.Struct(exitStatus=0)

    def fitNonLinearityAndBuildLinearizers(self, dataset, detector, tableArray=None, log=None):
        """Fit non-linearity function and build linearizer objects.

        Parameters
        ----------
        dataset : `lsst.cp.pipe.ptc.PhotonTransferCurveDataset`
            The dataset containing information such as the means, variances and exposure times.

        detector : `lsst.afw.cameraGeom.Detector`
            Detector object.

        tableArray : `np.array`, optional
            Optional. Look-up table array with size rows=nAmps and columns=DN values.
            It will be modified in-place if supplied.

        log : `lsst.log.Log`, optional
            Logger to handle messages.

        Returns
        -------
        dataset: `lsst.cp.pipe.ptc.PhotonTransferCurveDataset`
            This is the same dataset as the input paramter, however, it has been modified
            to include information such as the fit vectors and the fit parameters. See
            the class `PhotonTransferCurveDatase`.

        linArray : `list`
            List with (linearizer object, linearizer data type) as entries.
        """

        # Fit NonLinearity
        dataset = self.fitNonLinearity(dataset, tableArray=tableArray)

        # Produce linearizer
        now = datetime.datetime.utcnow()
        calibDate = now.strftime("%Y-%m-%d")

        linArray = []
        for linType, dataType in [("LOOKUPTABLE", 'linearizeLut'),
                                  ("LINEARIZEPOLYNOMIAL", 'linearizePolynomial'),
                                  ("LINEARIZESQUARED", 'linearizeSquared')]:

            if linType == "LOOKUPTABLE":
                tableArray = tableArray
            else:
                tableArray = None

            linearizer = self.buildLinearizerObject(dataset, detector, calibDate, linType,
                                                    instruName=self.config.instrumentName,
                                                    tableArray=tableArray,
                                                    log=log)
            linArray.append((linearizer, dataType))

        return dataset, linArray

    def fitCovariancesAstier(self, dataset, covariancesWithTagsArray):
        """Fit measured flat covariances to full model in Astier+19.

        Parameters
        ----------
        dataset : `lsst.cp.pipe.ptc.PhotonTransferCurveDataset`
            The dataset containing information such as the means, variances and exposure times.

        covariancesWithTagsArray : `numpy.recarray`
            Tuple with at least (mu, cov, var, i, j, npix), where:
            mu : 0.5*(m1 + m2), where:
                mu1: mean value of flat1
                mu2: mean value of flat2
            cov: covariance value at lag(i, j)
            var: variance(covariance value at lag(0, 0))
            i: lag dimension
            j: lag dimension
            npix: number of pixels used for covariance calculation.

        Returns
        -------
        dataset: `lsst.cp.pipe.ptc.PhotonTransferCurveDataset`
            This is the same dataset as the input paramter, however, it has been modified
            to include information such as the fit vectors and the fit parameters. See
            the class `PhotonTransferCurveDatase`.
        """

        covFits, covFitsNoB = fitData(covariancesWithTagsArray, maxMu=self.config.maxMeanSignal,
                                      maxMuElectrons=self.config.maxMeanSignal,
                                      r=self.config.maximumRangeCovariancesAstier)

        dataset.covariancesTuple = covariancesWithTagsArray
        dataset.covariancesFits = covFits
        dataset.covariancesFitsWithNoB = covFitsNoB
        dataset = self.getOutputPtcDataCovAstier(dataset, covFits)

        return dataset

    def getOutputPtcDataCovAstier(self, dataset, covFits):
        """Get output data for PhotonTransferCurveCovAstierDataset from CovFit objects.

        Parameters
        ----------
        dataset : `lsst.cp.pipe.ptc.PhotonTransferCurveDataset`
            The dataset containing information such as the means, variances and exposure times.

        covFits: `dict`
            Dictionary of CovFit objects, with amp names as keys.

        Returns
        -------
        dataset : `lsst.cp.pipe.ptc.PhotonTransferCurveDataset`
            This is the same dataset as the input paramter, however, it has been modified
            to include extra information such as the mask 1D array, gains, reoudout noise, measured signal,
            measured variance, modeled variance, a, and b coefficient matrices (see Astier+19) per amplifier.
            See the class `PhotonTransferCurveDatase`.
            """

        for i, amp in enumerate(covFits):
            fit = covFits[amp]
            meanVecFinal, varVecFinal, varVecModel, wc = fit.getNormalizedFitData(0, 0, divideByMu=False)
            gain = fit.getGain()
            dataset.visitMask[amp] = fit.getMaskVar()
            dataset.gain[amp] = gain
            dataset.gainErr[amp] = fit.getGainErr()
            dataset.noise[amp] = np.sqrt(np.fabs(fit.getRon()))
            dataset.noiseErr[amp] = fit.getRonErr()
            dataset.finalVars[amp].append(varVecFinal/(gain**2))
            dataset.finalModelVars[amp].append(varVecModel/(gain**2))
            dataset.finalMeans[amp].append(meanVecFinal/gain)
            dataset.aMatrix[amp].append(fit.getA())
            dataset.bMatrix[amp].append(fit.getB())

        return dataset

    def measureMeanVarCov(self, exposure1, exposure2, region=None, covAstierRealSpace=False):
        """Calculate the mean of each of two exposures and the variance and covariance of their difference.

        The variance is calculated via afwMath, and the covariance via the methods in Astier+19 (appendix A).
        In theory, var = covariance[0,0]. This should be validated, and in the future, we may decide to just
        keep one (covariance).

        Parameters
        ----------
        exposure1 : `lsst.afw.image.exposure.exposure.ExposureF`
            First exposure of flat field pair.

        exposure2 : `lsst.afw.image.exposure.exposure.ExposureF`
            Second exposure of flat field pair.

        region : `lsst.geom.Box2I`, optional
            Region of each exposure where to perform the calculations (e.g, an amplifier).

        covAstierRealSpace : `bool`, optional
            Should the covariannces in Astier+19 be calculated in real space or via FFT?
            See Appendix A of Astier+19.

        Returns
        -------
        mu : `float`
            0.5*(mu1 + mu2), where mu1, and mu2 are the clipped means of the regions in
            both exposures.

        varDiff : `float`
            Half of the clipped variance of the difference of the regions inthe two input
            exposures.

        covDiffAstier : `list`
            List with tuples of the form (dx, dy, var, cov, npix), where:
                dx : `int`
                    Lag in x
                dy : `int`
                    Lag in y
                var : `float`
                    Variance at (dx, dy).
                cov : `float`
                    Covariance at (dx, dy).
                nPix : `int`
                    Number of pixel pairs used to evaluate var and cov.
        """

        if region is not None:
            im1Area = exposure1.maskedImage[region]
            im2Area = exposure2.maskedImage[region]
        else:
            im1Area = exposure1.maskedImage
            im2Area = exposure2.maskedImage

        im1Area = afwMath.binImage(im1Area, self.config.binSize)
        im2Area = afwMath.binImage(im2Area, self.config.binSize)

        statsCtrl = afwMath.StatisticsControl()
        statsCtrl.setNumSigmaClip(self.config.nSigmaClipPtc)
        statsCtrl.setNumIter(self.config.nIterSigmaClipPtc)
        #  Clipped mean of images; then average of mean.
        mu1 = afwMath.makeStatistics(im1Area, afwMath.MEANCLIP, statsCtrl).getValue()
        mu2 = afwMath.makeStatistics(im2Area, afwMath.MEANCLIP, statsCtrl).getValue()
        mu = 0.5*(mu1 + mu2)

        # Take difference of pairs
        # symmetric formula: diff = (mu2*im1-mu1*im2)/(0.5*(mu1+mu2))
        temp = im2Area.clone()
        temp *= mu1
        diffIm = im1Area.clone()
        diffIm *= mu2
        diffIm -= temp
        diffIm /= mu

        varDiff = 0.5*(afwMath.makeStatistics(diffIm, afwMath.VARIANCECLIP, statsCtrl).getValue())

        # Get the mask and identify good pixels as '1', and the rest as '0'.
        w1 = np.where(im1Area.getMask().getArray() == 0, 1, 0)
        w2 = np.where(im2Area.getMask().getArray() == 0, 1, 0)

        w12 = w1*w2
        wDiff = np.where(diffIm.getMask().getArray() == 0, 1, 0)
        w = w12*wDiff

        maxRangeCov = self.config.maximumRangeCovariancesAstier
        if covAstierRealSpace:
            covDiffAstier = computeCovDirect(diffIm.getImage().getArray(), w, maxRangeCov)
        else:
            shapeDiff = diffIm.getImage().getArray().shape
            fftShape = (fftSize(shapeDiff[0] + maxRangeCov), fftSize(shapeDiff[1]+maxRangeCov))
            c = CovFft(diffIm.getImage().getArray(), w, fftShape, maxRangeCov)
            covDiffAstier = c.reportCovFft(maxRangeCov)

        return mu, varDiff, covDiffAstier

    def computeCovDirect(self, diffImage, weightImage, maxRange):
        """Compute covariances of diffImage in real space.

        For lags larger than ~25, it is slower than the FFT way.
        Taken from https://github.com/PierreAstier/bfptc/

        Parameters
        ----------
        diffImage : `numpy.array`
            Image to compute the covariance of.

        weightImage : `numpy.array`
            Weight image of diffImage (1's and 0's for good and bad pixels, respectively).

        maxRange : `int`
            Last index of the covariance to be computed.

        Returns
        -------
        outList : `list`
            List with tuples of the form (dx, dy, var, cov, npix), where:
                dx : `int`
                    Lag in x
                dy : `int`
                    Lag in y
                var : `float`
                    Variance at (dx, dy).
                cov : `float`
                    Covariance at (dx, dy).
                nPix : `int`
                    Number of pixel pairs used to evaluate var and cov.
        """
        outList = []
        var = 0
        # (dy,dx) = (0,0) has to be first
        for dy in range(maxRange + 1):
            for dx in range(0, maxRange + 1):
                if (dx*dy > 0):
                    cov1, nPix1 = self.covDirectValue(diffImage, weightImage, dx, dy)
                    cov2, nPix2 = self.covDirectValue(diffImage, weightImage, dx, -dy)
                    cov = 0.5*(cov1 + cov2)
                    nPix = nPix1 + nPix2
                else:
                    cov, nPix = self.covDirectValue(diffImage, weightImage, dx, dy)
                if (dx == 0 and dy == 0):
                    var = cov
                outList.append((dx, dy, var, cov, nPix))

        return outList

    def covDirectValue(self, diffImage, weightImage, dx, dy):
        """Compute covariances of diffImage in real space at lag (dx, dy).

        Taken from https://github.com/PierreAstier/bfptc/ (c.f., appendix of Astier+19).

        Parameters
        ----------
        diffImage : `numpy.array`
            Image to compute the covariance of.

        weightImage : `numpy.array`
            Weight image of diffImage (1's and 0's for good and bad pixels, respectively).

        dx : `int`
            Lag in x.

        dy : `int`
            Lag in y.

        Returns
        -------
        cov : `float`
            Covariance at (dx, dy)

        nPix : `int`
            Number of pixel pairs used to evaluate var and cov.
        """
        (nCols, nRows) = diffImage.shape
        # switching both signs does not change anything:
        # it just swaps im1 and im2 below
        if (dx < 0):
            (dx, dy) = (-dx, -dy)
        # now, we have dx >0. We have to distinguish two cases
        # depending on the sign of dy
        if dy >= 0:
            im1 = diffImage[dy:, dx:]
            w1 = weightImage[dy:, dx:]
            im2 = diffImage[:nCols - dy, :nRows - dx]
            w2 = weightImage[:nCols - dy, :nRows - dx]
        else:
            im1 = diffImage[:nCols + dy, dx:]
            w1 = weightImage[:nCols + dy, dx:]
            im2 = diffImage[-dy:, :nRows - dx]
            w2 = weightImage[-dy:, :nRows - dx]
        # use the same mask for all 3 calculations
        wAll = w1*w2
        # do not use mean() because weightImage=0 pixels would then count
        nPix = wAll.sum()
        im1TimesW = im1*wAll
        s1 = im1TimesW.sum()/nPix
        s2 = (im2*wAll).sum()/nPix
        p = (im1TimesW*im2).sum()/nPix
        cov = p - s1*s2

        return cov, nPix

    def buildLinearizerObject(self, dataset, detector, calibDate, linearizerType, instruName='',
                              tableArray=None, log=None):
        """Build linearizer object to persist.

        Parameters
        ----------
        dataset : `lsst.cp.pipe.ptc.PhotonTransferCurveDataset`
            The dataset containing the means, variances, and exposure times

        detector : `lsst.afw.cameraGeom.Detector`
            Detector object

        calibDate : `datetime.datetime`
            Calibration date

        linearizerType : `str`
            'LOOKUPTABLE', 'LINEARIZESQUARED', or 'LINEARIZEPOLYNOMIAL'

        instruName : `str`, optional
            Instrument name

        tableArray : `np.array`, optional
            Look-up table array with size rows=nAmps and columns=DN values

        log : `lsst.log.Log`, optional
            Logger to handle messages

        Returns
        -------
        linearizer : `lsst.ip.isr.Linearizer`
            Linearizer object
        """
        detName = detector.getName()
        detNum = detector.getId()
        if linearizerType == "LOOKUPTABLE":
            if tableArray is not None:
                linearizer = Linearizer(detector=detector, table=tableArray, log=log)
            else:
                raise RuntimeError("tableArray must be provided when creating a LookupTable linearizer")
        elif linearizerType in ("LINEARIZESQUARED", "LINEARIZEPOLYNOMIAL"):
            linearizer = Linearizer(log=log)
        else:
            raise RuntimeError("Invalid linearizerType {linearizerType} to build a Linearizer object. "
                               "Supported: 'LOOKUPTABLE', 'LINEARIZESQUARED', or 'LINEARIZEPOLYNOMIAL'")
        for i, amp in enumerate(detector.getAmplifiers()):
            ampName = amp.getName()
            if linearizerType == "LOOKUPTABLE":
                linearizer.linearityCoeffs[ampName] = [i, 0]
                linearizer.linearityType[ampName] = "LookupTable"
            elif linearizerType == "LINEARIZESQUARED":
                linearizer.linearityCoeffs[ampName] = [dataset.coefficientLinearizeSquared[ampName]]
                linearizer.linearityType[ampName] = "Squared"
            elif linearizerType == "LINEARIZEPOLYNOMIAL":
                linearizer.linearityCoeffs[ampName] = dataset.coefficientsLinearizePolynomial[ampName]
                linearizer.linearityType[ampName] = "Polynomial"
            linearizer.linearityBBox[ampName] = amp.getBBox()

        linearizer.validate()
        calibId = f"detectorName={detName} detector={detNum} calibDate={calibDate} ccd={detNum} filter=NONE"

        try:
            raftName = detName.split("_")[0]
            calibId += f" raftName={raftName}"
        except Exception:
            raftname = "NONE"
            calibId += f" raftName={raftname}"

        serial = detector.getSerial()
        linearizer.updateMetadata(instrumentName=instruName, detectorId=f"{detNum}",
                                  calibId=calibId, serial=serial, detectorName=f"{detName}")

        return linearizer

    @staticmethod
    def _initialParsForPolynomial(order):
        assert(order >= 2)
        pars = np.zeros(order, dtype=np.float)
        pars[0] = 10
        pars[1] = 1
        pars[2:] = 0.0001
        return pars

    @staticmethod
    def _boundsForPolynomial(initialPars):
        lowers = [np.NINF for p in initialPars]
        uppers = [np.inf for p in initialPars]
        lowers[1] = 0  # no negative gains
        return (lowers, uppers)

    @staticmethod
    def _boundsForAstier(initialPars):
        lowers = [np.NINF for p in initialPars]
        uppers = [np.inf for p in initialPars]
        return (lowers, uppers)

    @staticmethod
    def _getInitialGoodPoints(means, variances, maxDeviationPositive, maxDeviationNegative):
        """Return a boolean array to mask bad points.

        A linear function has a constant ratio, so find the median
        value of the ratios, and exclude the points that deviate
        from that by more than a factor of maxDeviationPositive/negative.
        Asymmetric deviations are supported as we expect the PTC to turn
        down as the flux increases, but sometimes it anomalously turns
        upwards just before turning over, which ruins the fits, so it
        is wise to be stricter about restricting positive outliers than
        negative ones.

        Too high and points that are so bad that fit will fail will be included
        Too low and the non-linear points will be excluded, biasing the NL fit."""
        ratios = [b/a for (a, b) in zip(means, variances)]
        medianRatio = np.median(ratios)
        ratioDeviations = [(r/medianRatio)-1 for r in ratios]

        # so that it doesn't matter if the deviation is expressed as positive or negative
        maxDeviationPositive = abs(maxDeviationPositive)
        maxDeviationNegative = -1. * abs(maxDeviationNegative)

        goodPoints = np.array([True if (r < maxDeviationPositive and r > maxDeviationNegative)
                              else False for r in ratioDeviations])
        return goodPoints

    def _makeZeroSafe(self, array, warn=True, substituteValue=1e-9):
        """"""
        nBad = Counter(array)[0]
        if nBad == 0:
            return array

        if warn:
            msg = f"Found {nBad} zeros in array at elements {[x for x in np.where(array==0)[0]]}"
            self.log.warn(msg)

        array[array == 0] = substituteValue
        return array

    def calculateLinearityResidualAndLinearizers(self, exposureTimeVector, meanSignalVector):
        """Calculate linearity residual and fit an n-order polynomial to the mean vs time curve
        to produce corrections (deviation from linear part of polynomial) for a particular amplifier
        to populate LinearizeLookupTable.
        Use the coefficients of this fit to calculate the correction coefficients for  LinearizePolynomial
        and LinearizeSquared."

        Parameters
        ---------

        exposureTimeVector: `list` of `float`
            List of exposure times for each flat pair

        meanSignalVector: `list` of `float`
            List of mean signal from diference image of flat pairs

        Returns
        -------
        dataset : `lsst.cp.pipe.ptc.LinearityResidualsAndLinearizersDataset`
            The dataset containing the fit parameters, the NL correction coefficients, and the
            LUT row for the amplifier at hand. Explicitly:

        dataset.polynomialLinearizerCoefficients : `list` of `float`
            Coefficients for LinearizePolynomial, where corrImage = uncorrImage + sum_i c_i uncorrImage^(2 +
            i).
            c_(j-2) = -k_j/(k_1^j) with units DN^(1-j) (c.f., Eq. 37 of 2003.05978). The units of k_j are
            DN/t^j, and they are fit from meanSignalVector = k0 + k1*exposureTimeVector +
            k2*exposureTimeVector^2 + ... + kn*exposureTimeVector^n, with
            n = "polynomialFitDegreeNonLinearity". k_0 and k_1 and degenerate with bias level and gain,
            and are not used by the non-linearity correction. Therefore, j = 2...n in the above expression
            (see `LinearizePolynomial` class in `linearize.py`.)

        dataset.quadraticPolynomialLinearizerCoefficient : `float`
            Coefficient for LinearizeSquared, where corrImage = uncorrImage + c0*uncorrImage^2.
            c0 = -k2/(k1^2), where k1 and k2 are fit from
            meanSignalVector = k0 + k1*exposureTimeVector + k2*exposureTimeVector^2 +...
                               + kn*exposureTimeVector^n, with n = "polynomialFitDegreeNonLinearity".

        dataset.linearizerTableRow : `list` of `float`
           One dimensional array with deviation from linear part of n-order polynomial fit
           to mean vs time curve. This array will be one row (for the particular amplifier at hand)
           of the table array for LinearizeLookupTable.

        dataset.meanSignalVsTimePolyFitPars  : `list` of `float`
            Parameters from n-order polynomial fit to meanSignalVector vs exposureTimeVector.

        dataset.meanSignalVsTimePolyFitParsErr : `list` of `float`
            Parameters from n-order polynomial fit to meanSignalVector vs exposureTimeVector.

        dataset.fractionalNonLinearityResidual : `list` of `float`
            Fractional residuals from the meanSignal vs exposureTime curve with respect to linear part of
            polynomial fit: 100*(linearPart - meanSignal)/linearPart, where
            linearPart = k0 + k1*exposureTimeVector.

        dataset.meanSignalVsTimePolyFitReducedChiSq  : `float`
            Reduced unweighted chi squared from polynomial fit to meanSignalVector vs exposureTimeVector.
        """

        # Lookup table linearizer
        parsIniNonLinearity = self._initialParsForPolynomial(self.config.polynomialFitDegreeNonLinearity + 1)
        if self.config.doFitBootstrap:
            parsFit, parsFitErr, reducedChiSquaredNonLinearityFit = fitBootstrap(parsIniNonLinearity,
                                                                                 exposureTimeVector,
                                                                                 meanSignalVector,
                                                                                 funcPolynomial)
        else:
            parsFit, parsFitErr, reducedChiSquaredNonLinearityFit = fitLeastSq(parsIniNonLinearity,
                                                                               exposureTimeVector,
                                                                               meanSignalVector,
                                                                               funcPolynomial)

        # LinearizeLookupTable:
        # Use linear part to get time at wich signal is maxAduForLookupTableLinearizer DN
        tMax = (self.config.maxAduForLookupTableLinearizer - parsFit[0])/parsFit[1]
        timeRange = np.linspace(0, tMax, self.config.maxAduForLookupTableLinearizer)
        signalIdeal = parsFit[0] + parsFit[1]*timeRange
        signalUncorrected = funcPolynomial(parsFit, timeRange)
        linearizerTableRow = signalIdeal - signalUncorrected  # LinearizerLookupTable has corrections
        # LinearizePolynomial and LinearizeSquared:
        # Check that magnitude of higher order (>= 3) coefficents of the polyFit are small,
        # i.e., less than threshold = 1e-10 (typical quadratic and cubic coefficents are ~1e-6
        # and ~1e-12).
        k1 = parsFit[1]
        polynomialLinearizerCoefficients = []
        for i, coefficient in enumerate(parsFit):
            c = -coefficient/(k1**i)
            polynomialLinearizerCoefficients.append(c)
            if np.fabs(c) > 1e-10:
                msg = f"Coefficient {c} in polynomial fit larger than threshold 1e-10."
                self.log.warn(msg)
        # Coefficient for LinearizedSquared. Called "c0" in linearize.py
        c0 = polynomialLinearizerCoefficients[2]

        # Fractional non-linearity residual, w.r.t linear part of polynomial fit
        linearPart = parsFit[0] + k1*exposureTimeVector
        fracNonLinearityResidual = 100*(linearPart - meanSignalVector)/linearPart

        dataset = LinearityResidualsAndLinearizersDataset([], None, [], [], [], [], None)
        dataset.polynomialLinearizerCoefficients = polynomialLinearizerCoefficients
        dataset.quadraticPolynomialLinearizerCoefficient = c0
        dataset.linearizerTableRow = linearizerTableRow
        dataset.meanSignalVsTimePolyFitPars = parsFit
        dataset.meanSignalVsTimePolyFitParsErr = parsFitErr
        dataset.fractionalNonLinearityResidual = fracNonLinearityResidual
        dataset.meanSignalVsTimePolyFitReducedChiSq = reducedChiSquaredNonLinearityFit

        return dataset

    def fitPtc(self, dataset, ptcFitType):
        """Fit the photon transfer curve to a polynimial or to Astier+19 approximation.

        Fit the photon transfer curve with either a polynomial of the order
        specified in the task config, or using the Astier approximation.

        Sigma clipping is performed iteratively for the fit, as well as an
        initial clipping of data points that are more than
        config.initialNonLinearityExclusionThreshold away from lying on a
        straight line. This other step is necessary because the photon transfer
        curve turns over catastrophically at very high flux (because saturation
        drops the variance to ~0) and these far outliers cause the initial fit
        to fail, meaning the sigma cannot be calculated to perform the
        sigma-clipping.

        Parameters
        ----------
        dataset : `lsst.cp.pipe.ptc.PhotonTransferCurveDataset`
            The dataset containing the means, variances and exposure times
        ptcFitType : `str`
            Fit a 'POLYNOMIAL' (degree: 'polynomialFitDegree') or
            'ASTIERAPPROXIMATION' (Eq. 16 of Astier+19) to the PTC

        Returns
        -------
        dataset: `lsst.cp.pipe.ptc.PhotonTransferCurveDataset`
            This is the same dataset as the input paramter, however, it has been modified
            to include information such as the fit vectors and the fit parameters. See
            the class `PhotonTransferCurveDatase`.
        """

        def errFunc(p, x, y):
            return ptcFunc(p, x) - y

        sigmaCutPtcOutliers = self.config.sigmaCutPtcOutliers
        maxIterationsPtcOutliers = self.config.maxIterationsPtcOutliers

        for i, ampName in enumerate(dataset.ampNames):
            timeVecOriginal = np.array(dataset.rawExpTimes[ampName])
            meanVecOriginal = np.array(dataset.rawMeans[ampName])
            varVecOriginal = np.array(dataset.rawVars[ampName])
            varVecOriginal = self._makeZeroSafe(varVecOriginal)

            mask = ((meanVecOriginal >= self.config.minMeanSignal) &
                    (meanVecOriginal <= self.config.maxMeanSignal))

            goodPoints = self._getInitialGoodPoints(meanVecOriginal, varVecOriginal,
                                                    self.config.initialNonLinearityExclusionThresholdPositive,
                                                    self.config.initialNonLinearityExclusionThresholdNegative)
            mask = mask & goodPoints

            if ptcFitType == 'ASTIERAPPROXIMATION':
                ptcFunc = funcAstier
                parsIniPtc = [-1e-9, 1.0, 10.]  # a00, gain, noise
                bounds = self._boundsForAstier(parsIniPtc)
            if ptcFitType == 'POLYNOMIAL':
                ptcFunc = funcPolynomial
                parsIniPtc = self._initialParsForPolynomial(self.config.polynomialFitDegree + 1)
                bounds = self._boundsForPolynomial(parsIniPtc)

            # Before bootstrap fit, do an iterative fit to get rid of outliers
            count = 1
            while count <= maxIterationsPtcOutliers:
                # Note that application of the mask actually shrinks the array
                # to size rather than setting elements to zero (as we want) so
                # always update mask itself and re-apply to the original data
                meanTempVec = meanVecOriginal[mask]
                varTempVec = varVecOriginal[mask]
                res = least_squares(errFunc, parsIniPtc, bounds=bounds, args=(meanTempVec, varTempVec))
                pars = res.x

                # change this to the original from the temp because the masks are ANDed
                # meaning once a point is masked it's always masked, and the masks must
                # always be the same length for broadcasting
                sigResids = (varVecOriginal - ptcFunc(pars, meanVecOriginal))/np.sqrt(varVecOriginal)
                newMask = np.array([True if np.abs(r) < sigmaCutPtcOutliers else False for r in sigResids])
                mask = mask & newMask

                nDroppedTotal = Counter(mask)[False]
                self.log.debug(f"Iteration {count}: discarded {nDroppedTotal} points in total for {ampName}")
                count += 1
                # objects should never shrink
                assert (len(mask) == len(timeVecOriginal) == len(meanVecOriginal) == len(varVecOriginal))

            dataset.visitMask[ampName] = mask  # store the final mask
            parsIniPtc = pars
            meanVecFinal = meanVecOriginal[mask]
            varVecFinal = varVecOriginal[mask]

            if Counter(mask)[False] > 0:
                self.log.info((f"Number of points discarded in PTC of amplifier {ampName}:" +
                               f" {Counter(mask)[False]} out of {len(meanVecOriginal)}"))

            if (len(meanVecFinal) < len(parsIniPtc)):
                msg = (f"\nSERIOUS: Not enough data points ({len(meanVecFinal)}) compared to the number of"
                       f"parameters of the PTC model({len(parsIniPtc)}). Setting {ampName} to BAD.")
                self.log.warn(msg)
                # The first and second parameters of initial fit are discarded (bias and gain)
                # for the final NL coefficients
                lenNonLinPars = self.config.polynomialFitDegreeNonLinearity - 1
                dataset.badAmps.append(ampName)
                dataset.gain[ampName] = np.nan
                dataset.gainErr[ampName] = np.nan
                dataset.noise[ampName] = np.nan
                dataset.noiseErr[ampName] = np.nan
                dataset.nonLinearity[ampName] = np.nan
                dataset.nonLinearityError[ampName] = np.nan
                dataset.fractionalNonLinearityResiduals[ampName] = np.nan
                dataset.coefficientLinearizeSquared[ampName] = np.nan
                dataset.ptcFitPars[ampName] = np.nan
                dataset.ptcFitParsError[ampName] = np.nan
                dataset.ptcFitReducedChiSquared[ampName] = np.nan
                dataset.coefficientsLinearizePolynomial[ampName] = [np.nan]*lenNonLinPars
                continue

            # Fit the PTC
            if self.config.doFitBootstrap:
                parsFit, parsFitErr, reducedChiSqPtc = fitBootstrap(parsIniPtc, meanVecFinal,
                                                                    varVecFinal, ptcFunc)
            else:
                parsFit, parsFitErr, reducedChiSqPtc = fitLeastSq(parsIniPtc, meanVecFinal,
                                                                  varVecFinal, ptcFunc)
            dataset.ptcFitPars[ampName] = parsFit
            dataset.ptcFitParsError[ampName] = parsFitErr
            dataset.ptcFitReducedChiSquared[ampName] = reducedChiSqPtc

            if ptcFitType == 'ASTIERAPPROXIMATION':
                ptcGain = parsFit[1]
                ptcGainErr = parsFitErr[1]
                ptcNoise = np.sqrt(np.fabs(parsFit[2]))
                ptcNoiseErr = 0.5*(parsFitErr[2]/np.fabs(parsFit[2]))*np.sqrt(np.fabs(parsFit[2]))
            if ptcFitType == 'POLYNOMIAL':
                ptcGain = 1./parsFit[1]
                ptcGainErr = np.fabs(1./parsFit[1])*(parsFitErr[1]/parsFit[1])
                ptcNoise = np.sqrt(np.fabs(parsFit[0]))*ptcGain
                ptcNoiseErr = (0.5*(parsFitErr[0]/np.fabs(parsFit[0]))*(np.sqrt(np.fabs(parsFit[0]))))*ptcGain
            dataset.gain[ampName] = ptcGain
            dataset.gainErr[ampName] = ptcGainErr
            dataset.noise[ampName] = ptcNoise
            dataset.noiseErr[ampName] = ptcNoiseErr
        if not dataset.ptcFitType:
            dataset.ptcFitType = ptcFitType

        return dataset

    def fitNonLinearity(self, dataset, tableArray=None):
        """Fit a polynomial to signal vs effective time curve to calculate linearity and residuals.

        Parameters
        ----------
        dataset : `lsst.cp.pipe.ptc.PhotonTransferCurveDataset`
            The dataset containing the means, variances and exposure times
        tableArray : `np.array`
            Optional. Look-up table array with size rows=nAmps and columns=DN values.
            It will be modified in-place if supplied.

        Returns
        -------
        dataset: `lsst.cp.pipe.ptc.PhotonTransferCurveDataset`
            This is the same dataset as the input paramter, however, it has been modified
            to include information such as the fit vectors and the fit parameters. See
            the class `PhotonTransferCurveDatase`.
        """
        for i, ampName in enumerate(dataset.ampNames):
            # If a mask is not found, use all points.
            if (len(dataset.visitMask[ampName]) == 0):
                self.log.warn(f"Mask not found for {ampName} in non-linearity fit. Using all points.")
                mask = np.repeat(True, len(dataset.rawExpTimes[ampName]))
            else:
                mask = dataset.visitMask[ampName]

            timeVecFinal = np.array(dataset.rawExpTimes[ampName])[mask]
            meanVecFinal = np.array(dataset.rawMeans[ampName])[mask]

            # Non-linearity residuals (NL of mean vs time curve): percentage, and fit to a quadratic function
            # In this case, len(parsIniNonLinearity) = 3 indicates that we want a quadratic fit
            datasetLinRes = self.calculateLinearityResidualAndLinearizers(timeVecFinal, meanVecFinal)

            # LinearizerLookupTable
            if tableArray is not None:
                tableArray[i, :] = datasetLinRes.linearizerTableRow
            dataset.nonLinearity[ampName] = datasetLinRes.meanSignalVsTimePolyFitPars
            dataset.nonLinearityError[ampName] = datasetLinRes.meanSignalVsTimePolyFitParsErr
            dataset.fractionalNonLinearityResiduals[ampName] = datasetLinRes.fractionalNonLinearityResidual
            dataset.nonLinearityReducedChiSquared[ampName] = datasetLinRes.meanSignalVsTimePolyFitReducedChiSq
            # Slice correction coefficients (starting at 2) for polynomial linearizer. The first
            # and second are reduntant  with the bias and gain, respectively,
            # and are not used by LinearizerPolynomial.
            polyLinCoeffs = np.array(datasetLinRes.polynomialLinearizerCoefficients[2:])
            dataset.coefficientsLinearizePolynomial[ampName] = polyLinCoeffs
            quadPolyLinCoeff = datasetLinRes.quadraticPolynomialLinearizerCoefficient
            dataset.coefficientLinearizeSquared[ampName] = quadPolyLinCoeff

        return dataset
