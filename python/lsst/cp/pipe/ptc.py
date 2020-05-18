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
import os
from matplotlib.backends.backend_pdf import PdfPages
from sqlite3 import OperationalError
from collections import Counter
from dataclasses import dataclass

import lsst.afw.math as afwMath
import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase
from lsst.ip.isr import IsrTask
from .utils import (NonexistentDatasetTaskDataIdContainer, PairedVisitListTaskRunner,
                    checkExpLengthEqual, validateIsrConfig)
from scipy.optimize import leastsq, least_squares
import numpy.polynomial.polynomial as poly

from lsst.ip.isr.linearize import Linearizer
import datetime

from .astierCovPtcUtils import (find_mask, fft_size, compute_cov_fft, load_fits, 
                                save_fits, fit_data)
from .ptc_plots import make_all_plots


class MeasurePhotonTransferCurveTaskConfig(pexConfig.Config):
    """Config class for photon transfer curve measurement task"""
    isr = pexConfig.ConfigurableField(
        target=IsrTask,
        doc="""Task to perform instrumental signature removal.""",
    )
    isrMandatorySteps = pexConfig.ListField(
        dtype=str,
        doc="isr operations that must be performed for valid results. Raises if any of these are False.",
        default=['doAssembleCcd']
    )
    isrForbiddenSteps = pexConfig.ListField(
        dtype=str,
        doc="isr operations that must NOT be performed for valid results. Raises if any of these are True",
        default=['doFlat', 'doFringe', 'doBrighterFatter', 'doUseOpticsTransmission',
                 'doUseFilterTransmission', 'doUseSensorTransmission', 'doUseAtmosphereTransmission']
    )
    isrDesirableSteps = pexConfig.ListField(
        dtype=str,
        doc="isr operations that it is advisable to perform, but are not mission-critical." +
        " WARNs are logged for any of these found to be False.",
        default=['doBias', 'doDark', 'doCrosstalk', 'doDefect']
    )
    isrUndesirableSteps = pexConfig.ListField(
        dtype=str,
        doc="isr operations that it is *not* advisable to perform in the general case, but are not" +
        " forbidden as some use-cases might warrant them." +
        " WARNs are logged for any of these found to be True.",
        default=['doLinearize']
    )
    ccdKey = pexConfig.Field(
        dtype=str,
        doc="The key by which to pull a detector from a dataId, e.g. 'ccd' or 'detector'.",
        default='ccd',
    )
    makePlots = pexConfig.Field(
        dtype=bool,
        doc="Plot the PTC curves?",
        default=False,
    )
    ptcFitType = pexConfig.ChoiceField(
        dtype=str,
        doc="Fit PTC to approximation in Astier+19 (Equation 16) or to a polynomial.",
        default="POLYNOMIAL",
        allowed={
            "POLYNOMIAL": "n-degree polynomial (use 'polynomialFitDegree' to set 'n').",
            "ASTIERAPPROXIMATION": "Approximation in Astier+19 (Eq. 16)."
        }
    )
    covariancesAstier = pexConfig.Field(
        dtype=bool,
        doc="Compute full covariancesas in Astier+19?",
        default=False,
    )
    polynomialFitDegree = pexConfig.Field(
        dtype=int,
        doc="Degree of polynomial to fit the PTC, when 'ptcFitType'=POLYNOMIAL.",
        default=2,
    )
    polynomialFitDegreeNonLinearity = pexConfig.Field(
        dtype=int,
        doc="Degree of polynomial to fit the meanSignal vs exposureTime curve to produce" +
        " the table for LinearizeLookupTable.",
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
    linResidualTimeIndex = pexConfig.Field(
        dtype=int,
        doc="Index position in time array for reference time in linearity residual calculation.",
        default=2,
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
    # Linearity residual, Eq. 2.2. of Janesick (2001)
    linearityResidual: list
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
    When fitPtcAndNonLinearity() is run, a mask is built up, which is by definition
    always the same length as inputVisitPairs, rawExpTimes, rawMeans
    and rawVars, and is a list of bools, which are incrementally set to False
    as points are discarded from the fits.

    PTC fit parameters for polynomials are stored in a list in ascending order
    of polynomial term, i.e. par[0]*x^0 + par[1]*x + par[2]*x^2 etc
    with the length of the list corresponding to the order of the polynomial
    plus one.
    """
    def __init__(self, ampNames):
        # add items to __dict__ directly because __setattr__ is overridden

        # instance variables
        self.__dict__["ampNames"] = ampNames
        self.__dict__["badAmps"] = []

        # raw data variables
        self.__dict__["inputVisitPairs"] = {ampName: [] for ampName in ampNames}
        self.__dict__["visitMask"] = {ampName: [] for ampName in ampNames}
        self.__dict__["rawExpTimes"] = {ampName: [] for ampName in ampNames}
        self.__dict__["rawMeans"] = {ampName: [] for ampName in ampNames}
        self.__dict__["rawVars"] = {ampName: [] for ampName in ampNames}

        # fit information
        self.__dict__["ptcFitType"] = {ampName: "" for ampName in ampNames}
        self.__dict__["ptcFitPars"] = {ampName: [] for ampName in ampNames}
        self.__dict__["ptcFitParsError"] = {ampName: [] for ampName in ampNames}
        self.__dict__["ptcFitReducedChiSquared"] = {ampName: [] for ampName in ampNames}
        self.__dict__["nonLinearity"] = {ampName: [] for ampName in ampNames}
        self.__dict__["nonLinearityError"] = {ampName: [] for ampName in ampNames}
        self.__dict__["nonLinearityResiduals"] = {ampName: [] for ampName in ampNames}
        self.__dict__["fractionalNonLinearityResiduals"] = {ampName: [] for ampName in ampNames}
        self.__dict__["nonLinearityReducedChiSquared"] = {ampName: [] for ampName in ampNames}

        # final results
        self.__dict__["gain"] = {ampName: -1. for ampName in ampNames}
        self.__dict__["gainErr"] = {ampName: -1. for ampName in ampNames}
        self.__dict__["noise"] = {ampName: -1. for ampName in ampNames}
        self.__dict__["noiseErr"] = {ampName: -1. for ampName in ampNames}
        self.__dict__["coefficientsLinearizePolynomial"] = {ampName: [] for ampName in ampNames}
        self.__dict__["coefficientLinearizeSquared"] = {ampName: [] for ampName in ampNames}

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
        if self.visitMask[ampName] == []:
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
    Janesick 2007). This task calculates the PTC from a series of pairs of
    flat-field images; each pair taken at identical exposure times. The
    difference image of each pair is formed to eliminate fixed pattern noise,
    and then the variance of the difference image and the mean of the average image
    are used to produce the PTC. An n-degree polynomial or the approximation in Equation
    16 of Astier+19 ("The Shape of the Photon Transfer Curve of CCD sensors",
    arXiv:1905.08677) can be fitted to the PTC curve. These models include
    parameters such as the gain (e/DN) and readout noise.

    Linearizers to correct for signal-chain non-linearity are also calculated.
    The `Linearizer` class, in general, can support per-amp linearizers, but in this
    task this is not supported.
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
        self.makeSubtask("isr")
        plt.interactive(False)  # stop windows popping up when plotting. When headless, use 'agg' backend too
        validateIsrConfig(self.isr, self.config.isrMandatorySteps,
                          self.config.isrForbiddenSteps, self.config.isrDesirableSteps, checkTrim=False)
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
        and given a list of visit pairs at different exposure times,
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

        if self.config.covariancesAstier:
            tupleCovariancesWithTags = self.computeCovariancesAstier(dataRef, visitPairs, detector)
            try :
                fits, fits_nb = load_fits('fits.pkl')
            except :
                fits, fits_nb = fit_data(tupleCovariancesWithTags, 1.4e5, 1e5, 8)
                save_fits(fits, fits_nb, 'fits.pkl')
            
            if self.config.makePlots:
                self.plotCovariancesAstier(dataRef, fits, fits_nb)
        else:
            dataset, lookupTableArray = self.computeStandardPtcAndNonLinearity(dataRef, visitPairs, detector)
            if self.config.makePlots:
                self.plot(dataRef, dataset, ptcFitType=self.config.ptcFitType)

            # Save data, PTC fit, and NL fit dictionaries
            self.log.info(f"Writing PTC and NL data to {dataRef.getUri(write=True)}")
            dataRef.put(dataset, datasetType="photonTransferCurveDataset")

            butler = dataRef.getButler()
            self.log.info("Writing linearizers: \n "
                          "lookup table (linear component of polynomial fit), \n "
                          "polynomial (coefficients for a polynomial correction), \n "
                          "and squared linearizer (quadratic coefficient from polynomial)")

            detName = detector.getName()
            now = datetime.datetime.utcnow()
            calibDate = now.strftime("%Y-%m-%d")

            for linType, dataType in [("LOOKUPTABLE", 'linearizeLut'),
                                      ("LINEARIZEPOLYNOMIAL", 'linearizePolynomial'),
                                      ("LINEARIZESQUARED", 'linearizeSquared')]:

                if linType == "LOOKUPTABLE":
                    tableArray = lookupTableArray
                else:
                    tableArray = None

                linearizer = self.buildLinearizerObject(dataset, detector, calibDate, linType,
                                                        instruName=self.config.instrumentName,
                                                        tableArray=tableArray,
                                                        log=self.log)
                butler.put(linearizer, datasetType=dataType, dataId={'detector': detNum,
                           'detectorName': detName, 'calibDate': calibDate})

        self.log.info('Finished measuring PTC for in detector %s' % detNum)

        return pipeBase.Struct(exitStatus=0)

    def computeCovariancesAstier(self, dataRef, visitPairs, detector):
        """Iterate over pairs of flats to calculate full covariances from the difference image.
        This follows the treatment in Astier+19

        Parameters
        ----------
        dataRef : list of lsst.daf.persistence.ButlerDataRef
            dataRef for the detector for the visits to be fit.

        visitPairs : `iterable` of `tuple` of `int`
            Pairs of visit numbers to be processed together

        detector : `lsst.afw.cameraGeom.Detector`
            Detector object

        Returns
        -------

        """
        tupleRecords = []
        allTags = []

        self.log.info('Measuring Astier covariances using %s visits for detector %s' % (visitPairs,
                                                                                        detector.getId()))

        for (v1, v2) in visitPairs:
            # Perform ISR on each exposure
            dataRef.dataId['expId'] = v1
            exp1 = self.isr.runDataRef(dataRef).exposure
            dataRef.dataId['expId'] = v2
            exp2 = self.isr.runDataRef(dataRef).exposure
            del dataRef.dataId['expId']

            checkExpLengthEqual(exp1, exp2, v1, v2, raiseWithMessage=True)
            expTime = exp1.getInfo().getVisitInfo().getExposureTime()

            tupleRows = []
            for ampNumber, amp in enumerate(detector):
                # covs: (mu1, mu2, i, j, var, cov, npix)
                mu1, mu2, covs = self.getCovariancesAstier(exp1, exp2, region=amp.getBBox())
                tupleRows += [(mu1, mu2) + cov + (ampNumber, expTime, amp.getName()) for cov in covs]
                tags = ['mu1', 'mu2', 'i', 'j', 'var', 'cov', 'npix', 'ext', 'expTime', 'ampName']
                print ("Inside amp loop: len(tupleRows), len(tags)", len(tupleRows), len(tags))
            allTags += tags
            tupleRecords += tupleRows
            print ("Inside visits loop len(allTags), len(tupleRecords)",len(allTags), len(tupleRecords) )
        #print ("allTags: ", allTags)
        #print ("tupleRecords: ", tupleRecords)
        print ("After visits loops: len(allTags), len(tupleRecords)", len(allTags), len(tupleRecords))
        print ("len(tupleRecords[0]), tupleRecords[0]: ", len(tupleRecords[0]), tupleRecords[0])
        print ("len(allTags)[0], len(allTags), allTags[0]: ", len(allTags[0]), len(allTags), allTags[0])
        print ("len(allTags)[1], len(allTags), allTags[1]: ", len(allTags[1]), len(allTags), allTags[1])
        tupleCovariancesWithTags = np.core.records.fromrecords(tupleRecords, names=allTags)
        print ("tupleCovariancesWithTags: ", tupleCovariancesWithTags)

        return tupleCovariancesWithTags

    def getCovariancesAstier(self, exposure1, exposure2, region=None):
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
       
        # Get the mask and identify good pixels as '1', and the rest as '0'.
        w1 = np.where (im1Area.getMask().getArray() == 0, 1, 0)
        w2 = np.where (im2Area.getMask().getArray() == 0, 1, 0) 
   
        temp = find_mask(im1Area.getImage().getArray(), 5.0)
        w12 = w1*w2
        wDiff =  np.where (diffIm.getMask().getArray() == 0, 1, 0)
        w = w12*wDiff

        shapeDiff = diffIm.getImage().getArray().shape
        # Make this a parameter
        maxrangeCov = 8
        fft_shape = (fft_size(shapeDiff[0] + maxrangeCov), fft_size(shapeDiff[1]+maxrangeCov))
        covs = compute_cov_fft(diffIm.getImage().getArray(), w, fft_shape, maxrangeCov)
        
        return mu1, mu2, covs

    def computeStandardPtcAndNonLinearity(self, dataRef, visitPairs, detector):
        """Iterate over pairs of flats to calculate the standard PTC from the difference image.
        In this case, 'standard' refers to var(diff) vs mean(diff), as opposed to a more
        general calculation of the covariances of the difference image.
        In addition, the linearizer to correct for signal-chain non-linearity is calculated here
        too.

        Parameters
        ----------
        dataRef : list of lsst.daf.persistence.ButlerDataRef
            dataRef for the detector for the visits to be fit.

        visitPairs : `iterable` of `tuple` of `int`
            Pairs of visit numbers to be processed together

        detector : `lsst.afw.cameraGeom.Detector`
            Detector object

        Returns
        -------
        dataset : `PhotonTransferCurveDataset`
            Output data from the standard PTC and linearizer calculations

        lookupTableArray : `np.array`
            Linearizer look-up table array with size rows=nAmps and columns=DN values
        """
        amps = detector.getAmplifiers()
        ampNames = [amp.getName() for amp in amps]
        dataset = PhotonTransferCurveDataset(ampNames)

        self.log.info('Measuring PTC using %s visits for detector %s' % (visitPairs, detector.getId()))

        for (v1, v2) in visitPairs:
            # Perform ISR on each exposure
            dataRef.dataId['expId'] = v1
            exp1 = self.isr.runDataRef(dataRef).exposure
            dataRef.dataId['expId'] = v2
            exp2 = self.isr.runDataRef(dataRef).exposure
            del dataRef.dataId['expId']

            checkExpLengthEqual(exp1, exp2, v1, v2, raiseWithMessage=True)
            expTime = exp1.getInfo().getVisitInfo().getExposureTime()

            for amp in detector:
                mu, varDiff = self.measureMeanVarPair(exp1, exp2, region=amp.getBBox())
                ampName = amp.getName()

                dataset.rawExpTimes[ampName].append(expTime)
                dataset.rawMeans[ampName].append(mu)
                dataset.rawVars[ampName].append(varDiff)
                dataset.inputVisitPairs[ampName].append((v1, v2))

        numberAmps = len(detector.getAmplifiers())
        numberAduValues = self.config.maxAduForLookupTableLinearizer
        lookupTableArray = np.zeros((numberAmps, numberAduValues), dtype=np.float32)

        # Fit PTC and (non)linearity of signal vs time curve.
        # Fill up PhotonTransferCurveDataset object.
        # Fill up array for LUT linearizer.
        # Produce coefficients for Polynomial ans Squared linearizers.
        dataset = self.fitPtcAndNonLinearity(dataset, self.config.ptcFitType,
                                             tableArray=lookupTableArray)

        return dataset, lookupTableArray

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

    def measureMeanVarPair(self, exposure1, exposure2, region=None):
        """Calculate the mean signal of each of two exposures and the variance of their difference.

        Parameters
        ----------
        exposure1 : `lsst.afw.image.exposure.exposure.ExposureF`
            First exposure of flat field pair.

        exposure2 : `lsst.afw.image.exposure.exposure.ExposureF`
            Second exposure of flat field pair.

        region : `lsst.geom.Box2I`
            Region of each exposure where to perform the calculations (e.g, an amplifier).

        Return
        ------

        mu : `float`
            0.5*(mu1 + mu2), where mu1, and mu2 are the clipped means of the regions in
            both exposures.

        varDiff : `float`
            Half of the clipped variance of the difference of the regions inthe two input
            exposures.
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

        return mu, varDiff

    def _fitLeastSq(self, initialParams, dataX, dataY, function):
        """Do a fit and estimate the parameter errors using using scipy.optimize.leastq.

        optimize.leastsq returns the fractional covariance matrix. To estimate the
        standard deviation of the fit parameters, multiply the entries of this matrix
        by the unweighted reduced chi squared and take the square root of the diagonal elements.

        Parameters
        ----------
        initialParams : `list` of `float`
            initial values for fit parameters. For ptcFitType=POLYNOMIAL, its length
            determines the degree of the polynomial.

        dataX : `numpy.array` of `float`
            Data in the abscissa axis.

        dataY : `numpy.array` of `float`
            Data in the ordinate axis.

        function : callable object (function)
            Function to fit the data with.

        Return
        ------
        pFitSingleLeastSquares : `list` of `float`
            List with fitted parameters.

        pErrSingleLeastSquares : `list` of `float`
            List with errors for fitted parameters.

        reducedChiSqSingleLeastSquares : `float`
            Unweighted reduced chi squared
        """

        def errFunc(p, x, y):
            return function(p, x) - y

        pFit, pCov, infoDict, errMessage, success = leastsq(errFunc, initialParams,
                                                            args=(dataX, dataY), full_output=1, epsfcn=0.0001)

        if (len(dataY) > len(initialParams)) and pCov is not None:
            reducedChiSq = (errFunc(pFit, dataX, dataY)**2).sum()/(len(dataY)-len(initialParams))
            pCov *= reducedChiSq
        else:
            pCov = np.zeros((len(initialParams), len(initialParams)))
            pCov[:, :] = np.inf
            reducedChiSq = np.inf

        errorVec = []
        for i in range(len(pFit)):
            errorVec.append(np.fabs(pCov[i][i])**0.5)

        pFitSingleLeastSquares = pFit
        pErrSingleLeastSquares = np.array(errorVec)

        return pFitSingleLeastSquares, pErrSingleLeastSquares, reducedChiSq

    def _fitBootstrap(self, initialParams, dataX, dataY, function, confidenceSigma=1.):
        """Do a fit using least squares and bootstrap to estimate parameter errors.

        The bootstrap error bars are calculated by fitting 100 random data sets.

        Parameters
        ----------
        initialParams : `list` of `float`
            initial values for fit parameters. For ptcFitType=POLYNOMIAL, its length
            determines the degree of the polynomial.

        dataX : `numpy.array` of `float`
            Data in the abscissa axis.

        dataY : `numpy.array` of `float`
            Data in the ordinate axis.

        function : callable object (function)
            Function to fit the data with.

        confidenceSigma : `float`
            Number of sigmas that determine confidence interval for the bootstrap errors.

        Return
        ------
        pFitBootstrap : `list` of `float`
            List with fitted parameters.

        pErrBootstrap : `list` of `float`
            List with errors for fitted parameters.

        reducedChiSqBootstrap : `float`
            Reduced chi squared.
        """

        def errFunc(p, x, y):
            return function(p, x) - y

        # Fit first time
        pFit, _ = leastsq(errFunc, initialParams, args=(dataX, dataY), full_output=0)

        # Get the stdev of the residuals
        residuals = errFunc(pFit, dataX, dataY)
        sigmaErrTotal = np.std(residuals)

        # 100 random data sets are generated and fitted
        pars = []
        for i in range(100):
            randomDelta = np.random.normal(0., sigmaErrTotal, len(dataY))
            randomDataY = dataY + randomDelta
            randomFit, _ = leastsq(errFunc, initialParams,
                                   args=(dataX, randomDataY), full_output=0)
            pars.append(randomFit)
        pars = np.array(pars)
        meanPfit = np.mean(pars, 0)

        # confidence interval for parameter estimates
        nSigma = confidenceSigma
        errPfit = nSigma*np.std(pars, 0)
        pFitBootstrap = meanPfit
        pErrBootstrap = errPfit

        reducedChiSq = (errFunc(pFitBootstrap, dataX, dataY)**2).sum()/(len(dataY)-len(initialParams))
        return pFitBootstrap, pErrBootstrap, reducedChiSq

    def funcPolynomial(self, pars, x):
        """Polynomial function definition"""
        return poly.polyval(x, [*pars])

    def funcAstier(self, pars, x):
        """Single brighter-fatter parameter model for PTC; Equation 16 of Astier+19"""
        a00, gain, noise = pars
        return 0.5/(a00*gain*gain)*(np.exp(2*a00*x*gain)-1) + noise/(gain*gain)

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
            c_(j-2) = -k_j/(k_1^j) with units (DN^(1-j)). The units of k_j are DN/t^j, and they are fit from
            meanSignalVector = k0 + k1*exposureTimeVector + k2*exposureTimeVector^2 +...
                             + kn*exposureTimeVector^n, with n = "polynomialFitDegreeNonLinearity".
            k_0 and k_1 and degenerate with bias level and gain, and are not used by the non-linearity
            correction. Therefore, j = 2...n in the above expression (see `LinearizePolynomial` class in
            `linearize.py`.)

        dataset.quadraticPolynomialLinearizerCoefficient : `float`
            Coefficient for LinearizeSquared, where corrImage = uncorrImage + c0*uncorrImage^2.
            c0 = -k2/(k1^2), where k1 and k2 are fit from
            meanSignalVector = k0 + k1*exposureTimeVector + k2*exposureTimeVector^2 +...
                               + kn*exposureTimeVector^n, with n = "polynomialFitDegreeNonLinearity".

        dataset.linearizerTableRow : `list` of `float`
           One dimensional array with deviation from linear part of n-order polynomial fit
           to mean vs time curve. This array will be one row (for the particular amplifier at hand)
           of the table array for LinearizeLookupTable.

        dataset.linearityResidual : `list` of `float`
            Linearity residual from the mean vs time curve, defined as
            100*(1 - meanSignalReference/expTimeReference/(meanSignal/expTime).

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
            parsFit, parsFitErr, reducedChiSquaredNonLinearityFit = self._fitBootstrap(parsIniNonLinearity,
                                                                                       exposureTimeVector,
                                                                                       meanSignalVector,
                                                                                       self.funcPolynomial)
        else:
            parsFit, parsFitErr, reducedChiSquaredNonLinearityFit = self._fitLeastSq(parsIniNonLinearity,
                                                                                     exposureTimeVector,
                                                                                     meanSignalVector,
                                                                                     self.funcPolynomial)

        # LinearizeLookupTable:
        # Use linear part to get time at wich signal is maxAduForLookupTableLinearizer DN
        tMax = (self.config.maxAduForLookupTableLinearizer - parsFit[0])/parsFit[1]
        timeRange = np.linspace(0, tMax, self.config.maxAduForLookupTableLinearizer)
        signalIdeal = parsFit[0] + parsFit[1]*timeRange
        signalUncorrected = self.funcPolynomial(parsFit, timeRange)
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

        # Linearity residual
        linResidualTimeIndex = self.config.linResidualTimeIndex
        if exposureTimeVector[linResidualTimeIndex] == 0.0:
            raise RuntimeError("Reference time for linearity residual can't be 0.0")
        linResidual = 100*(1 - ((meanSignalVector[linResidualTimeIndex] /
                           exposureTimeVector[linResidualTimeIndex]) /
                           (meanSignalVector/exposureTimeVector)))

        # Fractional non-linearity residual, w.r.t linear part of polynomial fit
        linearPart = parsFit[0] + k1*exposureTimeVector
        fracNonLinearityResidual = 100*(linearPart - meanSignalVector)/linearPart

        dataset = LinearityResidualsAndLinearizersDataset([], None, [], [], [], [], [], None)
        dataset.polynomialLinearizerCoefficients = polynomialLinearizerCoefficients
        dataset.quadraticPolynomialLinearizerCoefficient = c0
        dataset.linearizerTableRow = linearizerTableRow
        dataset.linearityResidual = linResidual
        dataset.meanSignalVsTimePolyFitPars = parsFit
        dataset.meanSignalVsTimePolyFitParsErr = parsFitErr
        dataset.fractionalNonLinearityResidual = fracNonLinearityResidual
        dataset.meanSignalVsTimePolyFitReducedChiSq = reducedChiSquaredNonLinearityFit

        return dataset

    def fitPtcAndNonLinearity(self, dataset, ptcFitType, tableArray=None):
        """Fit the photon transfer curve and calculate linearity and residuals.

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
            'ASTIERAPPROXIMATION' to the PTC
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
                ptcFunc = self.funcAstier
                parsIniPtc = [-1e-9, 1.0, 10.]  # a00, gain, noise
                bounds = self._boundsForAstier(parsIniPtc)
            if ptcFitType == 'POLYNOMIAL':
                ptcFunc = self.funcPolynomial
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
            timeVecFinal = timeVecOriginal[mask]
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
                dataset.nonLinearityResiduals[ampName] = np.nan
                dataset.fractionalNonLinearityResiduals[ampName] = np.nan
                dataset.coefficientLinearizeSquared[ampName] = np.nan
                dataset.ptcFitPars[ampName] = np.nan
                dataset.ptcFitParsError[ampName] = np.nan
                dataset.ptcFitReducedChiSquared[ampName] = np.nan
                dataset.coefficientsLinearizePolynomial[ampName] = [np.nan]*lenNonLinPars
                tableArray[i, :] = [np.nan]*self.config.maxAduForLookupTableLinearizer
                continue

            # Fit the PTC
            if self.config.doFitBootstrap:
                parsFit, parsFitErr, reducedChiSqPtc = self._fitBootstrap(parsIniPtc, meanVecFinal,
                                                                          varVecFinal, ptcFunc)
            else:
                parsFit, parsFitErr, reducedChiSqPtc = self._fitLeastSq(parsIniPtc, meanVecFinal,
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
            dataset.ptcFitType[ampName] = ptcFitType

            # Non-linearity residuals (NL of mean vs time curve): percentage, and fit to a quadratic function
            # In this case, len(parsIniNonLinearity) = 3 indicates that we want a quadratic fit

            datasetLinRes = self.calculateLinearityResidualAndLinearizers(timeVecFinal, meanVecFinal)

            # LinearizerLookupTable
            if tableArray is not None:
                tableArray[i, :] = datasetLinRes.linearizerTableRow
            dataset.nonLinearity[ampName] = datasetLinRes.meanSignalVsTimePolyFitPars
            dataset.nonLinearityError[ampName] = datasetLinRes.meanSignalVsTimePolyFitParsErr
            dataset.nonLinearityResiduals[ampName] = datasetLinRes.linearityResidual
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

    def plotCovariancesAstier(self, dataRef, covAstierFits, covAstierFitsWithoutB):
        dirname = dataRef.getUri(datasetType='cpPipePlotRoot', write=True)
        if not os.path.exists(dirname):
            os.makedirs(dirname)

        detNum = dataRef.dataId[self.config.ccdKey]
        filename = f"PTC_COVS_ASTIER_det{detNum}.pdf"
        filenameFull = os.path.join(dirname, filename)
        with PdfPages(filenameFull) as pdfPages:
            make_all_plots(covAstierFits, covAstierFitsWithoutB, pdfPages)

    def plot(self, dataRef, dataset, ptcFitType):
        dirname = dataRef.getUri(datasetType='cpPipePlotRoot', write=True)
        if not os.path.exists(dirname):
            os.makedirs(dirname)

        detNum = dataRef.dataId[self.config.ccdKey]
        filename = f"PTC_det{detNum}.pdf"
        filenameFull = os.path.join(dirname, filename)
        with PdfPages(filenameFull) as pdfPages:
            self._plotPtc(dataset, ptcFitType, pdfPages)

    def _plotPtc(self, dataset, ptcFitType, pdfPages):
        """Plot PTC, linearity, and linearity residual per amplifier"""

        reducedChiSqPtc = dataset.ptcFitReducedChiSquared
        if ptcFitType == 'ASTIERAPPROXIMATION':
            ptcFunc = self.funcAstier
            stringTitle = (r"Var = $\frac{1}{2g^2a_{00}}(\exp (2a_{00} \mu g) - 1) + \frac{n_{00}}{g^2}$ "
                           r" ($chi^2$/dof = %g)" % (reducedChiSqPtc))
        if ptcFitType == 'POLYNOMIAL':
            ptcFunc = self.funcPolynomial
            stringTitle = r"Polynomial (degree: %g)" % (self.config.polynomialFitDegree)

        legendFontSize = 7
        labelFontSize = 7
        titleFontSize = 9
        supTitleFontSize = 18
        markerSize = 25

        # General determination of the size of the plot grid
        nAmps = len(dataset.ampNames)
        if nAmps == 2:
            nRows, nCols = 2, 1
        nRows = np.sqrt(nAmps)
        mantissa, _ = np.modf(nRows)
        if mantissa > 0:
            nRows = int(nRows) + 1
            nCols = nRows
        else:
            nRows = int(nRows)
            nCols = nRows

        f, ax = plt.subplots(nrows=nRows, ncols=nCols, sharex='col', sharey='row', figsize=(13, 10))
        f2, ax2 = plt.subplots(nrows=nRows, ncols=nCols, sharex='col', sharey='row', figsize=(13, 10))

        for i, (amp, a, a2) in enumerate(zip(dataset.ampNames, ax.flatten(), ax2.flatten())):
            meanVecOriginal = np.array(dataset.rawMeans[amp])
            varVecOriginal = np.array(dataset.rawVars[amp])
            mask = dataset.visitMask[amp]
            meanVecFinal = meanVecOriginal[mask]
            varVecFinal = varVecOriginal[mask]
            meanVecOutliers = meanVecOriginal[np.invert(mask)]
            varVecOutliers = varVecOriginal[np.invert(mask)]
            pars, parsErr = dataset.ptcFitPars[amp], dataset.ptcFitParsError[amp]

            if ptcFitType == 'ASTIERAPPROXIMATION':
                if len(meanVecFinal):
                    ptcA00, ptcA00error = pars[0], parsErr[0]
                    ptcGain, ptcGainError = pars[1], parsErr[1]
                    ptcNoise = np.sqrt(np.fabs(pars[2]))
                    ptcNoiseError = 0.5*(parsErr[2]/np.fabs(pars[2]))*np.sqrt(np.fabs(pars[2]))
                    stringLegend = (f"a00: {ptcA00:.2e}+/-{ptcA00error:.2e} 1/e"
                                    f"\n Gain: {ptcGain:.4}+/-{ptcGainError:.2e} e/DN"
                                    f"\n Noise: {ptcNoise:.4}+/-{ptcNoiseError:.2e} e \n")

            if ptcFitType == 'POLYNOMIAL':
                if len(meanVecFinal):
                    ptcGain, ptcGainError = 1./pars[1], np.fabs(1./pars[1])*(parsErr[1]/pars[1])
                    ptcNoise = np.sqrt(np.fabs(pars[0]))*ptcGain
                    ptcNoiseError = (0.5*(parsErr[0]/np.fabs(pars[0]))*(np.sqrt(np.fabs(pars[0]))))*ptcGain
                    stringLegend = (f"Noise: {ptcNoise:.4}+/-{ptcNoiseError:.2e} e \n"
                                    f"Gain: {ptcGain:.4}+/-{ptcGainError:.2e} e/DN \n")

                a.set_xlabel(r'Mean signal ($\mu$, DN)', fontsize=labelFontSize)
                a.set_ylabel(r'Variance (DN$^2$)', fontsize=labelFontSize)
                a.tick_params(labelsize=11)
                a.set_xscale('linear', fontsize=labelFontSize)
                a.set_yscale('linear', fontsize=labelFontSize)

                a2.set_xlabel(r'Mean Signal ($\mu$, DN)', fontsize=labelFontSize)
                a2.set_ylabel(r'Variance (DN$^2$)', fontsize=labelFontSize)
                a2.tick_params(labelsize=11)
                a2.set_xscale('log')
                a2.set_yscale('log')

            if len(meanVecFinal):  # Empty if the whole amp is bad, for example.
                minMeanVecFinal = np.min(meanVecFinal)
                maxMeanVecFinal = np.max(meanVecFinal)
                meanVecFit = np.linspace(minMeanVecFinal, maxMeanVecFinal, 100*len(meanVecFinal))
                minMeanVecOriginal = np.min(meanVecOriginal)
                maxMeanVecOriginal = np.max(meanVecOriginal)
                deltaXlim = maxMeanVecOriginal - minMeanVecOriginal

                a.plot(meanVecFit, ptcFunc(pars, meanVecFit), color='red')
                a.plot(meanVecFinal, pars[0] + pars[1]*meanVecFinal, color='green', linestyle='--')
                a.scatter(meanVecFinal, varVecFinal, c='blue', marker='o', s=markerSize)
                a.scatter(meanVecOutliers, varVecOutliers, c='magenta', marker='s', s=markerSize)
                a.text(0.03, 0.8, stringLegend, transform=a.transAxes, fontsize=legendFontSize)
                a.set_title(amp, fontsize=titleFontSize)
                a.set_xlim([minMeanVecOriginal - 0.2*deltaXlim, maxMeanVecOriginal + 0.2*deltaXlim])

                # Same, but in log-scale
                a2.plot(meanVecFit, ptcFunc(pars, meanVecFit), color='red')
                a2.scatter(meanVecFinal, varVecFinal, c='blue', marker='o', s=markerSize)
                a2.scatter(meanVecOutliers, varVecOutliers, c='magenta', marker='s', s=markerSize)
                a2.text(0.03, 0.8, stringLegend, transform=a2.transAxes, fontsize=legendFontSize)
                a2.set_title(amp, fontsize=titleFontSize)
                a2.set_xlim([minMeanVecOriginal, maxMeanVecOriginal])
            else:
                a.set_title(f"{amp} (BAD)", fontsize=titleFontSize)
                a2.set_title(f"{amp} (BAD)", fontsize=titleFontSize)

        f.suptitle("PTC \n Fit: " + stringTitle, fontsize=20)
        pdfPages.savefig(f)
        f2.suptitle("PTC (log-log)", fontsize=20)
        pdfPages.savefig(f2)

        # Plot mean vs time
        f, ax = plt.subplots(nrows=4, ncols=4, sharex='col', sharey='row', figsize=(13, 10))
        for i, (amp, a) in enumerate(zip(dataset.ampNames, ax.flatten())):
            meanVecFinal = np.array(dataset.rawMeans[amp])[dataset.visitMask[amp]]
            timeVecFinal = np.array(dataset.rawExpTimes[amp])[dataset.visitMask[amp]]

            a.set_xlabel('Time (sec)', fontsize=labelFontSize)
            a.set_ylabel(r'Mean signal ($\mu$, DN)', fontsize=labelFontSize)
            a.tick_params(labelsize=labelFontSize)
            a.set_xscale('linear', fontsize=labelFontSize)
            a.set_yscale('linear', fontsize=labelFontSize)

            if len(meanVecFinal):
                pars, parsErr = dataset.nonLinearity[amp], dataset.nonLinearityError[amp]
                k0, k0Error = pars[0], parsErr[0]
                k1, k1Error = pars[1], parsErr[1]
                k2, k2Error = pars[2], parsErr[2]
                stringLegend = (f"k0: {k0:.4}+/-{k0Error:.2e} DN\n k1: {k1:.4}+/-{k1Error:.2e} DN/t"
                                f"\n k2: {k2:.2e}+/-{k2Error:.2e} DN/t^2 \n")
                a.scatter(timeVecFinal, meanVecFinal)
                a.plot(timeVecFinal, self.funcPolynomial(pars, timeVecFinal), color='red')
                a.text(0.03, 0.75, stringLegend, transform=a.transAxes, fontsize=legendFontSize)
                a.set_title(f"{amp}", fontsize=titleFontSize)
            else:
                a.set_title(f"{amp} (BAD)", fontsize=titleFontSize)

        f.suptitle("Linearity \n Fit: Polynomial (degree: %g)"
                   % (self.config.polynomialFitDegreeNonLinearity),
                   fontsize=supTitleFontSize)
        pdfPages.savefig(f)

        # Plot linearity residual
        f, ax = plt.subplots(nrows=4, ncols=4, sharex='col', sharey='row', figsize=(13, 10))
        for i, (amp, a) in enumerate(zip(dataset.ampNames, ax.flatten())):
            meanVecFinal = np.array(dataset.rawMeans[amp])[dataset.visitMask[amp]]
            linRes = np.array(dataset.nonLinearityResiduals[amp])

            a.set_xlabel(r'Mean signal ($\mu$, DN)', fontsize=labelFontSize)
            a.set_ylabel('LR (%)', fontsize=labelFontSize)
            a.set_xscale('linear', fontsize=labelFontSize)
            a.set_yscale('linear', fontsize=labelFontSize)
            if len(meanVecFinal):
                a.axvline(x=timeVecFinal[self.config.linResidualTimeIndex], color='g', linestyle='--')
                a.scatter(meanVecFinal, linRes)
                a.set_title(f"{amp}", fontsize=titleFontSize)
                a.axhline(y=0, color='k')
                a.tick_params(labelsize=labelFontSize)
            else:
                a.set_title(f"{amp} (BAD)", fontsize=titleFontSize)

        f.suptitle(r"Linearity Residual: $100\times(1 - \mu_{\rm{ref}}/t_{\rm{ref}})/(\mu / t))$" + "\n" +
                   r"$t_{\rm{ref}}$: " + f"{timeVecFinal[2]} s", fontsize=supTitleFontSize)
        pdfPages.savefig(f)

        # Plot fractional non-linearity residual (w.r.t linear part of polynomial fit)
        f, ax = plt.subplots(nrows=4, ncols=4, sharex='col', sharey='row', figsize=(13, 10))
        for i, (amp, a) in enumerate(zip(dataset.ampNames, ax.flatten())):
            meanVecFinal = np.array(dataset.rawMeans[amp])[dataset.visitMask[amp]]
            fracLinRes = np.array(dataset.fractionalNonLinearityResiduals[amp])

            a.axhline(y=0, color='k')
            a.axvline(x=0, color='k', linestyle='-')
            a.set_xlabel(r'Mean signal ($\mu$, DN)', fontsize=labelFontSize)
            a.set_ylabel('Fractional nonlinearity (%)', fontsize=labelFontSize)
            a.tick_params(labelsize=labelFontSize)
            a.set_xscale('linear', fontsize=labelFontSize)
            a.set_yscale('linear', fontsize=labelFontSize)

            if len(meanVecFinal):
                a.scatter(meanVecFinal, fracLinRes, c='g')
                a.set_title(amp, fontsize=titleFontSize)
            else:
                a.set_title(f"{amp} (BAD)", fontsize=titleFontSize)

        f.suptitle(r"Fractional NL residual" + "\n" +
                   r"$100\times \frac{(k_0 + k_1*Time-\mu)}{k_0+k_1*Time}$",
                   fontsize=supTitleFontSize)
        pdfPages.savefig(f)

        return
