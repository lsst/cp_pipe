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

__all__ = ['ddict2dict', 'CovFastFourierTransform', 'getReadNoise', 'ampOffsetGainRatioFixup']


import galsim
import logging
import numpy as np
import itertools
import numpy.polynomial.polynomial as poly

from scipy.interpolate import Akima1DInterpolator
from scipy.optimize import leastsq, minimize
from scipy.stats import median_abs_deviation, norm

from lsst.ip.isr import isrMock
import lsst.afw.image
import lsst.afw.math


def sigmaClipCorrection(nSigClip):
    """Correct measured sigma to account for clipping.

    If we clip our input data and then measure sigma, then the
    measured sigma is smaller than the true value because real
    points beyond the clip threshold have been removed.  This is a
    small (1.5% at nSigClip=3) effect when nSigClip >~ 3, but the
    default parameters for measure crosstalk use nSigClip=2.0.
    This causes the measured sigma to be about 15% smaller than
    real.  This formula corrects the issue, for the symmetric case
    (upper clip threshold equal to lower clip threshold).

    Parameters
    ----------
    nSigClip : `float`
        Number of sigma the measurement was clipped by.

    Returns
    -------
    scaleFactor : `float`
        Scale factor to increase the measured sigma by.
    """
    varFactor = 1.0 - (2 * nSigClip * norm.pdf(nSigClip)) / (norm.cdf(nSigClip) - norm.cdf(-nSigClip))
    return 1.0 / np.sqrt(varFactor)


def calculateWeightedReducedChi2(measured, model, weightsMeasured, nData, nParsModel):
    """Calculate weighted reduced chi2.

    Parameters
    ----------
    measured : `list`
        List with measured data.
    model : `list`
        List with modeled data.
    weightsMeasured : `list`
        List with weights for the measured data.
    nData : `int`
        Number of data points.
    nParsModel : `int`
        Number of parameters in the model.

    Returns
    -------
    redWeightedChi2 : `float`
        Reduced weighted chi2.
    """
    wRes = (measured - model)*weightsMeasured
    return ((wRes*wRes).sum())/(nData-nParsModel)


def makeMockFlats(expTime, gain=1.0, readNoiseElectrons=5, fluxElectrons=1000,
                  randomSeedFlat1=1984, randomSeedFlat2=666, powerLawBfParams=[],
                  expId1=0, expId2=1, ampNames=[]):
    """Create a pair or mock flats with isrMock.

    Parameters
    ----------
    expTime : `float`
        Exposure time of the flats.
    gain : `float`, optional
        Gain, in e/ADU.
    readNoiseElectrons : `float`, optional
        Read noise rms, in electrons.
    fluxElectrons : `float`, optional
        Flux of flats, in electrons per second.
    randomSeedFlat1 : `int`, optional
        Random seed for the normal distrubutions for the mean signal
        and noise (flat1).
    randomSeedFlat2 : `int`, optional
        Random seed for the normal distrubutions for the mean signal
        and noise (flat2).
    powerLawBfParams : `list`, optional
        Parameters for `galsim.cdmodel.PowerLawCD` to simulate the
        brightter-fatter effect.
    expId1 : `int`, optional
        Exposure ID for first flat.
    expId2 : `int`, optional
        Exposure ID for second flat.
    ampNames : `list` [`str`], optional
        Names of amplifiers for filling in header information.

    Returns
    -------
    flatExp1 : `lsst.afw.image.exposure.ExposureF`
        First exposure of flat field pair.
    flatExp2 : `lsst.afw.image.exposure.ExposureF`
        Second exposure of flat field pair.

    Notes
    -----
    The parameters of `galsim.cdmodel.PowerLawCD` are `n, r0, t0, rx,
    tx, r, t, alpha`. For more information about their meaning, see
    the Galsim documentation
    https://galsim-developers.github.io/GalSim/_build/html/_modules/galsim/cdmodel.html  # noqa: W505
    and Gruen+15 (1501.02802).

    Example: galsim.cdmodel.PowerLawCD(8, 1.1e-7, 1.1e-7, 1.0e-8,
                                       1.0e-8, 1.0e-9, 1.0e-9, 2.0)
    """
    flatFlux = fluxElectrons  # e/s
    flatMean = flatFlux*expTime  # e
    readNoise = readNoiseElectrons  # e

    mockImageConfig = isrMock.IsrMock.ConfigClass()

    mockImageConfig.flatDrop = 0.99999
    mockImageConfig.isTrimmed = True

    flatExp1 = isrMock.FlatMock(config=mockImageConfig).run()
    flatExp2 = flatExp1.clone()
    (shapeY, shapeX) = flatExp1.getDimensions()
    flatWidth = np.sqrt(flatMean)

    rng1 = np.random.RandomState(randomSeedFlat1)
    flatData1 = rng1.normal(flatMean, flatWidth, (shapeX, shapeY)) + rng1.normal(0.0, readNoise,
                                                                                 (shapeX, shapeY))
    rng2 = np.random.RandomState(randomSeedFlat2)
    flatData2 = rng2.normal(flatMean, flatWidth, (shapeX, shapeY)) + rng2.normal(0.0, readNoise,
                                                                                 (shapeX, shapeY))
    # Simulate BF with power law model in galsim
    if len(powerLawBfParams):
        if not len(powerLawBfParams) == 8:
            raise RuntimeError("Wrong number of parameters for `galsim.cdmodel.PowerLawCD`. "
                               f"Expected 8; passed {len(powerLawBfParams)}.")
        cd = galsim.cdmodel.PowerLawCD(*powerLawBfParams)
        tempFlatData1 = galsim.Image(flatData1)
        temp2FlatData1 = cd.applyForward(tempFlatData1)

        tempFlatData2 = galsim.Image(flatData2)
        temp2FlatData2 = cd.applyForward(tempFlatData2)

        flatExp1.image.array[:] = temp2FlatData1.array/gain   # ADU
        flatExp2.image.array[:] = temp2FlatData2.array/gain  # ADU
    else:
        flatExp1.image.array[:] = flatData1/gain   # ADU
        flatExp2.image.array[:] = flatData2/gain   # ADU

    visitInfoExp1 = lsst.afw.image.VisitInfo(exposureTime=expTime)
    visitInfoExp2 = lsst.afw.image.VisitInfo(exposureTime=expTime)

    flatExp1.info.id = expId1
    flatExp1.getInfo().setVisitInfo(visitInfoExp1)
    flatExp2.info.id = expId2
    flatExp2.getInfo().setVisitInfo(visitInfoExp2)

    # Set ISR metadata.
    flatExp1.metadata["LSST ISR UNITS"] = "adu"
    flatExp2.metadata["LSST ISR UNITS"] = "adu"
    for ampName in ampNames:
        key = f"LSST ISR OVERSCAN RESIDUAL SERIAL STDEV {ampName}"
        value = readNoiseElectrons / gain

        flatExp1.metadata[key] = value
        flatExp2.metadata[key] = value

        key = f"LSST ISR OVERSCAN SERIAL MEDIAN {ampName}"
        flatExp1.metadata[key] = 25000.0  # adu
        flatExp2.metadata[key] = 25000.0  # adu

        key = f"LSST ISR AMPOFFSET PEDESTAL {ampName}"
        value = 0.0

        flatExp1.metadata[key] = value
        flatExp2.metadata[key] = value

    return flatExp1, flatExp2


def irlsFit(initialParams, dataX, dataY, function, weightsY=None, weightType='Cauchy', scaleResidual=True):
    """Iteratively reweighted least squares fit.

    This uses the `lsst.cp.pipe.utils.fitLeastSq`, but applies weights
    based on the Cauchy distribution by default.  Other weight options
    are implemented.  See e.g. Holland and Welsch, 1977,
    doi:10.1080/03610927708827533

    Parameters
    ----------
    initialParams : `list` [`float`]
        Starting parameters.
    dataX : `numpy.array`, (N,)
        Abscissa data.
    dataY : `numpy.array`, (N,)
        Ordinate data.
    function : callable
        Function to fit.
    weightsY : `numpy.array`, (N,)
        Weights to apply to the data.
    weightType : `str`, optional
        Type of weighting to use.  One of Cauchy, Anderson, bisquare,
        box, Welsch, Huber, logistic, or Fair.
    scaleResidual : `bool`, optional
        If true, the residual is scaled by the sqrt of the Y values.

    Returns
    -------
    polyFit : `list` [`float`]
        Final best fit parameters.
    polyFitErr : `list` [`float`]
        Final errors on fit parameters.
    chiSq : `float`
        Reduced chi squared.
    weightsY : `list` [`float`]
        Final weights used for each point.

    Raises
    ------
    RuntimeError :
        Raised if an unknown weightType string is passed.
    """
    if not weightsY:
        weightsY = np.ones_like(dataX)

    polyFit, polyFitErr, chiSq = fitLeastSq(initialParams, dataX, dataY, function, weightsY=weightsY)
    for iteration in range(10):
        resid = np.abs(dataY - function(polyFit, dataX))
        if scaleResidual:
            resid = resid / np.sqrt(dataY)
        if weightType == 'Cauchy':
            # Use Cauchy weighting.  This is a soft weight.
            # At [2, 3, 5, 10] sigma, weights are [.59, .39, .19, .05].
            Z = resid / 2.385
            weightsY = 1.0 / (1.0 + np.square(Z))
        elif weightType == 'Anderson':
            # Anderson+1972 weighting.  This is a hard weight.
            # At [2, 3, 5, 10] sigma, weights are [.67, .35, 0.0, 0.0].
            Z = resid / (1.339 * np.pi)
            weightsY = np.where(Z < 1.0, np.sinc(Z), 0.0)
        elif weightType == 'bisquare':
            # Beaton and Tukey (1974) biweight.  This is a hard weight.
            # At [2, 3, 5, 10] sigma, weights are [.81, .59, 0.0, 0.0].
            Z = resid / 4.685
            weightsY = np.where(Z < 1.0, 1.0 - np.square(Z), 0.0)
        elif weightType == 'box':
            # Hinich and Talwar (1975).  This is a hard weight.
            # At [2, 3, 5, 10] sigma, weights are [1.0, 0.0, 0.0, 0.0].
            weightsY = np.where(resid < 2.795, 1.0, 0.0)
        elif weightType == 'Welsch':
            # Dennis and Welsch (1976).  This is a hard weight.
            # At [2, 3, 5, 10] sigma, weights are [.64, .36, .06, 1e-5].
            Z = resid / 2.985
            weightsY = np.exp(-1.0 * np.square(Z))
        elif weightType == 'Huber':
            # Huber (1964) weighting.  This is a soft weight.
            # At [2, 3, 5, 10] sigma, weights are [.67, .45, .27, .13].
            Z = resid / 1.345
            weightsY = np.where(Z < 1.0, 1.0, 1 / Z)
        elif weightType == 'logistic':
            # Logistic weighting.  This is a soft weight.
            # At [2, 3, 5, 10] sigma, weights are [.56, .40, .24, .12].
            Z = resid / 1.205
            weightsY = np.tanh(Z) / Z
        elif weightType == 'Fair':
            # Fair (1974) weighting.  This is a soft weight.
            # At [2, 3, 5, 10] sigma, weights are [.41, .32, .22, .12].
            Z = resid / 1.4
            weightsY = (1.0 / (1.0 + (Z)))
        else:
            raise RuntimeError(f"Unknown weighting type: {weightType}")
        polyFit, polyFitErr, chiSq = fitLeastSq(initialParams, dataX, dataY, function, weightsY=weightsY)

    return polyFit, polyFitErr, chiSq, weightsY


def fitLeastSq(initialParams, dataX, dataY, function, weightsY=None):
    """Do a fit and estimate the parameter errors using using
    scipy.optimize.leastq.

    optimize.leastsq returns the fractional covariance matrix. To
    estimate the standard deviation of the fit parameters, multiply
    the entries of this matrix by the unweighted reduced chi squared
    and take the square root of the diagonal elements.

    Parameters
    ----------
    initialParams : `list` [`float`]
        initial values for fit parameters. For ptcFitType=POLYNOMIAL,
        its length determines the degree of the polynomial.
    dataX : `numpy.array`, (N,)
        Data in the abscissa axis.
    dataY : `numpy.array`, (N,)
        Data in the ordinate axis.
    function : callable object (function)
        Function to fit the data with.
    weightsY : `numpy.array`, (N,)
        Weights of the data in the ordinate axis.

    Return
    ------
    pFitSingleLeastSquares : `list` [`float`]
        List with fitted parameters.
    pErrSingleLeastSquares : `list` [`float`]
        List with errors for fitted parameters.

    reducedChiSqSingleLeastSquares : `float`
        Reduced chi squared, unweighted if weightsY is not provided.
    """
    if weightsY is None:
        weightsY = np.ones(len(dataX))

    def errFunc(p, x, y, weightsY=None):
        if weightsY is None:
            weightsY = np.ones(len(x))
        return (function(p, x) - y)*weightsY

    pFit, pCov, infoDict, errMessage, success = leastsq(errFunc, initialParams,
                                                        args=(dataX, dataY, weightsY), full_output=1,
                                                        epsfcn=0.0001)

    if (len(dataY) > len(initialParams)) and pCov is not None:
        reducedChiSq = calculateWeightedReducedChi2(dataY, function(pFit, dataX), weightsY, len(dataY),
                                                    len(initialParams))
        pCov *= reducedChiSq
    else:
        pCov = np.zeros((len(initialParams), len(initialParams)))
        pCov[:, :] = np.nan
        reducedChiSq = np.nan

    errorVec = []
    for i in range(len(pFit)):
        errorVec.append(np.fabs(pCov[i][i])**0.5)

    pFitSingleLeastSquares = pFit
    pErrSingleLeastSquares = np.array(errorVec)

    return pFitSingleLeastSquares, pErrSingleLeastSquares, reducedChiSq


def fitBootstrap(initialParams, dataX, dataY, function, weightsY=None, confidenceSigma=1.):
    """Do a fit using least squares and bootstrap to estimate parameter errors.

    The bootstrap error bars are calculated by fitting 100 random data sets.

    Parameters
    ----------
    initialParams : `list` [`float`]
        initial values for fit parameters. For ptcFitType=POLYNOMIAL,
        its length determines the degree of the polynomial.
    dataX : `numpy.array`, (N,)
        Data in the abscissa axis.
    dataY : `numpy.array`, (N,)
        Data in the ordinate axis.
    function : callable object (function)
        Function to fit the data with.
    weightsY : `numpy.array`, (N,), optional.
        Weights of the data in the ordinate axis.
    confidenceSigma : `float`, optional.
        Number of sigmas that determine confidence interval for the
        bootstrap errors.

    Return
    ------
    pFitBootstrap : `list` [`float`]
        List with fitted parameters.
    pErrBootstrap : `list` [`float`]
        List with errors for fitted parameters.
    reducedChiSqBootstrap : `float`
        Reduced chi squared, unweighted if weightsY is not provided.
    """
    if weightsY is None:
        weightsY = np.ones(len(dataX))

    def errFunc(p, x, y, weightsY):
        if weightsY is None:
            weightsY = np.ones(len(x))
        return (function(p, x) - y)*weightsY

    # Fit first time
    pFit, _ = leastsq(errFunc, initialParams, args=(dataX, dataY, weightsY), full_output=0)

    # Get the stdev of the residuals
    residuals = errFunc(pFit, dataX, dataY, weightsY)
    # 100 random data sets are generated and fitted
    pars = []
    for i in range(100):
        randomDelta = np.random.normal(0., np.fabs(residuals), len(dataY))
        randomDataY = dataY + randomDelta
        randomFit, _ = leastsq(errFunc, initialParams,
                               args=(dataX, randomDataY, weightsY), full_output=0)
        pars.append(randomFit)
    pars = np.array(pars)
    meanPfit = np.mean(pars, 0)

    # confidence interval for parameter estimates
    errPfit = confidenceSigma*np.std(pars, 0)
    pFitBootstrap = meanPfit
    pErrBootstrap = errPfit

    reducedChiSq = calculateWeightedReducedChi2(dataY, function(pFitBootstrap, dataX), weightsY, len(dataY),
                                                len(initialParams))
    return pFitBootstrap, pErrBootstrap, reducedChiSq


def funcPolynomial(pars, x):
    """Polynomial function definition
    Parameters
    ----------
    params : `list`
        Polynomial coefficients. Its length determines the polynomial order.

    x : `numpy.array`, (N,)
        Abscisa array.

    Returns
    -------
    y : `numpy.array`, (N,)
        Ordinate array after evaluating polynomial of order
        len(pars)-1 at `x`.
    """
    return poly.polyval(x, [*pars])


def funcAstier(pars, x):
    """Single brighter-fatter parameter model for PTC; Equation 16 of
    Astier+19.

    Parameters
    ----------
    params : `list`
        Parameters of the model: a00 (brightter-fatter), gain (e/ADU),
        and noise (e^2).
    x : `numpy.array`, (N,)
        Signal mu (ADU).

    Returns
    -------
    y : `numpy.array`, (N,)
        C_00 (variance) in ADU^2.
    """
    a00, gain, noiseSquared = pars
    return 0.5/(a00*gain*gain)*(np.exp(2*a00*x*gain)-1) + noiseSquared/(gain*gain)  # C_00


def arrangeFlatsByExpTime(exposureList, exposureIdList, log=None):
    """Arrange exposures by exposure time.

    Parameters
    ----------
    exposureList : `list` [`lsst.pipe.base.connections.DeferredDatasetRef`]
        Input list of exposure references.
    exposureIdList : `list` [`int`]
        List of exposure ids as obtained by dataId[`exposure`].
    log : `lsst.utils.logging.LsstLogAdapter`, optional
        Log object.

    Returns
    ------
    flatsAtExpTime : `dict` [`float`,
                     `list`[(`lsst.pipe.base.connections.DeferredDatasetRef`,
                              `int`)]]
        Dictionary that groups references to flat-field exposures
        (and their IDs) that have the same exposure time (seconds).
    """
    flatsAtExpTime = {}
    assert len(exposureList) == len(exposureIdList), "Different lengths for exp. list and exp. ID lists"
    for expRef, expId in zip(exposureList, exposureIdList):
        expTime = expRef.get(component='visitInfo').exposureTime
        if not np.isfinite(expTime) and log is not None:
            log.warning("Exposure %d has non-finite exposure time.", expId)
        listAtExpTime = flatsAtExpTime.setdefault(expTime, [])
        listAtExpTime.append((expRef, expId))

    return flatsAtExpTime


def arrangeFlatsByExpFlux(exposureList, exposureIdList, fluxKeyword, log=None):
    """Arrange exposures by exposure flux.

    Parameters
    ----------
    exposureList : `list` [`lsst.pipe.base.connections.DeferredDatasetRef`]
        Input list of exposure references.
    exposureIdList : `list` [`int`]
        List of exposure ids as obtained by dataId[`exposure`].
    fluxKeyword : `str`
        Header keyword that contains the flux per exposure.
    log : `lsst.utils.logging.LsstLogAdapter`, optional
        Log object.

    Returns
    -------
    flatsAtFlux : `dict` [`float`,
                  `list`[(`lsst.pipe.base.connections.DeferredDatasetRef`,
                          `int`)]]
        Dictionary that groups references to flat-field exposures
        (and their IDs) that have the same flux.
    """
    flatsAtExpFlux = {}
    assert len(exposureList) == len(exposureIdList), "Different lengths for exp. list and exp. ID lists"
    for expRef, expId in zip(exposureList, exposureIdList):
        # Get flux from header, assuming it is in the metadata.
        try:
            expFlux = expRef.get().getMetadata()[fluxKeyword]
        except KeyError:
            # If it's missing from the header, continue; it will
            # be caught and rejected when pairing exposures.
            expFlux = None
        if expFlux is None:
            if log is not None:
                log.warning("Exposure %d does not have valid header keyword %s.", expId, fluxKeyword)
            expFlux = np.nan
        listAtExpFlux = flatsAtExpFlux.setdefault(expFlux, [])
        listAtExpFlux.append((expRef, expId))

    return flatsAtExpFlux


def arrangeFlatsByExpId(exposureList, exposureIdList):
    """Arrange exposures by exposure ID.

    There is no guarantee that this will properly group exposures, but
    allows a sequence of flats that have different illumination
    (despite having the same exposure time) to be processed.

    Parameters
    ----------
    exposureList : `list`[`lsst.pipe.base.connections.DeferredDatasetRef`]
        Input list of exposure references.
    exposureIdList : `list`[`int`]
        List of exposure ids as obtained by dataId[`exposure`].

    Returns
    ------
    flatsAtExpId : `dict` [`float`,
                   `list`[(`lsst.pipe.base.connections.DeferredDatasetRef`,
                           `int`)]]
        Dictionary that groups references to flat-field exposures (and their
        IDs) sequentially by their exposure id.

    Notes
    -----

    This algorithm sorts the input exposure references by their exposure
    id, and then assigns each pair of exposure references (exp_j, exp_{j+1})
    to pair k, such that 2*k = j, where j is the python index of one of the
    exposure references (starting from zero).  By checking for the IndexError
    while appending, we can ensure that there will only ever be fully
    populated pairs.
    """
    flatsAtExpId = {}
    assert len(exposureList) == len(exposureIdList), "Different lengths for exp. list and exp. ID lists"
    # Sort exposures by expIds, which are in the second list `exposureIdList`.
    sortedExposures = sorted(zip(exposureList, exposureIdList), key=lambda pair: pair[1])

    for jPair, expTuple in enumerate(sortedExposures):
        if (jPair + 1) % 2:
            kPair = jPair // 2
            listAtExpId = flatsAtExpId.setdefault(kPair, [])
            try:
                listAtExpId.append(expTuple)
                listAtExpId.append(sortedExposures[jPair + 1])
            except IndexError:
                pass

    return flatsAtExpId


def extractCalibDate(calib):
    """Extract common calibration metadata values that will be written to
    output header.

    Parameters
    ----------
    calib : `lsst.afw.image.Exposure` or `lsst.ip.isr.IsrCalib`
        Calibration to pull date information from.

    Returns
    -------
    dateString : `str`
        Calibration creation date string to add to header.
    """
    if hasattr(calib, "getMetadata"):
        if 'CALIB_CREATION_DATE' in calib.getMetadata():
            return " ".join((calib.getMetadata().get("CALIB_CREATION_DATE", "Unknown"),
                             calib.getMetadata().get("CALIB_CREATION_TIME", "Unknown")))
        else:
            return " ".join((calib.getMetadata().get("CALIB_CREATE_DATE", "Unknown"),
                             calib.getMetadata().get("CALIB_CREATE_TIME", "Unknown")))
    else:
        return "Unknown Unknown"


class CovFastFourierTransform:
    """A class to compute (via FFT) the nearby pixels correlation function.

    Implements appendix of Astier+19.

    Parameters
    ----------
    diff : `numpy.array`
        Image where to calculate the covariances (e.g., the difference
        image of two flats).
    w : `numpy.array`
        Weight image (mask): it should consist of 1's (good pixel) and
        0's (bad pixels).
    fftShape : `tuple`
        2d-tuple with the shape of the FFT
    maxRangeCov : `int`
        Maximum range for the covariances.
    """

    def __init__(self, diff, w, fftShape, maxRangeCov):
        # check that the zero padding implied by "fft_shape"
        # is large enough for the required correlation range
        assert fftShape[0] > diff.shape[0]+maxRangeCov+1
        assert fftShape[1] > diff.shape[1]+maxRangeCov+1
        # for some reason related to numpy.fft.rfftn,
        # the second dimension should be even, so
        if fftShape[1]%2 == 1:
            fftShape = (fftShape[0], fftShape[1]+1)
        tIm = np.fft.rfft2(diff*w, fftShape)
        tMask = np.fft.rfft2(w, fftShape)
        # sum of  "squares"
        self.pCov = np.fft.irfft2(tIm*tIm.conjugate())
        # sum of values
        self.pMean = np.fft.irfft2(tIm*tMask.conjugate())
        # number of w!=0 pixels.
        self.pCount = np.fft.irfft2(tMask*tMask.conjugate())

    def cov(self, dx, dy):
        """Covariance for dx,dy averaged with dx,-dy if both non zero.

        Implements appendix of Astier+19.

        Parameters
        ----------
        dx : `int`
           Lag in x
        dy : `int`
           Lag in y

        Returns
        -------
        0.5*(cov1+cov2) : `float`
            Covariance at (dx, dy) lag
        npix1+npix2 : `int`
            Number of pixels used in covariance calculation.

        Raises
        ------
        ValueError if number of pixels for a given lag is 0.
        """
        # compensate rounding errors
        nPix1 = int(round(self.pCount[dy, dx]))
        if nPix1 == 0:
            raise ValueError(f"Could not compute covariance term {dy}, {dx}, as there are no good pixels.")
        cov1 = self.pCov[dy, dx]/nPix1-self.pMean[dy, dx]*self.pMean[-dy, -dx]/(nPix1*nPix1)
        if (dx == 0 or dy == 0):
            return cov1, nPix1
        nPix2 = int(round(self.pCount[-dy, dx]))
        if nPix2 == 0:
            raise ValueError("Could not compute covariance term {dy}, {dx} as there are no good pixels.")
        cov2 = self.pCov[-dy, dx]/nPix2-self.pMean[-dy, dx]*self.pMean[dy, -dx]/(nPix2*nPix2)
        return 0.5*(cov1+cov2), nPix1+nPix2

    def reportCovFastFourierTransform(self, maxRange):
        """Produce a list of tuples with covariances.

        Implements appendix of Astier+19.

        Parameters
        ----------
        maxRange : `int`
            Maximum range of covariances.

        Returns
        -------
        tupleVec : `list`
            List with covariance tuples.
        """
        tupleVec = []
        # (dy,dx) = (0,0) has to be first
        for dy in range(maxRange+1):
            for dx in range(maxRange+1):
                cov, npix = self.cov(dx, dy)
                if (dx == 0 and dy == 0):
                    var = cov
                tupleVec.append((dx, dy, var, cov, npix))
        return tupleVec


def getFitDataFromCovariances(i, j, mu, fullCov, fullCovModel, fullCovSqrtWeights, gain=1.0,
                              divideByMu=False, returnMasked=False):
    """Get measured signal and covariance, cov model, weigths, and mask at
    covariance lag (i, j).

    Parameters
    ----------
    i :  `int`
        Lag for covariance matrix.
    j : `int`
        Lag for covariance matrix.
    mu : `list`
        Mean signal values.
    fullCov : `list` of `numpy.array`
        Measured covariance matrices at each mean signal level in mu.
    fullCovSqrtWeights : `list` of `numpy.array`
        List of square root of measured covariances at each mean
        signal level in mu.
    fullCovModel : `list` of `numpy.array`
        List of modeled covariances at each mean signal level in mu.
    gain : `float`, optional
        Gain, in e-/ADU. If other than 1.0 (default), the returned
        quantities will be in electrons or powers of electrons.
    divideByMu : `bool`, optional
        Divide returned covariance, model, and weights by the mean
        signal mu?
    returnMasked : `bool`, optional
        Use mask (based on weights) in returned arrays (mu,
        covariance, and model)?

    Returns
    -------
    mu : `numpy.array`
        list of signal values at (i, j).
    covariance : `numpy.array`
        Covariance at (i, j) at each mean signal mu value (fullCov[:, i, j]).
    covarianceModel : `numpy.array`
        Covariance model at (i, j).
    weights : `numpy.array`
        Weights at (i, j).
    maskFromWeights : `numpy.array`, optional
        Boolean mask of the covariance at (i,j), where the weights
        differ from 0.
    """
    mu = np.array(mu)
    fullCov = np.array(fullCov)
    fullCovModel = np.array(fullCovModel)
    fullCovSqrtWeights = np.array(fullCovSqrtWeights)
    covariance = fullCov[:, i, j]*(gain**2)
    covarianceModel = fullCovModel[:, i, j]*(gain**2)
    weights = fullCovSqrtWeights[:, i, j]/(gain**2)

    maskFromWeights = weights != 0
    if returnMasked:
        weights = weights[maskFromWeights]
        covarianceModel = covarianceModel[maskFromWeights]
        mu = mu[maskFromWeights]
        covariance = covariance[maskFromWeights]

    if divideByMu:
        covariance /= mu
        covarianceModel /= mu
        weights *= mu
    return mu, covariance, covarianceModel, weights, maskFromWeights


def symmetrize(inputArray):
    """ Copy array over 4 quadrants prior to convolution.

    Parameters
    ----------
    inputarray : `numpy.array`
        Input array to symmetrize.

    Returns
    -------
    aSym : `numpy.array`
        Symmetrized array.
    """
    targetShape = list(inputArray.shape)
    r1, r2 = inputArray.shape[-1], inputArray.shape[-2]
    targetShape[-1] = 2*r1-1
    targetShape[-2] = 2*r2-1
    aSym = np.ndarray(tuple(targetShape))
    aSym[..., r2-1:, r1-1:] = inputArray
    aSym[..., r2-1:, r1-1::-1] = inputArray
    aSym[..., r2-1::-1, r1-1::-1] = inputArray
    aSym[..., r2-1::-1, r1-1:] = inputArray

    return aSym


def ddict2dict(d):
    """Convert nested default dictionaries to regular dictionaries.

    This is needed to prevent yaml persistence issues.

    Parameters
    ----------
    d : `defaultdict`
        A possibly nested set of `defaultdict`.

    Returns
    -------
    dict : `dict`
        A possibly nested set of `dict`.
    """
    for k, v in d.items():
        if isinstance(v, dict):
            d[k] = ddict2dict(v)
    return dict(d)


class Pol2D:
    """2D Polynomial Regression.

    Parameters
    ----------
    x : numpy.ndarray
        Input array for the x-coordinate.
    y : numpy.ndarray
        Input array for the y-coordinate.
    z : numpy.ndarray
        Input array for the dependent variable.
    order : int
        Order of the polynomial.
    w : numpy.ndarray, optional
        Weight array for weighted regression. Default is None.

    Notes
    -----
    Ported from by https://gitlab.in2p3.fr/astier/bfptc P. Astier.

    Example:
        >>> x = np.array([1, 2, 3])
        >>> y = np.array([4, 5, 6])
        >>> z = np.array([7, 8, 9])
        >>> order = 2
        >>> poly_reg = Pol2D(x, y, z, order)
        >>> result = poly_reg.eval(2.5, 5.5)
    """
    def __init__(self, x, y, z, order, w=None):
        """
        orderx : `int`
            Effective order in the x-direction.
        ordery : `int`
            Effective order in the y-direction.
        coeff : `numpy.ndarray`
            Coefficients of the polynomial regression.
        """
        self.orderx = min(order, x.shape[0] - 1)
        self.ordery = min(order, x.shape[1] - 1)
        G = self.monomials(x.ravel(), y.ravel())
        if w is None:
            self.coeff, _, rank, _ = np.linalg.lstsq(G, z.ravel(), rcond=None)
        else:
            self.coeff, _, rank, _ = np.linalg.lstsq((w.ravel() * G.T).T, z.ravel() * w.ravel(), rcond=None)

    def monomials(self, x, y):
        """
        Generate the monomials matrix for the given x and y.

        Parameters
        ----------
        x : numpy.ndarray
            Input array for the x-coordinate.
        y : numpy.ndarray
            Input array for the y-coordinate.

        Returns
        -------
        G : numpy.ndarray
            Monomials matrix.
        """
        ncols = (self.orderx + 1) * (self.ordery + 1)
        G = np.zeros(x.shape + (ncols,))
        ij = itertools.product(range(self.orderx + 1), range(self.ordery + 1))
        for k, (i, j) in enumerate(ij):
            G[..., k] = x**i * y**j
        return G

    def eval(self, x, y):
        """
        Evaluate the polynomial at the given x and y coordinates.

        Parameters
        ----------
        x : `float`
            x-coordinate for evaluation.
        y : `float`
            y-coordinate for evaluation.

        Returns
        -------
        result : `float`
            Result of the polynomial evaluation.
        """
        G = self.monomials(x, y)
        return np.dot(G, self.coeff)


class AstierSplineLinearityFitter:
    """Class to fit the Astier spline linearity model.

    This is a spline fit with photodiode data based on a model
    from Pierre Astier, referenced in June 2023 from
    https://me.lsst.eu/astier/bot/7224D/model_nonlin.py

    This model fits a spline with (optional) nuisance parameters
    to allow for different linearity coefficients with different
    photodiode settings.  The minimization is a least-squares
    fit with the residual of
    Sum[(S(mu_i) + mu_i)/(k_j * D_i - O) - 1]**2, where S(mu_i)
    is an Akima Spline function of mu_i, the observed flat-pair
    mean; D_j is the photo-diode measurement corresponding to
    that flat-pair; and k_j is a constant of proportionality
    which is over index j as it is allowed to
    be different based on different photodiode settings (e.g.
    CCOBCURR); and O is a constant offset to allow for light
    leaks (and is only fit if fit_offset=True). In
    addition, if config.doSplineFitTemperature is True then
    the fit will adjust mu such that
    mu = mu_input*(1 + alpha*(temperature_scaled))
    and temperature_scaled = T - T_ref. Finally, if
    config.doSplineFitTemporal is True then the fit will
    further adjust mu such that
    mu = mu_input*(1 + beta*(mjd_scaled))
    and mjd_scaled = mjd - mjd_ref.

    The fit has additional constraints to ensure that the spline
    goes through the (0, 0) point, as well as a normalization
    condition so that the average of the spline over the full
    range is 0. The normalization ensures that the spline only
    fits deviations from linearity, rather than the linear
    function itself which is degenerate with the gain.

    Parameters
    ----------
    nodes : `np.ndarray` (N,)
        Array of spline node locations.
    grouping_values : `np.ndarray` (M,)
        Array of values to group values for different proportionality
        constants (e.g. CCOBCURR).
    pd : `np.ndarray` (M,)
        Array of photodiode measurements.
    mu : `np.ndarray` (M,)
        Array of flat mean values.
    mask : `np.ndarray` (M,), optional
        Input mask (True is good point, False is bad point).
    log : `logging.logger`, optional
        Logger object to use for logging.
    fit_offset : `bool`, optional
        Fit a constant offset to allow for light leaks?
    fit_weights : `bool`, optional
        Fit for the weight parameters?
    weight_pars_start : `list` [ `float` ]
        Iterable of 2 weight parameters for weighed fit. These will
        be used as input if the weight parameters are not fit.
    fit_temperature : `bool`, optional
        Fit for temperature scaling?
    temperature_scaled : `np.ndarray` (M,), optional
        Input scaled temperature values (T - T_ref).
    max_signal_nearly_linear : `float`, optional
        Maximum signal that we are confident the input is nearly
        linear. This is used both for regularization, and for
        fitting the raw slope. Usually set to the ptc turnoff,
        above which we allow the spline to significantly deviate
        and do not demand the deviation to average to zero.
    fit_temporal : `bool`, optional
        Fit for temporal scaling?
    mjd_scaled : `np.ndarray` (M,), optional
        Input scaled mjd values (mjd - mjd_ref).
    """
    def __init__(
        self,
        nodes,
        grouping_values,
        pd,
        mu,
        mask=None,
        log=None,
        fit_offset=True,
        fit_weights=False,
        weight_pars_start=[1.0, 0.0],
        fit_temperature=False,
        temperature_scaled=None,
        max_signal_nearly_linear=None,
        fit_temporal=False,
        mjd_scaled=None,
    ):
        self._pd = pd
        self._mu = mu
        self._grouping_values = grouping_values
        self.log = log if log else logging.getLogger(__name__)
        self._fit_offset = fit_offset
        self._fit_weights = fit_weights
        self._weight_pars_start = weight_pars_start
        self._fit_temperature = fit_temperature
        self._fit_temporal = fit_temporal

        self._nodes = nodes
        if nodes[0] != 0.0:
            raise ValueError("First node must be 0.0")
        if not np.all(np.diff(nodes) > 0):
            raise ValueError("Nodes must be sorted with no repeats.")

        # Find the group indices.
        u_group_values = np.unique(self._grouping_values)
        self.ngroup = len(u_group_values)

        self.group_indices = []
        for i in range(self.ngroup):
            self.group_indices.append(np.where(self._grouping_values == u_group_values[i])[0])

        # Weight values.  Outliers will be set to 0.
        if mask is None:
            _mask = np.ones(len(mu), dtype=np.bool_)
        else:
            _mask = mask
        self._w = self.compute_weights(self._weight_pars_start, self._mu, _mask)

        if temperature_scaled is None:
            temperature_scaled = np.zeros(len(self._mu))
        else:
            if len(np.atleast_1d(temperature_scaled)) != len(self._mu):
                raise ValueError("temperature_scaled must be the same length as input mu.")
        self._temperature_scaled = temperature_scaled

        if mjd_scaled is None:
            mjd_scaled = np.zeros(len(self._mu))
        else:
            if len(np.atleast_1d(mjd_scaled)) != len(self._mu):
                raise ValueError("mjd_scaled must be the same length as input mu.")
        self._mjd_scaled = mjd_scaled

        # Values to regularize spline fit.
        if max_signal_nearly_linear is None:
            max_signal_nearly_linear = self._mu[self.mask].max()
        self._max_signal_nearly_linear = max_signal_nearly_linear
        self._x_regularize = np.linspace(0.0, self._max_signal_nearly_linear, 100)

        # Set up the indices for the fit parameters.
        self.par_indices = {
            "values": np.arange(len(self._nodes)),
            "groups": len(self._nodes) + np.arange(self.ngroup),
            "offset": np.zeros(0, dtype=np.int64),
            "weight_pars": np.zeros(0, dtype=np.int64),
            "temperature_coeff": np.zeros(0, dtype=np.int64),
            "temporal_coeff": np.zeros(0, dtype=np.int64),
        }
        if self._fit_offset:
            self.par_indices["offset"] = np.arange(1) + (
                len(self.par_indices["values"])
                + len(self.par_indices["groups"])
            )
        if self._fit_weights:
            self.par_indices["weight_pars"] = np.arange(2) + (
                len(self.par_indices["values"])
                + len(self.par_indices["groups"])
                + len(self.par_indices["offset"])
            )
        if self._fit_temperature:
            self.par_indices["temperature_coeff"] = np.arange(1) + (
                len(self.par_indices["values"])
                + len(self.par_indices["groups"])
                + len(self.par_indices["offset"])
                + len(self.par_indices["weight_pars"])
            )
        if self._fit_temporal:
            self.par_indices["temporal_coeff"] = np.arange(1) + (
                len(self.par_indices["values"])
                + len(self.par_indices["groups"])
                + len(self.par_indices["offset"])
                + len(self.par_indices["weight_pars"])
                + len(self.par_indices["temperature_coeff"])
            )

    @staticmethod
    def compute_weights(weight_pars, mu, mask):
        w = 1./np.sqrt(weight_pars[0]**2. + weight_pars[1]**2./mu)
        w[~mask] = 0.0

        return w

    def estimate_p0(self):
        """Estimate initial fit parameters.

        Returns
        -------
        p0 : `np.ndarray`
            Parameter array, with spline values (one for each node) followed
            by proportionality constants (one for each group); one extra
            for the offset O (if fit_offset was set to True); two extra
            for the weights (if fit_weights was set to True); one
            extra for the temperature coefficient (if fit_temperature was
            set to True); and one extra for the temporal coefficient (if
            fit_temporal was set to True).
        """
        npt = (len(self.par_indices["values"])
               + len(self.par_indices["groups"])
               + len(self.par_indices["offset"])
               + len(self.par_indices["weight_pars"])
               + len(self.par_indices["temperature_coeff"])
               + len(self.par_indices["temporal_coeff"]))
        p0 = np.zeros(npt)

        # Do a simple linear fit for each group.
        for i, indices in enumerate(self.group_indices):
            mask = self.mask[indices]
            mu = self._mu[indices][mask]
            pd = self._pd[indices][mask]
            to_fit = (mu < self._max_signal_nearly_linear)
            linfit = np.polyfit(pd[to_fit], mu[to_fit], 1)
            p0[self.par_indices["groups"][i]] = linfit[0]

        # Look at the residuals...
        ratio_model = self.compute_ratio_model(
            self._nodes,
            self.group_indices,
            self.par_indices,
            p0,
            self._pd,
            self._mu,
            self._temperature_scaled,
            self._mjd_scaled,
        )
        # ...and adjust the linear parameters accordingly.
        p0[self.par_indices["groups"]] *= np.median(ratio_model[self.mask])

        # Re-compute the residuals.
        ratio_model2 = self.compute_ratio_model(
            self._nodes,
            self.group_indices,
            self.par_indices,
            p0,
            self._pd,
            self._mu,
            self._temperature_scaled,
            self._mjd_scaled,
        )

        # And compute a first guess of the spline nodes.
        bins = np.searchsorted(self._nodes, self._mu[self.mask])
        tot_arr = np.zeros(len(self._nodes))
        n_arr = np.zeros(len(self._nodes), dtype=int)
        np.add.at(tot_arr, bins, ratio_model2[self.mask])
        np.add.at(n_arr, bins, 1)

        ratio = np.ones(len(self._nodes))
        ratio[n_arr > 0] = tot_arr[n_arr > 0]/n_arr[n_arr > 0]
        ratio[0] = 1.0
        p0[self.par_indices["values"]] = (ratio - 1) * self._nodes

        if self._fit_offset:
            p0[self.par_indices["offset"]] = 0.0

        if self._fit_weights:
            p0[self.par_indices["weight_pars"]] = self._weight_pars_start

        return p0

    @staticmethod
    def compute_ratio_model(
        nodes,
        group_indices,
        par_indices,
        pars,
        pd,
        mu,
        temperature_scaled,
        mjd_scaled,
        return_spline=False,
    ):
        """Compute the ratio model values.

        Parameters
        ----------
        nodes : `np.ndarray` (M,)
            Array of node positions.
        group_indices : `list` [`np.ndarray`]
            List of group indices, one array for each group.
        par_indices : `dict`
            Dictionary showing which indices in the pars belong to
            each set of fit values.
        pars : `np.ndarray`
            Parameter array, with spline values (one for each node) followed
            by proportionality constants (one for each group); one extra
            for the offset O (if fit_offset was set to True); two extra
            for the weights (if fit_weights was set to True); and one
            extra for the temperature coefficient (if fit_temperature was
            set to True).
        pd : `np.ndarray` (N,)
            Array of photodiode measurements.
        mu : `np.ndarray` (N,)
            Array of flat means.
        temperature_scaled : `np.ndarray` (N,)
            Array of scaled temperature values.
        mjd_scaled : `np.ndarray` (N,)
            Array of scaled mjd values.
        return_spline : `bool`, optional
            Return the spline interpolation as well as the model ratios?

        Returns
        -------
        ratio_models : `np.ndarray` (N,)
            Model ratio, (mu_i - S(mu_i) - O)/(k_j * D_i)
        spl : `scipy.interpolate.Akima1DInterpolator`
            Spline interpolator (returned if return_spline=True).
        """
        spl = Akima1DInterpolator(nodes, pars[par_indices["values"]], method="akima")

        # Check if we want to do just the left or both with temp scale.
        if len(par_indices["temperature_coeff"]) == 1:
            mu_corr = mu*(1. + pars[par_indices["temperature_coeff"]]*temperature_scaled)
        else:
            mu_corr = mu

        if len(par_indices["temporal_coeff"]) == 1:
            mu_corr = mu_corr*(1. + pars[par_indices["temporal_coeff"]]*mjd_scaled)

        numerator = mu_corr - spl(np.clip(mu_corr, nodes[0], nodes[-1]))
        if len(par_indices["offset"]) == 1:
            numerator -= pars[par_indices["offset"][0]]
        denominator = pd.copy()
        ngroup = len(group_indices)
        kj = pars[par_indices["groups"]]
        for j in range(ngroup):
            denominator[group_indices[j]] *= kj[j]

        if return_spline:
            return numerator / denominator, spl
        else:
            return numerator / denominator

    def fit(self, p0, min_iter=3, max_iter=20, max_rejection_per_iteration=5, n_sigma_clip=5.0):
        """
        Perform iterative fit for linear + spline model with offsets.

        Parameters
        ----------
        p0 : `np.ndarray`
            Initial parameter array, with spline values (one for each node)
            followed by proportionality constants (one for each group); one
            extra for the offset O (if fit_offset was set to True); two extra
            for the weights (if fit_weights was set to True); one
            extra for the temperature coefficient (if fit_temperature was
            set to True); and one extra for the temporal coefficient (if
            fit_temporal was set to True).
        min_iter : `int`, optional
            Minimum number of fit iterations.
        max_iter : `int`, optional
            Maximum number of fit iterations.
        max_rejection_per_iteration : `int`, optional
            Maximum number of points to reject per iteration.
        n_sigma_clip : `float`, optional
            Number of sigma to do clipping in each iteration.
        """
        init_params = p0
        for k in range(max_iter):
            params, cov_params, _, msg, ierr = leastsq(
                self,
                init_params,
                full_output=True,
                ftol=1e-5,
                maxfev=12000,
            )
            init_params = params.copy()

            # We need to cut off the constraints at the end (there are more
            # residuals than data points.)
            res = self(params)[: len(self._w)]
            std_res = median_abs_deviation(res[self.good_points], scale="normal")
            sample = len(self.good_points)

            # We don't want to reject too many outliers at once.
            if sample > max_rejection_per_iteration:
                sres = np.sort(np.abs(res))
                cut = max(sres[-max_rejection_per_iteration], std_res*n_sigma_clip)
            else:
                cut = std_res*n_sigma_clip

            outliers = np.abs(res) > cut
            self._w[outliers] = 0
            if outliers.sum() != 0:
                self.log.info(
                    "After iteration %d there are %d outliers (of %d).",
                    k,
                    outliers.sum(),
                    sample,
                )
            elif k >= min_iter:
                self.log.info("After iteration %d there are no more outliers.", k)
                break

        return params

    @property
    def mask(self):
        return (self._w > 0)

    @property
    def good_points(self):
        return self.mask.nonzero()[0]

    def compute_chisq_dof(self, pars):
        """Compute the chi-squared per degree of freedom for a set of pars.

        Parameters
        ----------
        pars : `np.ndarray`
            Parameter array.

        Returns
        -------
        chisq_dof : `float`
            Chi-squared per degree of freedom.
        """
        resids = self(pars)[0: len(self.mask)]
        chisq = np.sum(resids[self.mask]**2.)
        dof = self.mask.sum() - self.ngroup
        if self._fit_temporal:
            dof -= 1
        if self._fit_temperature:
            dof -= 1
        if self._fit_offset:
            dof -= 1
        if self._fit_weights:
            dof -= 2

        return chisq/dof

    def __call__(self, pars):

        ratio_model, spl = self.compute_ratio_model(
            self._nodes,
            self.group_indices,
            self.par_indices,
            pars,
            self._pd,
            self._mu,
            self._temperature_scaled,
            self._mjd_scaled,
            return_spline=True,
        )

        _mask = self.mask
        # Update the weights if we are fitting them.
        if self._fit_weights:
            self._w = self.compute_weights(pars[self.par_indices["weight_pars"]], self._mu, _mask)
        resid = self._w*(ratio_model - 1.0)

        # Ensure masked points have 0 residual.
        resid[~_mask] = 0.0

        constraint = [1e3 * np.mean(spl(np.clip(self._x_regularize, self._nodes[0], self._nodes[-1])))]
        # 0 should transform to 0
        constraint.append(spl(0)*1e10)
        # Use a Jeffreys prior on the weight if we are fitting it.
        if self._fit_weights:
            # This factor ensures that log(fact * w) is negative.
            fact = 1e-3 / self._w.max()
            # We only add non-zero weights to the constraint array.
            log_w = np.sqrt(-2.*np.log(fact*self._w[self._w > 0]))
            constraint = np.hstack([constraint, log_w])

        # Don't let it get to >5% correction.
        values = pars[self.par_indices["values"]]
        if np.abs(values[-1])/self._nodes[-1] > 0.25:
            extra_constraint = 1e10
        else:
            extra_constraint = 0

        return np.hstack([resid, constraint, extra_constraint])


def getReadNoise(exposure, ampName, taskMetadata=None, log=None):
    """Gets readout noise for an amp from ISR metadata.

    If possible, this attempts to get the now-standard headers
    added to the exposure itself.  If not found there, the ISR
    TaskMetadata is searched.  If neither of these has the value,
    warn and set the read noise to NaN.

    Parameters
    ----------
    exposure : `lsst.afw.image.Exposure`
        Exposure to check for read noise first.
    ampName : `str`
        Amplifier name.
    taskMetadata : `lsst.pipe.base.TaskMetadata`, optional
        List of exposures metadata from ISR for this exposure.
    log : `logging.logger`, optional
        Log for messages.

    Returns
    -------
    readNoise : `float`
        The read noise for this set of exposure/amplifier.
    """
    exposureMetadata = exposure.getMetadata()

    # Try from the exposure first.
    expectedKey = f"LSST ISR OVERSCAN RESIDUAL SERIAL STDEV {ampName}"
    if expectedKey in exposureMetadata:
        return exposureMetadata[expectedKey]

    # If not, try getting it from the task metadata.
    if taskMetadata:
        expectedKey = f"RESIDUAL STDEV {ampName}"
        if "isr" in taskMetadata:
            if expectedKey in taskMetadata["isr"]:
                return taskMetadata["isr"][expectedKey]

    log = log if log else logging.getLogger(__name__)
    log.warning("Median readout noise from ISR metadata for amp %s "
                "could not be found.", ampName)
    return np.nan


def ampOffsetGainRatioFixup(ptc, minAdu, maxAdu, log=None):
    """Apply gain ratio fixup using amp offsets.

    Parameters
    ----------
    ptc : `lsst.ip.isr.PhotonTransferCurveDataset`
        Input PTC. Will be modified in place.
    minAdu : `float`
        Minimum number of ADU in mean of amplifier to use for
        computing gain ratio fixup.
    maxAdu : `float`
        Maximum number of ADU in mean of amplifier to use for
        computing gain ratio fixup.
    log : `lsst.utils.logging.LsstLogAdapter`, optional
        Log object.
    """
    # We need to find the reference amplifier.
    # Find an amp near the middle to use as a pivot.
    if log is None:
        log = logging.getLogger(__name__)

    # We first check for which amps have gain measurements
    # (fully bad amps are filled with NaN for gain.)
    gainArray = np.zeros(len(ptc.ampNames))
    for i, ampName in enumerate(ptc.ampNames):
        gainArray[i] = ptc.gain[ampName]
    good, = np.where(np.isfinite(gainArray))

    if len(good) > 1:
        # This only works with more than 1 good amp.

        # We sort the gains and take the one that is closest
        # to the median to use as the reference amplifier for
        # gain ratios.
        st = np.argsort(gainArray[good])
        midAmp = good[st[int(0.5*len(good))]]
        midAmpName = ptc.ampNames[midAmp]

        log.info("Using amplifier %s as the reference for doLinearityGainRatioFixup.", midAmpName)

        # First pass, we need to compute the corrections.
        corrections = {}
        for ampName in ptc.ampNames:
            if not np.isfinite(ptc.gain[ampName]) or ampName == midAmpName:
                continue

            ratioPtc = ptc.gain[ampName] / ptc.gain[midAmpName]

            deltas = ptc.ampOffsets[ampName] - ptc.ampOffsets[midAmpName]
            use = (
                (ptc.expIdMask[ampName])
                & (np.isfinite(deltas))
                & (ptc.finalMeans[ampName] >= minAdu)
                & (ptc.finalMeans[ampName] <= maxAdu)
                & (np.isfinite(ptc.finalMeans[midAmpName]))
                & (ptc.expIdMask[midAmpName])
            )
            if use.sum() < 3:
                log.warning("Not enough good amp offset measurements to fix up amp %s "
                            "gains from amp ratios.", ampName)
                continue

            ratios = 1. / (deltas / ptc.finalMeans[midAmpName] + 1.0)
            ratio = np.median(ratios[use])
            corrections[ampName] = ratio / ratioPtc

        # For the final correction, we need to make sure that the
        # reference amplifier is included. By definition, it has a
        # correction factor of 1.0 before any final fix.
        corrections[midAmpName] = 1.0

        # Adjust the median correction to be 1.0 so we do not
        # change the gain of the detector on average.
        # This is needed in case the reference amplifier is
        # skewed in terms of offsets even though it has the median
        # gain.
        medCorrection = np.median([corrections[key] for key in corrections])

        for ampName in ptc.ampNames:
            if ampName not in corrections:
                continue

            correction = corrections[ampName] / medCorrection
            newGain = ptc.gain[ampName] * correction
            log.info(
                "Adjusting gain from amplifier %s by factor of %.5f (from %.5f to %.5f)",
                ampName,
                correction,
                ptc.gain[ampName],
                newGain,
            )
            # Copying the value should not be necessary, but we record
            # it just in case.
            ptc.gainUnadjusted[ampName] = ptc.gain[ampName]
            ptc.gain[ampName] = newGain
    else:
        log.warning("Cannot apply ampOffsetGainRatioFixup with fewer than 2 good amplifiers.")


class FlatGradientFitter:
    """Class to fit various large-scale flat-field gradients.

    This fitter will take arrays of x/y/value and (by default) fit a radial
    gradient, using a spline function at the specified nodes. The fitter
    will also fit out a nuisance parameter for the ratio of the ITL/E2V
    throughput (though in general this could work for any such focal plane).
    The focal plane origin is set by fp_centroid_x and fp_centroid_y which
    is the center of the radial gradient, though it may be modified if
    fit_centroid is True. In addition, the fitter may fit a linear gradient
    in x/y if fit_gradient is True. The "pivot" of the gradient is at
    fp_centroid_x/fp_centroid_y. Finally, the fitter may fit an additional
    gradient in the outer region where the focal plane radius is greater
    than outer_gradient_cut. This is controlled by fit_outer_gradient.
    This outer gradient is applied "on top of" the primary gradient.

    Parameters
    ----------
    nodes : `np.ndarray`
        Array of spline nodes for radial spline.
    x : `np.ndarray`
        Array of x values for points to fit (mm).
    y : `np.ndarray`
        Array of y values for points to fit (mm).
    value : `np.ndarray`
        Array of values describing the flat field to fit the gradient.
    itl_indices : `np.ndarray`
        Array of indices corresponding to the ITL detectors.
    constrain_zero : `bool`, optional
        Constrain the outermost radial spline value to 0?
    fit_centroid : `bool`, optional
        Fit an additional centroid offset?
    fit_gradient : `bool`, optional
        Fit an additional plane gradient?
    fit_outer_gradient : `bool`, optional
        Fit an additional plane gradient at large focal plane radii?
    outer_gradient_radius : `float`, optional
        Radius (mm) at which the outer gradient is valid.
    fp_centroid_x : `float`, optional
        Focal plane centroid x (mm).
    fp_centroid_y : `float`, optional
        Focal plane centroid y (mm).
    """
    def __init__(
        self,
        nodes,
        x,
        y,
        value,
        itl_indices,
        constrain_zero=True,
        fit_centroid=False,
        fit_gradient=False,
        fit_outer_gradient=False,
        outer_gradient_radius=325.0,
        fp_centroid_x=0.0,
        fp_centroid_y=0.0
    ):
        self._nodes = nodes
        self._x = x
        self._y = y
        self._value = value
        self._itl_indices = itl_indices
        self._fp_centroid_x = fp_centroid_x
        self._fp_centroid_y = fp_centroid_y
        self._outer_gradient_radius = outer_gradient_radius

        self._constrain_zero = constrain_zero

        self._fit_centroid = fit_centroid
        self._fit_gradient = fit_gradient
        self._fit_outer_gradient = fit_outer_gradient

        self.indices = {"spline": np.arange(len(nodes))}
        npar = len(nodes)

        self._fit_itl_ratio = False
        if len(itl_indices) > 0:
            self._fit_itl_ratio = True
            self.indices["itl_ratio"] = npar
            npar += 1

        radius = np.sqrt((self._x - self._fp_centroid_x)**2. + (self._y - self._fp_centroid_y)**2.)

        if fit_centroid:
            self.indices["centroid_delta"] = np.arange(2) + npar
            npar += 2
        else:
            self._radius = radius

        if fit_gradient:
            self.indices["gradient"] = np.arange(2) + npar
            npar += 2

        if fit_outer_gradient:
            self.indices["outer_gradient"] = np.arange(2) + npar
            npar += 2

        self._npar = npar

        self._bounds = [(None, None)]*npar

        if self._constrain_zero:
            self._bounds[self.indices["spline"][-1]] = (0.0, 0.0)

    def compute_p0(self):
        """Compute initial guess for fit parameters.

        Returns
        -------
        pars : `np.ndarray`
            Array of first guess fit parameters.
        """
        pars = np.zeros(self._npar)

        # Initial spline values
        radius = np.sqrt((self._x - self._fp_centroid_x)**2. + (self._y - self._fp_centroid_y)**2.)
        for i, index in enumerate(self.indices["spline"]):
            if i == 0:
                low = self._nodes[i]
            else:
                low = (self._nodes[i - 1] + self._nodes[i])/2.
            if i == (len(self._nodes) - 1):
                high = self._nodes[i]
            else:
                high = (self._nodes[i] + self._nodes[i + 1])/2.
            u = ((radius > low) & (radius < high))
            if u.sum() == 0:
                pars[index] = 0.0
            else:
                pars[index] = np.median(self._value[u])

        if self._constrain_zero:
            pars[self.indices["spline"][-1]] = 0.0

        spl = Akima1DInterpolator(self._nodes, pars[self.indices["spline"]], method="akima")
        model = spl(radius)
        resid_ratio = self._value / model

        if self._fit_itl_ratio:
            e2v_indices = np.delete(np.arange(len(self._value)), self._itl_indices)

            itl_inner = radius[self._itl_indices] < 0.8*np.max(radius)
            e2v_inner = radius[e2v_indices] < 0.8*np.max(radius)

            itl_median = np.nanmedian(resid_ratio[self._itl_indices][itl_inner])
            e2v_median = np.nanmedian(resid_ratio[e2v_indices][e2v_inner])

            pars[self.indices["itl_ratio"]] = itl_median / e2v_median

        if self._fit_centroid:
            pars[self.indices["centroid_delta"]] = [0.0, 0.0]

        if self._fit_gradient:
            resid_ratio = self._value / model

            ok = (np.isfinite(resid_ratio) & (model > 0.5))

            fit = np.polyfit(self._x[ok], resid_ratio[ok], 1)
            pars[self.indices["gradient"][0]] = fit[0]

            fit = np.polyfit(self._y[ok], resid_ratio[ok], 1)
            pars[self.indices["gradient"][1]] = fit[0]

        if self._fit_outer_gradient:
            pars[self.indices["outer_gradient"]] = [0.0, 0.0]

        return pars

    def fit(self, p0, fit_eps=1e-8, fit_gtol=1e-10):
        """Do a non-linear minimization to fit the parameters.

        Parameters
        ----------
        p0 : `np.ndarray`
            Array of initial parameter estimates.
        fit_eps : `float`, optional
            Value of ``eps`` to send to the scipy minimizer.
        fit_gtol : `float`, optional
            Value of ``gtol`` to send to the scipy minimizer.

        Returns
        -------
        pars : `np.ndarray`
            Array of parameters. Use ``fitter.indices`` for the
            dictionary to map parameters to subsets.
        """
        res = minimize(
            self,
            p0,
            method="L-BFGS-B",
            jac=False,
            bounds=self._bounds,
            options={
                "maxfun": 10000,
                "maxiter": 10000,
                "maxcor": 20,
                "eps": fit_eps,
                "gtol": fit_gtol,
            },
            callback=None,
        )
        pars = res.x

        return pars

    def compute_model(self, pars):
        """Compute the model given a set of parameters.

        Parameters
        ----------
        pars : `np.ndarray`
            Parameter array to compute model.

        Returns
        -------
        model_array : `np.ndarray`
            Array of model parameters at the input x/y.
        """
        spl = Akima1DInterpolator(self._nodes, pars[self.indices["spline"]], method="akima")
        if self._fit_centroid:
            centroid_delta = pars[self.indices["centroid_delta"]]
            centroid_x = self._fp_centroid_x + centroid_delta[0]
            centroid_y = self._fp_centroid_y + centroid_delta[1]
            radius = np.sqrt((self._x - centroid_x)**2. + (self._y - centroid_y)**2.)
        else:
            radius = self._radius

        model = spl(np.clip(radius, self._nodes[0], self._nodes[-1]))
        if self._fit_itl_ratio:
            model[self._itl_indices] *= pars[self.indices["itl_ratio"]]

        if self._fit_gradient:
            a, b = pars[self.indices["gradient"]]
            gradient = 1 + a*(self._x - self._fp_centroid_x) + b*(self._y - self._fp_centroid_y)
            model /= gradient

        if self._fit_outer_gradient:
            # The outer gradient is applied to the outer regions
            # in addition to the primary gradient.
            ao, bo = pars[self.indices["outer_gradient"]]
            gradiento = 1 + ao*(self._x - self._fp_centroid_x) + bo*(self._y - self._fp_centroid_y)
            outer = radius > self._outer_gradient_radius
            model[outer] /= gradiento[outer]

        return model

    def __call__(self, pars):
        """Compute the cost function for a set of parameters.

        Parameters
        ----------
        pars : `np.ndarray`
            Parameter array to compute model.

        Returns
        -------
        cost : `float`
            Cost value computed from the absolute deviation.
        """

        model = self.compute_model(pars)

        absdev = np.abs(self._value - model)
        t = np.sum(absdev.astype(np.float64))

        return t


class FlatGainRatioFitter:
    """Class to fit amplifier gain ratios, removing a simple gradient.

    This fitter will take arrays of x/y/amp_num/value and fit amplifier
    gain ratios, using one amplifier as the fixed point. The fitter
    uses a low-order chebyshev polynomial to fit out the gradient, with
    amp ratios on top of this.

    Parameters
    ----------
    bbox : `lsst.geom.Box2I`
        Bounding box for Chebyshev polynomial gradient.
    order : `int`
        Order of Chebyshev polynomial.
    x : `np.ndarray`
        Array of x values for points to fit (detector pixels).
    y : `np.ndarray`
        Array of y values for points to fit (detector pixels).
    amp_num : `np.ndarray`
        Array of amp numbers associated with each x/y pair.
    value : `np.ndarray`
        Flat value at each x/y pair.
    fixed_amp_num : `int`
        Amplifier number to keep fixed.
    """
    def __init__(self, bbox, order, x, y, amp_num, value, fixed_amp_num):
        self._bbox = bbox
        self._order = order
        self._x = x.astype(np.float64)
        self._y = y.astype(np.float64)
        self._amp_num = amp_num
        self._value = value.astype(np.float64)
        self._fixed_amp_num = fixed_amp_num

        self.indices = {"chebyshev": np.arange((order + 1) * (order + 1))}
        npar = len(self.indices["chebyshev"])

        self._amps = np.unique(amp_num)
        self._n_amp = len(self._amps)

        self.indices["amp_pars"] = np.arange(self._n_amp) + npar
        npar += self._n_amp

        self._npar = npar

        self._amp_indices = {}
        for i in range(self._n_amp):
            amp_num = self._amps[i]
            self._amp_indices[amp_num] = (self._amp_num == amp_num)

    def fit(self, n_iter=10):
        """Fit the amp ratio parameters.

        This uses an iterative fit, where it fits a Chebyshev gradient,
        computes amp ratios, and re-fits the gradient.

        Returns
        -------
        pars : `np.ndarray`
            Chebyshev parameters and amp offset parameters.
        """
        value = self._value.copy()

        control = lsst.afw.math.ChebyshevBoundedFieldControl()
        control.orderX = self._order
        control.orderY = self._order
        control.triangular = False

        pars = np.zeros(self._npar)

        pars[self.indices["amp_pars"]] = 1.0

        for i in range(n_iter):
            field = lsst.afw.math.ChebyshevBoundedField.fit(
                self._bbox,
                self._x,
                self._y,
                value,
                control,
            )

            pars[self.indices["chebyshev"][:]] = field.getCoefficients().ravel()

            ratio = self._value / field.evaluate(self._x, self._y)

            fixed_med = np.median(ratio[self._amp_indices[self._fixed_amp_num]])
            ratio /= fixed_med

            pars[self.indices["amp_pars"][self._fixed_amp_num]] = 1.0

            value = self._value.copy()

            for j in range(self._n_amp):
                amp_num = self._amps[j]

                pars[self.indices["amp_pars"][j]] = np.median(ratio[self._amp_indices[amp_num]])
                value[self._amp_indices[amp_num]] *= pars[self.indices["amp_pars"][j]]

        return pars

    def compute_model(self, pars):
        """Compute the gradient/amp ratio model for a given set of parameters.

        Parameters
        ----------
        pars : `np.ndarray`
            Chebyshev parameters and amp offset parameters.

        Returns
        -------
        model : `np.ndarray`
            The model at each x/y pair.
        """
        field = lsst.afw.math.ChebyshevBoundedField(
            self._bbox,
            pars[self.indices["chebyshev"]].reshape(self._order + 1, self._order + 1),
        )
        model = field.evaluate(self._x, self._y)

        for i in range(self._n_amp):
            amp_num = self._amps[i]

            model[self._amp_indices[amp_num]] *= pars[self.indices["amp_pars"][i]]

        return model

    def __call__(self, pars):
        """Compute the cost function for a set of parameters.

        Parameters
        ----------
        pars : `np.ndarray`
            Chebyshev parameters and amp offset parameters.

        Returns
        -------
        cost : `float`
            Cost value computed from the absolute deviation.
        """
        model = self.compute_model(pars)

        absdev = np.abs(self._value - model)
        t = np.sum(absdev.astype(np.float64))

        return t
