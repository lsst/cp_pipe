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

__all__ = ['ddict2dict', 'CovFastFourierTransform']

import numpy as np
from scipy.optimize import leastsq
import numpy.polynomial.polynomial as poly
from scipy.stats import median_abs_deviation, norm
import logging

from lsst.ip.isr import isrMock
import lsst.afw.image
import lsst.afw.math

import galsim


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
                  expId1=0, expId2=1):
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
    a00, gain, noise = pars
    return 0.5/(a00*gain*gain)*(np.exp(2*a00*x*gain)-1) + noise/(gain*gain)  # C_00


def arrangeFlatsByExpTime(exposureList, exposureIdList):
    """Arrange exposures by exposure time.

    Parameters
    ----------
    exposureList : `list` [`lsst.pipe.base.connections.DeferredDatasetRef`]
        Input list of exposure references.
    exposureIdList : `list` [`int`]
        List of exposure ids as obtained by dataId[`exposure`].

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
        listAtExpTime = flatsAtExpTime.setdefault(expTime, [])
        listAtExpTime.append((expRef, expId))

    return flatsAtExpTime


def arrangeFlatsByExpFlux(exposureList, exposureIdList, fluxKeyword):
    """Arrange exposures by exposure flux.

    Parameters
    ----------
    exposureList : `list` [`lsst.pipe.base.connections.DeferredDatasetRef`]
        Input list of exposure references.
    exposureIdList : `list` [`int`]
        List of exposure ids as obtained by dataId[`exposure`].
    fluxKeyword : `str`
        Header keyword that contains the flux per exposure.

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
        expFlux = expRef.get().getMetadata()[fluxKeyword]
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
        assert(fftShape[0] > diff.shape[0]+maxRangeCov+1)
        assert(fftShape[1] > diff.shape[1]+maxRangeCov+1)
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


class AstierSplineLinearityFitter:
    """Class to fit the Astier spline linearity model.

    This is a spline fit with photodiode data based on a model
    from Pierre Astier, referenced in June 2023 from
    https://me.lsst.eu/astier/bot/7224D/model_nonlin.py

    This model fits a spline with (optional) nuisance parameters
    to allow for different linearity coefficients with different
    photodiode settings.  The minimization is a least-squares
    fit with the residual of
    Sum[(S(mu_i) + mu_i)/(k_j * D_i) - 1]**2, where S(mu_i) is
    an Akima Spline function of mu_i, the observed flat-pair
    mean; D_j is the photo-diode measurement corresponding to
    that flat-pair; and k_j is a constant of proportionality
    which is over index j as it is allowed to
    be different based on different photodiode settings (e.g.
    CCOBCURR).

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
    """
    def __init__(self, nodes, grouping_values, pd, mu, mask=None, log=None):
        self._pd = pd
        self._mu = mu
        self._grouping_values = grouping_values
        self.log = log if log else logging.getLogger(__name__)

        self._nodes = nodes
        if nodes[0] != 0.0:
            raise ValueError("First node must be 0.0")
        if not np.all(np.diff(nodes) > 0):
            raise ValueError("Nodes must be sorted with no repeats.")

        # Check if sorted (raise otherwise)
        if not np.all(np.diff(self._grouping_values) >= 0):
            raise ValueError("Grouping values must be sorted.")

        _, uindex, ucounts = np.unique(self._grouping_values, return_index=True, return_counts=True)
        self.ngroup = len(uindex)

        self.group_indices = []
        for i in range(self.ngroup):
            self.group_indices.append(np.arange(uindex[i], uindex[i] + ucounts[i]))

        # Values to regularize spline fit.
        self._x_regularize = np.linspace(self._mu.min(), self._mu.max(), 100)

        # Outlier weight values.  Will be 1 (in) or 0 (out).
        self._w = np.ones(len(self._pd))

        if mask is not None:
            self._w[~mask] = 0.0

    def estimate_p0(self):
        """Estimate initial fit parameters.

        Returns
        -------
        p0 : `np.ndarray`
            Parameter array, with spline values (one for each node) followed
            by proportionality constants (one for each group).
        """
        npt = len(self._nodes) + self.ngroup
        p0 = np.zeros(npt)

        # Do a simple linear fit and set all the constants to this.
        linfit = np.polyfit(self._pd, self._mu, 1)
        p0[-self.ngroup:] = linfit[0]

        # Look at the residuals...
        ratio_model = self.compute_ratio_model(
            self._nodes,
            self.group_indices,
            p0,
            self._pd,
            self._mu,
        )
        # ...and adjust the linear parameters accordingly.
        p0[-self.ngroup:] *= np.median(ratio_model)

        # Re-compute the residuals.
        ratio_model2 = self.compute_ratio_model(
            self._nodes,
            self.group_indices,
            p0,
            self._pd,
            self._mu,
        )

        # And compute a first guess of the spline nodes.
        bins = np.searchsorted(self._nodes, self._mu)
        tot_arr = np.zeros(len(self._nodes))
        n_arr = np.zeros(len(self._nodes), dtype=int)
        np.add.at(tot_arr, bins, ratio_model2)
        np.add.at(n_arr, bins, 1)

        ratio = np.ones(len(self._nodes))
        ratio[n_arr > 0] = tot_arr[n_arr > 0]/n_arr[n_arr > 0]
        ratio[0] = 1.0
        p0[0: len(self._nodes)] = (ratio - 1) * self._nodes

        return p0

    @staticmethod
    def compute_ratio_model(nodes, group_indices, pars, pd, mu, return_spline=False):
        """Compute the ratio model values.

        Parameters
        ----------
        nodes : `np.ndarray` (M,)
            Array of node positions.
        group_indices : `list` [`np.ndarray`]
            List of group indices, one array for each group.
        pars : `np.ndarray`
            Parameter array, with spline values (one for each node) followed
            by proportionality constants (one for each group.)
        pd : `np.ndarray` (N,)
            Array of photodiode measurements.
        mu : `np.ndarray` (N,)
            Array of flat means.
        return_spline : `bool`, optional
            Return the spline interpolation as well as the model ratios?

        Returns
        -------
        ratio_models : `np.ndarray` (N,)
            Model ratio, (mu_i - S(mu_i))/(k_j * D_i)
        spl : `lsst.afw.math.thing`
            Spline interpolator (returned if return_spline=True).
        """
        spl = lsst.afw.math.makeInterpolate(
            nodes,
            pars[0: len(nodes)],
            lsst.afw.math.stringToInterpStyle("AKIMA_SPLINE"),
        )

        numerator = mu - spl.interpolate(mu)
        denominator = pd.copy()
        ngroup = len(group_indices)
        kj = pars[-ngroup:]
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
            Initial fit parameters (one for each knot, followed by one for
            each grouping).
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

    def __call__(self, pars):

        ratio_model, spl = self.compute_ratio_model(
            self._nodes,
            self.group_indices,
            pars,
            self._pd,
            self._mu,
            return_spline=True,
        )

        resid = self._w*(ratio_model - 1.0)

        constraint = [1e3 * np.mean(spl.interpolate(self._x_regularize))]
        # 0 should transform to 0
        constraint.append(spl.interpolate(0)*1e10)

        return np.hstack([resid, constraint])
