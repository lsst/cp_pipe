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

import numpy as np
from .astierCovPtcFit import CovFit

__all__ = ['CovFft']


class CovFft:
    """A class to compute (via FFT) the nearby pixels correlation function.

    Implements appendix of Astier+19.

    Parameters
    ----------
    diff: `numpy.array`
        Image where to calculate the covariances (e.g., the difference image of two flats).

    w: `numpy.array`
        Weight image (mask): it should consist of 1's (good pixel) and 0's (bad pixels).

    fftShape: `tuple`
        2d-tuple with the shape of the FFT

    maxRangeCov: `int`
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
        dx: `int`
           Lag in x

        dy: `int
           Lag in y

        Returns
        -------
        0.5*(cov1+cov2): `float`
            Covariance at (dx, dy) lag

        npix1+npix2: `int`
            Number of pixels used in covariance calculation.
        """
        # compensate rounding errors
        nPix1 = int(round(self.pCount[dy, dx]))
        cov1 = self.pCov[dy, dx]/nPix1-self.pMean[dy, dx]*self.pMean[-dy, -dx]/(nPix1*nPix1)
        if (dx == 0 or dy == 0):
            return cov1, nPix1
        nPix2 = int(round(self.pCount[-dy, dx]))
        cov2 = self.pCov[-dy, dx]/nPix2-self.pMean[-dy, dx]*self.pMean[dy, -dx]/(nPix2*nPix2)
        return 0.5*(cov1+cov2), nPix1+nPix2

    def reportCovFft(self, maxRange):
        """Produce a list of tuples with covariances.

        Implements appendix of Astier+19.

        Parameters
        ----------
        maxRange: `int`
            Maximum range of covariances.

        Returns
        -------
        tupleVec: `list`
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


def fftSize(s):
    """Calculate the size fof one dimension for the FFT"""
    x = int(np.log(s)/np.log(2.))
    return int(2**(x+1))


def computeCovDirect(diffImage, weightImage, maxRange):
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
                cov1, nPix1 = covDirectValue(diffImage, weightImage, dx, dy)
                cov2, nPix2 = covDirectValue(diffImage, weightImage, dx, -dy)
                cov = 0.5*(cov1 + cov2)
                nPix = nPix1 + nPix2
            else:
                cov, nPix = covDirectValue(diffImage, weightImage, dx, dy)
            if (dx == 0 and dy == 0):
                var = cov
            outList.append((dx, dy, var, cov, nPix))

    return outList


def covDirectValue(diffImage, weightImage, dx, dy):
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


class LoadParams:
    """
    A class to prepare covariances for the PTC fit.

    Parameters
    ----------
    r: `int`, optional
        Maximum lag considered (e.g., to eliminate data beyond a separation "r": ignored in the fit).

    subtractDistantValue: `bool`, optional
        Subtract a background to the measured covariances (mandatory for HSC flat pairs)?

    start: `int`, optional
        Distance beyond which the subtractDistant model is fitted.

    offsetDegree: `int`
        Polynomial degree for the subtraction model.

    Notes
    -----
    params = LoadParams(). "params" drives what happens in he fit. LoadParams provides default values.
    """
    def __init__(self):
        self.r = 8
        self.subtractDistantValue = False
        self.start = 5
        self.offsetDegree = 1


def parseData(dataset, params):
    """ Returns a list of CovFit objects, indexed by amp number.

    Params
    ------
    dataset : `lsst.ip.isr.ptcDataset.PhotonTransferCurveDataset`
        The PTC dataset containing the means, variances, and
        exposure times.

    params: `covAstierptcUtil.LoadParams`
        Object with values to drive the bahaviour of fits.

    Returns
    -------
    covFitList: `dict`
        Dictionary with amps as keys, and CovFit objects as values.
    """

    covFitList = {}
    for ampName in dataset.ampNames:
        # If there is a bad amp, don't fit it
        if ampName in dataset.badAmps:
            continue
        maskAtAmp = dataset.expIdMask[ampName]
        muAtAmp = dataset.rawMeans[ampName]
        covAtAmp = dataset.covariances[ampName]
        covSqrtWeightsAtAmp = dataset.covariancesSqrtWeights[ampName]

        if params.subtractDistantValue:
            c = CovFit(muAtAmp, covAtAmp, covSqrtWeightsAtAmp, params.r, maskAtAmp)
            c.subtractDistantOffset(params.r, params.start, params.offsetDegree)
        else:
            c = CovFit(muAtAmp, covAtAmp, covSqrtWeightsAtAmp, params.r, maskAtAmp)

        cc = c.copy()
        cc.initFit()  # allows to get a crude gain.
        covFitList[ampName] = cc

    return covFitList


def fitData(dataset, r=8):
    """Fit data to models in Astier+19.

    Parameters
    ----------
    dataset : `lsst.ip.isr.ptcDataset.PhotonTransferCurveDataset`
        The dataset containing the means, variances, and exposure times.

    r : `int`, optional
        Maximum lag considered (e.g., to eliminate data beyond a separation "r": ignored in the fit).

    Returns
    -------
    covFitList: `dict`
        Dictionary of CovFit objects, with amp names as keys.

    covFitNoBList: `dict`
       Dictionary of CovFit objects, with amp names as keys (b=0 in Eq. 20 of Astier+19).

    Notes
    -----
    The parameters of the full model for C_ij(mu) ("C_ij" and "mu" in ADU^2 and ADU, respectively)
    in Astier+19 (Eq. 20) are:

        "a" coefficients (r by r matrix), units: 1/e
        "b" coefficients (r by r matrix), units: 1/e
        noise matrix (r by r matrix), units: e^2
        gain, units: e/ADU

    "b" appears in Eq. 20 only through the "ab" combination, which is defined in this code as "c=ab".
    """

    lparams = LoadParams()
    lparams.subtractDistantValue = False
    lparams.r = r
    covFitList = parseData(dataset, lparams)
    covFitNoBList = {}  # [None]*(exts[-1]+1)
    for ext, c in covFitList.items():
        c.fitFullModel()
        covFitNoBList[ext] = c.copy()
        c.params['c'].release()
        c.fitFullModel()
    return covFitList, covFitNoBList


def getFitDataFromCovariances(i, j, mu, fullCov, fullCovModel, fullCovSqrtWeights, gain=1.0,
                              divideByMu=False, returnMasked=False):
    """Get measured signal and covariance, cov model, weigths, and mask at covariance lag (i, j).

    Parameters
    ----------
    i :  `int`
        Lag for covariance matrix.

    j: `int`
        Lag for covariance matrix.

    mu : `list`
        Mean signal values.

    fullCov: `list` of `numpy.array`
        Measured covariance matrices at each mean signal level in mu.

    fullCovSqrtWeights: `list` of `numpy.array`
        List of square root of measured covariances at each mean signal level in mu.

    fullCovModel : `list` of `numpy.array`
        List of modeled covariances at each mean signal level in mu.

    gain : `float`, optional
        Gain, in e-/ADU. If other than 1.0 (default), the returned quantities will be in
        electrons or powers of electrons.

    divideByMu: `bool`, optional
        Divide returned covariance, model, and weights by the mean signal mu?

    returnMasked : `bool`, optional
        Use mask (based on weights) in returned arrays (mu, covariance, and model)?

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
        Boolean mask of the covariance at (i,j), where the weights differ from 0.

    Notes
    -----
    This function is a method of the `CovFit` class.
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
