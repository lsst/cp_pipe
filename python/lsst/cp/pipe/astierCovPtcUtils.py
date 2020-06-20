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
    x = int(np.log(s)/np.log(2.))
    return int(2**(x+1))


def computeCovFft(diff, w, fftSize, maxRange):
    """Compute covariances via FFT

    Parameters
    ----------
    diff: `lsst.afw.image.exposure.exposure.ExposureF`
        Difference image from a pair of flats taken at the same exposure time.

    w: `numpy array`
        Mask array with 1's (good pixels) and 0's (bad pixels).

    fftSize: `tuple`
        Size of the DFT: (xSize, ySize)

    maxRange: `int`
        Maximum range of covariances

    Returns
    -------
    CovFft.reportCovFft(maxRange): `list`
        List with covariance tuples,
    """

    c = CovFft(diff, w, fftSize, maxRange)

    return c.reportCovFft(maxRange)


def findGroups(x, maxDiff):
    """Group data into bins, with at most maxDiff distance between bins.

    Parameters
    ----------
    x: `list`
        Data to bin.

    maxDiff: `int`
        Maximum distance between bins.

    Returns
    -------
    index: `list`
        Bin indices.
    """
    ix = np.argsort(x)
    xsort = np.sort(x)
    index = np.zeros_like(x, dtype=np.int32)
    xc = xsort[0]
    group = 0
    ng = 1

    for i in range(1, len(ix)):
        xval = xsort[i]
        if (xval - xc < maxDiff):
            xc = (ng*xc + xval)/(ng+1)
            ng += 1
            index[ix[i]] = group
        else:
            group += 1
            ng = 1
            index[ix[i]] = group
            xc = xval

    return index


def indexForBins(x, nBins):
    """Builds an index with regular binning. The result can be fed into binData.

    Parameters
    ----------
    x: `numpy.array`
        Data to bin.
    nBins: `int`
        Number of bin.

    Returns
    -------
    np.digitize(x, bins): `numpy.array`
        Bin indices.
    """

    bins = np.linspace(x.min(), x.max() + abs(x.max() * 1e-7), nBins + 1)

    return np.digitize(x, bins)


def binData(x, y, binIndex, wy=None):
    """Bin data (usually for display purposes).

    Patrameters
    -----------
    x: `numpy.array`
        Data to bin.

    y: `numpy.array`
        Data to bin.

    binIdex: `list`
        Bin number of each datum.

    wy: `numpy.array`
        Inverse rms of each datum to use when averaging (the actual weight is wy**2).

    Returns:
    -------

    xbin: `numpy.array`
        Binned data in x.

    ybin: `numpy.array`
        Binned data in y.

    wybin: `numpy.array`
        Binned weights in y, computed from wy's in each bin.

    sybin: `numpy.array`
        Uncertainty on the bin average, considering actual scatter, and ignoring weights.
    """

    if wy is None:
        wy = np.ones_like(x)
    binIndexSet = set(binIndex)
    w2 = wy*wy
    xw2 = x*(w2)
    xbin = np.array([xw2[binIndex == i].sum()/w2[binIndex == i].sum() for i in binIndexSet])

    yw2 = y*w2
    ybin = np.array([yw2[binIndex == i].sum()/w2[binIndex == i].sum() for i in binIndexSet])

    wybin = np.sqrt(np.array([w2[binIndex == i].sum() for i in binIndexSet]))
    sybin = np.array([y[binIndex == i].std()/np.sqrt(np.array([binIndex == i]).sum()) for i in binIndexSet])

    return xbin, ybin, wybin, sybin


class LoadParams:
    """
    A class to prepare covariances for the PTC fit.

    Parameters
    ----------
    r: `int`, optional
        Maximum lag considered (e.g., to eliminate data beyond a separation "r": ignored in the fit).

    maxMu: `float`, optional
        Maximum signal, in ADU (e.g., to eliminate data beyond saturation).

    maxMuElectrons: `float`, optional
        Maximum signal in electrons.

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
        self.maxMu = 1e9
        self.maxMuElectrons = 1e9
        self.subtractDistantValue = False
        self.start = 5
        self.offsetDegree = 1


def loadData(tupleName, params):
    """ Returns a list of CovFit objects, indexed by amp number.

    Params
    ------
    tupleName: `numpy.recarray`
        Recarray with rows with at least ( mu1, mu2, cov ,var, i, j, npix), where:
            mu1: mean value of flat1
            mu2: mean value of flat2
            cov: covariance value at lag (i, j)
            var: variance (covariance value at lag (0, 0))
            i: lag dimension
            j: lag dimension
            npix: number of pixels used for covariance calculation.

    params: `covAstierptcUtil.LoadParams`
        Object with values to drive the bahaviour of fits.

    Returns
    -------
    covFitList: `dict`
        Dictionary with amps as keys, and CovFit objects as values.
    """

    exts = np.array(np.unique(tupleName['ampName']), dtype=str)
    covFitList = {}
    for ext in exts:
        ntext = tupleName[tupleName['ampName'] == ext]
        if params.subtractDistantValue:
            c = CovFit(ntext, params.r)
            c.subtractDistantOffset(params.r, params.start, params.offsetDegree)
        else:
            c = CovFit(ntext, params.r)
        thisMaxMu = params.maxMu
        # Tune the maxMuElectrons cut
        for iter in range(3):
            cc = c.copy()
            cc.setMaxMu(thisMaxMu)
            cc.initFit()  # allows to get a crude gain.
            gain = cc.getGain()
            if (thisMaxMu*gain < params.maxMuElectrons):
                thisMaxMu = params.maxMuElectrons/gain
                continue
            cc.setMaxMuElectrons(params.maxMuElectrons)
            break
        covFitList[ext] = cc

    return covFitList


def fitData(tupleName, maxMu=1e9, maxMuElectrons=1e9, r=8):
    """Fit data to models in Astier+19.

    Parameters
    ----------
    tupleName: `numpy.recarray`
        Recarray with rows with at least ( mu1, mu2, cov ,var, i, j, npix), where:
            mu1: mean value of flat1
            mu2: mean value of flat2
            cov: covariance value at lag (i, j)
            var: variance (covariance value at lag (0, 0))
            i: lag dimension
            j: lag dimension
            npix: number of pixels used for covariance calculation.

    r: `int`, optional
        Maximum lag considered (e.g., to eliminate data beyond a separation "r": ignored in the fit).

    maxMu: `float`, optional
        Maximum signal, in ADU (e.g., to eliminate data beyond saturation).

    maxMuElectrons: `float`, optional
        Maximum signal in electrons.

    Returns
    -------
    covFitList: `dict`
        Dictionary of CovFit objects, with amp names as keys.

    covFitNoBList: `dict`
       Dictionary of CovFit objects, with amp names as keys (b=0 in Eq. 20 of Astier+19).
    """
    lparams = LoadParams()
    lparams.subtractDistantValue = False
    lparams.maxMu = maxMu
    lparams.maxMu = maxMuElectrons
    lparams.r = r

    covFitList = loadData(tupleName, lparams)
    covFitNoBList = {}  # [None]*(exts[-1]+1)
    for ext, c in covFitList.items():
        c.fit()
        covFitNoBList[ext] = c.copy()
        c.params['c'].release()
        c.fit()

    return covFitList, covFitNoBList


def CHI2(res, wy):
    wres = res*wy
    return (wres*wres).sum()
