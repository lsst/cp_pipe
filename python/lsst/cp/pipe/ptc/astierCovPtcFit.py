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
import copy
from scipy.signal import fftconvolve
from scipy.optimize import leastsq
from .astierCovFitParameters import FitParameters

import lsst.log as lsstLog

__all__ = ["CovFit"]


def makeCovArray(inputTuple, maxRangeFromTuple=8):
    """Make covariances array from tuple.

    Parameters
    ----------
    inputTuple: `numpy.ndarray`
        Structured array with rows with at least
        (mu, afwVar, cov, var, i, j, npix), where:

        mu : 0.5*(m1 + m2), where:
            mu1: mean value of flat1
            mu2: mean value of flat2
        afwVar: variance of difference flat, calculated with afw
        cov: covariance value at lag(i, j)
        var: variance(covariance value at lag(0, 0))
        i: lag dimension
        j: lag dimension
        npix: number of pixels used for covariance calculation.

    maxRangeFromTuple: `int`
        Maximum range to select from tuple.

    Returns
    -------
    cov: `numpy.array`
        Covariance arrays, indexed by mean signal mu.

    vCov: `numpy.array`
        Variance arrays, indexed by mean signal mu.

    muVals: `numpy.array`
        List of mean signal values.

    Notes
    -----

    The input tuple should contain  the following rows:
    (mu, cov, var, i, j, npix), with one entry per lag, and image pair.
    Different lags(i.e. different i and j) from the same
    image pair have the same values of mu1 and mu2. When i==j==0, cov
    = var.

    If the input tuple contains several video channels, one should
    select the data of a given channel *before* entering this
    routine, as well as apply(e.g.) saturation cuts.

    The routine returns cov[k_mu, j, i], vcov[(same indices)], and mu[k]
    where the first index of cov matches the one in mu.

    This routine implements the loss of variance due to clipping cuts
    when measuring variances and covariance, but this should happen
    inside the measurement code, where the cuts are readily available.

    """
    if maxRangeFromTuple is not None:
        cut = (inputTuple['i'] < maxRangeFromTuple) & (inputTuple['j'] < maxRangeFromTuple)
        cutTuple = inputTuple[cut]
    else:
        cutTuple = inputTuple
    # increasing mu order, so that we can group measurements with the same mu
    muTemp = cutTuple['mu']
    ind = np.argsort(muTemp)

    cutTuple = cutTuple[ind]
    # should group measurements on the same image pairs(same average)
    mu = cutTuple['mu']
    xx = np.hstack(([mu[0]], mu))
    delta = xx[1:] - xx[:-1]
    steps, = np.where(delta > 0)
    ind = np.zeros_like(mu, dtype=int)
    ind[steps] = 1
    ind = np.cumsum(ind)  # this acts as an image pair index.
    # now fill the 3-d cov array(and variance)
    muVals = np.array(np.unique(mu))
    i = cutTuple['i'].astype(int)
    j = cutTuple['j'].astype(int)
    c = 0.5*cutTuple['cov']
    n = cutTuple['npix']
    v = 0.5*cutTuple['var']
    # book and fill
    cov = np.ndarray((len(muVals), np.max(i)+1, np.max(j)+1))
    var = np.zeros_like(cov)
    cov[ind, i, j] = c
    var[ind, i, j] = v**2/n
    var[:, 0, 0] *= 2  # var(v) = 2*v**2/N

    return cov, var, muVals


def symmetrize(inputArray):
    """ Copy array over 4 quadrants prior to convolution.

    Parameters
    ----------
    inputarray: `numpy.array`
        Input array to symmetrize.

    Returns
    -------
    aSym: `numpy.array`
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


class CovFit:
    """A class to fit the models in Astier+19 to flat covariances.

    This code implements the model(and the fit thereof) described in
    Astier+19: https://arxiv.org/pdf/1905.08677.pdf

    Parameters
    ----------
    meanSignals : `list`[`float`]
        List with means of the difference image of two flats,
        for a particular amplifier in the detector.

    covariances : `list`[`numpy.array`]
        List with 2D covariance arrays at a given mean signal.

    covsSqrtWeights : `list`[`numpy.array`]
        List with 2D arrays with weights from `vcov as defined in
        `makeCovArray`: weight = 1/sqrt(vcov).

    maxRangeFromTuple: `int`, optional
        Maximum range to select from tuple.

    meanSignalMask: `list`[`bool`], optional
        Mask of mean signal 1D array. Use all entries if empty.
    """

    def __init__(self, meanSignals, covariances, covsSqrtWeights, maxRangeFromTuple=8, meanSignalsMask=[]):
        assert (len(meanSignals) == len(covariances))
        assert (len(covariances) == len(covsSqrtWeights))
        if len(meanSignalsMask) == 0:
            meanSignalsMask = np.repeat(True, len(meanSignals))
        self.mu = meanSignals[meanSignalsMask]
        self.cov = np.nan_to_num(covariances)[meanSignalsMask]
        # make it nan safe, replacing nan's with 0 in weights
        self.sqrtW = np.nan_to_num(covsSqrtWeights)[meanSignalsMask]
        self.r = maxRangeFromTuple
        self.logger = lsstLog.Log.getDefaultLogger()

    def copy(self):
        """Make a copy of params"""
        cop = copy.deepcopy(self)
        # deepcopy does not work for FitParameters.
        if hasattr(self, 'params'):
            cop.params = self.params.copy()
        return cop

    def initFit(self):
        """ Performs a crude parabolic fit of the data in order to start
        the full fit close to the solution.
        """
        # number of parameters for 'a' array.
        lenA = self.r*self.r
        # define parameters: c corresponds to a*b in Astier+19 (Eq. 20).
        self.params = FitParameters([('a', lenA), ('c', lenA), ('noise', lenA), ('gain', 1)])
        self.params['gain'] = 1.
        # c=0 in a first go.
        self.params['c'].fix(val=0.)
        # plumbing: extract stuff from the parameter structure
        a = self.params['a'].full.reshape(self.r, self.r)
        noise = self.params['noise'].full.reshape(self.r, self.r)
        gain = self.params['gain'].full[0]

        # iterate the fit to account for higher orders
        # the chi2 does not necessarily go down, so one could
        # stop when it increases
        oldChi2 = 1e30
        for _ in range(5):
            model = np.nan_to_num(self.evalCovModel())  # this computes the full model.
            # loop on lags
            for i in range(self.r):
                for j in range(self.r):
                    # fit a parabola for a given lag
                    parsFit = np.polyfit(self.mu, self.cov[:, i, j] - model[:, i, j],
                                         2, w=self.sqrtW[:, i, j])
                    # model equation(Eq. 20) in Astier+19:
                    a[i, j] += parsFit[0]
                    noise[i, j] += parsFit[2]*gain*gain
                    if(i + j == 0):
                        gain = 1./(1/gain+parsFit[1])
                        self.params['gain'].full[0] = gain
            chi2 = self.chi2()
            if chi2 > oldChi2:
                break
            oldChi2 = chi2

        return

    def getParamValues(self):
        """Return an array of free parameter values (it is a copy)."""
        return self.params.free + 0.

    def setParamValues(self, p):
        """Set parameter values."""
        self.params.free = p
        return

    def evalCovModel(self, mu=None):
        """Computes full covariances model (Eq. 20 of Astier+19).

        Parameters
        ----------
        mu: `numpy.array`, optional
            List of mean signals.

        Returns
        -------
        covModel: `numpy.array`
            Covariances model.

        Notes
        -----
        By default, computes the covModel for the mu's stored(self.mu).

        Returns cov[Nmu, self.r, self.r]. The variance for the PTC is
        cov[:, 0, 0].  mu and cov are in ADUs and ADUs squared. To use
        electrons for both, the gain should be set to 1. This routine
        implements the model in Astier+19 (1905.08677).

        The parameters of the full model for C_ij(mu) ("C_ij" and "mu"
        in ADU^2 and ADU, respectively) in Astier+19 (Eq. 20) are:

        "a" coefficients (r by r matrix), units: 1/e
        "b" coefficients (r by r matrix), units: 1/e
        noise matrix (r by r matrix), units: e^2
        gain, units: e/ADU

        "b" appears in Eq. 20 only through the "ab" combination, which
        is defined in this code as "c=ab".

        """
        sa = (self.r, self.r)
        a = self.params['a'].full.reshape(sa)
        c = self.params['c'].full.reshape(sa)
        gain = self.params['gain'].full[0]
        noise = self.params['noise'].full.reshape(sa)
        # pad a with zeros and symmetrize
        aEnlarged = np.zeros((int(sa[0]*1.5)+1, int(sa[1]*1.5)+1))
        aEnlarged[0:sa[0], 0:sa[1]] = a
        aSym = symmetrize(aEnlarged)
        # pad c with zeros and symmetrize
        cEnlarged = np.zeros((int(sa[0]*1.5)+1, int(sa[1]*1.5)+1))
        cEnlarged[0:sa[0], 0:sa[1]] = c
        cSym = symmetrize(cEnlarged)
        a2 = fftconvolve(aSym, aSym, mode='same')
        a3 = fftconvolve(a2, aSym, mode='same')
        ac = fftconvolve(aSym, cSym, mode='same')
        (xc, yc) = np.unravel_index(np.abs(aSym).argmax(), a2.shape)
        range = self.r
        a1 = a[np.newaxis, :, :]
        a2 = a2[np.newaxis, xc:xc + range, yc:yc + range]
        a3 = a3[np.newaxis, xc:xc + range, yc:yc + range]
        ac = ac[np.newaxis, xc:xc + range, yc:yc + range]
        c1 = c[np.newaxis, ::]
        if mu is None:
            mu = self.mu
        # assumes that mu is 1d
        bigMu = mu[:, np.newaxis, np.newaxis]*gain
        # c(=a*b in Astier+19) also has a contribution to the last
        # term, that is absent for now.
        covModel = (bigMu/(gain*gain)*(a1*bigMu+2./3.*(bigMu*bigMu)*(a2 + c1)
                    + (1./3.*a3 + 5./6.*ac)*(bigMu*bigMu*bigMu)) + noise[np.newaxis, :, :]/gain**2)
        # add the Poisson term, and the read out noise (variance)
        covModel[:, 0, 0] += mu/gain

        return covModel

    def getA(self):
        """'a' matrix from Astier+19(e.g., Eq. 20)"""
        return self.params['a'].full.reshape(self.r, self.r)

    def getB(self):
        """'b' matrix from Astier+19(e.g., Eq. 20)"""
        return self.params['c'].full.reshape(self.r, self.r)/self.getA()

    def getC(self):
        """'c'='ab' matrix from Astier+19(e.g., Eq. 20)"""
        return np.array(self.params['c'].full.reshape(self.r, self.r))

    def _getCovParams(self, what):
        """Get covariance matrix of parameters from fit"""
        indices = self.params[what].indexof()
        i1 = indices[:, np.newaxis]
        i2 = indices[np.newaxis, :]
        if self.covParams is not None:
            covp = self.covParams[i1, i2]
        else:
            covp = None
        return covp

    def getACov(self):
        """Get covariance matrix of "a" coefficients from fit"""
        if self._getCovParams('a') is not None:
            cova = self._getCovParams('a').reshape((self.r, self.r, self.r, self.r))
        else:
            cova = None
        return cova

    def getASig(self):
        """Square root of diagonal of the parameter covariance of the fitted
        "a" matrix

        """
        if self._getCovParams('a') is not None:
            sigA = np.sqrt(self._getCovParams('a').diagonal()).reshape((self.r, self.r))
        else:
            sigA = None
        return sigA

    def getBCov(self):
        """Get covariance matrix of "a" coefficients from fit
        b = c /a
        """
        covb = self._getCovParams('c')
        aval = self.getA().flatten()
        factor = np.outer(aval, aval)
        covb /= factor
        return covb.reshape((self.r, self.r, self.r, self.r))

    def getCCov(self):
        """Get covariance matrix of "c" coefficients from fit"""
        cova = self._getCovParams('c')
        return cova.reshape((self.r, self.r, self.r, self.r))

    def getGainErr(self):
        """Get error on fitted gain parameter"""
        if self._getCovParams('gain') is not None:
            gainErr = np.sqrt(self._getCovParams('gain')[0][0])
        else:
            gainErr = 0.0
        return gainErr

    def getNoiseCov(self):
        """Get covariances of noise matrix from fit"""
        covNoise = self._getCovParams('noise')
        return covNoise.reshape((self.r, self.r, self.r, self.r))

    def getNoiseSig(self):
        """Square root of diagonal of the parameter covariance of the fitted
        "noise" matrix

        """
        if self._getCovParams('noise') is not None:
            covNoise = self._getCovParams('noise')
            noise = np.sqrt(covNoise.diagonal()).reshape((self.r, self.r))
        else:
            noise = None
        return noise

    def getGain(self):
        """Get gain (e/ADU)"""
        return self.params['gain'].full[0]

    def getRon(self):
        """Get readout noise (e^2)"""
        return self.params['noise'].full[0]

    def getRonErr(self):
        """Get error on readout noise parameter"""
        ronSqrt = np.sqrt(np.fabs(self.getRon()))
        if self.getNoiseSig() is not None:
            noiseSigma = self.getNoiseSig()[0][0]
            ronErr = 0.5*(noiseSigma/np.fabs(self.getRon()))*ronSqrt
        else:
            ronErr = np.nan
        return ronErr

    def getNoise(self):
        """Get noise matrix"""
        return self.params['noise'].full.reshape(self.r, self.r)

    def getMaskCov(self, i, j):
        """Get mask of Cov[i,j]"""
        weights = self.sqrtW[:, i, j]
        mask = weights != 0
        return mask

    def setAandB(self, a, b):
        """Set "a" and "b" coeffcients forfull Astier+19 model
        (Eq. 20). "c=a*b".

        """
        self.params['a'].full = a.flatten()
        self.params['c'].full = a.flatten()*b.flatten()
        return

    def chi2(self):
        """Calculate weighted chi2 of full-model fit."""
        return(self.weightedRes()**2).sum()

    def weightedRes(self, params=None):
        """Weighted residuals.

        Notes
        -----
        To be used via:
        c = CovFit(meanSignals, covariances, covsSqrtWeights)
        c.initFit()
        coeffs, cov, _, mesg, ierr = leastsq(c.weightedRes,
                                             c.getParamValues(),
                                             full_output=True)
        """
        if params is not None:
            self.setParamValues(params)
        covModel = np.nan_to_num(self.evalCovModel())
        weightedRes = (covModel-self.cov)*self.sqrtW

        return weightedRes.flatten()

    def fitFullModel(self, pInit=None):
        """Fit measured covariances to full model in Astier+19 (Eq. 20)

        Parameters
        ----------
        pInit : `list`
            Initial parameters of the fit.
            len(pInit) = #entries(a) + #entries(c) + #entries(noise) + 1
            len(pInit) = r^2 + r^2 + r^2 + 1, where "r" is the maximum lag
              considered for the covariances calculation, and the extra "1"
              is the gain.
            If "b" is 0, then "c" is 0, and len(pInit) will have r^2 fewer
              entries.

        Returns
        -------
        params : `np.array`
            Fit parameters (see "Notes" below).

        Notes
        -----
        The parameters of the full model for C_ij(mu) ("C_ij" and "mu"
        in ADU^2 and ADU, respectively) in Astier+19 (Eq. 20) are:

            "a" coefficients (r by r matrix), units: 1/e
            "b" coefficients (r by r matrix), units: 1/e
            noise matrix (r by r matrix), units: e^2
            gain, units: e/ADU

        "b" appears in Eq. 20 only through the "ab" combination, which
        is defined in this code as "c=ab".

        """

        if pInit is None:
            pInit = self.getParamValues()
        params, paramsCov, _, mesg, ierr = leastsq(self.weightedRes, pInit, full_output=True)
        self.covParams = paramsCov

        return params

    def getFitData(self, i, j, divideByMu=False, unitsElectrons=False, returnMasked=False):
        """Get measured signal and covariance, cov model, weigths, and mask at
        covariance lag (i, j).

        Parameters
        ---------
        i: `int`
            Lag for covariance matrix.

        j: `int`
            Lag for covariance matrix.

        divideByMu: `bool`, optional
            Divide covariance, model, and weights by signal mu?

        unitsElectrons : `bool`, optional
            mu, covariance, and model are in ADU (or powers of ADU) If this
            parameter is true, these are multiplied by the adequte
            factors of the gain to return quantities in electrons (or
            powers of electrons).

        returnMasked : `bool`, optional
            Use mask (based on weights) in returned arrays (mu,
            covariance, and model)?

        Returns
        -------
        mu: `numpy.array`
            list of signal values (mu).

        covariance: `numpy.array`
            Covariance arrays, indexed by mean signal mu (self.cov[:, i, j]).

        covarianceModel: `numpy.array`
            Covariance model (model).

        weights: `numpy.array`
            Weights (self.sqrtW)

        mask : `numpy.array`, optional
            Boolean mask of the covariance at (i,j).

        Notes
        -----
        Using a CovFit object, selects from (i, j) and returns
        mu*gain, self.cov[:, i, j]*gain**2 model*gain**2, and
        self.sqrtW/gain**2 in electrons or ADU if
        unitsElectrons=False.

        """
        if unitsElectrons:
            gain = self.getGain()
        else:
            gain = 1.0

        mu = self.mu*gain
        covariance = self.cov[:, i, j]*(gain**2)
        covarianceModel = self.evalCovModel()[:, i, j]*(gain**2)
        weights = self.sqrtW[:, i, j]/(gain**2)

        # select data used for the fit:
        mask = self.getMaskCov(i, j)
        if returnMasked:
            weights = weights[mask]
            covarianceModel = covarianceModel[mask]
            mu = mu[mask]
            covariance = covariance[mask]

        if divideByMu:
            covariance /= mu
            covarianceModel /= mu
            weights *= mu

        return mu, covariance, covarianceModel, weights, mask
