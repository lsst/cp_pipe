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

import numpy as np
import copy
import scipy.optimize as opt
from scipy.signal import fftconvolve
from scipy.optimize import leastsq
from .astierCovPtcUtils import *
from .astierCovFitParameters import FitParameters

__all__=["covFit", "WeightedRes"]


"""
This code implements the model (and the fit thereof) described in
https://arxiv.org/pdf/1905.08677.pdf
For the time being it uses as input a numpy recarray which contains
one row per covariance and per pair: see routine makeCovArray
"""

def mad(data, axis=0, scale=1.4826):
    """Median of absolute deviation along a given axis.  

    Parameters
    ----------
    data : `numpy array`
        Input numpy array

    axis : `int`, optional
        Dimension along which to calculate MAD

    scale : `float`
         Scale factor that depends on the distribution. An estimator
         of the standard deviation is given by sigma=scale*MAD (scale=1.4826
         for a Gaussian distribution)
    
    Returns
    -------
    sigma : `float`
        Ewtimator of the standard deviation (sigma=scale*MAD).        
    """

    if data.ndim == 1:
        med = np.ma.median(data)
        ret = np.ma.median(np.abs(data-med))
    else:
        med = np.ma.median(data, axis=axis)
        if axis>0:
            sw = np.ma.swapaxes(data, 0, axis)
        else:
            sw = data
        ret = np.ma.median(np.abs(sw-med), axis=0)
    sigma = scale*ret
    
    return sigma

def aCoeffsComputeOldFashion(fit, muEl):
    """Compute the "a" coefficients of the Antilogus model the old way: 1501.01577 (eq. 16).

    The 'old way' refers to the treatement in, e.g., 1501.01577 (eq. 16), as opposed to the more complete
    model in Astier+19 (1905.08677).

    Parameters
    ---------
    fit : `lsst.cp.pipe.astierCovPtcFit.fit`
        Fit object
    
    Returns
    -------
    aCoeffsOld : `numpy.array`
        Slope of cov/var at a given flux mu in electrons.

    Notes
    -----
    Returns the "a" array, computed this way, to be compared to the actual a_array from the full model
    (fit.geA()).
    """

    gain = fit.getGain()
    muAdu = np.array([muEl/gain])
    model = fit.evalCovModel(muAdu)
    var = model[0,0,0]
    # The model is in ADU**2, so is var, mu is in adu.
    # So for a result in electrons^-1, we have to convert mu to electrons
    return model[0,:,:]/(var*mu_el)


def makeCovArray(inputTuple, maxRangeFromTuple=8):
    """Make covariances array from tuple.
   
    Parameters
    ----------
    inputTuple : `tuple`
        Tuple with at least ( mu1, mu2, cov ,var, i, j, npix), where:
        mu1 : mean value of flat1
        mu2 : mean value of flat2
        cov : covariance value at lag (i, j)
        var : variance (covariance value at lag (0, 0))
        i : lag dimension
        j : lag dimension
        npix : number of pixels used for covariance calculation.

    maxRangeFromTuple : `int`
        Maximum range to select from tuple.

    Returns
    -------
    cov : `numpy.array`
        Covariance arrays, indexed by mean signal mu. 

    vCov : `numpy.array`
        Variance arrays, indexed by mean signal mu.

    muVals : `numpy.array`
        List of mean signal values.

    Notes
    -----

    The input tuple should contain, at least, the following rows:
    (mu1, mu2, cov ,var, i, j, npix), with one entry per lag, and image pair.
    Different lags (i.e. different i and j) from the same
    image pair have the same values of mu1 and mu2. When i==j==0, cov
    = var.

    If the input tuple contains several video channels, one should
    select the data of a given channel *before* entering this
    routine, as well as apply (e.g.) saturation cuts.

    The routine returns cov[k_mu, j, i] , vcov[(same indices)], and mu[k]
    where the first index of cov matches the one in mu.

    This routine implements the loss of variance due to
    clipping cuts when measuring variances and covariance, but this should happen inside 
    the measurement code, where the cuts are readily available.

    """
    if maxRangeFromTuple is not None:
        cut = (inputTuple['i'] < maxRangeFromTuple) & (inputTuple['j'] < maxRangeFromTuple)
        cutTuple = inputTuple[cut]
    else:
        cutTuple = inputTuple
    # increasing mu order, so that we can group measurements with the same mu
    muTemp = (cutTuple['mu1'] + cutTuple['mu2'])*0.5
    ind = np.argsort(muTemp)
    cutTuple = cutTuple[ind]
    # should group measurements on the same image pairs(same average)
    mu = 0.5*(cutTuple['mu1'] + cutTuple['mu2'])
    xx = np.hstack(([mu[0]], mu))
    delta = xx[1:] - xx[:-1]
    steps, = np.where(delta>0)
    ind = np.zeros_like(mu, dtype = int)
    ind[steps] = 1
    ind = np.cumsum(ind)  # this acts as an image pair index.
    # now fill the 3-d cov array (and variance)
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
    var[:, 0,0] *= 2 # var(v) = 2*v**2/N
    # compensate for loss of variance and covariance due to outlier elimination (sigma clipping)
    # when computing variances (cut to 4 sigma): 1 per mill for variances and twice as
    # much for covariances:
    fact = 1.0 #1.00107
    cov *= fact*fact
    cov[:, 0,0] /= fact

    return cov, var, muVals


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
    aSym[..., r2-1:   , r1-1:   ]= inputArray
    aSym[..., r2-1:   , r1-1::-1]= inputArray
    aSym[..., r2-1::-1, r1-1::-1]= inputArray
    aSym[..., r2-1::-1, r1-1:   ]= inputArray
    
    return aSym


class covFit:
    """A class to fit the models in Astier+19 to flat covariances."""
    
    def __init__(self, inputTuple, maxRangeFromTuple=8) :
        """ 
        Parameters
        ----------
        inputTuple : `tuple`
            Tuple with at least ( mu1, mu2, cov ,var, i, j, npix), where:
            
            mu1 : mean value of flat1
            mu2 : mean value of flat2
            cov : covariance value at lag (i, j)
            var : variance (covariance value at lag (0, 0))
            i : lag dimension
            j : lag dimension
            npix : number of pixels used for covariance calculation.

        maxRangeFromTuple : `int`
            Maximum range to select from tuple.
        """
        self.cov, self.vcov, self.mu = makeCovArray(inputTuple, maxRangeFromTuple)
        self.sqrtW  = 1./np.sqrt(self.vcov)
        self.r = self.cov.shape[1]

    def subtractDistantOffset(self, maxLag=8, startLag=15, polDegree=1) :
        """Subtract a background/offset to the measured covariances.
        
        Parameters
        ---------
        maxLag : `int`
            Maximum lag considered
        
        startLag : `int`
            First lag from where to start the offset subtraction.

        polDegree : ``int`
            Degree of 2D polynomial to fit to covariance to define offse to be subtracted. 
        """
        
        assert(start < self.r)
        for k in range(len(self.mu)) :
            # Make a copy because it is going to be altered
            w = self.sqrtW[k,...] + 0.
            sh = w.shape
            i, j =  np.meshgrid(range(sh[0]), range(sh[1]), indexing='ij')
            # kill the core for the fit
            w[:start, :start] = 0
            poly = pol2d.pol2d(i,j,self.cov[k,...], degree+1, w=w)
            back = poly.eval(i,j)
            self.cov[k,...] -= back
        self.r = r
        self.cov = self.cov[:,:r,:r]
        self.vcov = self.vcov[:,:r,:r]
        self.sqrtW = self.sqrtW[:,:r,:r]

    def setMaxMu(self, maxMu):
        """Select signal level based on max average signal in ADU"""
        # mus are sorted at construction
        index = self.mu < maxMu
        k = index.sum()
        self.mu = self.mu[:k]
        self.cov = self.cov[:k,...]
        self.vcov = self.vcov[:k,...]
        self.sqrtW = self.sqrtW[:k,...]

    def setMaxMuElectrons(self,maxMuEl):
        """Select signal level based on max average signal in electrons"""
        g = self.getGain()
        kill = (self.mu*g > maxMuEl)
        self.sqrtW[kill,:,:] = 0

    def copy(self):
        cop = copy.deepcopy(self)
        # deepcopy does not work for FitParameters.
        if hasattr(self, 'params'):
            cop.params = self.params.copy()
        return cop

    def initFit(self):
        """
        Performs a crude parabolic fit of the data in order to start
        the full fit close to the solution.
        """
        # number of parameters for 'a' array.
        lenA = self.r*self.r
        # define parameters : c corresponds to a*b in the Astier+19.
        self.params = FitParameters([('a', lenA), ('c', lenA), ('noise', lenA), ('gain', 1)])
        self.params['gain'] = 1.
        # obvious: c=0 in a first go.
        self.params['c'].fix(val = 0.)
        # plumbing: extract stuff from the parameter structure
        a = self.params['a'].full.reshape(self.r, self.r)
        noise = self.params['noise'].full.reshape(self.r,self.r)
        gain = self.params['gain'].full[0]
        
        # iterate the fit to account for higher orders
        # the chi2 does not necessarily go down, so one could
        # stop when it increases

        for iter in range(5): 
            model = self.evalCovModel() # this computes the full model.
            # loop on lags
            for i in range(self.r) :
                for j in range(self.r) :
                    # fit a given lag with a parabola
                    p = np.polyfit(self.mu, self.cov[:,i,j] - model[:,i,j],
                                   2, w = self.sqrtW[:,i,j])
                    # model equation (Eq. 20) in Astier+19:
                    a[i,j] += p[0]
                    noise[i,j] += p[2]*gain*gain
                    if (i + j == 0) :
                        gain  = 1./(1/gain+p[1])
                        self.params['gain'].full[0] = gain
                    #if (i+j==0) : print(p, gain, a[0,0])
            #print('iter,chi2 a00 gain = ', iter, self.chi2(), a[0,0], gain)

    def getParamValues(self):
        """
        Return an array of free parameter values (it is a copy).
        """
        return self.params.free + 0.

    def setParamValues(self, p):
        self.params.free = p

    def evalCovModel(self, mu = None) :
        """Computes full covariances model (Eq. 20 of Astier+19).
   
        Parameters
        ----------
        mu : `numpy.array`, optional
            List of mean signals.

        Returns
        -------
        covModel : `numpy.array`
            Covariances model.

        Notes
        -----
    
        By default, computes the covModel for the mu's stored (self.mu).
        
        Returns cov[Nmu, self.r, self.r]. The variance for the PTC is cov[:, 0, 0].
        mu and cov are in ADUs and ADUs squared. To use electrons for both,
        the gain should be set to 1. This routine implements the model in Astier+19 (1905.08677).
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
        a2 = fftconvolve(aSym, aSym, mode = 'same')
        a3 = fftconvolve(a2, aSym, mode = 'same')
        ac = fftconvolve(aSym, cSym, mode = 'same')
        (xc,yc) = np.unravel_index(np.abs(aSym).argmax(), a2.shape)
        range = self.r
        a1 = a[np.newaxis, :,:]
        a2 = a2[np.newaxis, xc:xc + range, yc:yc + range]
        a3 = a3[np.newaxis, xc:xc + range, yc:yc + range]
        ac = ac[np.newaxis, xc:xc + range, yc:yc + range]
        c1 = c[np.newaxis, : : ]
        if mu is None: 
            mu = self.mu
        # assumes that mu is 1d
        bigMu = mu[:, np.newaxis, np.newaxis]*gain
        # c (=a*b in Astier+19) also has a contribution to the last term, that is absent for now.
        covModel = (bigMu/(gain*gain)*(a1*bigMu+2./3.*(bigMu*bigMu)*(a2+c1) + 
                   (1./3.*a3+5./6.*ac)*(bigMu*bigMu*bigMu)) + 
                   noise[np.newaxis,:,:]/gain**2)
        # add the Poisson term, and the read out noise (variance, obviously)
        covModel[:,0,0] += mu/gain
        
        return covModel

    def getA(self):
        """'a' matrix from Astier+19 (e.g., Eq. 20)"""
        return self.params['a'].full.reshape(self.r, self.r)

    def getB(self):
        """'b' matrix from Astier+19 (e.g., Eq. 20)"""
        return self.params['c'].full.reshape(self.r, self.r)/self.getA()

    def getC(self):
        """'c'='ab' matrix from Astier+19 (e.g., Eq. 20)"""
        return self.params['c'].full.reshape(self.r, self.r)

    def _getCovParams(self,what):
        indices = self.params[what].indexof()
        #indicesp = [i for i in range(len(indices)) if indices[i]>=0 ]
        i1 = indices[:,np.newaxis]
        i2 = indices[np.newaxis, :]
        covp = self.covParams[i1,i2]
        return covp

    def getACov(self) :
        cova = self._getCovParams('a')
        return cova.reshape((self.r, self.r, self.r,self.r))

    def getASig(self) :
        cova = self._getCovParams('a')
        return np.sqrt(cova.diagonal()).reshape((self.r, self.r))

    def getBCov(self) :
        # b = c/a
        covb = self._getCovParams('c')
        aval = self.getA().flatten()
        factor = np.outer(aval,aval)
        covb /= factor
        return covb.reshape((self.r, self.r, self.r,self.r))

    def getCCov(self) :
        cova = self._getCovParams('c')
        return cova.reshape((self.r, self.r, self.r,self.r))

    def getGain(self) :
        return self.params['gain'].full[0]

    def getRon(self) :
        return self.params['noise'].full[0]

    def getNoise(self) :
        return self.params['noise'].full.reshape(self.r, self.r)

    def setAandB(self, a, b) :
        self.params['a'].full = a.flatten()
        self.params['c'].full = a.flatten()*b.flatten()

    def chi2(self) :
        return (self.weightedRes()**2).sum()

    def wres(self, params = None) :
        if params is not None:
            self.setParamValues(params)
        model = self.evalCovModel()
        return ((model-self.cov)*self.sqrtW)

    def weightedRes(self, params = None):
        """
        to be used via:
        c = covFit(nt)
        c.initFit()
        coeffs, cov, _, mesg, ierr = leastsq(c.weightedRes, c.getParamValues(), full_output=True )
        works nicely indeed.
        """
        return self.wres(params).flatten()

    def fit(self, p0 = None, nsig = 5) :
        if p0 is None:
            p0 = self.getParamValues()
        nOutliers = 1
        #while (nOutliers != 0) :
        for _  in [1,2]:
            coeffs, covParams, _, mesg, ierr = leastsq(self.weightedRes, p0, full_output=True)
            print ("coeffs: ", coeffs)
            wres = self.weightedRes(coeffs)
            # Do not count the outliers as significant
            sig = mad(wres[wres != 0])
            mask = (np.abs(wres) > (nsig * sig))
            self.sqrtW.flat[mask] = 0 #flatten makes a copy
            nOutliers = mask.sum()
            print(" dropped %d outliers (/%d)"%(nOutliers, len(mask)))

        if ierr not in [1,2,3,4] :
            print("minimisation failed ", mesg)
            raise RuntimeError(mesg)
        self.covParams = covParams
        return coeffs

    def ndof(self):
        """Number of degrees of freedom
        
        Returns
        -------
        mask.sum() - len(self.params.free) : `int`
            Number of usable pixels - number of parameters of fit.
        """
        mask = self.sqrtW != 0

        return mask.sum() - len(self.params.free)

    def getNormalizedFitData(self, i, j, divideByMu=False) :
        """Get measured signal and covariance, cov model and wigths
        
        Parameters
        ---------
        i : `int`
            Lag for covariance

        j : `int`
            Lag for covariance
   
        divideByMu : `bool`, optional
            Divide covariance, model, and weights by signal mu?

        Returns
        -------
        mu : `numpy.array`
            list of signal values (mu*gain).

        covariance : `numpy.array`
            Covariance arrays, indexed by mean signal mu (self.cov[:,i,j]*gain**2).
        
        model : `numpy.array` 
            Covariance model (model*gain**2)

        weights : `numpy.array`
            Weights (self.sqrtW/gain**2) 
        
        Notes
        -----
        Using a covFit object, selects from (i,j) and returns 
        mu*gain, self.cov[:,i,j]*gain**2 model*gain**2, and self.sqrtW/gain**2
        """
        gain = self.getGain()
        print ("GAIN: ", gain)
        mu = self.mu*gain
        covariance = self.cov[:,i,j]*(gain**2)
        model = self.evalCovModel()[:,i,j]*(gain**2)
        weights = self.sqrtW[:,i,j]/(gain**2)
        # select data used for the fit:
        mask = weights != 0
        weights = weights[mask]
        model = model[mask]
        print ("mu before mask: ", mu)
        mu = mu[mask]
        print ("mu after mask: ", mu)
        covariance = covariance[mask]
        if (divideByMu) :
            covariance /= mu
            model /= mu
            weights *= mu

        return mu , covariance , model, weights

    def __call__(self, params) :
        self.setParamValues(params)
        chi2 = self.chi2()
        #print('chi2 = ',chi2)
        
        return chi2


class WeightedRes:
    def __init__(self, model, x, y, sigma=None):
        self.x = x
        self.y = y
        self.model = model
        self.sigma= sigma

# could probably be more efficient (i.e. have two different functions)
    def __call__(self, params) :
        if self.sigma is None:
            res = self.y-self.model(self.x,*params)
        else :
            res = (self.y-self.model(self.x,*params))/self.sigma
        return res
