import numpy as np
import matplotlib.pyplot as pl
import math

import scipy.interpolate as interp
from .astierCovPtcFit import covFit

import itertools
import pickle

#__all__ = ["cov_fft"]

def findMask(im, nsig, w=None) :
    if w is None:
        w = np.ones(im.shape)
    #  mu is used for sake of numerical precision in the sigma
    #  computation, and is needed (at full precision) to identify outliers.
    count = w.sum()
    # different from (w*im).mean()
    mu = (w*im).sum()/count
    # same comment for the variance.
    sigma = np.sqrt((((im-mu)*w)**2).sum()/count)
    for iter in range(3):
        outliers = np.where(np.abs((im-mu)*w) > nsig*sigma)
        w[outliers] = 0
        #  does not work :
        # count = im.size-ouliers[0].size
        count = w.sum()
        mu = (w*im).sum()/count
        newsig = np.sqrt((((im-mu)*w)**2).sum()/count)
        if (np.abs(sigma-newsig) < 0.02*sigma):
            sigma = newsig
            break
        sigma = newsig
    return w

def fftSize(s):
    x = int(np.log(s)/np.log(2.))
    return int(2**(x+1))

def saveFits(fits, fits_nob, filename) :
    with open(filename, 'wb') as f:
        pickle.dump(fits,f)
        pickle.dump(fits_nob,f)
    
def loadFits(filename):
    with open(filename, 'rb') as f:
        fits = pickle.load(f)
        fits_nb = pickle.load(f)
    return fits,fits_nb


class covFft :
    def __init__(self, diff, w, fftShape, maxRangeCov):
        """
        This class computed (via FFT), the nearby pixel correlation function.
        The range is controlled by maxRangeCov, as well as
        the actual FFT shape. It  assumes that w consists of 1's (good pix) and 0's (bad).
        
        Implements appendix of Astier+19.
        """

        # check that the zero padding implied by "fft_shape"
        # is large enough for the required correlation range 
        assert(fftShape[0] > diff.shape[0]+maxRangeCov+1)
        assert(fftShape[1] > diff.shape[1]+maxRangeCov+1)
        # for some reason related to numpy.fft.rfftn,
        # the second dimension should be even, so
        if fftShape[1]%2 == 1 :
            fftShape = (fftShape[0], fftShape[1]+1)
        tIm = np.fft.rfft2(diff*w, fftShape)
        tMask = np.fft.rfft2(w, fftShape)
        # sum of  "squares"
        self.pCov = np.fft.irfft2(tIm*tIm.conjugate())
        # sum of values
        self.pMean= np.fft.irfft2(tIm*tMask.conjugate())
        # number of w!=0 pixels.
        self.pCount= np.fft.irfft2(tMask*tMask.conjugate())

    def cov(self, dx,dy) :
        """Covariance for dx,dy averaged with dx,-dy if both non zero.
        
        Implements appendix of Astier+19.

        Parameters
        ----------
        dx : `int`
           Lag in x

        dy : `int
           Lag in y

        Returns
        -------
        0.5*(cov1+cov2) : `float`
            Covariance at (dx, dy) lag

        npix1+npix2 : `int`
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
            for dx in range(0, maxRange+1):
                cov, npix = self.cov(dx, dy)
                if (dx == 0 and dy == 0):
                    var = cov
                tupleVec.append((dx, dy, var, cov, npix))
        return tupleVec

def computeCovFft(diff, w, fftSize, maxRange):
    """Compute covariances via FFT
    
    Parameters
    ----------
    diff : `lsst.afw.image.exposure.exposure.ExposureF`
        Difference image from a pair of flats taken at the same exposure time.
    
    w : `numpy array`
        Mask array with 1's (good pixels) and 0's (bad pixels).
    
    fftSize : `tuple`
        Size of the DFT: (xSize, ySize)

    maxRange : `int`
        Maximum range of covariances

    Returns
    -------
    covFft.reportCovFft(maxRange) : `list`
        List with covariance tuples,
    """

    c = covFft(diff, w, fftSize, maxRange)
    return c.reportCovFft(maxRange)

def find_groups(x, maxdiff):
    """
    group data into bins, with at most maxdiff distance between bins.
    returns bin indices
    """
    ix = np.argsort(x)
    xsort = np.sort(x)
    index = np.zeros_like(x, dtype=np.int32)
    xc = xsort[0] 
    group = 0
    ng = 1
    for i in range(1,len(ix)) :
        xval = xsort[i]
        if (xval-xc < maxdiff) :
            xc = (ng*xc+xval)/(ng+1)
            ng += 1
            index[ix[i]] = group
        else :
            group+=1
            ng=1
            index[ix[i]] = group
            xc = xval
    return index

def index_for_bins(x, nbins) :
    """
    just builds an index with regular binning
    The result can be fed into bin_data
    """
    print ("Index for bin x: ", x)
    bins = np.linspace(x.min(), x.max() + abs(x.max() * 1e-7), nbins + 1)
    return np.digitize(x, bins)


def bin_data(x,y, bin_index, wy=None):
    """
    Bin data (usually for display purposes).
    x and y is the data to bin, bin_index should contain the bin number of each datum, and wy is the inverse of rms of each datum to use when averaging.
    (actual weight is wy**2)

    Returns 4 arrays : xbin (average x) , ybin (average y), wybin (computed from wy's in this bin), sybin (uncertainty on the bin average, considering actual scatter, ignoring weights) 
    """
    if wy is  None : wy = np.ones_like(x)
    bin_index_set = set(bin_index)
    w2 = wy*wy
    xw2 = x*(w2)
    xbin= np.array([xw2[bin_index == i].sum()/w2[bin_index == i].sum() for i in bin_index_set])
    yw2 = y*w2
    ybin= np.array([yw2[bin_index == i].sum()/w2[bin_index == i].sum() for i in bin_index_set])
    wybin = np.sqrt(np.array([w2[bin_index == i].sum() for i in bin_index_set]))
    # not sure about this one...
    #sybin= np.array([yw2[bin_index == i].std()/w2[bin_index == i].sum() for i in bin_index_set])
    sybin= np.array([y[bin_index == i].std()/np.sqrt(np.array([bin_index==i]).sum()) for i in bin_index_set])
    return xbin, ybin, wybin, sybin


class pol2d :
    def __init__(self, x,y,z,order, w=None):
        self.orderx = min(order,x.shape[0]-1)
        self.ordery = min(order,x.shape[1]-1)
        G = self.monomials(x.ravel(), y.ravel())
        if w is None:
            self.coeff,_,rank,_ = np.linalg.lstsq(G,z.ravel())
        else :
            self.coeff,_,rank,_ = np.linalg.lstsq((w.ravel()*G.T).T,z.ravel()*w.ravel())

    def monomials(self, x, y) :
        ncols = (self.orderx+1)*(self.ordery+1)
        G = np.zeros(x.shape + (ncols,))
        ij = itertools.product(range(self.orderx+1), range(self.ordery+1))
        for k, (i,j) in enumerate(ij):
            G[...,k] = x**i * y**j
        return G
            
    def eval(self, x, y) :
        G = self.monomials(x,y)
        return np.dot(G, self.coeff)


class load_params:
    """
    Prepare covariances for the PTC fit:
    - eliminate data beyond saturation
    - eliminate data beyond r (ignored in the fit
    - optionnaly (subtractDistantValue) subtract the extrapolation from distant covariances to closer ones, separately for each pair.
    - start: beyond which the modl is fitted
    - offset_degree: polynomila degree for the subtraction model
    """
    def __init__(self):
        self.r = 8
        self.maxMu = 2e5
        self.maxMuElectrons = 1e5
        self.subtractDistantValue = True
        self.start=12
        self.offset_degree = 1        

def load_data(tuple_name,params) :
    """
    Returns a list of covFits, indexed by amp number.
    tuple_name can be an actual tuple (rec array), rather than a file name containing a tuple.

    params drives what happens....  the class load_params provides default values
    params.r : max lag considered
    params.maxMu : maxMu in ADU's

    params.subtractDistantValue: boolean that says if one wants to subtract a background to the measured covariances (mandatory for HSC flat pairs).
    Then there are two more needed parameters: start, offset_degree

    """
    if (tuple_name.__class__ == str) :
        nt = np.load(tuple_name) 
    else :
        nt = tuple_name
    exts = np.array(np.unique(nt['ext']), dtype = int)
    covFitList = {}
    for ext in exts :
        print('extension=', ext)
        ntext = nt[nt['ext'] == ext]
        if params.subtractDistantValue :
            c = covFit(ntext,r=None)
            c.subtract_distant_offset(params.r, params.start, params.offset_degree)
        else :
            c = covFit(ntext, params.r)
        thisMaxMu = params.maxMu            
        # tune the maxMuElectrons cut
        for iter in range(3) : 
            cc = c.copy()
            cc.setMaxMu(thisMaxMu)
            cc.initFit()# allows to get a crude gain.
            gain = cc.getGain()
            if (thisMaxMu*gain < params.maxMuElectrons) :
                thisMaxMu = params.maxMuElectrons/gain
                continue
            cc.setMaxMuElectrons(params.maxMuElectrons)
            break
        covFitList[ext] = cc
    return covFitList

def fitData(tuple_name, maxMu = 1.4e5, maxMuElectrons = 1e5, r=8) :
    """
    The first argument can be a tuple, instead of the name of a tuple file.
    returns 2 dictionnaries, one of full fits, and one with b=0

    The behavior of this routine should be controlled by other means.
    """
    lparams = load_params()
    lparams.subtractDistantValue = False
    lparams.maxMu = maxMu
    lparams.maxMu = maxMuElectrons
    lparams.r = r
    covFitList = load_data(tuple_name, lparams)
    # exts = [i for i in range(len(covFitList)) if covFitList[i] is not None]
    alist = []
    blist = []
    covFitNoBList = {} # [None]*(exts[-1]+1)
    for ext,c in covFitList.items() :
        print('fitting channel %d'%ext)
        print ("c: ", c)
        c.fit()
        print ("after c.fit()")
        covFitNoBList[ext] = c.copy()
        c.params['c'].release()
        c.fit()
        a = c.getA()
        alist.append(a)
        print(a[0:3, 0:3])
        b = c.getB()
        blist.append(b)
        print(b[0:3, 0:3])
    a = np.asarray(alist)
    b = np.asarray(blist)
    for i in range(2):
        for j in range(2) :
            print(i,j,'a = %g +/- %g'%(a[:,i,j].mean(), a[:,i,j].std()),
                  'b = %g +/- %g'%(b[:,i,j].mean(), b[:,i,j].std()))
    return covFitList, covFitNoBList


# subtract the "long distance" offset from measured covariances



def CHI2(res,wy):
    wres = res*wy
    return (wres*wres).sum()
    

    
# pass fixed arguments using curve_fit:    
# https://stackoverflow.com/questions/10250461/passing-additional-arguments-using-scipy-optimize-curve-fit



def select_from_tuple(t, i, j, ext):
    cut = (t['i'] == i) & (t['j'] == j) & (t['ext'] == ext)
    return t[cut]


def eval_nonlin(tuple, knots = 20, verbose = False, fullOutput=False):
    """
    it will be faster if the tuple only contains the variances
    return value: a dictionnary of correction spline functions (one per amp)
    """
    amps = np.unique(tuple['ext'].astype(int))
    res={}
    if fullOutput:
        x = {}
        y = {}
    for i in amps :
        t = tuple[tuple['ext'] == i]
        clap = t['c1']
        mu = t['mu1']
        if fullOutput :
            res[i], x[i], y[i] = fit_nonlin_corr(mu,clap, knots=knots, verbose=verbose, fullOutput=fullOutput)
        else :
            res[i] = fit_nonlin_corr(mu,clap, knots=knots, verbose=verbose, fullOutput=fullOutput)
    if fullOutput:
        return res,x,y
    else :
        return res

    
# I don't know who wrote it...
def mad(data, axis=0, scale=1.4826):
    """
    Median of absolute deviation along a given axis.  

    Normalized to match the definition of the sigma for a gaussian
    distribution.
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
    return scale * ret



#    mcc = interp.splev(x, s) # model values
#    dd = interp.splder(s)  # model derivative
#    der = interp.splev(x,dd) # model derivative values

def fit_nonlin_corr(xin, yclapin, knots = 20, loop = 20, verbose = False, fullOutput=False):
    """
    xin : the data to be "linearized"
    yclapin : the (hopefully) linear reference
    returns a  spline that can be used for correction uisng "scikit.splev"
    if full_output==True, returns spline,x,y  (presumably to plot)
    """
    # do we need outlier rejection ?
    # the xin has to be sorted, although the doc does not say it....
    index = xin.argsort()
    x = xin[index]
    yclap = yclapin[index]
    chi2_mask = np.isfinite(yclap) # yclap = nan kills the whole thing
    xx = x
    yyclap = yclap
    for i in range(loop):
        xx = xx[chi2_mask]
        # first fit the scaled difference between the two channels we are comparing
        yyclap = yyclap[chi2_mask]
        length = xx[-1]-xx[0]
        t = np.linspace(xx[0]+1e-5*length, xx[-1]-1e-5*length, knots)
        s = interp.splrep(xx, yyclap, task=-1, t=t)        
        model = interp.splev(xx, s)     # model values
        res = model - yyclap
        sig = mad(res)
        res = np.abs(res)
        if (res> (5 * sig)).sum()>0 : # remove one at a time
            chi2_mask = np.ones(len(xx)).astype(bool)
            chi2_mask[np.argmax(res)] = False
            continue
        else : break
    # normalize so that the gain does not change too much
    yyclap_norm = yyclap * xx.mean()/yyclap.mean()
    s = interp.splrep(xx, yyclap_norm, task=-1, t=t[1:-2])
    model = interp.splev(xx, s)     # model values
    # compute gain residuals
    print("nonlin gain residuals : %g"%(model/yyclap_norm-1).std())
    if verbose :     
        print("fit_nonlin loops=%d sig=%f res.max = %f"%(i,sig, res.max()))
    if fullOutput :
        return s, xx, yyclap_norm
    return s

def correct_tuple_for_nonlin(tuple, nonlin_corr=None, verbose=False, draw = False):
    """
    Compute the non-linearity correction  using the
    'c1' field, and applies it.
    Return value: corrected tuple
    """
    t00 = tuple[(tuple['i'] == 0)  & (tuple['j'] == 0)]
    if nonlin_corr == None :
        if draw :
            nonlin_corr = eval_nonlin_draw(t00, verbose=verbose)
        else :
            nonlin_corr = eval_nonlin(t00, verbose=verbose)
    amps = np.unique(t00['ext']).astype(int)
    # sort the tuple by amp, i.e. extension
    index = tuple['ext'].argsort()
    stuple=tuple[index]
    tuple = stuple # release memory ? not sure
    # find out where each amp starts and ends in the tuple
    ext = stuple['ext']
    diff = ext[1:] - ext[:-1]
    boundaries = [0]+ [ i+1 for i in range(len(diff)) if diff[i]!=0]+[len(ext)]
    amps = np.unique(ext).astype(int)
    start = [None]*int(amps.max()+1)
    end = [None]*int(amps.max()+1)
    for k in range(len(boundaries)-1):
        b = boundaries[k]
        amp = int(ext[b])
        start[amp]= b
        end[amp] = boundaries[k+1]
        # print amp,start[amp], end[amp], ext[start[amp]],ext[end[amp]-1]
    # now applyr the nonlinearity correction        
    for amp in amps:
        tamp = stuple[stuple['ext'] == amp]
        x = 0.5*(tamp['mu1'] + tamp['mu2'])
        iamp = int(amp)
        s = nonlin_corr[iamp]
        mu_corr = interp.splev(x, s) # model values
        dd = interp.splder(s)  # model derivative
        der = interp.splev(x,dd) # model derivative values
        stuple['mu1'][start[iamp]:end[iamp]] = mu_corr
        stuple['mu2'][start[iamp]:end[iamp]] = mu_corr
        stuple['var'][start[iamp]:end[iamp]] *= (der**2)
        stuple['cov'][start[iamp]:end[iamp]] *= (der**2)
    return stuple

#    return mcc, cvc, pix

def apply_quality_cuts(nt0, saturationAdu = 1.35e5, sigPedestal = 3):
    """
    dispersion of the pedestal and saturation
    """
    cut = (nt0['sp1']<sigPedestal)  & (nt0['sp2']<sigPedestal) & (nt0['mu1']<saturationAdu)
    
    return nt0[cut]


import astropy.io.fits as pf

def dump_a_fits(fits) :
    a = np.array([f.getA() for f in fits.values()]).mean(axis=0)
    siga = np.array([f.getASig() for f in fits.values()]).mean(axis=0)
    pf.writeto('a.fits', a, overwrite=True)
    pf.writeto('siga.fits', siga, overwrite=True)
    
