from __future__ import print_function
from builtins import zip
from builtins import str
from builtins import range
import math
import os
import re
import sys
import pickle
#from mpl_toolkits.mplot3d import Axes3D
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
pyplot = plt
import numpy as np
try:
    import scipy
    import scipy.interpolate
    from scipy.integrate import romb
    from scipy.interpolate import griddata
except ImportError:
    scipy = None

from scipy import stats
import lsst.daf.base as dafBase
import lsst.pex.config as pexConfig
import lsst.afw.cameraGeom as afwCG
import lsst.afw.cameraGeom.utils as afwCGUtils
import lsst.afw.detection as afwDet
import lsst.afw.geom as afwGeom
import lsst.afw.image as afwImage
import lsst.afw.math as afwMath
import lsst.afw.display.ds9 as ds9
import lsst.afw.display.utils as ds9Utils
import lsst.meas.algorithms as measAlg
import lsst.afw.table as afwTable
from lsst.daf.persistence import Butler
import ctypes
from lsst.obs.subaru.crosstalk import CrosstalkTask
from lsst.obs.subaru.isr import SubaruIsrTask
from hsc.pipe.base.butler import getDataRef
import random
import lsst.pex.exceptions as pexExcept


# This is code preforms some preliminary operations and then calls the main correlation calculation code. This is used for calculating the xcorr after setting the gains.
def xcorrFromVisit(butler, v1, v2, ccds=[1], n=5, border=10, plot=False, zmax=.04, fig=None, display=False, GAIN=None, sigma=5):
    """Return an xcorr from a given pair of visits (and ccds)"""

    try:
        v1[0]
    except TypeError:
        v1 = [v1]
    try:
        v2[0]
    except TypeError:
        v2 = [v2]

    try:
        ccds[0]
    except TypeError:
        ccds = [ccds]

    ims = [None, None]
    means = [None, None]
    for i, vs in enumerate([v1, v2, ]):
        for v in vs:
            for ccd in ccds:
                tmp = isr(butler, v, ccd)
                if ims[i] is None:
                    ims[i] = tmp
                    im = ims[i].getMaskedImage()
                else:
                    im += tmp.getMaskedImage()

        nData = len(ccds)*len(vs)
        if nData > 1:
            im /= nData
        means[i] = afwMath.makeStatistics(im, afwMath.MEANCLIP).getValue()
        if display:
            ds9.mtv(trim(ims[i]), frame=i, title=v)
    mean = np.mean(means)
    xcorrImg, means1 = xcorr(*ims, Visits=[v1, v2], n=n, border=border, frame=len(ims)
                             if display else None, CCD=[ccds[0]], GAIN=GAIN, sigma=sigma)

    if plot:
        plotXcorr(xcorrImg.clone(), (means1[0]+means[1]), title=r"Visits %s; %s, CCDs %s  $\langle{I}\rangle = %.3f$ (%s) Var = %.4f" %
                  (getNameOfSet(v1), getNameOfSet(v2), getNameOfSet(ccds),
                   (means1[0]+means[1]), ims[0].getFilter().getName(), float(xcorrImg.getArray()[0, 0])/(means1[0]+means[1])), zmax=zmax, fig=fig, SAVE=True, fileName="/home/wcoulton/HSC/Graphs/I/Correlation_Functions/Xcorr_visit_"+str(v1[0])+"_"+str(v2[0])+"_ccd_"+str(ccds[0])+".png")
    return xcorrImg, means1


#Some simple code to perform some simple ISR
def isr(butler, v, ccd):
    dataId = {'visit': v, 'ccd': ccd}
    dataRef = getDataRef(butler, dataId)
    config = SubaruIsrTask.ConfigClass()
   # config.load(os.path.join(os.environ["OBS_SUBARU_DIR"], "config", "isr.py"))
   # config.load(os.path.join(os.environ["OBS_SUBARU_DIR"], "config", "hsc", "isr.py"))

    config.doFlat = False
    config.doGuider = False
    config.doSaturation = True
    config.doWrite = False
    config.doDefect = True
    config.qa.doThumbnailOss = False
    config.qa.doThumbnailFlattened = False
    config.doFringe = False
    config.fringe.filters = ['y', ]
    config.overscanFitType = "AKIMA_SPLINE"
    config.overscanPolyOrder = 30
    config.doBias = True # Overscan is fairly efficient at removing bias level, but leaves a line in the middle
    config.doDark = True # Required especially around CCD 33
    config.crosstalk.retarget(CrosstalkTask)
    config.crosstalk.value.coeffs.values = [0.0e-6, -125.0e-6, -149.0e-6, -156.0e-6, -124.0e-6, 0.0e-6, -
                                            132.0e-6, -157.0e-6, -171.0e-6, -134.0e-6, 0.0e-6, -153.0e-6, -157.0e-6, -151.0e-6, -137.0e-6, 0.0e-6, ]
    isr = SubaruIsrTask(config=config)
    exp = isr.run(dataRef).exposure
    return exp


"""Calculate the cross-correlation of two images im1 and im2 (using robust measures of the covariance).
    This is designed to be called through xcorrFromVisit as that performs some simple ISR.
    Maximum lag is n, and ignore border pixels around the outside. Sigma is the number of sigma passed to sig cut.
    GAIN allows user specified GAINS to be used otherwise the default gains are used.
    The biasCorr parameter is used to correct from the bias of our measurements introduced by the sigma cuts. This was calculated using the sim. code at the bottom.
    This function returns one quater of the correlation function, the sum of the means of the two images and the individual means of the images
    """


def xcorr(im1, im2, Visits, n=5, border=20, frame=None, CCD=[1], GAIN=None, sigma=5, biasCorr=0.9241):

    sctrl = afwMath.StatisticsControl()
    sctrl.setNumSigmaClip(sigma)
    ims = [im1, im2]
    means = [None, None]
    means1 = [None, None]
    for i, im in enumerate(ims):
        ccd = afwCG.cast_Ccd(im.getDetector())
        try:
            frameId = int(re.sub(r"^SUPA0*", "", im.getMetadata().get("FRAMEID")))
        except:
            frameId = -1
        #
        # Starting with an Exposure, MaskedImage, or Image trim the data and convert to float
        #
        for attr in ("getMaskedImage", "getImage"):
            if hasattr(im, attr):
                im = getattr(im, attr)()
        try:
            im = im.convertF()
        except AttributeError:
            pass
       # im = trim(im, ccd)
        means[i] = afwMath.makeStatistics(im[border:-border, border:-border],
                                          afwMath.MEANCLIP, sctrl).getValue()
        temp = im.clone()
        # Rescale each amp by the appropriate gain and subtract the mean.
        for j, a in enumerate(ccd):
            smi = im[a.getDataSec(True)]
            smiTemp = temp[a.getDataSec(True)]
            mean = afwMath.makeStatistics(smi, afwMath.MEANCLIP, sctrl).getValue()
            if GAIN == None:
                gain = a.getElectronicParams().getGain()
            else:
                gain = GAIN[j]
           # gain/=gain
            smi *= gain
            print(mean*gain, afwMath.makeStatistics(smi, afwMath.MEANCLIP, sctrl).getValue())
            smi -= mean*gain
            smiTemp *= gain
        means1[i] = afwMath.makeStatistics(
            temp[border:-border, border:-border], afwMath.MEANCLIP, sctrl).getValue()
        print(afwMath.makeStatistics(temp[border:-border, border:-border], afwMath.MEANCLIP, sctrl).getValue())
    #    print(afwMath.makeStatistics(temp, afwMath.MEANCLIP,sctrl).getValue()-afwMath.makeStatistics(temp[0:-n,0:-n], afwMath.MEANCLIP,sctrl).getValue())
    im1, im2 = ims
    #
    # Actually diff the images
    #
    diff = ims[0].clone()
    diff = diff.getMaskedImage().getImage()
    diff -= ims[1].getMaskedImage().getImage()

    diff = diff[border:-border, border:-border]
   # diff.writeFits("./Data/Diff_CCD_"+str(CCD)+".fits")
   #
    # Subtract background.  It should be a constant, but it isn't always
    #
    binsize = 128
    nx = diff.getWidth()//binsize
    ny = diff.getHeight()//binsize
    bctrl = afwMath.BackgroundControl(nx, ny, sctrl, afwMath.MEANCLIP)
    bkgd = afwMath.makeBackground(diff, bctrl)
    diff -= bkgd.getImageF(afwMath.Interpolate.CUBIC_SPLINE, afwMath.REDUCE_INTERP_ORDER)
   # diff.writeFits("./Data/Diff_backsub_CCD_"+str(CCD)+".fits")
    if frame is not None:
        ds9.mtv(diff, frame=frame, title="diff")

    if False:
        global diffim
        diffim = diff
    if False:
        print(afwMath.makeStatistics(diff, afwMath.MEDIAN, sctrl).getValue())
        print(afwMath.makeStatistics(diff, afwMath.VARIANCECLIP, sctrl).getValue(), np.var(diff.getArray()))
    #
    # Measure the correlations
    #
    dim0 = diff[0: -n, : -n]
    dim0 -= afwMath.makeStatistics(dim0, afwMath.MEANCLIP, sctrl).getValue()
    w, h = dim0.getDimensions()
    xcorr = afwImage.ImageF(n + 1, n + 1)
    for di in range(n + 1):
        for dj in range(n + 1):
            dim_ij = diff[di:di + w, dj: dj + h].clone()
            dim_ij -= afwMath.makeStatistics(dim_ij, afwMath.MEANCLIP, sctrl).getValue()
            dim_ij *= dim0
            xcorr[di, dj] = afwMath.makeStatistics(dim_ij, afwMath.MEANCLIP, sctrl).getValue()/(biasCorr)
    L = np.shape(xcorr.getArray())[0]-1
    XCORR = np.zeros([2*L+1, 2*L+1])
    for i in range(L+1):
        for j in range(L+1):
            XCORR[i+L, j+L] = xcorr.getArray()[i, j]
            XCORR[-i+L, j+L] = xcorr.getArray()[i, j]
            XCORR[i+L, -j+L] = xcorr.getArray()[i, j]
            XCORR[-i+L, -j+L] = xcorr.getArray()[i, j]
    print(sum(means1), xcorr.getArray()[0, 0], np.sum(XCORR), xcorr.getArray()[0, 0]/sum(means1), np.sum(XCORR)/sum(means1))
    return (xcorr, means1)

#This program is used to plot the correlation functions


def plotXcorr(xcorr, mean, zmax=0.05, title=None, fig=None, SAVE=False, fileName=None):
    try:
        xcorr = xcorr.getArray()
    except:
        pass

    xcorr /= float(mean)
   # xcorr.getArray()[0,0]=abs(xcorr.getArray()[0,0]-1)

    if fig is None:
        fig = plt.figure()
    else:
        fig.clf()

    ax = fig.add_subplot(111, projection='3d')
    ax.azim = 30
    ax.elev = 20

    nx, ny = np.shape(xcorr)

    xpos, ypos = np.meshgrid(np.arange(nx), np.arange(ny))
    xpos = xpos.flatten()
    ypos = ypos.flatten()
    zpos = np.zeros(nx*ny)
    dz = xcorr.flatten()
    dz[dz > zmax] = zmax

    ax.bar3d(xpos, ypos, zpos, 1, 1, dz, color='b', zsort='max', sort_zpos=100)
    if xcorr[0, 0] > zmax:
        ax.bar3d([0], [0], [zmax], 1, 1, 1e-4, color='c')

    ax.set_xlabel("row")
    ax.set_ylabel("column")
    ax.set_zlabel(r"$\langle{(F_i - \bar{F})(F_i - \bar{F})}\rangle/\bar{F}$")

    if title:
        fig.suptitle(title)
    if SAVE == True:
        fig.savefig(fileName)
    #plt.close(fig)
    return fig, ax


def getNameOfSet(vals):
    """Convert a list of numbers into a string, merging consecutive values"""
    if not vals:
        return ""

    def addPairToName(valName, val0, val1):
        """Add a pair of values, val0 and val1, to the valName list"""
        sval1 = str(val1)
        if val0 != val1:
            pre = os.path.commonprefix([str(val0), sval1])
            sval1 = int(sval1[len(pre):])
        valName.append("%s-%s" % (val0, sval1) if val1 != val0 else str(val0))

    valName = []
    val0 = vals[0]
    val1 = val0
    for val in vals[1:]:
        if isinstance(val, int) and val == val1 + 1:
            val1 = val
        else:
            addPairToName(valName, val0, val1)
            val0 = val
            val1 = val0

    addPairToName(valName, val0, val1)

    return ", ".join(valName)


"""This is similiar to the xcorr function above except is used in the calculation of the amp gains.
    It is run on two visit numbers and the ccds that you are interested in. It will calculate the correlations in the individual amps without rescaling any gains. 
    From this you can generate a photon transfer curve and deduce the gain.
    This code runs some basic ISR on the images.
    Note that border pixels are discard only from the edge of the ccd and not from the boundary between amps.
    This returns the sum of the means, variance, one quarter of the xcorr and the original gain for each amp.
 """


def gainInvest(butler, v1, v2, ccds=[12], n=5, border=10, plot=False, zmax=.05, fig=None, display=False, sigma=5, biasCorr=0.9241):

    try:
        v1[0]
    except TypeError:
        v1 = [v1]
    try:
        v2[0]
    except TypeError:
        v2 = [v2]
    try:
        ccds[0]
    except TypeError:
        ccds = [ccds]
    ims = [None, None]
    means = [None, None]
    for i, vs in enumerate([v1, v2, ]):
        for v in vs:
            for ccd in ccds:
                tmp = isr(butler, v, ccd)
                if ims[i] is None:
                    ims[i] = tmp
                    im = ims[i].getMaskedImage()
                else:
                    im += tmp.getMaskedImage()

        nData = len(ccds)*len(vs)
        if nData > 1:
            im /= nData
        if display:
            ds9.mtv(trim(ims[i]), frame=i, title=v)

    means = [None, None]
    means1 = [[], []]
    gains = []
    sctrl = afwMath.StatisticsControl()
    sctrl.setNumSigmaClip(sigma)
    CCD = afwCG.cast_Ccd(ims[0].getDetector())
    for i, im in enumerate(ims):

        ccd = afwCG.cast_Ccd(im.getDetector())
        try:
            frameId = int(re.sub(r"^SUPA0*", "", im.getMetadata().get("FRAMEID")))
        except:
            frameId = -1
        #
        # Starting with an Exposure, MaskedImage, or Image trim the data and convert to float
        #
        for attr in ("getMaskedImage", "getImage"):
            if hasattr(im, attr):
                im = getattr(im, attr)()
        try:
            im = im.convertF()
        except AttributeError:
            pass
       # im = trim(im, ccd)
       # ims[i]=ims[i][border:-border,border:-border]
        means[i] = afwMath.makeStatistics(im, afwMath.MEANCLIP, sctrl).getValue()
        for j, a in enumerate(ccd):
            smi = im[a.getDataSec(True)]
            if j == 0:
                mean = afwMath.makeStatistics(smi[border:, border:-border], afwMath.MEANCLIP).getValue()
            elif j == 3:
                mean = afwMath.makeStatistics(smi[:-border, border:-border], afwMath.MEANCLIP).getValue()
            else:
                mean = afwMath.makeStatistics(smi[:, border:-border], afwMath.MEANCLIP).getValue()
            gain = a.getElectronicParams().getGain()
            means1[i].append(mean)
            if i == 0:
                gains.append(gain)
            #means1[i].append(mean*gain)
            #smi*=gain
            #smi-=mean*gain
            smi -= mean
    diff = ims[0].clone()
    diff = diff.getMaskedImage().getImage()
    diff -= ims[1].getMaskedImage().getImage()

    temp = diff[border:-border, border:-border]

    #
    # Subtract background.  It should be a constant, but it isn't always (e.g. some SuprimeCam flats)
    #
    binsize = 128
    nx = temp.getWidth()//binsize
    ny = temp.getHeight()//binsize
    bctrl = afwMath.BackgroundControl(nx, ny, sctrl, afwMath.MEANCLIP)
    bkgd = afwMath.makeBackground(temp, bctrl)
    diff[border:-border, border:-
         border] -= bkgd.getImageF(afwMath.Interpolate.CUBIC_SPLINE, afwMath.REDUCE_INTERP_ORDER)
    Var = []
    CorVar = []
    # For each amp calculate the correlation
    for i, a in enumerate(CCD):
        borderL = 0
        borderR = 0
        if i == 0:
            borderL = border
        if i == 3:
            borderR = border
        smi = diff[a.getDataSec(True)].clone()
       # dim0 = smi[border:-border-n,border:-border-n]
        dim0 = smi[borderL:-borderR-n, border:-border-n]
        dim0 -= afwMath.makeStatistics(dim0, afwMath.MEANCLIP, sctrl).getValue()
        w, h = dim0.getDimensions()
        xcorr = afwImage.ImageF(n + 1, n + 1)
        for di in range(n + 1):
            for dj in range(n + 1):
                dim_ij = smi[borderL+di:borderL+di + w, border+dj: border+dj + h].clone()
               # dim_ij = smi[border+di:border+di + w, border+dj: border+dj + h].clone()
                dim_ij -= afwMath.makeStatistics(dim_ij, afwMath.MEANCLIP, sctrl).getValue()
                dim_ij *= dim0
                xcorr[di, dj] = afwMath.makeStatistics(dim_ij, afwMath.MEANCLIP, sctrl).getValue()/(biasCorr)
        Var.append(xcorr.getArray()[0, 0])
        L = np.shape(xcorr.getArray())[0]-1
        XCORR = np.zeros([2*L+1, 2*L+1])
        for I in range(L+1):
            for J in range(L+1):
                XCORR[I+L, J+L] = xcorr.getArray()[I, J]
                XCORR[-I+L, J+L] = xcorr.getArray()[I, J]
                XCORR[I+L, -J+L] = xcorr.getArray()[I, J]
                XCORR[-I+L, -J+L] = xcorr.getArray()[I, J]
        CorVar.append(np.sum(XCORR))
        print(means1[0][i], means1[1][i], means1[0][i]+means1[1][i], Var[i], CorVar[i])
    return ([i+j for i, j in zip(means1[1], means1[0])], Var, CorVar, gains)


# For future, merge with the above function!!
""" This calculates the xcorr in the amps after correcting for the gain (either default or user supplied).
    This is useful for investigating the kernel in each amp. independently.
    It is run on two visit numbers and the ccds that you are interested in. It will calculate the correlations in the individual amps without rescaling any gains. 
    From this you can generate a photon transfer curve and deduce the gain.
    This code runs some basic ISR on the images.
    Note that border pixels are discard only from the edge of the ccd and not from the boundary between amps.
    This returns the sum of the means, variance, one quarter of the xcorr and the original gain for each amp.
 """


def ampCorrelation(butler, v1, v2, ccds=[12], n=5, border=20, plot=False, zmax=.05, fig=None, display=False, GAINS=None, sigma=5, biasCorr=0.9241):
    """Return an xcorr from a given pair of visits (and ccds)"""
    try:
        v1[0]
    except TypeError:
        v1 = [v1]
    try:
        v2[0]
    except TypeError:
        v2 = [v2]
    try:
        ccds[0]
    except TypeError:
        ccds = [ccds]
    ims = [None, None]
    means = [None, None]
    for i, vs in enumerate([v1, v2, ]):
        for v in vs:
            for ccd in ccds:
                tmp = isr(butler, v, ccd)
                if ims[i] is None:
                    ims[i] = tmp
                    im = ims[i].getMaskedImage()
                else:
                    im += tmp.getMaskedImage()

        nData = len(ccds)*len(vs)
        if nData > 1:
            im /= nData
        if display:
            ds9.mtv(trim(ims[i]), frame=i, title=v)

    means = [None, None]
    means1 = [[], []]
    sctrl = afwMath.StatisticsControl()
    sctrl.setNumSigmaClip(sigma)
    CCD = afwCG.cast_Ccd(ims[0].getDetector())
    for i, im in enumerate(ims):

        ccd = afwCG.cast_Ccd(im.getDetector())
        try:
            frameId = int(re.sub(r"^SUPA0*", "", im.getMetadata().get("FRAMEID")))
        except:
            frameId = -1
        #
        # Starting with an Exposure, MaskedImage, or Image trim the data and convert to float
        #
        for attr in ("getMaskedImage", "getImage"):
            if hasattr(im, attr):
                im = getattr(im, attr)()
        try:
            im = im.convertF()
        except AttributeError:
            pass
       # im = trim(im, ccd)
       # ims[i]=ims[i][border:-border,border:-border]
        means[i] = afwMath.makeStatistics(im, afwMath.MEANCLIP, sctrl).getValue()
        for j, a in enumerate(ccd):
            smi = im[a.getDataSec(True)]
            if j == 0:
                mean = afwMath.makeStatistics(smi[border:, border:-border], afwMath.MEANCLIP).getValue()
            elif j == 3:
                mean = afwMath.makeStatistics(smi[:-border, border:-border], afwMath.MEANCLIP).getValue()
            else:
                mean = afwMath.makeStatistics(smi[:, border:-border], afwMath.MEANCLIP).getValue()
            if GAINS is not None:
                gain = GAINS[0]
            else:
                gain = a.getElectronicParams().getGain()
            means1[i].append(mean*gain)
            smi *= gain
            smi -= mean*gain
    diff = ims[0].clone()
    diff = diff.getMaskedImage().getImage()
    diff -= ims[1].getMaskedImage().getImage()

    #
    # Subtract background.  It should be a constant, but it isn't always (e.g. some SuprimeCam flats)
    #
    binsize = 128
    nx = diff.getWidth()//binsize
    ny = diff.getHeight()//binsize
    bctrl = afwMath.BackgroundControl(nx, ny, sctrl, afwMath.MEANCLIP)
    bkgd = afwMath.makeBackground(diff, bctrl)
    diff -= bkgd.getImageF(afwMath.Interpolate.CUBIC_SPLINE, afwMath.REDUCE_INTERP_ORDER)
    Var = []
    CorVar = []
    for i, a in enumerate(CCD):
        borderL = 0
        borderR = 0
        if i == 0:
            borderL = border
        if i == 3:
            borderR = border
        smi = diff[a.getDataSec(True)].clone()
       # dim0 = smi[border:-border-n,border:-border-n]
        dim0 = smi[borderL:-borderR-n, border:-border-n]

        dim0 -= afwMath.makeStatistics(dim0, afwMath.MEANCLIP, sctrl).getValue()
        w, h = dim0.getDimensions()
        xcorr = afwImage.ImageF(n + 1, n + 1)
        for di in range(n + 1):
            for dj in range(n + 1):
                dim_ij = smi[borderL+di:borderL+di + w, border+dj: border+dj + h].clone()
               # dim_ij = smi[border+di:border+di + w, border+dj: border+dj + h].clone()
                dim_ij -= afwMath.makeStatistics(dim_ij, afwMath.MEANCLIP, sctrl).getValue()
                dim_ij *= dim0
                xcorr[di, dj] = afwMath.makeStatistics(dim_ij, afwMath.MEANCLIP, sctrl).getValue()/(biasCorr)
        Var.append(xcorr.getArray()[0, 0])
        CorVar.append(xcorr.getArray())
        L = np.shape(xcorr.getArray())[0]-1
        XCORR = np.zeros([2*L+1, 2*L+1])
        for I in range(L+1):
            for J in range(L+1):
                XCORR[I+L, J+L] = xcorr.getArray()[I, J]
                XCORR[-I+L, J+L] = xcorr.getArray()[I, J]
                XCORR[I+L, -J+L] = xcorr.getArray()[I, J]
        print(means1[0][i], means1[1][i], means1[0][i]+means1[1][i], Var[i], np.sum(XCORR))
    return (means1[0], means1[1], [i+j for i, j in zip(means1[1], means1[0])], Var, CorVar)


#A best fit method which removes outliers. Useful when you have sufficiently large numbers of points on your PTC

def iterativeRegression(x, y, intercept=0, sigma=3):
    iterate = 1
    sctrl = afwMath.StatisticsControl()
    sctrl.setNumSigmaClip(sigma)
    if intercept:
        while iterate:
            print(iterate, np.shape(x))
            A = np.vstack([x, np.ones(len(x))]).T
            B, _, _, _ = np.linalg.lstsq(A, y)
            slope, intercept = B
            res = y-slope*x-intercept
            resMean = afwMath.makeStatistics(res, afwMath.MEANCLIP, sctrl).getValue()
            resSTD = np.sqrt(afwMath.makeStatistics(res, afwMath.VARIANCECLIP, sctrl).getValue())
            index = np.where((res > (resMean+sigma*resSTD)) | (res < resMean-sigma*resSTD))
            print(resMean, resSTD, np.max(res), sigma)
            if np.shape(np.where(index))[1] == 0:
                break
            x = np.delete(x, index)
            y = np.delete(y, index)

        return slope, intercept
    while iterate:
        print(iterate, np.shape(x))
        TEST = x[:, np.newaxis]
        slope, _, _, _ = np.linalg.lstsq(TEST, y)
        slope = slope[0]
        res = y-slope*x
        resMean = afwMath.makeStatistics(res, afwMath.MEANCLIP, sctrl).getValue()
        resSTD = np.sqrt(afwMath.makeStatistics(res, afwMath.VARIANCECLIP, sctrl).getValue())
        index = np.where((res > (resMean+sigma*resSTD)) | (res < resMean-sigma*resSTD))
        print(resMean, resSTD, np.max(res), sigma)
        if np.shape(np.where(index))[1] == 0:
            break
        x = np.delete(x, index)
        y = np.delete(y, index)

    return slope


""" This function uses the above functions to measure the gains. Pass the desired ccd(s), a butler and a set of flats of varying intensity. 
    The intercept option chooses the linear fitting option. The default fits Var=1/g mean, if non zero Var=1/g mean + const is fit.
    If saveDic is false no writing is done. Elsewise the gains are added to a dictionary containing the gains of each ccd in outputFile.
"""


def gainEst(SelCCDS, butler, Visits, intercept=0, saveDic=0, outputFile='/home/wcoulton/HSC/Data/WILLS_GAINS.pkl', figLocation='/home/wcoulton/HSC/Graphs/', plot=1):
    oGAINS = {}
    GAINS = {}
    try:
        SelCCDS[0]
    except:
        SelCCDS = [SelCCDS]
    #This cycles through the input ccds
    for CCDS in SelCCDS:
        AmpMeans = []
        AmpVariance = []
        AmpCorrVariance = []
        AmpGain = []
        oGAINS[CCDS] = []
        # This cycles through the input visits and calculates the xcorr in the individual amps. No gain correction is applied
        for I, VISITS in enumerate(Visits[:]):
            a, b, c, d = gainInvest(butler, VISITS[0], VISITS[1], n=8, ccds=CCDS, plot=plot)
            breaker = 0
            #So sanity checks. If these are failed more investigation is needed!
            for i, j in enumerate(a):
                if a[i]*10 < b[i] or a[i]*10 < c[i]:
                    print('\n\n\n\n Check this visit! ', VISITS, ' \n\n\n\n')
                    breaker += 1
            if breaker:
                continue
            if I == 0:
                for i in range(len(a)):
                    AmpMeans.append(np.array([]))
                    AmpVariance.append(np.array([]))
                    AmpCorrVariance.append(np.array([]))
                    AmpGain.append(np.array([]))
            for i, j in enumerate(a):
                if I == 0:
                    oGAINS[CCDS].append(d[i])
                if b[i]*1.3 < c[i] or b[i]*0.7 > c[i]:
                    continue
                AmpMeans[i] = np.append(AmpMeans[i], a[i])
                AmpVariance[i] = np.append(AmpVariance[i], b[i])
                AmpCorrVariance[i] = np.append(AmpCorrVariance[i], c[i])
                AmpGain[i] = np.append(AmpGain[i], d[i])
        # Use the resulting means and xcorr to find the gain. There are two options,
        fig = None
        GAINS[CCDS] = []
        for i in range(len(AmpMeans)):
            if fig is None:
                fig = plt.figure()
            else:
                fig.clf()
            ax = fig.add_subplot(111)
            slope2, intercept, r_value, p_value, std_err = stats.linregress(AmpMeans[i], AmpCorrVariance[i])
            TEST = AmpMeans[i][:, np.newaxis]
            slope = iterativeRegression(AmpMeans[i], AmpCorrVariance[i])
            slope3, intercept2 = iterativeRegression(AmpMeans[i], AmpCorrVariance[i], intercept=1)
            print("\n\n\n\n slope of fit: ", slope2, "intercept of fit: ", intercept, 'p value', p_value)
            print(" slope of second fit: ", slope, 'difference:', slope-slope2)
            print(" slope of third fit: ", slope3, 'difference:', slope-slope3)
            if intercept:
                slope = slope3

            if plot:
                ax.plot(AmpMeans[i], AmpCorrVariance[i], linestyle='None', marker='x', label='data')
                if intercept:
                    ax.plot(AmpMeans[i], AmpMeans[i]*slope+intercept2, label='fix')

                else:
                    ax.plot(AmpMeans[i], AmpMeans[i]*slope, label='fix')
                fig.savefig(figLocation+'/PTC_CCD_'+str(CCDS)+'_AMP_'+str(i)+'.pdf')
                #plt.show()
            GAINS[CCDS].append(1.0/slope)

        if saveDic:
            try:
                FILE = open(outputFile, 'r+b')
                FILE.seek(0)
                try:
                    STOREDGAINS = pickle.load(FILE)
                except EOFError:
                    STOREDGAINS = {}
            except IOError:
                FILE = open(outputFile, 'w')
                STOREDGAINS = {}
            STOREDGAINS[CCDS] = GAINS[CCDS]
            FILE.seek(0)
            FILE.truncate()
            pickle.dump(STOREDGAINS, FILE)
            FILE.close()
    print('\n\n\n GAINS ', GAINS, '\n\n', oGAINS)
    return (GAINS, oGAINS)


# An implementation of the successive over relaxation method as described in press et al Numerical Recipes (2007) section 20.5.1. See Press for more details
def SOR(source, dx=1.0, MAXIT=10000, eLevel=5.0e-14):
    #initialise function: Done to zero here. Setting boundary conditions too!
    func = np.zeros([source.shape[0]+2, source.shape[1]+2])

    resid = np.zeros([source.shape[0]+2, source.shape[1]+2])
    rhoSpe = np.cos(np.pi/source.shape[0]) #Here a square grid is assummed

    inError = 0
    #Calculate the initial error
    for i in range(1, func.shape[0]-1):
        for j in range(1, func.shape[1]-1):
            resid[i, j] = func[i, j-1]+func[i, j+1]+func[i-1, j]+func[i+1, j]-4*func[i, j]-source[i-1, j-1]
            # inError+=abs(resid[i,j])
    inError = np.sum(np.abs(resid))
    COUNT = 0
    omega = 1.0
    #Iterate until convergence. We perform two sweeps per cycle, updating 'odd' and 'even' points separately.
    while COUNT < MAXIT*2:
        outError = 0
        if COUNT%2 == 0:
            for i in range(1, func.shape[0]-1, 2):
                for j in range(1, func.shape[0]-1, 2):
                    resid[i, j] = float(func[i, j-1]+func[i, j+1]+func[i-1, j] +
                                        func[i+1, j]-4.0*func[i, j]-dx*dx*source[i-1, j-1])
                    func[i, j] += omega*resid[i, j]*.25
                    # outError+=float(abs(resid[i,j]))
            for i in range(2, func.shape[0]-1, 2):
                for j in range(2, func.shape[0]-1, 2):
                    resid[i, j] = float(func[i, j-1]+func[i, j+1]+func[i-1, j] +
                                        func[i+1, j]-4.0*func[i, j]-dx*dx*source[i-1, j-1])
                    func[i, j] += omega*resid[i, j]*.25
                    # outError+=float(abs(resid[i,j]))
        else:
            for i in range(1, func.shape[0]-1, 2):
                for j in range(2, func.shape[0]-1, 2):
                    resid[i, j] = float(func[i, j-1]+func[i, j+1]+func[i-1, j] +
                                        func[i+1, j]-4.0*func[i, j]-dx*dx*source[i-1, j-1])
                    func[i, j] += omega*resid[i, j]*.25
                    # outError+=float(abs(resid[i,j]))
            for i in range(2, func.shape[0]-1, 2):
                for j in range(1, func.shape[0]-1, 2):
                    resid[i, j] = float(func[i, j-1]+func[i, j+1]+func[i-1, j] +
                                        func[i+1, j]-4.0*func[i, j]-dx*dx*source[i-1, j-1])
                    func[i, j] += omega*resid[i, j]*.25
                    # outError+=float(abs(resid[i,j]))
        outError = np.sum(np.abs(resid))
        if outError < inError*eLevel:
            break
        if COUNT == 0:
            omega = 1.0/(1-rhoSpe*rhoSpe/2.0)
        else:
            omega = 1.0/(1-rhoSpe*rhoSpe*omega/4.0)
        COUNT += 1
    if COUNT == MAXIT*2:
        print("Did not converge ", COUNT, outError, inError*eLevel)
    else:
        print("Converged in ", COUNT, outError, inError*eLevel)
    return func[1:-1, 1:-1]


""" This code uses a xcorr with gain correction to calculate the kernel
    corr is one quarter of the full xcorr., means is the means of the two individual images.
    MAXIT and eLevel are parameters for deciding when to end the SOR, either after a certain num of iterations or after the error has been reduced by a factor eLevel
    LEVEL is a sanity check parameter. If this condition is violated there is something unexpected going on in the image.
"""


def kernelGen(corr, means, LEVEL=.20, MAXIT=10000, eLevel=5.0e-14, sigma=4.0):
    try:
        # Try to average over a set of possible inputs. This generates a simple function of the kernel that should be constant across the images and averages that.
        counter = 0
        Isource = []
        sctrl = afwMath.StatisticsControl()
        sctrl.setNumSigmaClip(sigma)
        for I, (mean1, mean2) in enumerate(means):
            if isinstance(corr[I], afwImage.ImageF):
                CORR = corr[I].getArray().copy()
            else:
                CORR = corr[I].copy()
            CORR[0, 0] -= (mean1+mean2)
            if CORR[0, 0] > 0:
                print('Unexpected value of the variance -mean!!!')
                continue
            CORR /= -float(1.0*(mean1**2+mean2**2))
            #print CORR.shape[0]
            #assume square...
            L = CORR.shape[0]-1
            l = L
            TIsource = np.zeros([2*L+1, 2*L+1])
            for i in range(L+1):
                for j in range(L+1):
                    TIsource[i+L, j+L] = CORR[i, j]
                    TIsource[-i+L, j+L] = CORR[i, j]
                    TIsource[i+L, -j+L] = CORR[i, j]
                    TIsource[-i+L, -j+L] = CORR[i, j]
            if np.abs(np.sum(TIsource))/np.sum(np.abs(TIsource)) > LEVEL:
                print('Sum of the xcorr is unexpectedly high. Investigate item num ', I, np.abs(np.sum(TIsource))/np.sum(np.abs(TIsource)))
                continue
            #Isource+=TIsource
            Isource.append(TIsource)
            counter += 1
        #Isource/=float(counter)
        IS = Isource[0].copy()
        IS[:, :] = 0
        Isource = np.transpose(Isource)
        for i in range(np.shape(IS)[0]):
            for j in range(np.shape(IS)[1]):
                IS[i, j] = afwMath.makeStatistics(Isource[i, j], afwMath.MEANCLIP, sctrl).getValue()
        Isource = IS
    except TypeError:
        Isource = 0
        if isinstance(corr, afwImage.ImageF):
            CORR = corr.getArray()
        else:
            CORR = corr
        CORR[0, 0] -= (means[0]+means[1])
        if CORR[0, 0] > 0:
            print('Unexpected value of the variance -mean!!!')
            return 0
        CORR /= -float(1.0*(means[0]**2+means[1]**2))
        #print CORR.shape[0]
        #assume square...
        L = CORR.shape[0]-1
        l = L
        TIsource = np.zeros([2*L+1, 2*L+1])
        for i in range(L+1):
            for j in range(L+1):
                TIsource[i+L, j+L] = CORR[i, j]
                TIsource[-i+L, j+L] = CORR[i, j]
                TIsource[i+L, -j+L] = CORR[i, j]
                TIsource[-i+L, -j+L] = CORR[i, j]
        if np.abs(np.sum(TIsource))/np.sum(np.abs(TIsource)) > LEVEL:
            print('Sum of the xcorr is unexpectedly high. Investigate here ', np.abs(np.sum(TIsource))/np.sum(np.abs(TIsource)) > LEVEL)
            return 0
        Isource += TIsource

    return SOR(Isource, 1, MAXIT, eLevel)


# This sim code is used to estimate the bias correction used above.


""" This function performs a simple xcorr from two images. It contains many elements of the actual code above (without individual amps and ISR removal )
     It takes two images, im and im2; n the max lag of the correlation function; border, the number of border pixels to discard; and sigma the sigma to use in the mean clip """


def xcorr_sim(im, im2, n=8, border=10, sigma=5):
    sctrl = afwMath.StatisticsControl()
    sctrl.setNumSigmaClip(sigma)

    for attr in ("getMaskedImage", "getImage"):
        if hasattr(im, attr):
            im = getattr(im, attr)()
        if hasattr(im2, attr):
            im2 = getattr(im2, attr)()

    try:
        im = im.convertF()
        im2 = im2.convertF()
    except AttributeError:
        pass
    means1 = [0, 0]
    means1[0] = afwMath.makeStatistics(im[border:-border, border:-border], afwMath.MEANCLIP, sctrl).getValue()
    means1[1] = afwMath.makeStatistics(im2[border:-border, border:-border],
                                       afwMath.MEANCLIP, sctrl).getValue()
    im -= means1[0]
    im2 -= means1[1]
    diff = im2.clone()
    diff -= im.clone()
    diff = diff[border:-border, border:-border]
    binsize = 128
    nx = diff.getWidth()//binsize
    ny = diff.getHeight()//binsize
    bctrl = afwMath.BackgroundControl(nx, ny, sctrl, afwMath.MEANCLIP)
    bkgd = afwMath.makeBackground(diff, bctrl)
    diff -= bkgd.getImageF(afwMath.Interpolate.CUBIC_SPLINE, afwMath.REDUCE_INTERP_ORDER)
    dim0 = diff[0: -n, : -n].clone()
    dim0 -= afwMath.makeStatistics(dim0, afwMath.MEANCLIP, sctrl).getValue()
    w, h = dim0.getDimensions()
    xcorr = afwImage.ImageD(n + 1, n + 1)
    for di in range(n + 1):
        for dj in range(n + 1):
            dim_ij = diff[di:di + w, dj: dj + h].clone()
            dim_ij -= afwMath.makeStatistics(dim_ij, afwMath.MEANCLIP, sctrl).getValue()

            dim_ij *= dim0
            xcorr[di, dj] = afwMath.makeStatistics(dim_ij, afwMath.MEANCLIP, sctrl).getValue()
    L = np.shape(xcorr.getArray())[0]-1
    XCORR = np.zeros([2*L+1, 2*L+1])
    for i in range(L+1):
        for j in range(L+1):
            XCORR[i+L, j+L] = xcorr.getArray()[i, j]
            XCORR[-i+L, j+L] = xcorr.getArray()[i, j]
            XCORR[i+L, -j+L] = xcorr.getArray()[i, j]
            XCORR[-i+L, -j+L] = xcorr.getArray()[i, j]
   # print (means1),xcorr.getArray()[0,0],np.sum(XCORR),xcorr.getArray()[0,0]/(np.sum(means1)),np.sum(XCORR)/(np.sum(means1))
    return (XCORR, xcorr, np.sum(means1), means1)


"""
This function fills images of specified size (nx and ny) with poisson points with means (in rangeMeans) before passing it to the above function with border and sig as above
Repeats specifies the number of times to run the simulations. If case is 1 then a correlation between x_{i,j} and x_{i+1,j+1} is artificially introduces by adding a*x_{i,j} to x_{i+1,j+1}
If seed is left to None the seed with be pulled from /dev/random. Else an int can be passed to see the random number generator.
"""


def xcorr_bias(rangeMeans=[87500, 70000, 111000], repeats=5, sig=5, border=3, seed=None, nx=2000, ny=4000, case=0, a=.1):
    if seed is None:
        with open("/dev/random", 'rb') as file:
            local_random = np.random.RandomState(int(file.read(4).encode('hex'), 16))
    else:
        local_random = np.random.RandomState(int(seed))
    MEANS = {}
    XCORRS = {}
    for M in rangeMeans:
        MEANS[M] = []
        XCORRS[M] = []

    if not case:
        for rep in range(repeats):
            for i, MEAN in enumerate(rangeMeans):

                im = afwImage.ImageD(nx, ny)
                im0 = afwImage.ImageD(nx, ny)
                #im.getArray()[:,:]=local_random.normal(MEAN,np.sqrt(MEAN),(ny,nx))
                #im0.getArray()[:,:]=local_random.normal(MEAN,np.sqrt(MEAN),(ny,nx))
                im.getArray()[:, :] = local_random.poisson(MEAN, (ny, nx))
                im0.getArray()[:, :] = local_random.poisson(MEAN, (ny, nx))
                XCORR, xcorr, means, MEANS1 = xcorr_sim(im, im0, border=border, sigma=sig)
                MEANS[MEAN].append(means)
                XCORRS[MEAN].append(xcorr)
            print('\n\n\n')
            for i, MEAN in enumerate(rangeMeans):
                print("Simulated/Expected:", MEAN, MEANS[MEAN][-1], XCORRS[MEAN][-1].getArray()[0, 0]/MEANS[MEAN][-1])
    else:
        for rep in range(repeats):
            for i, MEAN in enumerate(rangeMeans):
                im = afwImage.ImageD(nx, ny)
                im0 = afwImage.ImageD(nx, ny)
                #im.getArray()[:,:]=local_random.normal(MEAN,np.sqrt(MEAN),(ny,nx))
                #im0.getArray()[:,:]=local_random.normal(MEAN,np.sqrt(MEAN),(ny,nx))
                im.getArray()[:, :] = local_random.poisson(MEAN, (ny, nx))
                im.getArray()[1:, 1:] += a*im.getArray()[:-1, :-1]
                im0.getArray()[:, :] = local_random.poisson(MEAN, (ny, nx))
                im0.getArray()[1:, 1:] += a*im0.getArray()[:-1, :-1]
                XCORR, xcorr, means, MEANS1 = xcorr_sim(im, im0, border=border, sigma=sig)
                MEANS[MEAN].append(means)
                XCORRS[MEAN].append(xcorr)
            print('\n\n\n')
            for i, MEAN in enumerate(rangeMeans):
                print("Simulated/Expected:", MEANS[MEAN][-1], '\n', (XCORRS[MEAN][-1].getArray()[1, 1]/MEANS[MEAN][-1]*(1+a))/.1)
    return MEANS, XCORRS
