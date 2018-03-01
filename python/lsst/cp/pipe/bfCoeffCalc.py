#
# LSST Data Management System
#
# Copyright 2008-2017  AURA/LSST.
#
# This product includes software developed by the
# LSST Project (http://www.lsst.org/).
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
# You should have received a copy of the LSST License Statement and
# the GNU General Public License along with this program.  If not,
# see <https://www.lsstcorp.org/LegalNotices/>.
#

"""Calculation of brighter-fatter effect correlations and kernels."""
from __future__ import print_function

from builtins import zip
from builtins import str
from builtins import range
import os
import re
import pickle
from scipy import stats
import matplotlib as mpl
import numpy as np
# mpl.use('Agg')
# pyplot = plt
import matplotlib.pyplot as plt

import lsst.afw.image as afwImage
import lsst.afw.math as afwMath
import lsst.afw.display.ds9 as ds9
import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase
# import lsst.log as lsstLog
from lsst.obs.subaru.crosstalk import CrosstalkTask
from lsst.obs.subaru.isr import SubaruIsrTask
import lsstDebug
debug = lsstDebug.Info(__name__)

# try:
#     import scipy
#     import scipy.interpolate
# except ImportError:
#     scipy = None

OUTPUT_PATH = '/home/mfl/bf_testing/'


class BfTaskConfig(pexConfig.Config):
    """Config class for bright-fatter effect coefficient calculation."""

    maxIter = pexConfig.Field(
        dtype=int,
        doc="Maximum number of iterations... xxx",
        default=1000
    )
    xcorrCheckRejectLevel = pexConfig.Field(
        dtype=float,
        doc="Sanity check level for the sum of the input cross-correlations. Arrays which "
        "sum to greater than this are discarded before the clipped mean is calculated.",
        default=0.2
    )
    maxIterSOR = pexConfig.Field(
        dtype=int,
        doc="The maximum number of iterations allowed for the successive over-relaxation method",
        default=10000
    )
    eLevelSOR = pexConfig.Field(
        dtype=float,
        doc="The target residual error for the successive over-relaxation method",
        default=5.0e-14
    )
    kernelGenSigmaClip = pexConfig.Field(
        dtype=float,
        doc="Number of sigma to clip to, during pixel-wise clipping when generating the kernel",
        default=4
    )


class BfTask(pipeBase.CmdLineTask):
    """Bright-fatter effect coefficient calculation task.

    See http://ls.st/ldm-151 Chapter 4, Calibration Products Production for further details
    regarding the inputs and outputs.
    """

    ConfigClass = BfTaskConfig
    _DefaultName = "bf"

    def __init__(self, *args, **kwargs):
        """Constructor for the BfTask."""
        pipeBase.CmdLineTask.__init__(self, *args, **kwargs)

        self.config.validate()
        self.config.freeze()

    def run(self, dummy):
        """Docstring."""
        # self.xxx_test_estimateGains()
        self.xxx_test_generateKernel()

    def xxx_test_estimateGains(self):
        """Docstring."""
        import lsst.daf.persistence as dafPersist
        butler = dafPersist.Butler('/datasets/hsc/repo/')
        # visPairs = [(904606, 904608),
        #             (904610, 904612)]
        visPairs = [(904606, 904608),
                    (904610, 904612),
                    (904618, 904620),
                    (904614, 904616),
                    (904590, 904592),
                    (904602, 904604),
                    (904594, 904596),
                    (904588, 904590),
                    (904584, 904586),
                    (904626, 904628),
                    (904622, 904624),
                    (904630, 904632),
                    (904646, 904648),
                    (904642, 904644),
                    (904660, 904664),
                    (904662, 904662)]
        ignoreCcdList = [_ for _ in range(112)]
        ignoreCcdList.remove(41)
        self.estimateGains(butler, visPairs, ignoreCcdList)

    def xxx_test_generateKernel(self):
        """Docstring."""
        import pickle
        f = open('/home/mfl/bf_output/merlinTestXcorr.pkl', 'rb')
        xcorr, means = pickle.load(f)
        f.close()
        print('\n\n Level = %s\n\n'%self.config.xcorrCheckRejectLevel)
        kernel = self._generateKernel(xcorr, means)
        f = open('/home/mfl/bf_output/taskOutput_kernel.pkl', 'wb')
        pickle.dump(kernel, f)
        f.close()

    def xcorrFromVisit(self, butler, v1, v2, ccds=[1], n=5, border=10, plot=False,
                       zmax=.04, fig=None, display=False, GAIN=None, sigma=5):
        """Return an xcorr from a given pair of visits (and ccds).

        This is code preforms some preliminary operations and then calls the main correlation calc code.
        This is used for calculating the xcorr after setting the gains.
        """
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
            plotXcorr(xcorrImg.clone(), (means1[0]+means[1]), title=r"Visits %s; %s, CCDs %s  $\langle{I}\rangle"
                      " = %.3f$ (%s) Var = %.4f" %
                      (_getNameOfSet(v1), _getNameOfSet(v2), _getNameOfSet(ccds),
                       (means1[0]+means[1]), ims[0].getFilter().getName(), float(xcorrImg.getArray()[0, 0]) /
                       (means1[0]+means[1])), zmax=zmax, fig=fig, SAVE=True,
                      fileName=(os.path.join(OUTPUT_PATH, ("Xcorr_visit_" + str(v1[0])+"_"+str(v2[0])+"_ccd_" +
                                                           str(ccds[0])+".png"))))
        return xcorrImg, means1

    def isr(self, butler, v, ccd):  # TODO: Need to replace this with a retargetable ISR task
        """Some simple code to perform some simple ISR."""
        dataId = {'visit': v, 'ccd': ccd}
        dataRef = butler.dataRef('raw', dataId=dataId)
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
        config.overscanOrder = 30
        # Overscan is fairly efficient at removing bias level, but leaves a line in the middle
        config.doBias = True
        config.doDark = True  # Required especially around CCD 33
        config.crosstalk.retarget(CrosstalkTask)
        config.crosstalk.value.coeffs.values = [0.0e-6, -125.0e-6, -149.0e-6, -156.0e-6, -124.0e-6, 0.0e-6, -
                                                132.0e-6, -157.0e-6, -171.0e-6, -134.0e-6, 0.0e-6, -153.0e-6,
                                                -157.0e-6, -151.0e-6, -137.0e-6, 0.0e-6, ]
        isr = SubaruIsrTask(config=config)
        exp = isr.run(dataRef).exposure
        return exp

    def xcorr(self, im1, im2, Visits, n=5, border=20, frame=None, CCD=[1], GAIN=None, sigma=5, biasCorr=0.9241):
        """Calculate the cross-correlation of two images im1 and im2 (using robust measures of the covariance).

        This is designed to be called through xcorrFromVisit as that performs some simple ISR.
        Maximum lag is n, and ignore border pixels around the outside. Sigma is the number of sigma passed
        to sig cut.
        GAIN allows user specified GAINS to be used otherwise the default gains are used.
        The biasCorr parameter is used to correct from the bias of our measurements introduced by the sigma cuts.
        This was calculated using the sim. code at the bottom.
        This function returns one quater of the correlation function, the sum of the means of the two images and
        the individual means of the images
        """
        sctrl = afwMath.StatisticsControl()
        sctrl.setNumSigmaClip(sigma)
        ims = [im1, im2]
        means = [None, None]
        means1 = [None, None]
        for i, im in enumerate(ims):
            # ccd = afwCG.cast_Ccd(im.getDetector())
            ccd = im.getDetector()
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
                # smi = im[a.getDataSec(True)]
                # smiTemp = temp[a.getDataSec(True)]
                smi = im[a.getBBox()]
                smiTemp = temp[a.getBBox()]
                mean = afwMath.makeStatistics(smi, afwMath.MEANCLIP, sctrl).getValue()
                if GAIN == None:
                    # gain = a.getElectronicParams().getGain()
                    gain = a.getGain()
                else:
                    gain = GAIN[j]
                # gain/=gain
                smi *= gain
                print(mean*gain, afwMath.makeStatistics(smi, afwMath.MEANCLIP, sctrl).getValue())
                smi -= mean*gain
                smiTemp *= gain
            means1[i] = afwMath.makeStatistics(
                temp[border:-border, border:-border], afwMath.MEANCLIP, sctrl).getValue()
            print(afwMath.makeStatistics(temp[border:-border, border:-border],
                                         afwMath.MEANCLIP, sctrl).getValue())
        #    print(afwMath.makeStatistics(temp, afwMath.MEANCLIP,sctrl).getValue()-
        #          afwMath.makeStatistics(temp[0:-n,0:-n], afwMath.MEANCLIP,sctrl).getValue())
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
        print(sum(means1), xcorr.getArray()[0, 0], np.sum(XCORR), xcorr.getArray()[0, 0]/sum(means1),
              np.sum(XCORR)/sum(means1))
        return (xcorr, means1)

    def plotXcorr(self, xcorr, mean, zmax=0.05, title=None, fig=None, SAVE=False, fileName=None):
        """This program is used to plot the correlation functions."""
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

    @staticmethod
    def _getNameOfSet(vals):
        """Convert a list of numbers into a string, merging consecutive values."""
        if not vals:
            return ""

        def _addPairToName(valName, val0, val1):
            """Add a pair of values, val0 and val1, to the valName list."""
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
                _addPairToName(valName, val0, val1)
                val0 = val
                val1 = val0

        _addPairToName(valName, val0, val1)

        return ", ".join(valName)

    def _calcMeansAndVars(self, butler, v1, v2, ccd, n=5, border=10, plot=False, zmax=.05,
                          fig=None, display=False, sigma=5, biasCorr=0.9241):
        """Calculate the means, vars, covars, and retieve the nominal gains, for each amp in each ccd.

        This code runs using two visit numbers, and for ccd specified.
        It calculates the correlations in the individual amps without rescaling any gains.
        This allows a photon transfer curve to be generated and the gains measured.

        Images are assembled with use the isrTask, and basic isr is performed.
        Note that the isr task used MUST set the EDGE bits.[xxx need to change to using this, or change this]

        Parameters:
        -----------
        butler : `lsst.daf.persistence.butler`
            Butler for the repo containg the flats to be used
        v1 : `int`
            First visit of the visit pair
        v2 : `int`
            Second visit of the visit pair
        ccd : `string` or `int`
            Names of the ccds to use

        Returns
        -------
        means, vars, covars, gains : `tuple` of `lists`
            The sum of the means, variance, one quarter of the xcorr, and the original gain for each amp.
        """
        nomGains = []
        imMeans = [None, None]
        ampMeans = [[], []]

        # TODO_URGENT: turn this into a dict so that we don't get muddled up. Currently this is nonsense.
        # TODO: change to looping over ccds so that we don't hold all the isr-ed images
        ims = [self.isr(butler, v1, ccd), self.isr(butler, v2, ccd)]
        # if display:  # TODO: replace with lsstDebug
        #     ds9.mtv(trim(ims[i]), frame=i, title=v)

        sctrl = afwMath.StatisticsControl()
        sctrl.setNumSigmaClip(sigma)  # TODO: change to pexConfig option
        for imNum, im in enumerate(ims):
            ccd = im.getDetector()
            # Starting with an Exposure, MaskedImage, or Image trim the data and convert to float
            for attr in ("getMaskedImage", "getImage"):
                if hasattr(im, attr):
                    im = getattr(im, attr)()
            try:
                im = im.convertF()
            except AttributeError:
                self.log.warn("Failed to convert image %s to float"%imNum)  # xxx fatal? Raise?
                pass

            # calculate the sigma-clipped mean, excluding the borders
            # TODO: rewrite to use egde bits
            imMeans[imNum] = afwMath.makeStatistics(im, afwMath.MEANCLIP, sctrl).getValue()
            for ampNum, amp in enumerate(ccd):
                ampIm = im[amp.getBBox()]
                if ampNum == 0:
                    mean = afwMath.makeStatistics(ampIm[border:, border:-border],
                                                  afwMath.MEANCLIP).getValue()
                elif ampNum == 3:
                    mean = afwMath.makeStatistics(ampIm[:-border, border:-border],
                                                  afwMath.MEANCLIP).getValue()
                else:
                    mean = afwMath.makeStatistics(ampIm[:, border:-border], afwMath.MEANCLIP).getValue()
                nomGain = amp.getGain()
                ampMeans[imNum].append(mean)
                if imNum == 0:
                    nomGains.append(nomGain)
                ampIm -= mean

        diff = ims[0].clone()
        diff = diff.getMaskedImage().getImage()
        diff -= ims[1].getMaskedImage().getImage()

        temp = diff[border:-border, border:-border]

        # Subtract background.  It should be a constant, but it isn't always (e.g. some SuprimeCam flats)
        # TODO: Check how this looks, and if this is the "right" way to do this
        binsize = 128  # TODO: change to pexConfig option
        nx = temp.getWidth()//binsize
        ny = temp.getHeight()//binsize
        bctrl = afwMath.BackgroundControl(nx, ny, sctrl, afwMath.MEANCLIP)
        bkgd = afwMath.makeBackground(temp, bctrl)
        diff[border:-border, border:-border] -= bkgd.getImageF(afwMath.Interpolate.CUBIC_SPLINE,
                                                               afwMath.REDUCE_INTERP_ORDER)
        variances = []  # can't shadow builtin "vars"
        coVars = []
        # For each amp calculate the correlation
        CCD = ims[0].getDetector()  # xxx can you do this for a heterogenous focal plane? (answer: 100% no)
        for ampNum, amp in enumerate(CCD):
            borderL = 0
            borderR = 0
            if ampNum == 0:  # TODO: this needs rewriting for using edge bits to make camera agnostic
                borderL = border
            if ampNum == 3:
                borderR = border

            diffAmpIm = diff[amp.getBBox()].clone()  # xxx why is this a clone? move .clone() to next line?
            diffAmpImCrop = diffAmpIm[borderL:-borderR-n, border:-border-n]
            diffAmpImCrop -= afwMath.makeStatistics(diffAmpImCrop, afwMath.MEANCLIP, sctrl).getValue()
            w, h = diffAmpImCrop.getDimensions()
            xcorr = np.zeros((n + 1, n + 1), dtype=np.float64)

            # calculate the cross correlation
            for xlag in range(n + 1):
                for ylag in range(n + 1):
                    dim_xy = diffAmpIm[borderL+xlag:borderL+xlag + w, border+ylag: border+ylag + h].clone()
                    dim_xy -= afwMath.makeStatistics(dim_xy, afwMath.MEANCLIP, sctrl).getValue()
                    dim_xy *= diffAmpImCrop
                    xcorr[xlag, ylag] = afwMath.makeStatistics(dim_xy,
                                                               afwMath.MEANCLIP, sctrl).getValue()/(biasCorr)

            variances.append(xcorr[0, 0])
            xcorr_full = self._tileArray(xcorr)
            coVars.append(np.sum(xcorr_full))

            msg = "M1: " + str(ampMeans[0][ampNum])
            msg += " M2 " + str(ampMeans[1][ampNum])
            msg += " M_sum: " + str((ampMeans[0][ampNum])+ampMeans[1][ampNum])
            msg += " Var " + str(variances[ampNum])
            msg += " coVar: " + str(coVars[ampNum])
            self.log.info(msg)  # xxx change to debug or trace level
        return ([i+j for i, j in zip(ampMeans[1], ampMeans[0])], variances, coVars, nomGains)

    # For future, merge with the above function!!
    """ This calculates the xcorr in the amps after correcting for the gain (either default or user supplied).
        This is useful for investigating the kernel in each amp. independently.
        It is run on two visit numbers and the ccds that you are interested in. It will calculate the
        correlations in the individual amps without rescaling any gains.
        From this you can generate a photon transfer curve and deduce the gain.
        This code runs some basic ISR on the images.
        Note that border pixels are discard only from the edge of the ccd and not from the boundary between amps.
        This returns the sum of the means, variance, one quarter of the xcorr and the original gain for each amp.
     """
    # def ampCorrelation(self, butler, v1, v2, ccds=[12], n=5, border=20, plot=False, zmax=.05,
    #                    fig=None, display=False, GAINS=None, sigma=5, biasCorr=0.9241):
    #     """Return an xcorr from a given pair of visits (and ccds)."""
    #     try:
    #         v1[0]
    #     except TypeError:
    #         v1 = [v1]
    #     try:
    #         v2[0]
    #     except TypeError:
    #         v2 = [v2]
    #     try:
    #         ccds[0]
    #     except TypeError:
    #         ccds = [ccds]
    #     ims = [None, None]
    #     means = [None, None]
    #     for i, vs in enumerate([v1, v2, ]):
    #         for v in vs:
    #             for ccd in ccds:
    #                 tmp = isr(butler, v, ccd)
    #                 if ims[i] is None:
    #                     ims[i] = tmp
    #                     im = ims[i].getMaskedImage()
    #                 else:
    #                     im += tmp.getMaskedImage()

    #         nData = len(ccds)*len(vs)
    #         if nData > 1:
    #             im /= nData
    #         if display:
    #             ds9.mtv(trim(ims[i]), frame=i, title=v)

    #     means = [None, None]
    #     means1 = [[], []]
    #     sctrl = afwMath.StatisticsControl()
    #     sctrl.setNumSigmaClip(sigma)
    #     # CCD = afwCG.cast_Ccd(ims[0].getDetector())
    #     CCD = ims[0].getDetector()
    #     for i, im in enumerate(ims):

    #         # ccd = afwCG.cast_Ccd(im.getDetector())
    #         ccd = im.getDetector()
    #         try:
    #             frameId = int(re.sub(r"^SUPA0*", "", im.getMetadata().get("FRAMEID")))
    #         except:
    #             frameId = -1
    #         #
    #         # Starting with an Exposure, MaskedImage, or Image trim the data and convert to float
    #         #
    #         for attr in ("getMaskedImage", "getImage"):
    #             if hasattr(im, attr):
    #                 im = getattr(im, attr)()
    #         try:
    #             im = im.convertF()
    #         except AttributeError:
    #             pass
    #         # im = trim(im, ccd)
    #         # ims[i]=ims[i][border:-border,border:-border]
    #         means[i] = afwMath.makeStatistics(im, afwMath.MEANCLIP, sctrl).getValue()
    #         for j, a in enumerate(ccd):
    #             # smi = im[a.getDataSec(True)]
    #             smi = im[a.getBBox()]
    #             if j == 0:
    #                 mean = afwMath.makeStatistics(smi[border:, border:-border], afwMath.MEANCLIP).getValue()
    #             elif j == 3:
    #                 mean = afwMath.makeStatistics(smi[:-border, border:-border], afwMath.MEANCLIP).getValue()
    #             else:
    #                 mean = afwMath.makeStatistics(smi[:, border:-border], afwMath.MEANCLIP).getValue()
    #             if GAINS is not None:
    #                 gain = GAINS[0]
    #             else:
    #                 # gain = a.getElectronicParams().getGain()
    #                 gain = a.getGain()
    #             means1[i].append(mean*gain)
    #             smi *= gain
    #             smi -= mean*gain
    #     diff = ims[0].clone()
    #     diff = diff.getMaskedImage().getImage()
    #     diff -= ims[1].getMaskedImage().getImage()

    #     #
    #     # Subtract background.  It should be a constant, but it isn't always (e.g. some SuprimeCam flats)
    #     #
    #     binsize = 128
    #     nx = diff.getWidth()//binsize
    #     ny = diff.getHeight()//binsize
    #     bctrl = afwMath.BackgroundControl(nx, ny, sctrl, afwMath.MEANCLIP)
    #     bkgd = afwMath.makeBackground(diff, bctrl)
    #     diff -= bkgd.getImageF(afwMath.Interpolate.CUBIC_SPLINE, afwMath.REDUCE_INTERP_ORDER)
    #     Var = []
    #     CorVar = []
    #     for i, a in enumerate(CCD):
    #         borderL = 0
    #         borderR = 0
    #         if i == 0:
    #             borderL = border
    #         if i == 3:
    #             borderR = border
    #         # smi = diff[a.getDataSec(True)].clone()
    #         smi = diff[a.getBBox()].clone()
    #         # dim0 = smi[border:-border-n,border:-border-n]
    #         dim0 = smi[borderL:-borderR-n, border:-border-n]

    #         dim0 -= afwMath.makeStatistics(dim0, afwMath.MEANCLIP, sctrl).getValue()
    #         w, h = dim0.getDimensions()
    #         xcorr = afwImage.ImageF(n + 1, n + 1)
    #         for di in range(n + 1):
    #             for dj in range(n + 1):
    #                 dim_ij = smi[borderL+di:borderL+di + w, border+dj: border+dj + h].clone()
    #                 # dim_ij = smi[border+di:border+di + w, border+dj: border+dj + h].clone()
    #                 dim_ij -= afwMath.makeStatistics(dim_ij, afwMath.MEANCLIP, sctrl).getValue()
    #                 dim_ij *= dim0
    #                 xcorr[di, dj] = afwMath.makeStatistics(dim_ij, afwMath.MEANCLIP, sctrl).getValue()/(biasCorr)
    #         Var.append(xcorr.getArray()[0, 0])
    #         CorVar.append(xcorr.getArray())
    #         L = np.shape(xcorr.getArray())[0]-1
    #         XCORR = np.zeros([2*L+1, 2*L+1])
    #         for I in range(L+1):
    #             for J in range(L+1):
    #                 XCORR[I+L, J+L] = xcorr.getArray()[I, J]
    #                 XCORR[-I+L, J+L] = xcorr.getArray()[I, J]
    #                 XCORR[I+L, -J+L] = xcorr.getArray()[I, J]
    #         print(means1[0][i], means1[1][i], means1[0][i]+means1[1][i], Var[i], np.sum(XCORR))
    #     return (means1[0], means1[1], [i+j for i, j in zip(means1[1], means1[0])], Var, CorVar)

    def iterativeRegression(self, x, y, fixThroughOrigin=False, nSigmaClip=3, maxIter=10):
        """Use linear regression to fit a line of best fit, iteratively removing outliers.

        Useful when you have sufficiently large numbers of points on your PTC.
        Function iterates until either there are no outliers of nSigmaClip magnitude, or until the specified
        max number of iterations have been performed.

        Parameters:
        -----------
        x : `numpy.array`
            The independent variable
        y : `numpy.array`
            The dependent variable

        Returns:
        --------
        slope : `float`
            The slope of the line of best fit
        intercept : `float`
            The y-intercept of the line of best fit
        """
        nIter = 0
        sctrl = afwMath.StatisticsControl()
        sctrl.setNumSigmaClip(nSigmaClip)

        if fixThroughOrigin:
            while nIter < maxIter:  # TODO: change log levels to debug
                nIter += 1
                self.log.info("Origin fixed, iteration # %s, %s elements:"%(nIter, np.shape(x)[0]))
                TEST = x[:, np.newaxis]
                slope, _, _, _ = np.linalg.lstsq(TEST, y)
                slope = slope[0]
                res = y - slope * x
                resMean = afwMath.makeStatistics(res, afwMath.MEANCLIP, sctrl).getValue()
                resStd = np.sqrt(afwMath.makeStatistics(res, afwMath.VARIANCECLIP, sctrl).getValue())
                index = np.where((res > (resMean + nSigmaClip * resStd)) | (res < resMean - nSigmaClip * resStd))
                self.log.info("%.3f %.3f %.3f %.3f"%(resMean, resStd, np.max(res), nSigmaClip))
                if np.shape(np.where(index))[1] == 0 or (nIter >= maxIter):  # run out of points, or iterations
                    break
                x = np.delete(x, index)
                y = np.delete(y, index)

            return slope, 0

        while nIter < maxIter:  # TODO: change log levels to debug
            nIter += 1
            self.log.info("Iteration # %s, %s elements:"%(nIter, np.shape(x)[0]))
            xx = np.vstack([x, np.ones(len(x))]).T
            ret, _, _, _ = np.linalg.lstsq(xx, y)
            slope, intercept = ret
            res = y - slope*x - intercept
            resMean = afwMath.makeStatistics(res, afwMath.MEANCLIP, sctrl).getValue()
            resStd = np.sqrt(afwMath.makeStatistics(res, afwMath.VARIANCECLIP, sctrl).getValue())
            index = np.where((res > (resMean + nSigmaClip * resStd)) | (res < resMean - nSigmaClip * resStd))
            self.log.info("%.3f %.3f %.3f %.3f"%(resMean, resStd, np.max(res), nSigmaClip))
            if np.shape(np.where(index))[1] == 0 or (nIter >= maxIter):  # run out of points, or iterations
                break
            x = np.delete(x, index)
            y = np.delete(y, index)

        return slope, intercept

    # def iterativeRegressionOLDXXX(self, x, y, intercept=0, sigma=3):
    #     """A best fit method which removes outliers.

    #     Useful when you have sufficiently large numbers of points on your PTC.
    #     """
    #     import ipdb as pdb; pdb.set_trace()
    #     iterate = 1
    #     sctrl = afwMath.StatisticsControl()
    #     sctrl.setNumSigmaClip(sigma)
    #     if intercept:
    #         while iterate:
    #             print("Iteration # %s: %s"%(iterate, np.shape(x)))
    #             A = np.vstack([x, np.ones(len(x))]).T
    #             B, _, _, _ = np.linalg.lstsq(A, y)
    #             slope, intercept = B
    #             res = y-slope*x-intercept
    #             resMean = afwMath.makeStatistics(res, afwMath.MEANCLIP, sctrl).getValue()
    #             resSTD = np.sqrt(afwMath.makeStatistics(res, afwMath.VARIANCECLIP, sctrl).getValue())
    #             index = np.where((res > (resMean+sigma*resSTD)) | (res < resMean-sigma*resSTD))
    #             print(resMean, resSTD, np.max(res), sigma)
    #             if np.shape(np.where(index))[1] == 0:
    #                 break
    #             x = np.delete(x, index)
    #             y = np.delete(y, index)

    #         return slope, intercept
    #     while iterate:
    #         print("Iteration # %s: %s"%(iterate, np.shape(x)))
    #         TEST = x[:, np.newaxis]
    #         slope, _, _, _ = np.linalg.lstsq(TEST, y)
    #         slope = slope[0]
    #         res = y-slope*x
    #         resMean = afwMath.makeStatistics(res, afwMath.MEANCLIP, sctrl).getValue()
    #         resSTD = np.sqrt(afwMath.makeStatistics(res, afwMath.VARIANCECLIP, sctrl).getValue())
    #         index = np.where((res > (resMean+sigma*resSTD)) | (res < resMean-sigma*resSTD))
    #         print(resMean, resSTD, np.max(res), sigma)
    #         if np.shape(np.where(index))[1] == 0:
    #             break
    #         x = np.delete(x, index)
    #         y = np.delete(y, index)

    #     return slope

    def estimateGains(self, butler, visitPairs, ignoreCcdList=None, intercept=0, writeGains=True,
                      xxx_outputFile=os.path.join(OUTPUT_PATH, 'WILLS_GAINS.pkl'),
                      xxx_figLocation=OUTPUT_PATH, xxx_plot=True):
        """Estimate the gains of the specified CCD(s) using the specified visits.

        XXX The is really a ptcGainTask by Will. Should be moved to its own task really.

        Given a butler and list of flats of varying intensity, calculate the gain for each
        CCD specified using the PTC method.

        The intercept option chooses the linear fitting option. The default fits
        Var=1/g mean, if non zero Var=1/g mean + const is fit.
        By default, gains are persisted per-amplifier as a dictionary

        Parameters
        ----------
        butler : `lsst.daf.persistence.butler`
            Butler for the repo containg the flats to be used
        visitPairs : `list` of `tuple`
            List of visit-pairs to use, as [(v1,v2), (v3,v4)...]
        writeGains : `bool`
            Persist the calculated gain values
        ignoreCcdList : `list`
            List of CCD(s) to skip, by CCD number. Defaults to none, i.e. use all CCDs in camera

        Returns
        -------
        gains : `dict`
            Amplifier gain values, as calculated
        nominalGains : `dict`
            Amplifier gains, as given by the `detector` objects
        """
        gains = {}
        nomGains = {}

        camera = butler.get('camera')
        useCcds = [ccd.getId() for ccd in camera if ccd.getId() not in ignoreCcdList]
        assert(len(camera) == len(useCcds) + len(ignoreCcdList))
        self.log.info('Processing CCDs %s'%useCcds)

        # Loop over the CCDs, calculating a PTC for each amplifier.
        # Amplifier iteration is performed in _calcMeansAndVars()
        for ccd in useCcds:
            ampMeans = []
            ampVariances = []
            ampCorrVariances = []
            ampGains = []
            nomGains[ccd] = []
            # Cycles through the input visits and calculate the xcorr in the individual amps.
            # NB: no gain correction is applied
            for visPairNum, visPair in enumerate(visitPairs):
                _means, _vars, _covars, _gains = self._calcMeansAndVars(butler, visPair[0], visPair[1],
                                                                        n=8, ccd=ccd, plot=xxx_plot)
                breaker = 0
                # Do sanity checks; if these are failed more investigation is needed!
                for i, j in enumerate(_means):
                    if _means[i]*10 < _vars[i] or _means[i]*10 < _covars[i]:
                        self.log.warn('Sanity check failed; check visit %s'%visPair)
                        breaker += 1
                if breaker:
                    continue
                if visPairNum == 0:
                    for i in range(len(_means)):
                        ampMeans.append(np.array([]))
                        ampVariances.append(np.array([]))
                        ampCorrVariances.append(np.array([]))
                        ampGains.append(np.array([]))
                for i, j in enumerate(_means):
                    if visPairNum == 0:
                        nomGains[ccd].append(_gains[i])
                    if _vars[i]*1.3 < _covars[i] or _vars[i]*0.7 > _covars[i]:
                        continue
                    ampMeans[i] = np.append(ampMeans[i], _means[i])
                    ampVariances[i] = np.append(ampVariances[i], _vars[i])
                    ampCorrVariances[i] = np.append(ampCorrVariances[i], _covars[i])
                    ampGains[i] = np.append(ampGains[i], _gains[i])

            # TODO: Change the "intercept" option to a pexConfig option (or decide which is best and remove)
            # TODO: replace with lsstDebug
            fig = None
            gains[ccd] = []
            for i in range(len(ampMeans)):
                # TODO: move to inside the if: plot block below
                if fig is None:
                    fig = plt.figure()
                else:
                    fig.clf()
                ax = fig.add_subplot(111)
                slope2, intercept, r_value, p_value, std_err = stats.linregress(ampMeans[i],
                                                                                ampCorrVariances[i])
                slope, _ = self.iterativeRegression(ampMeans[i], ampCorrVariances[i], fixThroughOrigin=True)
                slope3, intercept2 = self.iterativeRegression(ampMeans[i], ampCorrVariances[i])
                # TODO: Change messages to say what these ARE, not just second/third fits
                self.log.info("slope of fit: %s intercept of fit: %s p value: %s"%(slope2,
                                                                                   intercept, p_value))
                self.log.info("slope of second fit: %s, difference:%s"%(slope, slope-slope2))
                self.log.info("slope of third  fit: %s, difference: %s"%(slope3, slope-slope3))
                if intercept:
                    slope = slope3

                if xxx_plot:  # TODO: replace with lsstDebug.Also, consider dumping based on p_value or X_sq?
                    ax.plot(ampMeans[i], ampCorrVariances[i], linestyle='None', marker='x', label='data')
                    if intercept:
                        ax.plot(ampMeans[i], ampMeans[i]*slope+intercept2, label='fix')

                    else:
                        ax.plot(ampMeans[i], ampMeans[i]*slope, label='fix')
                    fig.savefig(os.path.join(xxx_figLocation, ('PTC_CCD_'+str(ccd)+'_AMP_'+str(i)+'.pdf')))
                    # plt.show()
                gains[ccd].append(1.0/slope)

            if writeGains:  # TODO: replace with buttleable dataset. Also, should be using `with`
                try:
                    f = open(xxx_outputFile, 'r+b')
                    f.seek(0)
                    try:
                        storedGains = pickle.load(f)
                    except EOFError:
                        storedGains = {}
                except IOError:
                    f = open(xxx_outputFile, 'wb')
                    storedGains = {}
                storedGains[ccd] = gains[ccd]
                f.seek(0)
                f.truncate()
                pickle.dump(storedGains, f)
                f.close()
        self.log.info('gains %s\noGains %s'%(gains, nomGains))
        return (gains, nomGains)

    def _generateKernel(self, corrs, means, rejectLevel=None):
        """Generate the full kernel from a list of (gain-corrected) cross-correlations and means.

        Taking a list of quarter-image, gain-corrected cross-correlations, do a pixel-wise sigma-clipped
        mean of each, and tile into the full-sized kernel image.

        Each corr in corrs is one quarter of the full cross-correlation, and has been gain-corrected.
        Each mean in means is a tuple of the means of the two individual images, corresponding to that corr.

        Parameters:
        -----------
        corrs : `list` of `numpy.ndarray`, (Ny, Nx)
            A list of the quarter-image cross-correlations
        means : `list` of `tuples` of `floats`
            The means of the input images for each corr in corrs
        rejectLevel : `float`, optional
            This is essentially is a sanity check parameter.
            If this condition is violated there is something unexpected going on in the image, and it is
            discarded from the stack before the clipped-mean is calculated.

        Returns:
        --------
        kernel : `numpy.ndarray`, (Ny, Nx)
            The output kernel
        """
        if not rejectLevel:
            rejectLevel = self.config.xcorrCheckRejectLevel

        if not isinstance(corrs, list):  # we expect a list of arrays
            corrs = [corrs]

        # Try to average over a set of possible inputs. This generates a simple function of the kernel that
        # should be constant across the images, and averages that.
        xcorrList = []
        sctrl = afwMath.StatisticsControl()
        sctrl.setNumSigmaClip(self.config.kernelGenSigmaClip)

        for corrNum, ((mean1, mean2), corr) in enumerate(zip(means, corrs)):
            corr[0, 0] -= (mean1+mean2)
            if corr[0, 0] > 0:
                self.log.warn('Skipped item %s due to unexpected value of (variance-mean)!'%corrNum)
                continue
            corr /= -float(1.0*(mean1**2+mean2**2))

            fullCorr = self._tileArray(corr)

            # TODO: what is this block really testing? Is this what it should be doing? First line is fishy
            xcorrCheck = np.abs(np.sum(fullCorr))/np.sum(np.abs(fullCorr))
            if xcorrCheck > rejectLevel:
                self.log.warn("Sum of the xcorr is unexpectedly high. Investigate item num %s. \n"
                              "value = %s"%(corrNum, xcorrCheck))
                continue
            xcorrList.append(fullCorr)
        
        if not xcorrList:
            raise RuntimeError("Cannot generate kernel because all inputs were discarded. "
                               "Either the data is bad, or config.xcorrCheckRejectLevel is too high")

        # stack the individual xcorrs and apply a per-pixel clipped-mean
        meanXcorr = np.zeros_like(fullCorr)
        xcorrList = np.transpose(xcorrList)
        for i in range(np.shape(meanXcorr)[0]):
            for j in range(np.shape(meanXcorr)[1]):
                meanXcorr[i, j] = afwMath.makeStatistics(xcorrList[i, j], afwMath.MEANCLIP, sctrl).getValue()

        return self._SOR(meanXcorr)

    def _SOR(self, source, maxIter=None, eLevel=None):
        """An implementation of the successive over relaxation (SOR) method.

        Parameters:
        -----------
        source : `numpy.ndarray`, (Ny, Nx)
            The input array
        maxIter : `int`, optional
            Maximum number of iterations to attempt before aborting
        eLevel : `float`, optional
            The target error level factor at which we deem convergence to have occured

        Returns:
        --------
        output : `numpy.ndarray`, (Ny, Nx)
            The solution
        """
        if not maxIter:
            maxIter = self.config.maxIterSOR
        if not eLevel:
            eLevel = self.config.eLevelSOR

        # initialise, and set boundary conditions
        func = np.zeros([source.shape[0]+2, source.shape[1]+2])
        resid = np.zeros([source.shape[0]+2, source.shape[1]+2])
        rhoSpe = np.cos(np.pi/source.shape[0])  # Here a square grid is assummed

        inError = 0
        # Calculate the initial error
        for i in range(1, func.shape[0]-1):
            for j in range(1, func.shape[1]-1):
                resid[i, j] = (func[i, j-1]+func[i, j+1]+func[i-1, j] +
                               func[i+1, j]-4*func[i, j]-source[i-1, j-1])
        inError = np.sum(np.abs(resid))

        # Iterate until convergence
        # We perform two sweeps per cycle, updating 'odd' and 'even' points separately
        nIter = 0
        omega = 1.0
        dx = 1.0
        while nIter < maxIter*2:
            outError = 0
            if nIter%2 == 0:
                for i in range(1, func.shape[0]-1, 2):
                    for j in range(1, func.shape[0]-1, 2):
                        resid[i, j] = float(func[i, j-1]+func[i, j+1]+func[i-1, j] +
                                            func[i+1, j]-4.0*func[i, j]-dx*dx*source[i-1, j-1])
                        func[i, j] += omega*resid[i, j]*.25
                for i in range(2, func.shape[0]-1, 2):
                    for j in range(2, func.shape[0]-1, 2):
                        resid[i, j] = float(func[i, j-1]+func[i, j+1]+func[i-1, j] +
                                            func[i+1, j]-4.0*func[i, j]-dx*dx*source[i-1, j-1])
                        func[i, j] += omega*resid[i, j]*.25
            else:
                for i in range(1, func.shape[0]-1, 2):
                    for j in range(2, func.shape[0]-1, 2):
                        resid[i, j] = float(func[i, j-1]+func[i, j+1]+func[i-1, j] +
                                            func[i+1, j]-4.0*func[i, j]-dx*dx*source[i-1, j-1])
                        func[i, j] += omega*resid[i, j]*.25
                for i in range(2, func.shape[0]-1, 2):
                    for j in range(1, func.shape[0]-1, 2):
                        resid[i, j] = float(func[i, j-1]+func[i, j+1]+func[i-1, j] +
                                            func[i+1, j]-4.0*func[i, j]-dx*dx*source[i-1, j-1])
                        func[i, j] += omega*resid[i, j]*.25
            outError = np.sum(np.abs(resid))
            if outError < inError*eLevel:
                break
            if nIter == 0:
                omega = 1.0/(1-rhoSpe*rhoSpe/2.0)
            else:
                omega = 1.0/(1-rhoSpe*rhoSpe*omega/4.0)
            nIter += 1

        if nIter >= maxIter*2:
            self.log.warn("Did not converge in %s iterations.\noutError: %s, inError: "
                          "%s,"%(nIter//2, outError, inError*eLevel))
        else:
            self.log.info("Converged in %s iterations.\noutError: %s, inError: "
                          "%s", nIter//2, outError, inError*eLevel)
        return func[1:-1, 1:-1]



    # This sim code is used to estimate the bias correction used above.
    def xcorr_sim(self, im, im2, n=8, border=10, sigma=5):
        """Perform a simple xcorr from two images.

        It contains many elements of the actual code
        above (without individual amps and ISR removal )
        It takes two images, im and im2; n the max lag of the correlation function; border, the number of border
        pixels to discard; and sigma the sigma to use in the mean clip.
        """
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
        means1[0] = afwMath.makeStatistics(im[border:-border, border:-border],
                                           afwMath.MEANCLIP, sctrl).getValue()
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
        # print((means1),xcorr.getArray()[0,0],np.sum(XCORR),xcorr.getArray()[0,0]/
        #       (np.sum(means1)),np.sum(XCORR)/(np.sum(means1)))
        return (XCORR, xcorr, np.sum(means1), means1)


    def xcorr_bias(self, rangeMeans=[87500, 70000, 111000], repeats=5, sig=5,
                   border=3, seed=None, nx=2000, ny=4000, case=0, a=.1):
        """Fill images of specified size (nx and ny) with poisson points with means (in rangeMeans).

        before passing it to the above function with border and sig as above
        Repeats specifies the number of times to run the simulations.
        If case is 1 then a correlation between x_{i,j} and x_{i+1,j+1} is artificially introduced
        by adding a*x_{i,j} to x_{i+1,j+1}
        If seed is left to None the seed with be pulled from /dev/random.
        Else an int can be passed to see the random number generator.
        """
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
                    # im.getArray()[:,:]=local_random.normal(MEAN,np.sqrt(MEAN),(ny,nx))
                    # im0.getArray()[:,:]=local_random.normal(MEAN,np.sqrt(MEAN),(ny,nx))
                    im.getArray()[:, :] = local_random.poisson(MEAN, (ny, nx))
                    im0.getArray()[:, :] = local_random.poisson(MEAN, (ny, nx))
                    XCORR, xcorr, means, MEANS1 = xcorr_sim(im, im0, border=border, sigma=sig)
                    MEANS[MEAN].append(means)
                    XCORRS[MEAN].append(xcorr)
                print('\n\n\n')
                for i, MEAN in enumerate(rangeMeans):
                    print("Simulated/Expected:", MEAN, MEANS[MEAN][-1],
                          XCORRS[MEAN][-1].getArray()[0, 0]/MEANS[MEAN][-1])
        else:
            for rep in range(repeats):
                for i, MEAN in enumerate(rangeMeans):
                    im = afwImage.ImageD(nx, ny)
                    im0 = afwImage.ImageD(nx, ny)
                    # im.getArray()[:,:]=local_random.normal(MEAN,np.sqrt(MEAN),(ny,nx))
                    # im0.getArray()[:,:]=local_random.normal(MEAN,np.sqrt(MEAN),(ny,nx))
                    im.getArray()[:, :] = local_random.poisson(MEAN, (ny, nx))
                    im.getArray()[1:, 1:] += a*im.getArray()[:-1, :-1]
                    im0.getArray()[:, :] = local_random.poisson(MEAN, (ny, nx))
                    im0.getArray()[1:, 1:] += a*im0.getArray()[:-1, :-1]
                    XCORR, xcorr, means, MEANS1 = xcorr_sim(im, im0, border=border, sigma=sig)
                    MEANS[MEAN].append(means)
                    XCORRS[MEAN].append(xcorr)
                print('\n\n\n')
                for i, MEAN in enumerate(rangeMeans):
                    print("Simulated/Expected:", MEANS[MEAN][-1], '\n',
                          (XCORRS[MEAN][-1].getArray()[1, 1]/MEANS[MEAN][-1]*(1+a))/.1)
        return MEANS, XCORRS

    @staticmethod
    def _tileArray(in_array):
        """Given a square input quarter-image, tile/mirror it, returning the full image.

        Given an input of side-length n, of the form

        input = array([[1, 2, 3],
                       [4, 5, 6],
                       [7, 8, 9]])

        return an array of size 2n-1 as

        output = array([[ 9,  8,  7,  8,  9],
                        [ 6,  5,  4,  5,  6],
                        [ 3,  2,  1,  2,  3],
                        [ 6,  5,  4,  5,  6],
                        [ 9,  8,  7,  8,  9]])

        Parameters:
        -----------
        input : `np.array`
            The square input quarter-array

        Returns:
        --------
        output : `np.array`
            The full, tiled array
        """
        assert(in_array.shape[0] == in_array.shape[1])
        length = in_array.shape[0]-1
        output = np.zeros((2*length+1, 2*length+1))

        for i in range(length+1):
            for j in range(length+1):
                output[i+length, j+length] = in_array[i, j]
                output[-i+length, j+length] = in_array[i, j]
                output[i+length, -j+length] = in_array[i, j]
                output[-i+length, -j+length] = in_array[i, j]
        return output
