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
           'MeasurePhotonTransferCurveTaskConfig', ]

import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.backends.backend_pdf import PdfPages

# import lsstDebug
# import lsst.afw.image as afwImage
import lsst.afw.math as afwMath
import lsst.log as lsstLog
import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase
from lsst.ip.isr import IsrTask
from .utils import NonexistentDatasetTaskDataIdContainer, PairedVisitListTaskRunner, checkExpLengthEqual, \
    validateIsrConfig
from scipy.optimize import leastsq
import numpy.polynomial.polynomial as poly
from copy import deepcopy

class MeasurePhotonTransferCurveTaskConfig(pexConfig.Config):
    """Config class for photon transfer curve measurement task"""
    isr = pexConfig.ConfigurableField(
        target=IsrTask,
        doc="""Task to perform instrumental signature removal""",
    )
    isrMandatorySteps = pexConfig.ListField(
        dtype=str,
        doc="isr operations that must be performed for valid results. Raises if any of these are False",
        default=['doAssembleCcd']
    )
    isrForbiddenSteps = pexConfig.ListField(
        dtype=str,
        doc="isr operations that must NOT be performed for valid results. Raises if any of these are True",
        default=['doFlat', 'doFringe', 'doAddDistortionModel', 'doBrighterFatter', 'doUseOpticsTransmission',
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
        doc="The key by which to pull a detector from a dataId, e.g. 'ccd' or 'detector'",
        default='ccd',
    )
    imageTypeKey = pexConfig.Field(
        dtype=str,
        doc="The key by which to check whether images are darks or flats",
        default='imgType',
    )
    makePlots = pexConfig.Field(
        dtype=bool,
        doc="Plot the PTC curves?",
        default=False,
    )
    typeFitPtc = pexConfig.ChoiceField(
        dtype=str,
        doc="Fit PTC to approx. in Astier+19 (Eq. 16) or to a polynomial",
        default="polynomial",
        allowed={
            "polynomial": "n-degree polynomial (use 'degPolynomial' to set 'n')",
            "astierSingleBfPar": "Approx. in Astier+19 (Eq. 16)"
        }
    )
    degPolynomial = pexConfig.Field(
        dtype=int,
        doc="Degree of polynomial to fit the PTC",
        default=2,
    )
    binSize = pexConfig.Field(
        dtype=int,
        doc="Bin the image by this factor in both dimensions",
        default=1,
    )
    minMeanSignal = pexConfig.Field(
        dtype=float,
        doc="Minimun value of mean signal (in ADU) to consider",
        default=0,
    )
    maxMeanSignal = pexConfig.Field(
        dtype=float,
        doc="Maximum value to of mean signal (in ADU) to consider",
        default=9e6,
    )


class MeasurePhotonTransferCurveTask(pipeBase.CmdLineTask):
    """A class to calculate, fit, and plot a PTC from a set of flat pairs.

    The Photon Transfer Curve (var(signal) vs mean(signal)) is a standard tool
    used in astronomical detectors characterization (e.g., Janesick 2001,
    Janesick 2007). This task calculates the PTC from a series of pairs of
    flat-field images; each pair taken at identical exposure times. The
    difference image of each pair is formed to elliminate fixed pattern noise,
    and then the
    variance of the difference image and the mean of the average image are used
    to produce the PTC. An n-degree polynomial or the approximation in Equation
    16 of Astier+19 ("The Shape of the Photon Transfer Curve of CCD sensors",
    arXiv:1905.08677) can be fitted to the PTC curve. These models include
    parameters such as the gain (e/ADU) and readout noise.

    Parameters
    ----------

    *args: `list`
        Positional arguments passed to the Task constructor. None used at this
        time.
    **kwargs: `dict`
        Keyword arguments passed on to the Task constructor. None used at this
        time

    """

    RunnerClass = PairedVisitListTaskRunner
    ConfigClass = MeasurePhotonTransferCurveTaskConfig
    _DefaultName = "measurePhotonTransferCurve"

    def __init__(self, *args, **kwargs):
        pipeBase.CmdLineTask.__init__(self, *args, **kwargs)
        self.makeSubtask("isr")
        self.isr.log.setLevel(lsstLog.WARN)  # xxx consider this level

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
        parser.add_id_argument("--id", datasetType="measurePhotonTransferCurveDataset",
                               ContainerClass=NonexistentDatasetTaskDataIdContainer,
                               help="The ccds to use, e.g. --id ccd=0..100")
        return parser

    @pipeBase.timeMethod
    def runDataRef(self, dataRef, visitPairs):
        """Run the Photon Transfer Curve (PTC) measurement task.

        For a dataRef (which is each detector here),
        and given a list of visit pairs (ascending in exposure time),
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
        ampInfoCat = detector.getAmpInfoCatalog()
        ampNames = [amp.getName() for amp in ampInfoCat]
        dataDict = {key: {} for key in ampNames}
        fitVectorsDict = {key: ([], [], []) for key in ampNames}

        self.log.info('Measuring PTC using %s visits for detector %s' % (visitPairs, detNum))

        sctrl = afwMath.StatisticsControl()
        sctrl.setNumSigmaClip(5)
        sctrl.setNumIter(3)

        for (v1, v2) in visitPairs:
            dataRef.dataId['visit'] = v1
            exp1 = self.isr.runDataRef(dataRef).exposure
            dataRef.dataId['visit'] = v2
            exp2 = self.isr.runDataRef(dataRef).exposure
            del dataRef.dataId['visit']

            checkExpLengthEqual(exp1, exp2, v1, v2, raiseWithMessage=True)
            expTime = exp1.getInfo().getVisitInfo().getExposureTime()

            for amp in detector:
                im1Area = exp1.maskedImage[amp.getBBox()]
                im1Area = afwMath.binImage(im1Area, self.config.binSize)

                im2Area = exp2.maskedImage[amp.getBBox()]
                im2Area = afwMath.binImage(im2Area, self.config.binSize)

                # Take difference of pairs
                diffIm = im1Area.clone()
                # Scale second image to have same mean as first
                im2Scale = im2Area.clone()
                im2Scale *= afwMath.makeStatistics(im1Area, afwMath.MEAN).getValue()
                im2Scale /= afwMath.makeStatistics(im2Area, afwMath.MEAN).getValue()
                diffIm -= im2Scale

                # Take average of pairs
                avIm = im1Area.clone()
                avIm += im2Area.clone()
                avIm *= 0.5

                meanDiff = afwMath.makeStatistics(avIm, afwMath.MEAN).getValue()
                varDiff = 0.5*(afwMath.makeStatistics(diffIm, afwMath.VARIANCE).getValue())

                data = dict(expTime=expTime, meanClip=meanDiff, varClip=varDiff)

                ampName = amp.getName()
                dataDict[ampName][(v1, v2)] = data

                fitVectorsDict[ampName][0].append(expTime)
                fitVectorsDict[ampName][1].append(meanDiff)
                fitVectorsDict[ampName][2].append(varDiff)

        # Fit PTC  and (non)linearity of signal vs time curve
        fitPtcDict, nlDict = self.fitPtcAndNl(fitVectorsDict, typeFitPtc=self.config.typeFitPtc)
        finalDict = {"data": dataDict, "ptc": fitPtcDict, "nl": nlDict}

        # Save data, PTC fit, and NL fit dictionaries
        self.log.info(f"Writing PTC and NL data to {dataRef.getUri(write=True)}")
        dataRef.put(finalDict)

        if self.config.makePlots:
            self.plot(dataRef, data, fitPtcDict, nlDict, typeFitPtc=self.config.typeFitPtc)

        self.log.info('Finished measuring PTC for in detector %s' % detNum)

        return pipeBase.Struct(exitStatus=0)

    def fitBootstrap(self, pIni, dataX, dataY, function, confidenceSigma=1.):
        """
        Do a fit using least squares and boostrap to estimate parameter errors. 
        Boostrap: 100 random datasets are generated and fitted
        
        Parameters
        ----------
        pIni: list of floats with initial values for fit parameters. For  
        typeFitPtc=polynomial, its length determines the degree of the polynomial.

        dataX: array with data in the abscisa axis

        dataY: array with data in the ordinate axis

        function: model or function to fit the data with

        confidenceSigma: number of sigmas that determine confidence interval for 
                         boostrap errors.

        Return
        ------
        pFitBootstrap: list with fitted parameters
        
        pErrBootstrap: list with errors for fitted parameters
        """

        def errFunc(p, x, y):
            return function(p, x) - y

        # Fit first time
        pFit, pErr = leastsq(errFunc, pIni, args=(dataX, dataY), full_output=0)

        # Get the stdev of the residuals
        residuals = errFunc(pFit, dataX, dataY)
        sigmaErrTotal = np.std(residuals)

        # 100 random data sets are generated and fitted
        pars = []
        for i in range(100):
            randomDelta = np.random.normal(0., sigmaErrTotal, len(dataY))
            randomDataY = dataY + randomDelta
            randomFit, randomCov = \
                leastsq(errFunc, pIni, args=(dataX, randomDataY), full_output=0)
            pars.append(randomFit)
        pars = np.array(pars)
        meanPfit = np.mean(pars, 0)

        # confidence interval for parameter estimates
        Nsigma = confidenceSigma
        errPfit = Nsigma * np.std(pars, 0)
        pFitBootstrap = meanPfit
        pErrBootstrap = errPfit
        return pFitBootstrap, pErrBootstrap

    def funcPolynomial(self, pars, x):
        """Polynomial function definition"""
        return poly.polyval(x, [*pars])

    def funcAstier(self, pars, x):
        """Single brightter-fatter parameter model for PTC; Eq. 16 of Astier+19"""
        a00, gain, noise = pars
        return 0.5/(a00*gain*gain)*(1 - np.exp(-2*a00*x*gain)) + noise/(gain*gain)


    def fitPtcAndNl(self, fitVectorsDict, typeFitPtc='polynomial'):
        """Function to fit PTC, and calculate linearity and linearity residual

        Parameters
        ----------
        fitVectorsDict: `dict`
            Dict. with exposure time, mean, and variance vectors in a tuple
        typeFitPtc: `str`
            Fit a 'polynomial' (degree: 'degPolynomial') or 'astierSingleBfPar' to the PTC

        Returns
        -------
        fitPtcDict: `dict`
            Dictionary of the form fitPtcDict[amp] =
            (meanVec, varVec, parsFit, parsFitErr, index)
        nlDict: `dict`
            Dictionary of the form nlDict[amp] =
            (timeVec, meanVec, linResidual, parsFit, parsFitErr)
        """
        if typeFitPtc == 'astierSingleBfPar':
            ptcFunc = self.funcAstier
            parsIniPtc = [1e-9, 0.7, 10.]
        if typeFitPtc == 'polynomial':
            ptcFunc = self.funcPolynomial
            parsIniPtc = np.repeat(1., self.config.degPolynomial + 1)

        parsIniNl = [10., 100., 1.]
        fitPtcDict = {key: {} for key in fitVectorsDict}
        nlDict = {key: {} for key in fitVectorsDict}

        def errFunc(p, x, y):
            return ptcFunc(p, x) - y


        for amp in fitVectorsDict:
            timeVec, meanVec, varVec = fitVectorsDict[amp]
            timeVec = np.array(timeVec)
            meanVec = np.array(meanVec)
            varVec = np.array(varVec)
            index = (meanVec > self.config.minMeanSignal) & (meanVec <= self.config.maxMeanSignal)

            #  Before boostrap fit, do an iterative fit to get rid of outliers in PTC 
            indexOld=[]
            count, sigmaCut = 1, 5
            while (index != indexOld) and count < 5: 
                pars, cov = leastsq (errFunc, parsIniPtc, args=(meanVec[index], varVec[index]), full_output=0)
                sigResids = (varVec - ptcFunc(pars, meanVec))/np.sqrt(varVec)      
                indexOld = deepcopy (index)
                index = list(np.where(np.abs(sigResids) < sigmaCut)[0])
                count+= 1

            parsIniPtc = pars
            timeVec, meanVec, varVec = timeVec[index], meanVec[index], varVec[index]
            #  Boostrap fit
            parsFit, parsFitErr = self.fitBootstrap(parsIniPtc, meanVec, varVec, ptcFunc)

            fitPtcDict[amp] = (meanVec, varVec, parsFit, parsFitErr, index)

            # Non-linearity residuals (NL of mean vs time curve): percentage, and fit to quadratic function
            # len(parsIniNl) = 3 indicates that we want a quadratic fit
            parsFit, parsFitErr = self.fitBootstrap(parsIniNl, timeVec, meanVec, self.funcPolynomial) 
            linResidual = 100*(1 - ((meanVec[2]/timeVec[2])/(meanVec/timeVec)))
            nlDict[amp] = (timeVec, meanVec, linResidual, parsFit, parsFitErr)

        return fitPtcDict, nlDict

    def plot(self, dataRef, data, fitPtcDict, nlDict, typeFitPtc='polynomial'):
        dirname = dataRef.getUri(datasetType='cpPipePlotRoot', write=True)
        if not os.path.exists(dirname):
            os.makedirs(dirname)

        detNum = dataRef.dataId[self.config.ccdKey]
        filename = f"PTC_det{detNum}.pdf"
        filenameFull = os.path.join(dirname, filename)
        with PdfPages(filenameFull) as pdfPages:
            self._plotPtc(data, fitPtcDict, nlDict, typeFitPtc, pdfPages)

    def _plotPtc(self, data, fitPtcDict, nlDict, typeFitPtc, pdfPages):
        """Plot PTC, linearity, and linearity residual per amplifier"""

        if typeFitPtc == 'astierSingleBfPar':
            ptcFunc = self.funcAstier
            stringTitle = r"Var = $\frac{1}{2g^2a_{00}}(\exp (2a_{00} \mu g) - 1) + \frac{n_{00}}{g^2}$"

        if typeFitPtc == 'polynomial':
            ptcFunc = self.funcPolynomial
            stringTitle = f"Polynomial (degree: {self.config.degPolynomial})"
        
        legendFontSize = 7.5

        # General determination of size of plot grid
        nAmps = len (fitPtcDict)
        if nAmps == 2:
            nRows, nCols = 2, 1
        nRows = np.sqrt(nAmps)
        mantissa, _ = np.modf(nRows)
        if mantissa > 0: 
            nRows = int (nRows) + 1
            nCols = nRows 
        else:
            nRows = int(nRows)
            nCols = nRows

        f, ax = plt.subplots(nrows=nRows, ncols=nCols, sharex='col', sharey='row', figsize=(13, 10))
        f2, ax2 = plt.subplots(nrows=nRows, ncols=nCols, sharex='col', sharey='row', figsize=(13, 10))

        for i, (amp, a, a2) in enumerate(zip(fitPtcDict, ax.flatten(), ax2.flatten())):
            meanVec, varVec = fitPtcDict[amp][0], fitPtcDict[amp][1]
            pars, parsErr = fitPtcDict[amp][2], fitPtcDict[amp][3]

            if typeFitPtc == 'astierSingleBfPar':
                ptcA00, ptcA00error = pars[0], parsErr[0]
                ptcGain, ptcGainError = pars[1], parsErr[1]
                ptcNoise = np.sqrt(np.fabs(pars[2]))
                ptcNoiseError = 0.5*(parsErr[2]/np.fabs(pars[2]))*np.sqrt(np.fabs(pars[2]))
                stringLegend = f"a00: {ptcA00:.2e}+/-{ptcA00error:.2e}" + \
                               f"\n Gain: {ptcGain:.4}+/-{ptcGainError:.2e}" + \
                               f"\n Noise: {ptcNoise:.4}+/-{ptcNoiseError:.2e}"

            if typeFitPtc == 'polynomial':
                ptcGain, ptcGainError = 1./pars[1], np.fabs(1./pars[1])*(parsErr[1]/pars[1])
                ptcNoise = np.sqrt(np.fabs(pars[0]))*ptcGain
                ptcNoiseError = (0.5*(parsErr[0]/np.fabs(pars[0]))*(np.sqrt(np.fabs(pars[0]))))*ptcGain
                stringLegend = f"Noise: {ptcNoise:.4}+/-{ptcNoiseError:.2e} \n" \
                    + f"Gain: {ptcGain:.4}+/-{ptcGainError:.2e}"

            a.plot(meanVec, ptcFunc(pars, meanVec), color='red')
            a.scatter(meanVec, varVec)
            #if i in [12, 13, 14, 15]:  # Last row in 4x4 grid
            a.set_xlabel('Mean signal ($\mu$, ADU)', fontsize=8)
            a.set_xticks(meanVec)
            #if i in [0, 4, 8, 12]:  # First column in 4x4 grid
            a.set_ylabel(r'Variance (ADU$^2$)', fontsize=8)
            a.tick_params(labelsize=11)
            a.text(0.03, 0.8, stringLegend, transform=a.transAxes, fontsize=legendFontSize)
            a.set_xscale('linear', fontsize=7)
            a.set_yscale('linear', fontsize=7)
            a.set_title(amp, fontsize=10)

            # Same, but in log-scale
            a2.plot(meanVec, ptcFunc(pars, meanVec), color='red')
            a2.scatter(meanVec, varVec)
            a2.set_xlabel('Mean Signal ($\mu$, ADU)', fontsize=8)
            a2.set_xticks(meanVec)
            a2.set_ylabel(r'Variance (ADU$^2$)', fontsize=8)
            a2.tick_params(labelsize=11)
            a2.text(0.03, 0.8, stringLegend, transform=a2.transAxes, fontsize=legendFontSize)
            a2.set_xscale('log')
            a2.set_yscale('log')
            a2.set_title(amp, fontsize=10)

        #f.tight_layout()
        f.suptitle(f"PTC \n Fit: " + stringTitle, fontsize=20)
        pdfPages.savefig(f)
        #f2.tight_layout()
        f2.suptitle(f"PTC (log-log)", fontsize=20)
        pdfPages.savefig(f2)

        # Plot mean vs time
        f, ax = plt.subplots(nrows=4, ncols=4, sharex='col', sharey='row', figsize=(13, 10))
        for i, (amp, a) in enumerate(zip(fitPtcDict, ax.flatten())):
            timeVec, meanVec = nlDict[amp][0], nlDict[amp][1]
            pars, cov = nlDict[amp][3], nlDict[amp][4]
            c0, c0Error = pars[0], parsErr[0]
            c1, c1Error = pars[1], parsErr[1]
            c2, c2Error = pars[2], parsErr[2]
            stringLegend = f"c0: {c0:.4}+/-{c0Error:.2e}\n c1: {c1:.4}+/-{c1Error:.2e}" \
                + f"\n c2(NL): {c2:.2e}+/-{c2Error:.2e}"
            a.scatter(timeVec, meanVec)
            a.plot(timeVec, self.funcPolynomial(pars, timeVec), color='red')
            #if i in [12, 13, 14, 15]:  # Last row in 4x4 grid
            a.set_xlabel('Time (sec)', fontsize=10)
            a.set_xticks(timeVec)
            #if i in [0, 4, 8, 12]:  # First column in 4x4 grid
            a.set_ylabel('Mean signal ($\mu$, ADU)', fontsize=10)
            a.tick_params(labelsize=11)
            a.text(0.03, 0.75, stringLegend, transform=a.transAxes, fontsize=legendFontSize)
            a.set_xscale('linear', fontsize=7)
            a.set_yscale('linear', fontsize=7)
            a.set_title(amp, fontsize=10)

        #f.tight_layout()
        f.suptitle("Linearity \n Fit: " + r"$\mu = c_0 + c_1 t + c_2 t^2$", fontsize=20)
        pdfPages.savefig()

        # Plot linearity residual
        f, ax = plt.subplots(nrows=4, ncols=4, sharex='col', sharey='row', figsize=(13, 10))
        for i, (amp, a) in enumerate(zip(fitPtcDict, ax.flatten())):
            meanVec, linRes = nlDict[amp][1], nlDict[amp][2]
            a.scatter(meanVec, linRes)
            a.axhline(y=0, color='k')
            a.axvline(x=timeVec[2], color ='g', linestyle = '--')
            a.set_xlabel(r'Mean signal ($\mu$, ADU)', fontsize=10)
            a.set_xticks(meanVec)
            a.set_ylabel('LR (%)', fontsize=10)
            a.tick_params(labelsize=11)
            a.set_xscale('linear', fontsize=7)
            a.set_yscale('linear', fontsize=7)
            a.set_title(amp, fontsize=10)

        #f.tight_layout()
        f.suptitle(r"Linearity Residual: $100(1 - \mu_{\rm{ref}}/t_{\rm{ref}})/(\mu / t))$" + "\n" +
                r"$t_{\rm{ref}}$: " + f"{timeVec[2]} s", fontsize=20)
        pdfPages.savefig()

        return
