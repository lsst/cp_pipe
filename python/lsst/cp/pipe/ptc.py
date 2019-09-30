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

import lsst.afw.math as afwMath
import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase
from lsst.ip.isr import IsrTask
from .utils import (NonexistentDatasetTaskDataIdContainer, PairedVisitListTaskRunner,
                    checkExpLengthEqual, validateIsrConfig)
from scipy.optimize import leastsq
import numpy.polynomial.polynomial as poly


class MeasurePhotonTransferCurveTaskConfig(pexConfig.Config):
    """Config class for photon transfer curve measurement task"""
    isr = pexConfig.ConfigurableField(
        target=IsrTask,
        doc="""Task to perform instrumental signature removal.""",
    )
    isrMandatorySteps = pexConfig.ListField(
        dtype=str,
        doc="isr operations that must be performed for valid results. Raises if any of these are False.",
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
        doc="The key by which to pull a detector from a dataId, e.g. 'ccd' or 'detector'.",
        default='ccd',
    )
    makePlots = pexConfig.Field(
        dtype=bool,
        doc="Plot the PTC curves?.",
        default=False,
    )
    ptcFitType = pexConfig.ChoiceField(
        dtype=str,
        doc="Fit PTC to approximation in Astier+19 (Equation 16) or to a polynomial.",
        default="POLYNOMIAL",
        allowed={
            "POLYNOMIAL": "n-degree polynomial (use 'polynomialFitDegree' to set 'n').",
            "ASTIERAPPROXIMATION": "Approximation in Astier+19 (Eq. 16)."
        }
    )
    polynomialFitDegree = pexConfig.Field(
        dtype=int,
        doc="Degree of polynomial to fit the PTC, when 'ptcFitType'=POLYNOMIAL.",
        default=2,
    )
    binSize = pexConfig.Field(
        dtype=int,
        doc="Bin the image by this factor in both dimensions.",
        default=1,
    )
    minMeanSignal = pexConfig.Field(
        dtype=float,
        doc="Minimum value of mean signal (in ADU) to consider.",
        default=0,
    )
    maxMeanSignal = pexConfig.Field(
        dtype=float,
        doc="Maximum value to of mean signal (in ADU) to consider.",
        default=9e6,
    )
    sigmaCutPtcOutliers = pexConfig.Field(
        dtype=float,
        doc="Sigma cut for outlier rejection in PTC.",
        default=4.0,
    )
    maxIterationsPtcOutliers = pexConfig.Field(
        dtype=int,
        doc="Maximum number of iterations for outlier rejection in PTC.",
        default=2,
    )
    doFitBootstrap = pexConfig.Field(
        dtype=bool,
        doc="Use bootstrap for the PTC fit parameters and errors?.",
        default=False,
    )
    linResidualTimeIndex = pexConfig.Field(
        dtype=int,
        doc="Index position in time array for reference time in linearity residual calculation.",
        default=2,
    )


class MeasurePhotonTransferCurveTask(pipeBase.CmdLineTask):
    """A class to calculate, fit, and plot a PTC from a set of flat pairs.

    The Photon Transfer Curve (var(signal) vs mean(signal)) is a standard tool
    used in astronomical detectors characterization (e.g., Janesick 2001,
    Janesick 2007). This task calculates the PTC from a series of pairs of
    flat-field images; each pair taken at identical exposure times. The
    difference image of each pair is formed to eliminate fixed pattern noise,
    and then the variance of the difference image and the mean of the average image
    are used to produce the PTC. An n-degree polynomial or the approximation in Equation
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
        time.

    """

    RunnerClass = PairedVisitListTaskRunner
    ConfigClass = MeasurePhotonTransferCurveTaskConfig
    _DefaultName = "measurePhotonTransferCurve"

    def __init__(self, *args, **kwargs):
        pipeBase.CmdLineTask.__init__(self, *args, **kwargs)
        self.makeSubtask("isr")
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
        parser.add_id_argument("--id", datasetType="measurePhotonTransferCurveGainAndNoise",
                               ContainerClass=NonexistentDatasetTaskDataIdContainer,
                               help="The ccds to use, e.g. --id ccd=0..100")
        return parser

    @pipeBase.timeMethod
    def runDataRef(self, dataRef, visitPairs):
        """Run the Photon Transfer Curve (PTC) measurement task.

        For a dataRef (which is each detector here),
        and given a list of visit pairs at different exposure times,
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

        for (v1, v2) in visitPairs:
            # Perform ISR on each exposure
            dataRef.dataId['visit'] = v1
            exp1 = self.isr.runDataRef(dataRef).exposure
            dataRef.dataId['visit'] = v2
            exp2 = self.isr.runDataRef(dataRef).exposure
            del dataRef.dataId['visit']

            checkExpLengthEqual(exp1, exp2, v1, v2, raiseWithMessage=True)
            expTime = exp1.getInfo().getVisitInfo().getExposureTime()

            for amp in detector:
                mu, varDiff = self.measureMeanVarPair(exp1, exp2, region=amp.getBBox())
                data = dict(expTime=expTime, meanClip=mu, varClip=varDiff)
                ampName = amp.getName()
                dataDict[ampName][(v1, v2)] = data
                fitVectorsDict[ampName][0].append(expTime)
                fitVectorsDict[ampName][1].append(mu)
                fitVectorsDict[ampName][2].append(varDiff)

        # Fit PTC and (non)linearity of signal vs time curve
        fitPtcDict, nlDict, gainDict, noiseDict = self.fitPtcAndNl(fitVectorsDict,
                                                                   ptcFitType=self.config.ptcFitType)
        allDict = {"data": dataDict, "ptc": fitPtcDict, "nl": nlDict}
        gainNoiseNlDict = {"gain": gainDict, "noise": noiseDict, "nl": nlDict}

        if self.config.makePlots:
            self.plot(dataRef, fitPtcDict, nlDict, ptcFitType=self.config.ptcFitType)

        # Save data, PTC fit, and NL fit dictionaries
        self.log.info(f"Writing PTC and NL data to {dataRef.getUri(write=True)}")
        dataRef.put(gainNoiseNlDict, datasetType="measurePhotonTransferCurveGainAndNoise")
        dataRef.put(allDict, datasetType="measurePhotonTransferCurveDatasetAll")

        self.log.info('Finished measuring PTC for in detector %s' % detNum)

        return pipeBase.Struct(exitStatus=0)

    def measureMeanVarPair(self, exposure1, exposure2, region=None):
        """Calculate the mean signal of two exposures and the variance of their difference.

        Parameters
        ----------
        exposure1 : `lsst.afw.image.exposure.exposure.ExposureF`
            First exposure of flat field pair.

        exposure2 : `lsst.afw.image.exposure.exposure.ExposureF`
            Second exposure of flat field pair.

        region : `lsst.geom.Box2I`
            Region of each exposure where to perform the calculations (e.g, an amplifier).

        Return
        ------

        mu : `np.float`
            0.5*(mu1 + mu2), where mu1, and mu2 are the clipped means of the regions in
            both exposures.

        varDiff : `np.float`
            Half of the clipped variance of the difference of the regions inthe two input
            exposures.
        """

        if region is not None:
            im1Area = exposure1.maskedImage[region]
            im2Area = exposure2.maskedImage[region]
        else:
            im1Area = exposure1.maskedImage
            im2Area = exposure2.maskedImage

        im1Area = afwMath.binImage(im1Area, self.config.binSize)
        im2Area = afwMath.binImage(im2Area, self.config.binSize)

        #  Clipped mean of images; then average of mean.
        mu1 = afwMath.makeStatistics(im1Area, afwMath.MEANCLIP).getValue()
        mu2 = afwMath.makeStatistics(im2Area, afwMath.MEANCLIP).getValue()
        mu = 0.5*(mu1 + mu2)

        # Take difference of pairs
        # symmetric formula: diff = (mu2*im1-mu1*im2)/(0.5*(mu1+mu2))
        temp = im2Area.clone()
        temp *= mu1
        diffIm = im1Area.clone()
        diffIm *= mu2
        diffIm -= temp
        diffIm /= mu

        varDiff = 0.5*(afwMath.makeStatistics(diffIm, afwMath.VARIANCECLIP).getValue())

        return mu, varDiff

    def _fitLeastSq(self, initialParams, dataX, dataY, function):
        """Do a fit and estimate the parameter errors using using scipy.optimize.leastq.

        optimize.leastsq returns the fractional covariance matrix. To estimate the
        standard deviation of the fit parameters, multiply the entries of this matrix
        by the reduced chi squared and take the square root of the diagon al elements.

        Parameters
        ----------
        initialParams : list of np.float
            initial values for fit parameters. For ptcFitType=POLYNOMIAL, its length
            determines the degree of the polynomial.

        dataX : np.array of np.float
            Data in the abscissa axis.

        dataY : np.array of np.float
            Data in the ordinate axis.

        function : callable object (function)
            Function to fit the data with.

        Return
        ------
        pFitSingleLeastSquares : list of np.float
            List with fitted parameters.

        pErrSingleLeastSquares : list of np.float
            List with errors for fitted parameters.
        """

        def errFunc(p, x, y):
            return function(p, x) - y

        pFit, pCov, infoDict, errMessage, success = leastsq(errFunc, initialParams,
                                                            args=(dataX, dataY), full_output=1, epsfcn=0.0001)

        if (len(dataY) > len(initialParams)) and pCov is not None:
            reducedChiSq = (errFunc(pFit, dataX, dataY)**2).sum()/(len(dataY)-len(initialParams))
            pCov *= reducedChiSq
        else:
            pCov = np.inf

        errorVec = []
        for i in range(len(pFit)):
            errorVec.append(np.fabs(pCov[i][i])**0.5)

        pFitSingleLeastSquares = pFit
        pErrSingleLeastSquares = np.array(errorVec)

        return pFitSingleLeastSquares, pErrSingleLeastSquares

    def _fitBootstrap(self, initialParams, dataX, dataY, function, confidenceSigma=1.):
        """Do a fit using least squares and bootstrap to estimate parameter errors.

        The bootstrap error bars are calculated by fitting 100 random data sets.

        Parameters
        ----------
        initialParams : list of np.float
            initial values for fit parameters. For ptcFitType=POLYNOMIAL, its length
            determines the degree of the polynomial.

        dataX : np.array of np.float
            Data in the abscissa axis.

        dataY : np.array of np.float
            Data in the ordinate axis.

        function : callable object (function)
            Function to fit the data with.

        confidenceSigma : np.float
            Number of sigmas that determine confidence interval for the bootstrap errors.

        Return
        ------
        pFitBootstrap : list of np.float
            List with fitted parameters.

        pErrBootstrap : list of np.float
            List with errors for fitted parameters.
        """

        def errFunc(p, x, y):
            return function(p, x) - y

        # Fit first time
        pFit, _ = leastsq(errFunc, initialParams, args=(dataX, dataY), full_output=0)

        # Get the stdev of the residuals
        residuals = errFunc(pFit, dataX, dataY)
        sigmaErrTotal = np.std(residuals)

        # 100 random data sets are generated and fitted
        pars = []
        for i in range(100):
            randomDelta = np.random.normal(0., sigmaErrTotal, len(dataY))
            randomDataY = dataY + randomDelta
            randomFit, _ = leastsq(errFunc, initialParams,
                                   args=(dataX, randomDataY), full_output=0)
            pars.append(randomFit)
        pars = np.array(pars)
        meanPfit = np.mean(pars, 0)

        # confidence interval for parameter estimates
        nSigma = confidenceSigma
        errPfit = nSigma*np.std(pars, 0)
        pFitBootstrap = meanPfit
        pErrBootstrap = errPfit
        return pFitBootstrap, pErrBootstrap

    def funcPolynomial(self, pars, x):
        """Polynomial function definition"""
        return poly.polyval(x, [*pars])

    def funcAstier(self, pars, x):
        """Single brighter-fatter parameter model for PTC; Equation 16 of Astier+19"""
        a00, gain, noise = pars
        return 0.5/(a00*gain*gain)*(np.exp(2*a00*x*gain)-1) + noise/(gain*gain)

    def fitPtcAndNl(self, fitVectorsDict, ptcFitType='POLYNOMIAL'):
        """Function to fit PTC, and calculate linearity and linearity residual

        Parameters
        ----------
        fitVectorsDicti : `dict`
            Dictionary with exposure time, mean, and variance vectors in a tuple
        ptcFitType : `str`
            Fit a 'POLYNOMIAL' (degree: 'polynomialFitDegree') or '
            ASTIERAPPROXIMATION' to the PTC

        Returns
        -------
        fitPtcDict : `dict`
            Dictionary of the form fitPtcDict[amp] =
            (meanVec, varVec, parsFit, parsFitErr, index)
        nlDict : `dict`
            Dictionary of the form nlDict[amp] =
            (timeVec, meanVec, linResidual, parsFit, parsFitErr)
        """
        if ptcFitType == 'ASTIERAPPROXIMATION':
            ptcFunc = self.funcAstier
            parsIniPtc = [-1e-9, 1.0, 10.]  # a00, gain, noise
        if ptcFitType == 'POLYNOMIAL':
            ptcFunc = self.funcPolynomial
            parsIniPtc = np.repeat(1., self.config.polynomialFitDegree + 1)

        parsIniNl = [1., 1., 1.]
        fitPtcDict = {key: {} for key in fitVectorsDict}
        nlDict = {key: {} for key in fitVectorsDict}
        gainDict = {key: {} for key in fitVectorsDict}
        noiseDict = {key: {} for key in fitVectorsDict}

        def errFunc(p, x, y):
            return ptcFunc(p, x) - y

        maxIterationsPtcOutliers = self.config.maxIterationsPtcOutliers
        for amp in fitVectorsDict:
            timeVec, meanVec, varVec = fitVectorsDict[amp]
            timeVecOriginal = np.array(timeVec)
            meanVecOriginal = np.array(meanVec)
            varVecOriginal = np.array(varVec)
            index0 = ((meanVecOriginal > self.config.minMeanSignal) &
                      (meanVecOriginal <= self.config.maxMeanSignal))
            #  Before bootstrap fit, do an iterative fit to get rid of outliers in PTC
            count = 1
            sigmaCutPtcOutliers = self.config.sigmaCutPtcOutliers
            maxIterationsPtcOutliers = self.config.maxIterationsPtcOutliers
            timeTempVec = timeVecOriginal[index0]
            meanTempVec = meanVecOriginal[index0]
            varTempVec = varVecOriginal[index0]
            while count <= maxIterationsPtcOutliers:
                pars, cov = leastsq(errFunc, parsIniPtc, args=(meanTempVec,
                                    varTempVec), full_output=0)
                sigResids = (varTempVec -
                             ptcFunc(pars, meanTempVec))/np.sqrt(varTempVec)
                index = list(np.where(np.abs(sigResids) < sigmaCutPtcOutliers)[0])
                timeTempVec = timeTempVec[index]
                meanTempVec = meanTempVec[index]
                varTempVec = varTempVec[index]
                count += 1

            parsIniPtc = pars
            timeVecFinal, meanVecFinal, varVecFinal = timeTempVec, meanTempVec, varTempVec
            if (len(meanVecFinal) - len(meanVecOriginal)) > 0:
                self.log.info((f"Number of points discarded in PTC of amplifier {amp}:" +
                               "{len(meanVecFinal)-len(meanVecOriginal)} out of {len(meanVecOriginal)}"))

            if (len(meanVecFinal) < len(parsIniPtc)):
                raise RuntimeError(f"Not enough data points ({len(meanVecFinal)}) compared to the number of" +
                                   "parameters of the PTC model({len(parsIniPtc)}).")
            # Fit the PTC
            if self.config.doFitBootstrap:
                parsFit, parsFitErr = self._fitBootstrap(parsIniPtc, meanVecFinal, varVecFinal, ptcFunc)
            else:
                parsFit, parsFitErr = self._fitLeastSq(parsIniPtc, meanVecFinal, varVecFinal, ptcFunc)

            fitPtcDict[amp] = (timeVecOriginal, meanVecOriginal, varVecOriginal, timeVecFinal,
                               meanVecFinal, varVecFinal, parsFit, parsFitErr)

            if ptcFitType == 'ASTIERAPPROXIMATION':
                ptcGain = parsFit[1]
                ptcGainErr = parsFitErr[1]
                ptcNoise = np.sqrt(np.fabs(parsFit[2]))
                ptcNoiseErr = 0.5*(parsFitErr[2]/np.fabs(parsFit[2]))*np.sqrt(np.fabs(parsFit[2]))
            if ptcFitType == 'POLYNOMIAL':
                ptcGain = 1./parsFit[1]
                ptcGainErr = np.fabs(1./parsFit[1])*(parsFitErr[1]/parsFit[1])
                ptcNoise = np.sqrt(np.fabs(parsFit[0]))*ptcGain
                ptcNoiseErr = (0.5*(parsFitErr[0]/np.fabs(parsFit[0]))*(np.sqrt(np.fabs(parsFit[0]))))*ptcGain

            gainDict[amp] = (ptcGain, ptcGainErr)
            noiseDict[amp] = (ptcNoise, ptcNoiseErr)

            # Non-linearity residuals (NL of mean vs time curve): percentage, and fit to a quadratic function
            # In this case, len(parsIniNl) = 3 indicates that we want a quadratic fit
            if self.config.doFitBootstrap:
                parsFit, parsFitErr = self._fitBootstrap(parsIniNl, timeVecFinal, meanVecFinal,
                                                         self.funcPolynomial)
            else:
                parsFit, parsFitErr = self._fitLeastSq(parsIniNl, timeVecFinal, meanVecFinal,
                                                       self.funcPolynomial)
            linResidualTimeIndex = self.config.linResidualTimeIndex
            if timeVecFinal[linResidualTimeIndex] == 0.0:
                raise RuntimeError("Reference time for linearity residual can't be 0.0")
            linResidual = 100*(1 - ((meanVecFinal[linResidualTimeIndex] /
                                     timeVecFinal[linResidualTimeIndex]) / (meanVecFinal/timeVecFinal)))
            nlDict[amp] = (timeVecFinal, meanVecFinal, linResidual, parsFit, parsFitErr)

        return fitPtcDict, nlDict, gainDict, noiseDict

    def plot(self, dataRef, fitPtcDict, nlDict, ptcFitType='POLYNOMIAL'):
        dirname = dataRef.getUri(datasetType='cpPipePlotRoot', write=True)
        if not os.path.exists(dirname):
            os.makedirs(dirname)

        detNum = dataRef.dataId[self.config.ccdKey]
        filename = f"PTC_det{detNum}.pdf"
        filenameFull = os.path.join(dirname, filename)
        with PdfPages(filenameFull) as pdfPages:
            self._plotPtc(fitPtcDict, nlDict, ptcFitType, pdfPages)

    def _plotPtc(self, fitPtcDict, nlDict, ptcFitType, pdfPages):
        """Plot PTC, linearity, and linearity residual per amplifier"""

        if ptcFitType == 'ASTIERAPPROXIMATION':
            ptcFunc = self.funcAstier
            stringTitle = r"Var = $\frac{1}{2g^2a_{00}}(\exp (2a_{00} \mu g) - 1) + \frac{n_{00}}{g^2}$"

        if ptcFitType == 'POLYNOMIAL':
            ptcFunc = self.funcPolynomial
            stringTitle = f"Polynomial (degree: {self.config.polynomialFitDegree})"

        legendFontSize = 7.5
        labelFontSize = 8
        titleFontSize = 10
        supTitleFontSize = 18

        # General determination of the size of the plot grid
        nAmps = len(fitPtcDict)
        if nAmps == 2:
            nRows, nCols = 2, 1
        nRows = np.sqrt(nAmps)
        mantissa, _ = np.modf(nRows)
        if mantissa > 0:
            nRows = int(nRows) + 1
            nCols = nRows
        else:
            nRows = int(nRows)
            nCols = nRows

        f, ax = plt.subplots(nrows=nRows, ncols=nCols, sharex='col', sharey='row', figsize=(13, 10))
        f2, ax2 = plt.subplots(nrows=nRows, ncols=nCols, sharex='col', sharey='row', figsize=(13, 10))

        # fitPtcDict[amp] = (timeVecOriginal, meanVecOriginal, varVecOriginal, timeVecFinal,
        #                    meanVecFinal, varVecFinal, parsFit, parsFitErr)
        for i, (amp, a, a2) in enumerate(zip(fitPtcDict, ax.flatten(), ax2.flatten())):
            meanVecOriginal, varVecOriginal = fitPtcDict[amp][1], fitPtcDict[amp][2]
            meanVecFinal, varVecFinal = fitPtcDict[amp][4], fitPtcDict[amp][5]
            meanVecOutliers = np.setdiff1d(meanVecOriginal, meanVecFinal)
            varVecOutliers = np.setdiff1d(varVecOriginal, varVecFinal)
            pars, parsErr = fitPtcDict[amp][6], fitPtcDict[amp][7]

            if ptcFitType == 'ASTIERAPPROXIMATION':
                ptcA00, ptcA00error = pars[0], parsErr[0]
                ptcGain, ptcGainError = pars[1], parsErr[1]
                ptcNoise = np.sqrt(np.fabs(pars[2]))
                ptcNoiseError = 0.5*(parsErr[2]/np.fabs(pars[2]))*np.sqrt(np.fabs(pars[2]))
                stringLegend = (f"a00: {ptcA00:.2e}+/-{ptcA00error:.2e}"
                                f"\n Gain: {ptcGain:.4}+/-{ptcGainError:.2e}"
                                f"\n Noise: {ptcNoise:.4}+/-{ptcNoiseError:.2e}")

            if ptcFitType == 'POLYNOMIAL':
                ptcGain, ptcGainError = 1./pars[1], np.fabs(1./pars[1])*(parsErr[1]/pars[1])
                ptcNoise = np.sqrt(np.fabs(pars[0]))*ptcGain
                ptcNoiseError = (0.5*(parsErr[0]/np.fabs(pars[0]))*(np.sqrt(np.fabs(pars[0]))))*ptcGain
                stringLegend = (f"Noise: {ptcNoise:.4}+/-{ptcNoiseError:.2e} \n"
                                f"Gain: {ptcGain:.4}+/-{ptcGainError:.2e}")

            minMeanVecFinal = np.min(meanVecFinal)
            maxMeanVecFinal = np.max(meanVecFinal)
            meanVecFit = np.linspace(minMeanVecFinal, maxMeanVecFinal, 100*len(meanVecFinal))
            minMeanVecOriginal = np.min(meanVecOriginal)
            maxMeanVecOriginal = np.max(meanVecOriginal)
            deltaXlim = maxMeanVecOriginal - minMeanVecOriginal

            a.plot(meanVecFit, ptcFunc(pars, meanVecFit), color='red')
            a.plot(meanVecFinal, pars[0] + pars[1]*meanVecFinal, color='green', linestyle='--')
            a.scatter(meanVecFinal, varVecFinal, c='blue', marker='o')
            a.scatter(meanVecOutliers, varVecOutliers, c='magenta', marker='s')
            a.set_xlabel(r'Mean signal ($\mu$, ADU)', fontsize=labelFontSize)
            a.set_xticks(meanVecOriginal)
            a.set_ylabel(r'Variance (ADU$^2$)', fontsize=labelFontSize)
            a.tick_params(labelsize=11)
            a.text(0.03, 0.8, stringLegend, transform=a.transAxes, fontsize=legendFontSize)
            a.set_xscale('linear', fontsize=labelFontSize)
            a.set_yscale('linear', fontsize=labelFontSize)
            a.set_title(amp, fontsize=titleFontSize)
            a.set_xlim([minMeanVecOriginal - 0.2*deltaXlim, maxMeanVecOriginal + 0.2*deltaXlim])

            # Same, but in log-scale
            a2.plot(meanVecFit, ptcFunc(pars, meanVecFit), color='red')
            a2.scatter(meanVecFinal, varVecFinal, c='blue', marker='o')
            a2.scatter(meanVecOutliers, varVecOutliers, c='magenta', marker='s')
            a2.set_xlabel(r'Mean Signal ($\mu$, ADU)', fontsize=labelFontSize)
            a2.set_ylabel(r'Variance (ADU$^2$)', fontsize=labelFontSize)
            a2.tick_params(labelsize=11)
            a2.text(0.03, 0.8, stringLegend, transform=a2.transAxes, fontsize=legendFontSize)
            a2.set_xscale('log')
            a2.set_yscale('log')
            a2.set_title(amp, fontsize=titleFontSize)
            a2.set_xlim([minMeanVecOriginal, maxMeanVecOriginal])

        f.suptitle(f"PTC \n Fit: " + stringTitle, fontsize=20)
        pdfPages.savefig(f)
        f2.suptitle(f"PTC (log-log)", fontsize=20)
        pdfPages.savefig(f2)

        # Plot mean vs time
        f, ax = plt.subplots(nrows=4, ncols=4, sharex='col', sharey='row', figsize=(13, 10))
        for i, (amp, a) in enumerate(zip(fitPtcDict, ax.flatten())):
            timeVecFinal, meanVecFinal = nlDict[amp][0], nlDict[amp][1]
            pars, _ = nlDict[amp][3], nlDict[amp][4]
            c0, c0Error = pars[0], parsErr[0]
            c1, c1Error = pars[1], parsErr[1]
            c2, c2Error = pars[2], parsErr[2]
            stringLegend = f"c0: {c0:.4}+/-{c0Error:.2e}\n c1: {c1:.4}+/-{c1Error:.2e}" \
                + f"\n c2(NL): {c2:.2e}+/-{c2Error:.2e}"
            a.scatter(timeVecFinal, meanVecFinal)
            a.plot(timeVecFinal, self.funcPolynomial(pars, timeVecFinal), color='red')
            a.set_xlabel('Time (sec)', fontsize=labelFontSize)
            a.set_xticks(timeVecFinal)
            a.set_ylabel(r'Mean signal ($\mu$, ADU)', fontsize=labelFontSize)
            a.tick_params(labelsize=labelFontSize)
            a.text(0.03, 0.75, stringLegend, transform=a.transAxes, fontsize=legendFontSize)
            a.set_xscale('linear', fontsize=labelFontSize)
            a.set_yscale('linear', fontsize=labelFontSize)
            a.set_title(amp, fontsize=titleFontSize)

        f.suptitle("Linearity \n Fit: " + r"$\mu = c_0 + c_1 t + c_2 t^2$", fontsize=supTitleFontSize)
        pdfPages.savefig()

        # Plot linearity residual
        f, ax = plt.subplots(nrows=4, ncols=4, sharex='col', sharey='row', figsize=(13, 10))
        for i, (amp, a) in enumerate(zip(fitPtcDict, ax.flatten())):
            meanVecFinal, linRes = nlDict[amp][1], nlDict[amp][2]
            a.scatter(meanVecFinal, linRes)
            a.axhline(y=0, color='k')
            a.axvline(x=timeVecFinal[self.config.linResidualTimeIndex], color ='g', linestyle = '--')
            a.set_xlabel(r'Mean signal ($\mu$, ADU)', fontsize=labelFontSize)
            a.set_xticks(meanVecFinal)
            a.set_ylabel('LR (%)', fontsize=labelFontSize)
            a.tick_params(labelsize=labelFontSize)
            a.set_xscale('linear', fontsize=labelFontSize)
            a.set_yscale('linear', fontsize=labelFontSize)
            a.set_title(amp, fontsize=titleFontSize)

        f.suptitle(r"Linearity Residual: $100(1 - \mu_{\rm{ref}}/t_{\rm{ref}})/(\mu / t))$" + "\n" +
                   r"$t_{\rm{ref}}$: " + f"{timeVecFinal[2]} s", fontsize=supTitleFontSize)
        pdfPages.savefig()

        return
