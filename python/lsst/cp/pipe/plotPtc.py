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

__all__ = ['PlotPhotonTransferCurveTask']

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import gridspec
import os
from matplotlib.backends.backend_pdf import PdfPages

import lsst.ip.isr as isr

from .utils import (funcAstier, funcPolynomial,
                    calculateWeightedReducedChi2)
from matplotlib.ticker import MaxNLocator

from .astierCovPtcFit import computeApproximateAcoeffs
from .astierCovPtcUtils import getFitDataFromCovariances

from lsst.ip.isr import PhotonTransferCurveDataset

import lsst.log as lsstLog


class PlotPhotonTransferCurveTask():
    """A class to plot the dataset from MeasurePhotonTransferCurveTask.

    Parameters
    ----------

    datasetFileName : `str`
        datasetPtc (lsst.ip.isr.PhotonTransferCurveDataset) file
        name (fits).

    linearizerFileName : `str`, optional
        linearizer (isr.linearize.Linearizer) file
        name (fits).

    outDir : `str`, optional
        Path to the output directory where the final PDF will
        be placed.

    detNum : `int`, optional
        Detector number.

    signalElectronsRelativeA : `float`, optional
        Signal value for relative systematic bias between different
        methods of estimating a_ij (Fig. 15 of Astier+19).

    plotNormalizedCovariancesNumberOfBins : `float`, optional
        Number of bins in `plotNormalizedCovariancesNumber` function
        (Fig. 8, 10., of Astier+19).
    """

    def __init__(self, datasetFilename, linearizerFileName=None,
                 outDir='.', detNum=999, signalElectronsRelativeA=75000,
                 plotNormalizedCovariancesNumberOfBins=10):
        self.datasetFilename = datasetFilename
        self.linearizerFileName = linearizerFileName
        self.detNum = detNum
        self.signalElectronsRelativeA = signalElectronsRelativeA
        self.plotNormalizedCovariancesNumberOfBins = plotNormalizedCovariancesNumberOfBins
        self.outDir = outDir

    def runDataRef(self):
        """Run the Photon Transfer Curve (PTC) plotting measurement task.
        """

        datasetFile = self.datasetFilename
        datasetPtc = PhotonTransferCurveDataset.readFits(datasetFile)

        dirname = self.outDir
        if not os.path.exists(dirname):
            os.makedirs(dirname)

        detNum = self.detNum
        filename = f"PTC_det{detNum}.pdf"
        filenameFull = os.path.join(dirname, filename)

        if self.linearizerFileName:
            linearizer = isr.linearize.Linearizer.readFits(self.linearizerFileName)
        else:
            linearizer = None
        self.run(filenameFull, datasetPtc, linearizer=linearizer, log=lsstLog)

        return

    def run(self, filenameFull, datasetPtc, linearizer=None, log=None):
        """Make the plots for the PTC task"""
        ptcFitType = datasetPtc.ptcFitType
        with PdfPages(filenameFull) as pdfPages:
            if ptcFitType in ["FULLCOVARIANCE", ]:
                self.covAstierMakeAllPlots(datasetPtc, pdfPages, log=log)
            elif ptcFitType in ["EXPAPPROXIMATION", "POLYNOMIAL"]:
                self._plotStandardPtc(datasetPtc, ptcFitType, pdfPages)
            else:
                raise RuntimeError(f"The input dataset had an invalid dataset.ptcFitType: {ptcFitType}. \n" +
                                   "Options: 'FULLCOVARIANCE', EXPAPPROXIMATION, or 'POLYNOMIAL'.")
            if linearizer:
                self._plotLinearizer(datasetPtc, linearizer, pdfPages)

        return

    def covAstierMakeAllPlots(self, dataset, pdfPages,
                              log=None):
        """Make plots for MeasurePhotonTransferCurve task when doCovariancesAstier=True.

        This function call other functions that mostly reproduce the plots in Astier+19.
        Most of the code is ported from Pierre Astier's repository https://github.com/PierreAstier/bfptc

        Parameters
        ----------
        dataset : `lsst.ip.isr.ptcDataset.PhotonTransferCurveDataset`
            The dataset containing the necessary information to produce the plots.

        pdfPages: `matplotlib.backends.backend_pdf.PdfPages`
            PDF file where the plots will be saved.

        log : `lsst.log.Log`, optional
            Logger to handle messages
        """
        mu = dataset.finalMeans
        # dictionaries with ampNames as keys
        fullCovs = dataset.covariances
        fullCovsModel = dataset.covariancesModel
        fullCovWeights = dataset.covariancesSqrtWeights
        aDict = dataset.aMatrix
        bDict = dataset.bMatrix
        fullCovsNoB = dataset.covariances
        fullCovsModelNoB = dataset.covariancesModelNoB
        fullCovWeightsNoB = dataset.covariancesSqrtWeights
        aDictNoB = dataset.aMatrixNoB
        gainDict = dataset.gain
        noiseDict = dataset.noise

        self.plotCovariances(mu, fullCovs, fullCovsModel, fullCovWeights, fullCovsNoB, fullCovsModelNoB,
                             fullCovWeightsNoB, gainDict, noiseDict, aDict, bDict, pdfPages)
        self.plotNormalizedCovariances(0, 0, mu, fullCovs, fullCovsModel, fullCovWeights, fullCovsNoB,
                                       fullCovsModelNoB, fullCovWeightsNoB, pdfPages,
                                       offset=0.01, topPlot=True,
                                       numberOfBins=self.plotNormalizedCovariancesNumberOfBins,
                                       log=log)
        self.plotNormalizedCovariances(0, 1, mu, fullCovs, fullCovsModel, fullCovWeights, fullCovsNoB,
                                       fullCovsModelNoB, fullCovWeightsNoB, pdfPages,
                                       numberOfBins=self.plotNormalizedCovariancesNumberOfBins,
                                       log=log)
        self.plotNormalizedCovariances(1, 0, mu, fullCovs, fullCovsModel, fullCovWeights, fullCovsNoB,
                                       fullCovsModelNoB, fullCovWeightsNoB, pdfPages,
                                       numberOfBins=self.plotNormalizedCovariancesNumberOfBins,
                                       log=log)
        self.plot_a_b(aDict, bDict, pdfPages)
        self.ab_vs_dist(aDict, bDict, pdfPages, bRange=4)
        self.plotAcoeffsSum(aDict, bDict, pdfPages)
        self.plotRelativeBiasACoeffs(aDict, aDictNoB, fullCovsModel, fullCovsModelNoB,
                                     self.signalElectronsRelativeA, gainDict, pdfPages, maxr=4)

        return

    @staticmethod
    def plotCovariances(mu, covs, covsModel, covsWeights, covsNoB, covsModelNoB, covsWeightsNoB,
                        gainDict, noiseDict, aDict, bDict, pdfPages):
        """Plot covariances and models: Cov00, Cov10, Cov01.

        Figs. 6 and 7 of Astier+19

        Parameters
        ----------
        mu : `dict`, [`str`, `list`]
            Dictionary keyed by amp name with mean signal values.

        covs : `dict`, [`str`, `list`]
            Dictionary keyed by amp names containing a list of measued covariances per mean flux.

        covsModel : `dict`, [`str`, `list`]
            Dictionary keyed by amp names containinging covariances model (Eq. 20 of Astier+19) per mean flux.

        covsWeights : `dict`, [`str`, `list`]
            Dictionary keyed by amp names containinging sqrt. of covariances weights.

        covsNoB : `dict`, [`str`, `list`]
            Dictionary keyed by amp names containing a list of measued covariances per mean flux ('b'=0 in
            Astier+19).

        covsModelNoB : `dict`, [`str`, `list`]
            Dictionary keyed by amp names containing covariances model (with 'b'=0 in Eq. 20 of Astier+19)
            per mean flux.

        covsWeightsNoB : `dict`, [`str`, `list`]
            Dictionary keyed by amp names containing sqrt. of covariances weights ('b' = 0 in Eq. 20 of
            Astier+19).

        gainDict : `dict`, [`str`, `float`]
            Dictionary keyed by amp names containing the gains in e-/ADU.

        noiseDict : `dict`, [`str`, `float`]
            Dictionary keyed by amp names containing the rms redout noise in e-.

        aDict : `dict`, [`str`, `numpy.array`]
            Dictionary keyed by amp names containing 'a' coefficients (Eq. 20 of Astier+19).

        bDict : `dict`, [`str`, `numpy.array`]
            Dictionary keyed by amp names containing 'b' coefficients (Eq. 20 of Astier+19).

        pdfPages: `matplotlib.backends.backend_pdf.PdfPages`
            PDF file where the plots will be saved.
        """

        legendFontSize = 6.5
        labelFontSize = 7
        titleFontSize = 9
        supTitleFontSize = 18
        markerSize = 25

        nAmps = len(covs)
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
        fResCov00, axResCov00 = plt.subplots(nrows=nRows, ncols=nCols, sharex='col', sharey='row',
                                             figsize=(13, 10))
        fCov01, axCov01 = plt.subplots(nrows=nRows, ncols=nCols, sharex='col', sharey='row', figsize=(13, 10))
        fCov10, axCov10 = plt.subplots(nrows=nRows, ncols=nCols, sharex='col', sharey='row', figsize=(13, 10))

        assert(len(covsModel) == nAmps)
        assert(len(covsWeights) == nAmps)

        assert(len(covsNoB) == nAmps)
        assert(len(covsModelNoB) == nAmps)
        assert(len(covsWeightsNoB) == nAmps)

        for i, (amp, a, a2, aResVar, a3, a4) in enumerate(zip(covs, ax.flatten(),
                                                              ax2.flatten(), axResCov00.flatten(),
                                                              axCov01.flatten(), axCov10.flatten())):

            muAmp, cov, model, weight = mu[amp], covs[amp], covsModel[amp], covsWeights[amp]
            if not np.isnan(np.array(cov)).all():  # If all the entries are np.nan, this is a bad amp.
                aCoeffs, bCoeffs = np.array(aDict[amp]), np.array(bDict[amp])
                gain, noise = gainDict[amp], noiseDict[amp]
                (meanVecFinal, varVecFinal, varVecModelFinal,
                    varWeightsFinal, _) = getFitDataFromCovariances(0, 0, muAmp, cov, model, weight,
                                                                    returnMasked=True)

                # Get weighted reduced chi2
                chi2FullModelVar = calculateWeightedReducedChi2(varVecFinal, varVecModelFinal,
                                                                varWeightsFinal, len(meanVecFinal), 4)

                (meanVecFinalCov01, varVecFinalCov01, varVecModelFinalCov01,
                    _, _) = getFitDataFromCovariances(0, 0, muAmp, cov, model, weight, returnMasked=True)

                (meanVecFinalCov10, varVecFinalCov10, varVecModelFinalCov10,
                    _, _) = getFitDataFromCovariances(1, 0, muAmp, cov, model, weight, returnMasked=True)

                # cuadratic fit for residuals below
                par2 = np.polyfit(meanVecFinal, varVecFinal, 2, w=varWeightsFinal)
                varModelFinalQuadratic = np.polyval(par2, meanVecFinal)
                chi2QuadModelVar = calculateWeightedReducedChi2(varVecFinal, varModelFinalQuadratic,
                                                                varWeightsFinal, len(meanVecFinal), 3)

                # fit with no 'b' coefficient (c = a*b in Eq. 20 of Astier+19)
                covNoB, modelNoB, weightNoB = covsNoB[amp], covsModelNoB[amp], covsWeightsNoB[amp]
                (meanVecFinalNoB, varVecFinalNoB, varVecModelFinalNoB,
                 varWeightsFinalNoB, _) = getFitDataFromCovariances(0, 0, muAmp, covNoB, modelNoB,
                                                                    weightNoB, returnMasked=True)

                chi2FullModelNoBVar = calculateWeightedReducedChi2(varVecFinalNoB, varVecModelFinalNoB,
                                                                   varWeightsFinalNoB, len(meanVecFinalNoB),
                                                                   3)
                stringLegend = (f"Gain: {gain:.4} e/ADU \n" +
                                f"Noise: {noise:.4} e \n" +
                                r"$a_{00}$: %.3e 1/e"%aCoeffs[0, 0] +
                                "\n" + r"$b_{00}$: %.3e 1/e"%bCoeffs[0, 0] +
                                f"\nLast in fit: {meanVecFinal[-1]:.7} ADU ")
                minMeanVecFinal = np.nanmin(meanVecFinal)
                maxMeanVecFinal = np.nanmax(meanVecFinal)
                deltaXlim = maxMeanVecFinal - minMeanVecFinal

                a.set_xlabel(r'Mean signal ($\mu$, ADU)', fontsize=labelFontSize)
                a.set_ylabel(r'Variance (ADU$^2$)', fontsize=labelFontSize)
                a.tick_params(labelsize=11)
                a.set_xscale('linear')
                a.set_yscale('linear')
                a.scatter(meanVecFinal, varVecFinal, c='blue', marker='o', s=markerSize)
                a.plot(meanVecFinal, varVecModelFinal, color='red', lineStyle='-')
                a.text(0.03, 0.7, stringLegend, transform=a.transAxes, fontsize=legendFontSize)
                a.set_title(amp, fontsize=titleFontSize)
                a.set_xlim([minMeanVecFinal - 0.2*deltaXlim, maxMeanVecFinal + 0.2*deltaXlim])

                # Same as above, but in log-scale
                a2.set_xlabel(r'Mean Signal ($\mu$, ADU)', fontsize=labelFontSize)
                a2.set_ylabel(r'Variance (ADU$^2$)', fontsize=labelFontSize)
                a2.tick_params(labelsize=11)
                a2.set_xscale('log')
                a2.set_yscale('log')
                a2.plot(meanVecFinal, varVecModelFinal, color='red', lineStyle='-')
                a2.scatter(meanVecFinal, varVecFinal, c='blue', marker='o', s=markerSize)
                a2.text(0.03, 0.7, stringLegend, transform=a2.transAxes, fontsize=legendFontSize)
                a2.set_title(amp, fontsize=titleFontSize)
                a2.set_xlim([minMeanVecFinal, maxMeanVecFinal])

                # Residuals var - model
                aResVar.set_xlabel(r'Mean signal ($\mu$, ADU)', fontsize=labelFontSize)
                aResVar.set_ylabel(r'Residuals (ADU$^2$)', fontsize=labelFontSize)
                aResVar.tick_params(labelsize=11)
                aResVar.set_xscale('linear')
                aResVar.set_yscale('linear')
                aResVar.plot(meanVecFinal, varVecFinal - varVecModelFinal, color='blue', lineStyle='-',
                             label=r'Full fit ($\chi_{\rm{red}}^2$: %g)'%chi2FullModelVar)
                aResVar.plot(meanVecFinal, varVecFinal - varModelFinalQuadratic, color='red', lineStyle='-',
                             label=r'Quadratic fit ($\chi_{\rm{red}}^2$: %g)'%chi2QuadModelVar)
                aResVar.plot(meanVecFinalNoB, varVecFinalNoB - varVecModelFinalNoB, color='green',
                             lineStyle='-',
                             label=r'Full fit (b=0) ($\chi_{\rm{red}}^2$: %g)'%chi2FullModelNoBVar)
                aResVar.axhline(color='black')
                aResVar.set_title(amp, fontsize=titleFontSize)
                aResVar.set_xlim([minMeanVecFinal - 0.2*deltaXlim, maxMeanVecFinal + 0.2*deltaXlim])
                aResVar.legend(fontsize=7)

                a3.set_xlabel(r'Mean signal ($\mu$, ADU)', fontsize=labelFontSize)
                a3.set_ylabel(r'Cov01 (ADU$^2$)', fontsize=labelFontSize)
                a3.tick_params(labelsize=11)
                a3.set_xscale('linear')
                a3.set_yscale('linear')
                a3.scatter(meanVecFinalCov01, varVecFinalCov01, c='blue', marker='o', s=markerSize)
                a3.plot(meanVecFinalCov01, varVecModelFinalCov01, color='red', lineStyle='-')
                a3.set_title(amp, fontsize=titleFontSize)
                a3.set_xlim([minMeanVecFinal - 0.2*deltaXlim, maxMeanVecFinal + 0.2*deltaXlim])

                a4.set_xlabel(r'Mean signal ($\mu$, ADU)', fontsize=labelFontSize)
                a4.set_ylabel(r'Cov10 (ADU$^2$)', fontsize=labelFontSize)
                a4.tick_params(labelsize=11)
                a4.set_xscale('linear')
                a4.set_yscale('linear')
                a4.scatter(meanVecFinalCov10, varVecFinalCov10, c='blue', marker='o', s=markerSize)
                a4.plot(meanVecFinalCov10, varVecModelFinalCov10, color='red', lineStyle='-')
                a4.set_title(amp, fontsize=titleFontSize)
                a4.set_xlim([minMeanVecFinal - 0.2*deltaXlim, maxMeanVecFinal + 0.2*deltaXlim])

            else:
                a.set_title(f"{amp} (BAD)", fontsize=titleFontSize)
                a2.set_title(f"{amp} (BAD)", fontsize=titleFontSize)
                a3.set_title(f"{amp} (BAD)", fontsize=titleFontSize)
                a4.set_title(f"{amp} (BAD)", fontsize=titleFontSize)

        f.suptitle("PTC from covariances as in Astier+19 \n Fit: Eq. 20, Astier+19",
                   fontsize=supTitleFontSize)
        pdfPages.savefig(f)
        f2.suptitle("PTC from covariances as in Astier+19 (log-log) \n Fit: Eq. 20, Astier+19",
                    fontsize=supTitleFontSize)
        pdfPages.savefig(f2)
        fResCov00.suptitle("Residuals (data-model) for Cov00 (Var)", fontsize=supTitleFontSize)
        pdfPages.savefig(fResCov00)
        fCov01.suptitle("Cov01 as in Astier+19 (nearest parallel neighbor covariance) \n" +
                        " Fit: Eq. 20, Astier+19", fontsize=supTitleFontSize)
        pdfPages.savefig(fCov01)
        fCov10.suptitle("Cov10 as in Astier+19 (nearest serial neighbor covariance) \n" +
                        "Fit: Eq. 20, Astier+19", fontsize=supTitleFontSize)
        pdfPages.savefig(fCov10)

        return

    def plotNormalizedCovariances(self, i, j, inputMu, covs, covsModel, covsWeights, covsNoB, covsModelNoB,
                                  covsWeightsNoB, pdfPages, offset=0.004,
                                  numberOfBins=10, plotData=True, topPlot=False, log=None):
        """Plot C_ij/mu vs mu.

        Figs. 8, 10, and 11 of Astier+19

        Parameters
        ----------
        i : `int`
            Covariane lag

        j : `int`
            Covariance lag

        inputMu : `dict`, [`str`, `list`]
            Dictionary keyed by amp name with mean signal values.

        covs : `dict`, [`str`, `list`]
            Dictionary keyed by amp names containing a list of measued covariances per mean flux.

        covsModel : `dict`, [`str`, `list`]
            Dictionary keyed by amp names containinging covariances model (Eq. 20 of Astier+19) per mean flux.

        covsWeights : `dict`, [`str`, `list`]
            Dictionary keyed by amp names containinging sqrt. of covariances weights.

        covsNoB : `dict`, [`str`, `list`]
            Dictionary keyed by amp names containing a list of measued covariances per mean flux ('b'=0 in
            Astier+19).

        covsModelNoB : `dict`, [`str`, `list`]
            Dictionary keyed by amp names containing covariances model (with 'b'=0 in Eq. 20 of Astier+19)
            per mean flux.

        covsWeightsNoB : `dict`, [`str`, `list`]
            Dictionary keyed by amp names containing sqrt. of covariances weights ('b' = 0 in Eq. 20 of
            Astier+19).

        expIdMask : `dict`, [`str`, `list`]
            Dictionary keyed by amp names containing the masked exposure pairs.

        pdfPages: `matplotlib.backends.backend_pdf.PdfPages`
            PDF file where the plots will be saved.

        offset : `float`, optional
            Constant offset factor to plot covariances in same panel (so they don't overlap).

        numberOfBins : `int`, optional
            Number of bins for top and bottom plot.

        plotData : `bool`, optional
            Plot the data points?

        topPlot : `bool`, optional
            Plot the top plot with the covariances, and the bottom plot with the model residuals?

        log : `lsst.log.Log`, optional
            Logger to handle messages.
        """
        if not topPlot:
            fig = plt.figure(figsize=(8, 10))
            gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1])
            gs.update(hspace=0)
            ax0 = plt.subplot(gs[0])
            plt.setp(ax0.get_xticklabels(), visible=False)
        else:
            fig = plt.figure(figsize=(8, 8))
            ax0 = plt.subplot(111)
            ax0.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
        ax0.tick_params(axis='both', labelsize='x-large')
        mue, rese, wce = [], [], []
        mueNoB, reseNoB, wceNoB = [], [], []
        for counter, amp in enumerate(covs):
            muAmp, fullCov, fullCovModel, fullCovWeight = (inputMu[amp], covs[amp], covsModel[amp],
                                                           covsWeights[amp])
            if len(fullCov) == 0:
                continue
            mu, cov, model, weightCov, _ = getFitDataFromCovariances(i, j, muAmp, fullCov, fullCovModel,
                                                                     fullCovWeight, divideByMu=True,
                                                                     returnMasked=True)

            mue += list(mu)
            rese += list(cov - model)
            wce += list(weightCov)

            fullCovNoB, fullCovModelNoB, fullCovWeightNoB = (covsNoB[amp], covsModelNoB[amp],
                                                             covsWeightsNoB[amp])
            if len(fullCovNoB) == 0:
                continue
            (muNoB, covNoB, modelNoB,
                weightCovNoB, _) = getFitDataFromCovariances(i, j, muAmp, fullCovNoB, fullCovModelNoB,
                                                             fullCovWeightNoB, divideByMu=True,
                                                             returnMasked=True)

            mueNoB += list(muNoB)
            reseNoB += list(covNoB - modelNoB)
            wceNoB += list(weightCovNoB)

            # the corresponding fit
            fit_curve, = plt.plot(mu, model + counter*offset, '-', linewidth=4.0)
            # bin plot. len(mu) = no binning
            gind = self.indexForBins(mu, numberOfBins)

            xb, yb, wyb, sigyb = self.binData(mu, cov, gind, weightCov)
            plt.errorbar(xb, yb+counter*offset, yerr=sigyb, marker='o', linestyle='none', markersize=6.5,
                         color=fit_curve.get_color(), label=f"{amp} (N: {len(mu)})")
            # plot the data
            if plotData:
                points, = plt.plot(mu, cov + counter*offset, '.', color=fit_curve.get_color())
            plt.legend(loc='upper right', fontsize=8)
        # end loop on amps
        mue = np.array(mue)
        rese = np.array(rese)
        wce = np.array(wce)
        mueNoB = np.array(mueNoB)
        reseNoB = np.array(reseNoB)
        wceNoB = np.array(wceNoB)

        plt.xlabel(r"$\mu (el)$", fontsize='x-large')
        plt.ylabel(r"$Cov{%d%d}/\mu + Cst (el)$"%(i, j), fontsize='x-large')
        if (not topPlot):
            gind = self.indexForBins(mue, numberOfBins)
            xb, yb, wyb, sigyb = self.binData(mue, rese, gind, wce)

            ax1 = plt.subplot(gs[1], sharex=ax0)
            ax1.errorbar(xb, yb, yerr=sigyb, marker='o', linestyle='none', label='Full fit')
            gindNoB = self.indexForBins(mueNoB, numberOfBins)
            xb2, yb2, wyb2, sigyb2 = self.binData(mueNoB, reseNoB, gindNoB, wceNoB)

            ax1.errorbar(xb2, yb2, yerr=sigyb2, marker='o', linestyle='none', label='b = 0')
            ax1.tick_params(axis='both', labelsize='x-large')
            plt.legend(loc='upper left', fontsize='large')
            # horizontal line at zero
            plt.plot(xb, [0]*len(xb), '--', color='k')
            plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
            plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
            plt.xlabel(r'$\mu (el)$', fontsize='x-large')
            plt.ylabel(r'$Cov{%d%d}/\mu$ -model (el)'%(i, j), fontsize='x-large')
        plt.tight_layout()
        plt.suptitle(f"Nbins: {numberOfBins}")
        # overlapping y labels:
        fig.canvas.draw()
        labels0 = [item.get_text() for item in ax0.get_yticklabels()]
        labels0[0] = u''
        ax0.set_yticklabels(labels0)
        pdfPages.savefig(fig)

        return

    @staticmethod
    def plot_a_b(aDict, bDict, pdfPages, bRange=3):
        """Fig. 12 of Astier+19

        Color display of a and b arrays fits, averaged over channels.

        Parameters
        ----------
        aDict : `dict`, [`numpy.array`]
            Dictionary keyed by amp names containing the fitted 'a' coefficients from the model
            in Eq. 20 of Astier+19 (if `ptcFitType` is `FULLCOVARIANCE`).

        bDict : `dict`, [`numpy.array`]
            Dictionary keyed by amp names containing the fitted 'b' coefficients from the model
            in Eq. 20 of Astier+19 (if `ptcFitType` is `FULLCOVARIANCE`).

        pdfPages: `matplotlib.backends.backend_pdf.PdfPages`
            PDF file where the plots will be saved.

        bRange : `int`
            Maximum lag for b arrays.
        """
        a, b = [], []
        for amp in aDict:
            if np.isnan(aDict[amp]).all():
                continue
            a.append(aDict[amp])
            b.append(bDict[amp])
        a = np.array(a).mean(axis=0)
        b = np.array(b).mean(axis=0)
        fig = plt.figure(figsize=(7, 11))
        ax0 = fig.add_subplot(2, 1, 1)
        im0 = ax0.imshow(np.abs(a.transpose()), origin='lower', norm=mpl.colors.LogNorm())
        ax0.tick_params(axis='both', labelsize='x-large')
        ax0.set_title(r'$|a|$', fontsize='x-large')
        ax0.xaxis.set_ticks_position('bottom')
        cb0 = plt.colorbar(im0)
        cb0.ax.tick_params(labelsize='x-large')

        ax1 = fig.add_subplot(2, 1, 2)
        ax1.tick_params(axis='both', labelsize='x-large')
        ax1.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
        im1 = ax1.imshow(1e6*b[:bRange, :bRange].transpose(), origin='lower')
        cb1 = plt.colorbar(im1)
        cb1.ax.tick_params(labelsize='x-large')
        ax1.set_title(r'$b \times 10^6$', fontsize='x-large')
        ax1.xaxis.set_ticks_position('bottom')
        plt.tight_layout()
        pdfPages.savefig(fig)

        return

    @staticmethod
    def ab_vs_dist(aDict, bDict, pdfPages, bRange=4):
        """Fig. 13 of Astier+19.

        Values of a and b arrays fits, averaged over amplifiers, as a function of distance.

        Parameters
        ----------
        aDict : `dict`, [`numpy.array`]
            Dictionary keyed by amp names containing the fitted 'a' coefficients from the model
            in Eq. 20 of Astier+19 (if `ptcFitType` is `FULLCOVARIANCE`).

        bDict : `dict`, [`numpy.array`]
            Dictionary keyed by amp names containing the fitted 'b' coefficients from the model
            in Eq. 20 of Astier+19 (if `ptcFitType` is `FULLCOVARIANCE`).

        pdfPages: `matplotlib.backends.backend_pdf.PdfPages`
            PDF file where the plots will be saved.

        bRange : `int`
            Maximum lag for b arrays.
        """
        assert (len(aDict) == len(bDict))
        a = []
        for amp in aDict:
            if np.isnan(aDict[amp]).all():
                continue
            a.append(aDict[amp])
        a = np.array(a)
        y = a.mean(axis=0)
        sy = a.std(axis=0)/np.sqrt(len(aDict))
        i, j = np.indices(y.shape)
        upper = (i >= j).ravel()
        r = np.sqrt(i**2 + j**2).ravel()
        y = y.ravel()
        sy = sy.ravel()
        fig = plt.figure(figsize=(6, 9))
        ax = fig.add_subplot(211)
        ax.set_xlim([0.5, r.max()+1])
        ax.errorbar(r[upper], y[upper], yerr=sy[upper], marker='o', linestyle='none', color='b',
                    label='$i>=j$')
        ax.errorbar(r[~upper], y[~upper], yerr=sy[~upper], marker='o', linestyle='none', color='r',
                    label='$i<j$')
        ax.legend(loc='upper center', fontsize='x-large')
        ax.set_xlabel(r'$\sqrt{i^2+j^2}$', fontsize='x-large')
        ax.set_ylabel(r'$a_{ij}$', fontsize='x-large')
        ax.set_yscale('log')
        ax.tick_params(axis='both', labelsize='x-large')

        #
        axb = fig.add_subplot(212)
        b = []
        for amp in bDict:
            if np.isnan(bDict[amp]).all():
                continue
            b.append(bDict[amp])
        b = np.array(b)
        yb = b.mean(axis=0)
        syb = b.std(axis=0)/np.sqrt(len(bDict))
        ib, jb = np.indices(yb.shape)
        upper = (ib > jb).ravel()
        rb = np.sqrt(i**2 + j**2).ravel()
        yb = yb.ravel()
        syb = syb.ravel()
        xmin = -0.2
        xmax = bRange
        axb.set_xlim([xmin, xmax+0.2])
        cutu = (r > xmin) & (r < xmax) & (upper)
        cutl = (r > xmin) & (r < xmax) & (~upper)
        axb.errorbar(rb[cutu], yb[cutu], yerr=syb[cutu], marker='o', linestyle='none', color='b',
                     label='$i>=j$')
        axb.errorbar(rb[cutl], yb[cutl], yerr=syb[cutl], marker='o', linestyle='none', color='r',
                     label='$i<j$')
        plt.legend(loc='upper center', fontsize='x-large')
        axb.set_xlabel(r'$\sqrt{i^2+j^2}$', fontsize='x-large')
        axb.set_ylabel(r'$b_{ij}$', fontsize='x-large')
        axb.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
        axb.tick_params(axis='both', labelsize='x-large')
        plt.tight_layout()
        pdfPages.savefig(fig)

        return

    @staticmethod
    def plotAcoeffsSum(aDict, bDict, pdfPages):
        """Fig. 14. of Astier+19

        Cumulative sum of a_ij as a function of maximum separation. This plot displays the average over
        channels.

        Parameters
        ----------
        aDict : `dict`, [`numpy.array`]
            Dictionary keyed by amp names containing the fitted 'a' coefficients from the model
            in Eq. 20 of Astier+19 (if `ptcFitType` is `FULLCOVARIANCE`).

        bDict : `dict`, [`numpy.array`]
            Dictionary keyed by amp names containing the fitted 'b' coefficients from the model
            in Eq. 20 of Astier+19 (if `ptcFitType` is `FULLCOVARIANCE`).

        pdfPages: `matplotlib.backends.backend_pdf.PdfPages`
            PDF file where the plots will be saved.
        """
        assert (len(aDict) == len(bDict))
        a, b = [], []
        for amp in aDict:
            if np.isnan(aDict[amp]).all() or np.isnan(bDict[amp]).all():
                continue
            a.append(aDict[amp])
            b.append(bDict[amp])
        a = np.array(a).mean(axis=0)
        b = np.array(b).mean(axis=0)
        fig = plt.figure(figsize=(7, 6))
        w = 4*np.ones_like(a)
        w[0, 1:] = 2
        w[1:, 0] = 2
        w[0, 0] = 1
        wa = w*a
        indices = range(1, a.shape[0]+1)
        sums = [wa[0:n, 0:n].sum() for n in indices]
        ax = plt.subplot(111)
        ax.plot(indices, sums/sums[0], 'o', color='b')
        ax.set_yscale('log')
        ax.set_xlim(indices[0]-0.5, indices[-1]+0.5)
        ax.set_ylim(None, 1.2)
        ax.set_ylabel(r'$[\sum_{|i|<n\  &\  |j|<n} a_{ij}] / |a_{00}|$', fontsize='x-large')
        ax.set_xlabel('n', fontsize='x-large')
        ax.tick_params(axis='both', labelsize='x-large')
        plt.tight_layout()
        pdfPages.savefig(fig)

        return

    @staticmethod
    def plotRelativeBiasACoeffs(aDict, aDictNoB, fullCovsModel, fullCovsModelNoB, signalElectrons,
                                gainDict, pdfPages, maxr=None):
        """Fig. 15 in Astier+19.

        Illustrates systematic bias from estimating 'a'
        coefficients from the slope of correlations as opposed to the
        full model in Astier+19.

        Parameters
        ----------
        aDict: `dict`
            Dictionary of 'a' matrices (Eq. 20, Astier+19), with amp names as keys.

        aDictNoB: `dict`
            Dictionary of 'a' matrices ('b'= 0 in Eq. 20, Astier+19), with amp names as keys.

        fullCovsModel : `dict`, [`str`, `list`]
            Dictionary keyed by amp names containing covariances model per mean flux.

        fullCovsModelNoB : `dict`, [`str`, `list`]
            Dictionary keyed by amp names containing covariances model (with 'b'=0 in Eq. 20 of
            Astier+19) per mean flux.

        signalElectrons : `float`
            Signal at which to evaluate the a_ij coefficients.

        pdfPages: `matplotlib.backends.backend_pdf.PdfPages`
            PDF file where the plots will be saved.

        gainDict : `dict`, [`str`, `float`]
            Dicgionary keyed by amp names with the gains in e-/ADU.

        maxr : `int`, optional
            Maximum lag.
        """

        fig = plt.figure(figsize=(7, 11))
        title = [f"'a' relative bias at {signalElectrons} e", "'a' relative bias (b=0)"]
        data = [(aDict, fullCovsModel), (aDictNoB, fullCovsModelNoB)]

        for k, pair in enumerate(data):
            diffs = []
            amean = []
            for amp in pair[0]:
                covModel = pair[1][amp]
                if np.isnan(covModel).all():
                    continue
                aOld = computeApproximateAcoeffs(covModel, signalElectrons, gainDict[amp])
                a = pair[0][amp]
                amean.append(a)
                diffs.append((aOld-a))
            amean = np.array(amean).mean(axis=0)
            diff = np.array(diffs).mean(axis=0)
            diff = diff/amean
            diff = diff[:]
            # The difference should be close to zero
            diff[0, 0] = 0
            if maxr is None:
                maxr = diff.shape[0]
            diff = diff[:maxr, :maxr]
            ax0 = fig.add_subplot(2, 1, k+1)
            im0 = ax0.imshow(diff.transpose(), origin='lower')
            ax0.yaxis.set_major_locator(MaxNLocator(integer=True))
            ax0.xaxis.set_major_locator(MaxNLocator(integer=True))
            ax0.tick_params(axis='both', labelsize='x-large')
            plt.colorbar(im0)
            ax0.set_title(title[k])

        plt.tight_layout()
        pdfPages.savefig(fig)

        return

    def _plotStandardPtc(self, dataset, ptcFitType, pdfPages):
        """Plot PTC, var/signal vs signal, linearity, and linearity residual per amplifier.

        Parameters
        ----------
        dataset : `lsst.ip.isr.ptcDataset.PhotonTransferCurveDataset`
            The dataset containing the means, variances, exposure times, and mask.

        ptcFitType : `str`
            Type of the model fit to the PTC. Options: 'FULLCOVARIANCE', EXPAPPROXIMATION, or 'POLYNOMIAL'.

        pdfPages: `matplotlib.backends.backend_pdf.PdfPages`
            PDF file where the plots will be saved.
        """

        if ptcFitType == 'EXPAPPROXIMATION':
            ptcFunc = funcAstier
            stringTitle = (r"Var = $\frac{1}{2g^2a_{00}}(\exp (2a_{00} \mu g) - 1) + \frac{n_{00}}{g^2}$ ")
        elif ptcFitType == 'POLYNOMIAL':
            ptcFunc = funcPolynomial
            for key in dataset.ptcFitPars:
                deg = len(dataset.ptcFitPars[key]) - 1
                break
            stringTitle = r"Polynomial (degree: %g)" % (deg)
        else:
            raise RuntimeError(f"The input dataset had an invalid dataset.ptcFitType: {ptcFitType}. \n"
                               "Options: 'FULLCOVARIANCE', EXPAPPROXIMATION, or 'POLYNOMIAL'.")

        legendFontSize = 6.5
        labelFontSize = 8
        titleFontSize = 9
        supTitleFontSize = 18
        markerSize = 25

        # General determination of the size of the plot grid
        nAmps = len(dataset.ampNames)
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
        f3, ax3 = plt.subplots(nrows=nRows, ncols=nCols, sharex='col', sharey='row', figsize=(13, 10))

        for i, (amp, a, a2, a3) in enumerate(zip(dataset.ampNames, ax.flatten(), ax2.flatten(),
                                             ax3.flatten())):
            meanVecOriginal = np.ravel(np.array(dataset.rawMeans[amp]))
            varVecOriginal = np.ravel(np.array(dataset.rawVars[amp]))
            mask = np.ravel(np.array(dataset.expIdMask[amp]))
            if np.isnan(mask[0]):  # All NaNs the whole amp is bad
                a.set_title(f"{amp} (BAD)", fontsize=titleFontSize)
                a2.set_title(f"{amp} (BAD)", fontsize=titleFontSize)
                a3.set_title(f"{amp} (BAD)", fontsize=titleFontSize)
                continue
            else:
                mask = mask.astype(bool)
            meanVecFinal = meanVecOriginal[mask]
            varVecFinal = varVecOriginal[mask]
            meanVecOutliers = meanVecOriginal[np.invert(mask)]
            varVecOutliers = varVecOriginal[np.invert(mask)]
            pars, parsErr = np.array(dataset.ptcFitPars[amp]), np.array(dataset.ptcFitParsError[amp])
            ptcRedChi2 = dataset.ptcFitChiSq[amp]
            if ptcFitType == 'EXPAPPROXIMATION':
                if len(meanVecFinal):
                    ptcA00, ptcA00error = pars[0], parsErr[0]
                    ptcGain, ptcGainError = pars[1], parsErr[1]
                    ptcNoise = np.sqrt((pars[2]))  # pars[2] is in (e-)^2
                    ptcNoiseAdu = ptcNoise*(1./ptcGain)
                    ptcNoiseError = 0.5*(parsErr[2]/np.fabs(pars[2]))*np.sqrt(np.fabs(pars[2]))
                    stringLegend = (f"a00: {ptcA00:.2e}+/-{ptcA00error:.2e} 1/e"
                                    f"\nGain: {ptcGain:.4}+/-{ptcGainError:.2e} e/ADU"
                                    f"\nNoise: {ptcNoise:.4}+/-{ptcNoiseError:.2e} e\n"
                                    r"$\chi^2_{\rm{red}}$: " + f"{ptcRedChi2:.4}"
                                    f"\nLast in fit: {meanVecFinal[-1]:.7} ADU ")

            if ptcFitType == 'POLYNOMIAL':
                if len(meanVecFinal):
                    ptcGain, ptcGainError = 1./pars[1], np.fabs(1./pars[1])*(parsErr[1]/pars[1])
                    ptcNoiseAdu = np.sqrt((pars[0]))  # pars[0] is in ADU^2
                    ptcNoise = ptcNoiseAdu*ptcGain
                    ptcNoiseError = (0.5*(parsErr[0]/np.fabs(pars[0]))*(np.sqrt(np.fabs(pars[0]))))*ptcGain
                    stringLegend = (f"Gain: {ptcGain:.4}+/-{ptcGainError:.2e} e/ADU\n"
                                    f"Noise: {ptcNoise:.4}+/-{ptcNoiseError:.2e} e\n"
                                    r"$\chi^2_{\rm{red}}$: " + f"{ptcRedChi2:.4}"
                                    f"\nLast in fit: {meanVecFinal[-1]:.7} ADU ")

            a.set_xlabel(r'Mean signal ($\mu$, ADU)', fontsize=labelFontSize)
            a.set_ylabel(r'Variance (ADU$^2$)', fontsize=labelFontSize)
            a.tick_params(labelsize=11)
            a.set_xscale('linear')
            a.set_yscale('linear')

            a2.set_xlabel(r'Mean Signal ($\mu$, ADU)', fontsize=labelFontSize)
            a2.set_ylabel(r'Variance (ADU$^2$)', fontsize=labelFontSize)
            a2.tick_params(labelsize=11)
            a2.set_xscale('log')
            a2.set_yscale('log')

            a3.set_xlabel(r'Mean signal ($\mu$, ADU)', fontsize=labelFontSize)
            a3.set_ylabel(r'Variance/$\mu$ (ADU)', fontsize=labelFontSize)
            a3.tick_params(labelsize=11)
            a3.set_xscale('log')
            a3.set_yscale('linear')

            minMeanVecFinal = np.nanmin(meanVecFinal)
            maxMeanVecFinal = np.nanmax(meanVecFinal)
            meanVecFit = np.linspace(minMeanVecFinal, maxMeanVecFinal, 100*len(meanVecFinal))
            minMeanVecOriginal = np.nanmin(meanVecOriginal)
            maxMeanVecOriginal = np.nanmax(meanVecOriginal)
            deltaXlim = maxMeanVecOriginal - minMeanVecOriginal
            a.plot(meanVecFit, ptcFunc(pars, meanVecFit), color='red')
            a.plot(meanVecFinal, ptcNoiseAdu**2 + (1./ptcGain)*meanVecFinal, color='green',
                   linestyle='--')
            a.scatter(meanVecFinal, varVecFinal, c='blue', marker='o', s=markerSize)
            a.scatter(meanVecOutliers, varVecOutliers, c='magenta', marker='s', s=markerSize)
            a.text(0.03, 0.66, stringLegend, transform=a.transAxes, fontsize=legendFontSize)
            a.set_title(amp, fontsize=titleFontSize)
            a.set_xlim([minMeanVecOriginal - 0.2*deltaXlim, maxMeanVecOriginal + 0.2*deltaXlim])

            # Same, but in log-scale
            a2.plot(meanVecFit, ptcFunc(pars, meanVecFit), color='red')
            a2.scatter(meanVecFinal, varVecFinal, c='blue', marker='o', s=markerSize)
            a2.scatter(meanVecOutliers, varVecOutliers, c='magenta', marker='s', s=markerSize)
            a2.text(0.03, 0.66, stringLegend, transform=a2.transAxes, fontsize=legendFontSize)
            a2.set_title(amp, fontsize=titleFontSize)
            a2.set_xlim([minMeanVecOriginal, maxMeanVecOriginal])

            # Var/mu vs mu
            a3.plot(meanVecFit, ptcFunc(pars, meanVecFit)/meanVecFit, color='red')
            a3.scatter(meanVecFinal, varVecFinal/meanVecFinal, c='blue', marker='o', s=markerSize)
            a3.scatter(meanVecOutliers, varVecOutliers/meanVecOutliers, c='magenta', marker='s',
                       s=markerSize)
            a3.text(0.05, 0.1, stringLegend, transform=a3.transAxes, fontsize=legendFontSize)
            a3.set_title(amp, fontsize=titleFontSize)
            a3.set_xlim([minMeanVecOriginal - 0.2*deltaXlim, maxMeanVecOriginal + 0.2*deltaXlim])

        f.suptitle("PTC \n Fit: " + stringTitle, fontsize=supTitleFontSize)
        pdfPages.savefig(f)
        f2.suptitle("PTC (log-log)", fontsize=supTitleFontSize)
        pdfPages.savefig(f2)
        f3.suptitle(r"Var/$\mu$", fontsize=supTitleFontSize)
        pdfPages.savefig(f3)

        return

    def _plotLinearizer(self, dataset, linearizer, pdfPages):
        """Plot linearity and linearity residual per amplifier

        Parameters
        ----------
        dataset : `lsst.ip.isr.ptcDataset.PhotonTransferCurveDataset`
            The dataset containing the means, variances, exposure times, and mask.

        linearizer : `lsst.ip.isr.Linearizer`
            Linearizer object
        """
        legendFontSize = 7
        labelFontSize = 7
        titleFontSize = 9
        supTitleFontSize = 18

        # General determination of the size of the plot grid
        nAmps = len(dataset.ampNames)
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

        # Plot mean vs time (f1), and fractional residuals (f2)
        f, ax = plt.subplots(nrows=nRows, ncols=nCols, sharex='col', sharey='row', figsize=(13, 10))
        f2, ax2 = plt.subplots(nrows=nRows, ncols=nCols, sharex='col', sharey='row', figsize=(13, 10))
        for i, (amp, a, a2) in enumerate(zip(dataset.ampNames, ax.flatten(), ax2.flatten())):
            mask = dataset.expIdMask[amp]
            if np.isnan(mask[0]):
                a.set_title(f"{amp} (BAD)", fontsize=titleFontSize)
                a2.set_title(f"{amp} (BAD)", fontsize=titleFontSize)
                continue
            else:
                mask = mask.astype(bool)
            meanVecFinal = np.array(dataset.rawMeans[amp])[mask]
            timeVecFinal = np.array(dataset.rawExpTimes[amp])[mask]

            a.set_xlabel('Time (sec)', fontsize=labelFontSize)
            a.set_ylabel(r'Mean signal ($\mu$, ADU)', fontsize=labelFontSize)
            a.tick_params(labelsize=labelFontSize)
            a.set_xscale('linear')
            a.set_yscale('linear')

            a2.axhline(y=0, color='k')
            a2.axvline(x=0, color='k', linestyle='-')
            a2.set_xlabel(r'Mean signal ($\mu$, ADU)', fontsize=labelFontSize)
            a2.set_ylabel('Fractional nonlinearity (%)', fontsize=labelFontSize)
            a2.tick_params(labelsize=labelFontSize)
            a2.set_xscale('linear')
            a2.set_yscale('linear')

            pars, parsErr = linearizer.fitParams[amp], linearizer.fitParamsErr[amp]
            k0, k0Error = pars[0], parsErr[0]
            k1, k1Error = pars[1], parsErr[1]
            k2, k2Error = pars[2], parsErr[2]
            linRedChi2 = linearizer.fitChiSq[amp]
            stringLegend = (f"k0: {k0:.4}+/-{k0Error:.2e} ADU\nk1: {k1:.4}+/-{k1Error:.2e} ADU/t"
                            f"\nk2: {k2:.2e}+/-{k2Error:.2e} ADU/t^2\n"
                            r"$\chi^2_{\rm{red}}$: " + f"{linRedChi2:.4}")
            a.scatter(timeVecFinal, meanVecFinal)
            a.plot(timeVecFinal, funcPolynomial(pars, timeVecFinal), color='red')
            a.text(0.03, 0.75, stringLegend, transform=a.transAxes, fontsize=legendFontSize)
            a.set_title(f"{amp}", fontsize=titleFontSize)

            linearPart = k0 + k1*timeVecFinal
            fracLinRes = 100*(linearPart - meanVecFinal)/linearPart
            a2.plot(meanVecFinal, fracLinRes, c='g')
            a2.set_title(f"{amp}", fontsize=titleFontSize)

        f.suptitle("Linearity \n Fit: Polynomial (degree: %g)"
                   % (len(pars)-1),
                   fontsize=supTitleFontSize)
        f2.suptitle(r"Fractional NL residual" + "\n" +
                    r"$100\times \frac{(k_0 + k_1*Time-\mu)}{k_0+k_1*Time}$",
                    fontsize=supTitleFontSize)
        pdfPages.savefig(f)
        pdfPages.savefig(f2)

    @staticmethod
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

    @staticmethod
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

    @staticmethod
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
        sybin = np.array([y[binIndex == i].std()/np.sqrt(np.array([binIndex == i]).sum())
                         for i in binIndexSet])

        return xbin, ybin, wybin, sybin
