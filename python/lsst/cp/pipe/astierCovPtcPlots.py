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

import matplotlib.pyplot as plt
from matplotlib import gridspec
import numpy as np
import matplotlib as mpl
from matplotlib.ticker import MaxNLocator

from .astierCovPtcFit import aCoeffsComputeOldFashion
from .astierCovPtcUtils import (binData, indexForBins, CHI2)


def covAstierMakeAllPlots(covFits, covFitsNoB, covTuple, pdfPages,
                          log=None, maxMu=1e9, maxMuElectrons=1e9, maxCovlag=8):
    """Make plots for MeasurePhotonTransferCurve task when doCovariancesAstier=True.

    This function call other functions that mostly reproduce the plots in Astier+19.
    Most of the code is ported from Pierre Astier's repository https://github.com/PierreAstier/bfptc

    Parameters
    ----------
    covFits: `dict`
        Dictionary of CovFit objects, with amp names as keys.

    covFitsNoB: `dict`
       Dictionary of CovFit objects, with amp names as keys (b=0 in Eq. 20 of Astier+19).

    covTuple: `numpy.recarray`
        Recarray with rows with at least( mu1, mu2, cov, var, i, j, npix), where:
            mu1: mean value of flat1
            mu2: mean value of flat2
            cov: covariance value at lag(i, j)
            var: variance(covariance value at lag(0, 0))
            i: lag dimension
            j: lag dimension
            npix: number of pixels used for covariance calculation.

    pdfPages: `matplotlib.backends.backend_pdf.PdfPages`
        PDF file where the plots will be saved.

    log : `lsst.log.Log`, optional
        Logger to handle messages

    maxMu: `float`, optional
        Maximum signal, in ADU.

    maxMuElectrons: `float`, optional
       Maximum signal, in electrons.

    maxCovLag: `int`, optional
        maximum lag in covariances.
    """

    plotCovariances(covFits, pdfPages)
    plotNormalizedCovariances(covFits, covFitsNoB, 0, 0, pdfPages, offset=0.01, topPlot=True, log=log)
    plotNormalizedCovariances(covFits, covFitsNoB, 0, 1, pdfPages, log=log)
    plotNormalizedCovariances(covFits, covFitsNoB, 1, 0, pdfPages, log=log)
    plot_a_b(covFits, pdfPages)
    ab_vs_dist(covFits, pdfPages, brange=4)
    plotAcoeffsSum(covFits, pdfPages)
    plotRelativeBiasACoeffs(covFits, covFitsNoB, maxMuElectrons, pdfPages)


def plotCovariances(covFits, pdfPages):
    """Plot covariances and models: Cov00, Cov10, Cov01.

    Figs. 6 and 7 of Astier+19
    """

    legendFontSize = 7
    labelFontSize = 7
    titleFontSize = 9
    supTitleFontSize = 18
    markerSize = 25

    nAmps = len(covFits)
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

    for i, (fitPair, a, a2, aResVar, a3, a4) in enumerate(zip(covFits.items(), ax.flatten(), ax2.flatten(),
                                                              axResCov00.flatten(), axCov01.flatten(),
                                                              axCov10.flatten())):

        amp = fitPair[0]
        fit = fitPair[1]

        meanVecFinal, varVecFinal, varVecModel, wc = fit.getNormalizedFitData(0, 0)
        meanVecFinalCov01, varVecFinalCov01, varVecModelCov01, wcCov01 = fit.getNormalizedFitData(0, 1)
        meanVecFinalCov10, varVecFinalCov10, varVecModelCov10, wcCov10 = fit.getNormalizedFitData(1, 0)

        # cuadratic fit for residuals below
        par2 = np.polyfit(meanVecFinal, varVecFinal, 2, w=wc)
        varModelQuadratic = np.polyval(par2, meanVecFinal)

        # fit with no 'b' coefficient (c = a*b in Eq. 20 of Astier+19)
        fitNoB = fit.copy()
        fitNoB.params['c'].fix(val=0)
        fitNoB.fit()
        meanVecFinalNoB, varVecFinalNoB, varVecModelNoB, wcNoB = fitNoB.getNormalizedFitData(0, 0)

        if len(meanVecFinal):  # Empty if the whole amp is bad, for example.
            stringLegend = (f"Gain: {fit.getGain():.4} e/DN \n Noise: {np.sqrt(fit.getRon()):.4} e \n" +
                            r"$a_{00}$: %.3e 1/e"%fit.getA()[0, 0] +
                            "\n" + r"$b_{00}$: %.3e 1/e"%fit.getB()[0, 0])
            minMeanVecFinal = np.min(meanVecFinal)
            maxMeanVecFinal = np.max(meanVecFinal)
            deltaXlim = maxMeanVecFinal - minMeanVecFinal

            a.set_xlabel(r'Mean signal ($\mu$, DN)', fontsize=labelFontSize)
            a.set_ylabel(r'Variance (DN$^2$)', fontsize=labelFontSize)
            a.tick_params(labelsize=11)
            a.set_xscale('linear', fontsize=labelFontSize)
            a.set_yscale('linear', fontsize=labelFontSize)
            a.scatter(meanVecFinal, varVecFinal, c='blue', marker='o', s=markerSize)
            a.plot(meanVecFinal, varVecModel, color='red', lineStyle='-')
            a.text(0.03, 0.7, stringLegend, transform=a.transAxes, fontsize=legendFontSize)
            a.set_title(amp, fontsize=titleFontSize)
            a.set_xlim([minMeanVecFinal - 0.2*deltaXlim, maxMeanVecFinal + 0.2*deltaXlim])

            # Same as above, but in log-scale
            a2.set_xlabel(r'Mean Signal ($\mu$, DN)', fontsize=labelFontSize)
            a2.set_ylabel(r'Variance (DN$^2$)', fontsize=labelFontSize)
            a2.tick_params(labelsize=11)
            a2.set_xscale('log')
            a2.set_yscale('log')
            a2.plot(meanVecFinal, varVecModel, color='red', lineStyle='-')
            a2.scatter(meanVecFinal, varVecFinal, c='blue', marker='o', s=markerSize)
            a2.text(0.03, 0.7, stringLegend, transform=a2.transAxes, fontsize=legendFontSize)
            a2.set_title(amp, fontsize=titleFontSize)
            a2.set_xlim([minMeanVecFinal, maxMeanVecFinal])

            # Residuals var - model
            aResVar.set_xlabel(r'Mean signal ($\mu$, DN)', fontsize=labelFontSize)
            aResVar.set_ylabel(r'Residuals (DN$^2$)', fontsize=labelFontSize)
            aResVar.tick_params(labelsize=11)
            aResVar.set_xscale('linear', fontsize=labelFontSize)
            aResVar.set_yscale('linear', fontsize=labelFontSize)
            aResVar.scatter(meanVecFinal, varVecFinal - varVecModel, c='blue', marker='.',
                            s=markerSize, label='Full fit')
            aResVar.scatter(meanVecFinal, varVecFinal - varModelQuadratic, c='red', marker='.',
                            s=markerSize, label='Quadratic fit')
            aResVar.scatter(meanVecFinal, varVecFinalNoB - varVecModelNoB, c='green', marker='.',
                            s=markerSize, label='Full fit with b=0')
            aResVar.axhline(color='black')
            aResVar.set_title(amp, fontsize=titleFontSize)
            aResVar.set_xlim([minMeanVecFinal - 0.2*deltaXlim, maxMeanVecFinal + 0.2*deltaXlim])
            aResVar.legend(fontsize=7)

            a3.set_xlabel(r'Mean signal ($\mu$, DN)', fontsize=labelFontSize)
            a3.set_ylabel(r'Cov01 (DN$^2$)', fontsize=labelFontSize)
            a3.tick_params(labelsize=11)
            a3.set_xscale('linear', fontsize=labelFontSize)
            a3.set_yscale('linear', fontsize=labelFontSize)
            a3.scatter(meanVecFinalCov01, varVecFinalCov01, c='blue', marker='o', s=markerSize)
            a3.plot(meanVecFinalCov01, varVecModelCov01, color='red', lineStyle='-')
            a3.set_title(amp, fontsize=titleFontSize)
            a3.set_xlim([minMeanVecFinal - 0.2*deltaXlim, maxMeanVecFinal + 0.2*deltaXlim])

            a4.set_xlabel(r'Mean signal ($\mu$, DN)', fontsize=labelFontSize)
            a4.set_ylabel(r'Cov10 (DN$^2$)', fontsize=labelFontSize)
            a4.tick_params(labelsize=11)
            a4.set_xscale('linear', fontsize=labelFontSize)
            a4.set_yscale('linear', fontsize=labelFontSize)
            a4.scatter(meanVecFinalCov10, varVecFinalCov10, c='blue', marker='o', s=markerSize)
            a4.plot(meanVecFinalCov10, varVecModelCov10, color='red', lineStyle='-')
            a4.set_title(amp, fontsize=titleFontSize)
            a4.set_xlim([minMeanVecFinal - 0.2*deltaXlim, maxMeanVecFinal + 0.2*deltaXlim])

        else:
            a.set_title(f"{amp} (BAD)", fontsize=titleFontSize)
            a2.set_title(f"{amp} (BAD)", fontsize=titleFontSize)
            a3.set_title(f"{amp} (BAD)", fontsize=titleFontSize)
            a4.set_title(f"{amp} (BAD)", fontsize=titleFontSize)

    f.suptitle("PTC from covariances as in Astier+19 \n Fit: Eq. 20, Astier+19", fontsize=supTitleFontSize)
    pdfPages.savefig(f)
    f2.suptitle("PTC from covariances as in Astier+19 (log-log) \n Fit: Eq. 20, Astier+19",
                fontsize=supTitleFontSize)
    pdfPages.savefig(f2)
    fResCov00.suptitle("Residuals (data- model) for Cov00 (Var)", fontsize=supTitleFontSize)
    pdfPages.savefig(fResCov00)
    fCov01.suptitle("Cov01 as in Astier+19 (nearest parallel neighbor covariance) \n Fit: Eq. 20, Astier+19",
                    fontsize=supTitleFontSize)
    pdfPages.savefig(fCov01)
    fCov10.suptitle("Cov10 as in Astier+19 (nearest serial neighbor covariance) \n Fit: Eq. 20, Astier+19",
                    fontsize=supTitleFontSize)
    pdfPages.savefig(fCov10)


def plotNormalizedCovariances(covFits, covFitsNoB, i, j, pdfPages, offset=0.004, figname=None,
                              plotData=True, topPlot=False, log=None):
    """Plot C_ij/mu vs mu.

    Figs. 8, 10, and 11 of Astier+19
    """

    lchi2, la, lb, lcov = [], [], [], []

    if (not topPlot):
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
    for counter, (amp, fit) in enumerate(covFits.items()):
        mu, c, model, wc = fit.getNormalizedFitData(i, j, divideByMu=True)
        chi2 = CHI2(c-model, wc)/(len(mu)-3)
        chi2bin = 0
        mue += list(mu)
        rese += list(c - model)
        wce += list(wc)

        fitNoB = covFitsNoB[amp]
        muNoB, cNoB, modelNoB, wcNoB = fitNoB.getNormalizedFitData(i, j, divideByMu=True)
        mueNoB += list(muNoB)
        reseNoB += list(cNoB - modelNoB)
        wceNoB += list(wcNoB)

        # the corresponding fit
        fit_curve, = plt.plot(mu, model + counter*offset, '-', linewidth=4.0)
        # bin plot. len(mu) = no binning
        gind = indexForBins(mu, len(mu))

        xb, yb, wyb, sigyb = binData(mu, c, gind, wc)
        chi2bin = (sigyb*wyb).mean()  # chi2 of enforcing the same value in each bin
        plt.errorbar(xb, yb+counter*offset, yerr=sigyb, marker='o', linestyle='none', markersize=6.5,
                     color=fit_curve.get_color(), label=f"{amp}")
        # plot the data
        if plotData:
            points, = plt.plot(mu, c + counter*offset, '.', color=fit_curve.get_color())
        plt.legend(loc='upper right', fontsize=8)
        aij = fit.getA()[i, j]
        bij = fit.getB()[i, j]
        la.append(aij)
        lb.append(bij)
        lcov.append(fit.getACov()[i, j, i, j])
        lchi2.append(chi2)
        log.info('%s: slope %g b %g  chi2 %f chi2bin %f'%(amp, aij, bij, chi2, chi2bin))
    # end loop on amps
    la = np.array(la)
    lb = np.array(lb)
    lcov = np.array(lcov)
    lchi2 = np.array(lchi2)
    mue = np.array(mue)
    rese = np.array(rese)
    wce = np.array(wce)
    mueNoB = np.array(mueNoB)
    reseNoB = np.array(reseNoB)
    wceNoB = np.array(wceNoB)

    plt.xlabel(r"$\mu (el)$", fontsize='x-large')
    plt.ylabel(r"$C_{%d%d}/\mu + Cst (el)$"%(i, j), fontsize='x-large')
    if (not topPlot):
        gind = indexForBins(mue, len(mue))
        xb, yb, wyb, sigyb = binData(mue, rese, gind, wce)

        ax1 = plt.subplot(gs[1], sharex=ax0)
        ax1.errorbar(xb, yb, yerr=sigyb, marker='o', linestyle='none', label='Full fit')
        gindNoB = indexForBins(mueNoB, len(mueNoB))
        xb2, yb2, wyb2, sigyb2 = binData(mueNoB, reseNoB, gindNoB, wceNoB)

        ax1.errorbar(xb2, yb2, yerr=sigyb2, marker='o', linestyle='none', label='b = 0')
        ax1.tick_params(axis='both', labelsize='x-large')
        plt.legend(loc='upper left', fontsize='large')
        # horizontal line at zero
        plt.plot(xb, [0]*len(xb), '--', color='k')
        plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
        plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
        plt.xlabel(r'$\mu (el)$', fontsize='x-large')
        plt.ylabel(r'$C_{%d%d}/\mu$ -model (el)'%(i, j), fontsize='x-large')
    plt.tight_layout()

    # overlapping y labels:
    fig.canvas.draw()
    labels0 = [item.get_text() for item in ax0.get_yticklabels()]
    labels0[0] = u''
    ax0.set_yticklabels(labels0)
    pdfPages.savefig(fig)


def plotAcoeffsSum(covFits, pdfPages):
    """Fig. 14. of Astier+19

    Cumulative sum of a_ij as a function of maximum separation. This plot displays the average over channels.
    """
    a, b = [], []
    for amp, fit in covFits.items():
        a.append(fit.getA())
        b.append(fit.getB())
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


def plot_a_b(covFits, pdfPages, brange=3):
    a, b = [], []
    for amp, fit in covFits.items():
        a.append(fit.getA())
        b.append(fit.getB())
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
    #
    ax1 = fig.add_subplot(2, 1, 2)
    ax1.tick_params(axis='both', labelsize='x-large')
    ax1.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
    im1 = ax1.imshow(1e6*b[:brange, :brange].transpose(), origin='lower')
    cb1 = plt.colorbar(im1)
    cb1.ax.tick_params(labelsize='x-large')
    ax1.set_title(r'$b \times 10^6$', fontsize='x-large')
    ax1.xaxis.set_ticks_position('bottom')
    plt.tight_layout()
    pdfPages.savefig(fig)


def plotPtcAndResiduals(covFits, ampName, pdfPages):
    """Figure 7 in Astier+19"""
    fit = covFits[ampName]
    fig = plt.figure(figsize=(6, 12))
    gs = gridspec.GridSpec(4, 1, height_ratios=[3, 1, 1, 1])
    gs.update(hspace=0)  # stack subplots
    fontsize = 'x-large'
    # extract the data and model
    mu, var, model, w = fit.getNormalizedFitData(0, 0, divideByMu=False)

    # var vs mu
    ax0 = plt.subplot(gs[0])
    # allows factors of 10 on the scale
    ax0.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    ax0.set_ylabel("$C_{00}$ (el$^2$)", fontsize=fontsize)
    plt.plot(mu, var, '.', label='data')
    plt.plot(mu, model, '-', label='full model')
    plt.setp(ax0.get_xticklabels(), visible=False)
    plt.legend(loc='upper left', fontsize='large')
    #
    # residuals
    gind = indexForBins(mu, 50)

    ax1 = plt.subplot(gs[1], sharex=ax0)
    xb, yb, wyb, sigyb = binData(mu, var - model, gind, w)
    plt.errorbar(xb, yb, yerr=sigyb, marker='.', ls='none')
    # draw a line at y=0:
    plt.plot([0, mu.max()], [0, 0], ls='--', color='k')
    plt.setp(ax1.get_xticklabels(), visible=False)
    ax1.text(0.1, 0.85, 'Residuals to full fit',
             verticalalignment='top', horizontalalignment='left',
             transform=ax1.transAxes, fontsize=15)
    #
    #  quadratic fit
    ax2 = plt.subplot(gs[2], sharex=ax0, sharey=ax1)
    par2 = np.polyfit(mu, var, 2, w=w)
    m2 = np.polyval(par2, mu)
    xb, yb, wyb, sigyb = binData(mu, var-m2, gind, w)
    plt.errorbar(xb, yb, yerr=sigyb, marker='.', color='r', ls='none')
    plt.plot([0, mu.max()], [0, 0], ls='--', color='k')
    plt.setp(ax2.get_xticklabels(), visible=False)
    ax2.text(0.1, 0.85, 'Quadratic fit',
             verticalalignment='top', horizontalalignment='left',
             transform=ax2.transAxes, fontsize=15)

    # fit with b=0
    ax3 = plt.subplot(gs[3], sharex=ax0, sharey=ax1)
    fit_nob = fit.copy()
    fit_nob.params['c'].fix(val=0)
    fit_nob.fit()
    mu, var, model, w = fit_nob.getNormalizedFitData(0, 0, divideByMu=False)

    xb, yb, wyb, sigyb = binData(mu, var-model, gind, w)
    plt.errorbar(xb, yb, yerr=sigyb, marker='.', color='g', ls='none')
    plt.plot([0, mu.max()], [0, 0], ls='--', color='k')
    ax3.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
    plt.xlabel(r'$\mu$ (el)', fontsize=fontsize)
    ax3.text(0.1, 0.85, 'b=0',
             verticalalignment='top', horizontalalignment='left',
             transform=ax3.transAxes, fontsize=15)

    plt.tight_layout()
    # remove the 'largest' y label (unelegant overwritings occur)
    for ax in [ax1, ax2, ax3]:
        plt.setp(ax.get_yticklabels()[-1], visible=False)

    fig.suptitle("PTC from covariances and model residuals for amplifier %s"%ampName, fontsize=11)
    pdfPages.savefig(fig)


def ab_vs_dist(covFits, pdfPages, brange=4):
    a = np.array([f.getA() for f in covFits.values()])
    y = a.mean(axis=0)
    sy = a.std(axis=0)/np.sqrt(len(covFits))
    i, j = np.indices(y.shape)
    upper = (i >= j).ravel()
    r = np.sqrt(i**2 + j**2).ravel()
    y = y.ravel()
    sy = sy.ravel()
    fig = plt.figure(figsize=(6, 9))
    ax = fig.add_subplot(211)
    ax.set_xlim([0.5, r.max()+1])
    ax.errorbar(r[upper], y[upper], yerr=sy[upper], marker='o', linestyle='none', color='b', label='$i>=j$')
    ax.errorbar(r[~upper], y[~upper], yerr=sy[~upper], marker='o', linestyle='none', color='r', label='$i<j$')
    ax.legend(loc='upper center', fontsize='x-large')
    ax.set_xlabel(r'$\sqrt{i^2+j^2}$', fontsize='x-large')
    ax.set_ylabel(r'$a_{ij}$', fontsize='x-large')
    ax.set_yscale('log')
    ax.tick_params(axis='both', labelsize='x-large')

    #
    axb = fig.add_subplot(212)
    b = np.array([f.getB() for f in covFits.values()])
    yb = b.mean(axis=0)
    syb = b.std(axis=0)/np.sqrt(len(covFits))
    ib, jb = np.indices(yb.shape)
    upper = (ib > jb).ravel()
    rb = np.sqrt(i**2 + j**2).ravel()
    yb = yb.ravel()
    syb = syb.ravel()
    xmin = -0.2
    xmax = brange
    axb.set_xlim([xmin, xmax+0.2])
    cutu = (r > xmin) & (r < xmax) & (upper)
    cutl = (r > xmin) & (r < xmax) & (~upper)
    axb.errorbar(rb[cutu], yb[cutu], yerr=syb[cutu], marker='o', linestyle='none', color='b', label='$i>=j$')
    axb.errorbar(rb[cutl], yb[cutl], yerr=syb[cutl], marker='o', linestyle='none', color='r', label='$i<j$')
    plt.legend(loc='upper center', fontsize='x-large')
    axb.set_xlabel(r'$\sqrt{i^2+j^2}$', fontsize='x-large')
    axb.set_ylabel(r'$b_{ij}$', fontsize='x-large')
    axb.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    axb.tick_params(axis='both', labelsize='x-large')
    plt.tight_layout()
    pdfPages.savefig(fig)


def plotRelativeBiasACoeffs(covFits, covFitsNoB, mu_el, pdfPages, maxr=None):
    """Illustrates systematic bias from estimating 'a'
    coefficients from the slope of correlations as opposed to the
    full model in Astier+19.

    Corresponds to Fig. 15 in Astier+19.
    """

    fig = plt.figure(figsize=(7, 11))
    title = ["'a' relative bias", "'a' relative bias (b=0)"]
    data = [covFits, covFitsNoB]

    for k in range(2):
        diffs = []
        amean = []
        for fit in data[k].values():
            if fit is None:
                continue
            aOld = aCoeffsComputeOldFashion(fit, mu_el)
            a = fit.getA()
            amean.append(a)
            diffs.append((aOld-a))
        amean = np.array(amean).mean(axis=0)
        diff = np.array(diffs).mean(axis=0)
        diff = diff/amean
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
