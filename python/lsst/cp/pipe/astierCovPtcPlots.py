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
from .astierCovPtcUtils import (findGroups, binData, indexForBins, CHI2)


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

    plotStandardPtc(covFits, pdfPages)
    # do_cov_exposure_plot(covFits['C10'], pdfPages)
    #for amp in covFits:
    makeFigSixPlot(covTuple, 'C10', pdfPages)
    # for amp in covFits:
    plotPtcAndResiduals(covFits, 'C10', pdfPages)
    
    ptc_table(covFits, covFitsNoB, 0, 0)
    plotNormalizedCovariances(covFits, covFitsNoB, 0, 0, pdfPages, offset=0.01, top_plot=True, log=log)
    plotNormalizedCovariances(covFits, covFitsNoB, 0, 1, pdfPages, log=log)
    plotNormalizedCovariances(covFits, covFitsNoB, 1, 0, pdfPages, log=log)
    plot_a_b(covFits, pdfPages)
    ab_vs_dist(covFits, pdfPages, brange=4)
    #######  make_distant_cov_plot(covFits, covariancesTuple, pdfPages)
    plot_a_sum(covFits, pdfPages)
    plotRelativeBiasACoeffs(covFits, covFitsNoB, maxMuElectrons, pdfPages)


def plotStandardPtc(covFits, pdfPages):
    """Plot PTC from covariances (var = cov[0, 0])"""

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

    for i, (fitPair, a, a2) in enumerate(zip(covFits.items(), ax.flatten(), ax2.flatten())):

        amp = fitPair[0]
        fit = fitPair[1]

        meanVecFinal, varVecFinal, varVecModel, wc = fit.getNormalizedFitData(0, 0, divideByMu=False)

        if len(meanVecFinal):  # Empty if the whole amp is bad, for example.
            stringLegend = (f"Gain: {fit.getGain():.4} e/DN \n Noise: {np.sqrt(fit.getRon()):.4} e \n" +
                            r"$a_{00}$: %.2e"%fit.getA()[0,0] +"\n" +r"$b_{00}$: %.2e" %fit.getB()[0,0])
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
            a.text(0.04, 0.8, stringLegend, transform=a.transAxes, fontsize=legendFontSize)
            a.set_title(amp, fontsize=titleFontSize)
            a.set_xlim([minMeanVecFinal - 0.2*deltaXlim, maxMeanVecFinal + 0.2*deltaXlim])

            # Same, but in log-scale
            a2.set_xlabel(r'Mean Signal ($\mu$, DN)', fontsize=labelFontSize)
            a2.set_ylabel(r'Variance (DN$^2$)', fontsize=labelFontSize)
            a2.tick_params(labelsize=11)
            a2.set_xscale('log')
            a2.set_yscale('log')
            a2.plot(meanVecFinal, varVecModel, color='red', lineStyle='-')
            a2.scatter(meanVecFinal, varVecFinal, c='blue', marker='o', s=markerSize)
            a2.text(0.04, 0.8, stringLegend, transform=a2.transAxes, fontsize=legendFontSize)
            a2.set_title(amp, fontsize=titleFontSize)
            a2.set_xlim([minMeanVecFinal, maxMeanVecFinal])
        else:
            a.set_title(f"{amp} (BAD)", fontsize=titleFontSize)
            a2.set_title(f"{amp} (BAD)", fontsize=titleFontSize)

    f.suptitle("PTC from covariances as in Astier+19 \n Fit: Eq. 20, Astier+19", fontsize=12)
    pdfPages.savefig(f)
    f2.suptitle("PTC PTC from covariances as in Astier+19 (log-log) \n Fit: Eq. 20, Astier+19", fontsize=12)
    pdfPages.savefig(f2)


def plotNormalizedCovariances(covFits, covFitsNoB, i, j, pdfPages, offset=0.004, figname=None, 
               plot_data=True, top_plot=False, log=None):
    """Plot C_ij/mu vs mu"""

    lchi2, la, lb, lcov = [],[], [], []

    if (not top_plot):
        fig = plt.figure(figsize=(8,10))
        gs = gridspec.GridSpec(2,1, height_ratios=[3, 1])
        gs.update(hspace=0)
        ax0=plt.subplot(gs[0])
        plt.setp(ax0.get_xticklabels(), visible=False)
    else:
        fig = plt.figure(figsize=(8,8))
        ax0 = plt.subplot(111)
        ax0.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    ax0.tick_params(axis='both', labelsize='x-large')
    mue, rese, wce = [], [], []
    mue_nb, rese_nb, wce_nb = [], [], []
    for counter, (amp, fit) in enumerate(covFits.items()):
        mu, c, model, wc = fit.getNormalizedFitData(i, j, divideByMu = True)
        chi2 = CHI2(c-model,wc)/(len(mu)-3)
        chi2bin= 0
        mue += list(mu)
        rese += list(c - model)
        wce += list(wc)

        fit_nb = covFitsNoB[amp]
        mu_nb, c_nb, model_nb, wc_nb = fit_nb.getNormalizedFitData(i, j, divideByMu = True)
        mue_nb += list(mu_nb)
        rese_nb += list(c_nb - model_nb)
        wce_nb += list(wc_nb)

        
        # the corresponding fit
        fit_curve, = plt.plot(mu,model + counter*offset, '-', linewidth=4.0)
        # bin plot 
        
        gind = indexForBins(mu, len(mu)) #group  25
        
        
    
        xb, yb, wyb, sigyb = binData(mu,c,gind, wc) # group 
        chi2bin = (sigyb*wyb).mean() # chi2 of enforcing the same value in each bin
        z = plt.errorbar(xb,yb+counter*offset,yerr=sigyb, marker = 'o', linestyle='none', markersize = 6.5,
                color=fit_curve.get_color(), label=f"ch {amp}")
        # plot the data
        if plot_data:
            points, = plt.plot(mu,c+counter*offset, '.', color = fit_curve.get_color())

        aij = fit.getA()[i,j]
        bij = fit.getB()[i,j]
        la.append(aij)
        lb.append(bij)
        lcov.append(fit.getACov()[i,j,i,j])
        lchi2.append(chi2)
        log.info('%s: slope %g b %g  chi2 %f chi2bin %f'%(amp, aij , bij, chi2, chi2bin))
    # end loop on amps
    la = np.array(la)
    lb = np.array(lb)
    lcov = np.array(lcov)
    lchi2 = np.array(lchi2)
    mue = np.array(mue)
    rese = np.array(rese)
    wce = np.array(wce)
    mue_nb = np.array(mue_nb)
    rese_nb = np.array(rese_nb)
    wce_nb = np.array(wce_nb)
 
    plt.xlabel("$\mu (el)$",fontsize='x-large')
    plt.ylabel("$C_{%d%d}/\mu + Cst (el)$"%(i,j),fontsize='x-large')
    if (not top_plot):
        #gind = group.findGroups(mue, 2000.)
        gind = indexForBins(mue, len(mue)) #25)
        xb, yb, wyb, sigyb = binData(mue,rese , gind, wce)
        #plt.errorbar(xb,yb,yerr=sigyb, fmt='o', label='data')
        print('yb0 %g'%yb[0])
    
        ax1 = plt.subplot(gs[1], sharex = ax0)
        ax1.errorbar(xb,yb, yerr=sigyb, marker='o', linestyle='none', label='full fit')
        gind_nb = indexForBins(mue_nb, len(mue_nb) )# 25)
        xb2, yb2, wyb2, sigyb2 = binData(mue_nb,rese_nb , gind_nb, wce_nb)
        print('yb0 %g %g'%(yb[0],yb2[0]))
    
        ax1.errorbar(xb2,yb2, yerr=sigyb2, marker='o', linestyle='none', label='b = 0')
        ax1.tick_params(axis='both', labelsize='x-large')
        plt.legend(loc='upper left', fontsize='large')    
        # horizontal line at zero
        plt.plot(xb,[0]*len(xb),'--', color = 'k')
        #plt.plot(xb,model,'--')
        plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
        plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
        plt.xlabel('$\mu (el)$',fontsize='x-large')
        plt.ylabel('$C_{%d%d}/\mu$ -model (el)'%(i,j),fontsize='x-large')
    plt.tight_layout()

    a_expected_rms = np.sqrt(lcov.mean())
    # overlapping y labels:
    fig.canvas.draw()
    labels0 = [item.get_text() for item in ax0.get_yticklabels()]
    labels0[0] = u''
    ax0.set_yticklabels(labels0)
    pdfPages.savefig(fig)

def plot_chi2_diff(covFits, covFits_nob):
    chi2_diff = []
    for amp in covFits.keys():
        dchi2 =  ((covFits_nob[amp].wres())**2).sum(axis=0)-((covFits[amp].wres())**2).sum(axis=0)
        chi2_diff.append(dchi2)
    chi2_diff = np.array(chi2_diff).mean(axis=0)
    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(111)
    im = ax.imshow(chi2_diff.transpose(), origin='lower', norm = mpl.colors.LogNorm()) 
    plt.colorbar(im )
    ax.set_title(r'$\delta \chi^2$ for $b \neq 0$',fontsize = 'x-large') 
    

def plot_a_sum(covFits, pdfPages):
    a, b = [],[]
    for amp,fit in covFits.items():
        a.append(fit.getA())
        b.append(fit.getB())
    a = np.array(a).mean(axis=0)
    b = np.array(b).mean(axis=0)
    fig = plt.figure(figsize=(7,6))
    w = 4*np.ones_like(a)
    w[0,1:] = 2
    w[1:,0] = 2
    w[0,0] = 1
    wa = w*a
    indices = range(1,a.shape[0]+1)
    sums = [wa[0:n,0:n].sum() for n in indices]
    ax = plt.subplot(111)
    ax.plot(indices,sums/sums[0],'o',color='b')
    ax.set_yscale('log')
    ax.set_xlim(indices[0]-0.5, indices[-1]+0.5)
    ax.set_ylim(None, 1.2)
    ax.set_ylabel('$[\sum_{|i|<n\  &\  |j|<n} a_{ij}] / |a_{00}|$',fontsize='x-large')
    ax.set_xlabel('n',fontsize='x-large')
    ax.tick_params(axis='both', labelsize='x-large')
    plt.tight_layout()
    pdfPages.savefig(fig)
    

def plot_a_b(covFits, pdfPages, brange=3):
    a, b = [],[]
    for amp,fit in covFits.items():
        a.append(fit.getA())
        b.append(fit.getB())
    a = np.array(a).mean(axis=0)
    b = np.array(b).mean(axis=0)
    fig = plt.figure(figsize=(7,11))
    ax0 = fig.add_subplot(2,1,1)
    im0 = ax0.imshow(np.abs(a.transpose()), origin='lower', norm = mpl.colors.LogNorm())
    ax0.tick_params(axis='both', labelsize='x-large')
    ax0.set_title('$|a|$', fontsize='x-large')
    ax0.xaxis.set_ticks_position('bottom')
    cb0 = plt.colorbar(im0)
    cb0.ax.tick_params(labelsize='x-large')
    #
    ax1 = fig.add_subplot(2,1,2)
    ax1.tick_params(axis='both', labelsize='x-large')
    ax1.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
    im1 = ax1.imshow(1e6*b[:brange,:brange].transpose(), origin='lower' )
    cb1 = plt.colorbar(im1)
    cb1.ax.tick_params(labelsize='x-large')
    ax1.set_title(r'$b \times 10^6$', fontsize='x-large')
    ax1.xaxis.set_ticks_position('bottom')
    plt.tight_layout()
    pdfPages.savefig(fig)
    

def ptc_table(covFits, covFitsNoB, i=0, j=0):
    amps = covFits.keys()
    # collect arrays of everything, for stats 
    chi2_tot = np.array([covFits[amp].chi2()/covFits[amp].ndof() for amp in amps])
    a_00 = np.array([covFits[amp].getA()[i,j] for amp in amps])
    sa_00 = np.array([covFits[amp].getASig()[i,j] for amp in amps])
    b_00 = np.array([covFits[amp].getB()[i,j] for amp in amps])
    n = np.sqrt(np.array([covFits[amp].getNoise()[i,j] for amp in amps]))
    gains = np.array([covFits[amp].getGain() for amp in amps])
    chi2_2 = []
    chi2_3 = []
    chi2 = []
    chi2_nb = []
    ndof = []
    for amp in amps:
        mu,var,model,w = covFits[amp].getNormalizedFitData(i,j, divideByMu=False)
        par2 = np.polyfit(mu, var, 2, w = w)
        m2 = np.polyval(par2, mu)
        chi2_2.append(CHI2(var-m2,w)/(len(var)-3))
        par3 = np.polyfit(mu, var, 3, w = w)
        m3 = np.polyval(par3, mu)
        chi2_3.append(CHI2(var-m3,w)/(len(var)-4))
        chi2.append(((covFits[amp].wres()[: ,i,j])**2).sum())
        chi2_nb.append(((covFitsNoB[amp].wres()[: ,i,j])**2).sum())
        ndof.append(len(covFits[amp].mu-4))
        
    chi2_2 = np.array(chi2_2)
    chi2_3 = np.array(chi2_3)
    chi2 = np.array(chi2)
    chi2_nb = np.array(chi2_nb)
    ndof=np.array(ndof)
    chi2_diff = chi2_nb-chi2
    chi2_nb /= ndof
    chi2 /= ndof
    stuff = [a_00, b_00, gains, chi2, chi2_nb, chi2_diff, chi2_2, chi2_3, n , sa_00] 
    names = ['a_%d%d'%(i,j), 'b_%d%d'%(i,j), 'gains', 'chi2', 'chi2_nb', 'chi2_diff', 'chi2_2', 'chi2_3', 'n', 'sa_00']
    print ("PTC Table: ")
    for x,n in zip(stuff, names):
        print('%s: %g %g'%(n,x.mean(), x.std()))
        
def do_cov_exposure_plot(fit, pdfPages, profile_plot=True):
    # Argument is expected to be a covFit
    li = [0,1,1, 0]
    lj = [1,1,0, 0]
    fig=plt.figure(figsize=(8,8))
    for (i,j) in zip(li,lj):
        mu,var,model,w = fit.getNormalizedFitData(i,j, divideByMu=False)

        if profile_plot: 
            gind = findGroups(mu, 1000.)
            xb, yb, wyb, sigyb = binData(mu, var, gind, w)
        else:
            xb,yb,wyb,sigyb = mu, var, mu/np.sqrt(var), np.sqrt(var)/mu
        ax = plt.subplot(2,2,i-2*j+3)

        ax.errorbar(xb,yb,yerr=sigyb, marker = 'o', linestyle='none', markersize = 7)
        ax.plot(mu,var,'.', alpha=0.5)
        ax.set_xlabel('$\mu$ (el)', fontsize='large')
        ax.set_ylabel('$C_{%d%d}/\mu$ (el)'%(i,j), fontsize='large')
        ax.ticklabel_format(style='sci', axis='x', scilimits=(0,0))

    plt.tight_layout()
    pdfPages.savefig(fig)

    
def plot_ptc_data(nt, i=0, j=0):
    amps = set(nt['ampName'].astype(str))
    plt.figure(figsize=(10,10))
    for k,amp in enumerate(amps):
        ax = plt.subplot(4,4,k+1)
        cut = (nt['i']==i)&(nt['j']==j) & (nt['ampName'] == amp)
        nt_amp = nt[cut]
        ax.plot(nt_amp['mu1'], nt_amp['cov']/nt_amp['mu1'], '.')
        ax.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
        ax.text(0.15, 0.85, 'amp %d'%amp, verticalalignment='top', horizontalalignment='left',transform=ax.transAxes, fontsize=15)
    plt.tight_layout()
    plt.show()


def plotPtcAndResiduals(covFits, ampName, pdfPages):
    """Figure 7 in Astier+19"""
    fit = covFits[ampName]
    fig=plt.figure(figsize=(6,12))
    gs = gridspec.GridSpec(4,1, height_ratios=[3, 1, 1, 1])
    gs.update(hspace=0) # stack subplots
    fontsize = 'x-large'
    # extract the data and model
    mu,var,model,w = fit.getNormalizedFitData(0,0, divideByMu=False)

    # var vs mu
    ax0 = plt.subplot(gs[0])
    # allows factors of 10 on the scale
    ax0.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    ax0.set_ylabel("$C_{00}$ (el$^2$)",fontsize=fontsize)
    plt.plot(mu, var, '.', label='data')
    plt.plot(mu, model, '-', label='full model')
    plt.setp(ax0.get_xticklabels(), visible=False)
    # plt.xlabel('$\mu$',fontsize=fontsize)
    plt.legend(loc='upper left',fontsize='large')
    #
    # residuals
    gind = indexForBins(mu, 50)

    ax1 = plt.subplot(gs[1], sharex = ax0)
    xb, yb, wyb, sigyb = binData(mu, var - model, gind, w)
    plt.errorbar(xb, yb, yerr=sigyb, marker='.', ls='none')
    # draw a line at y=0: 
    plt.plot([0, mu.max()], [0,0], ls='--', color= 'k')
    plt.setp(ax1.get_xticklabels(), visible=False)
    ax1.text(0.1, 0.85, 'Residuals to full fit',
        verticalalignment='top', horizontalalignment='left',
             transform=ax1.transAxes, fontsize=15)
    #
    #  quadratic fit
    ax2 = plt.subplot(gs[2], sharex = ax0, sharey=ax1)
    par2 = np.polyfit(mu, var, 2, w = w)
    m2 = np.polyval(par2, mu)
    chi2_2 = CHI2(var-m2,w)/(len(var)-3)
    par3 = np.polyfit(mu, var, 3, w = w)
    m3 = np.polyval(par3, mu)
    chi2_3 = CHI2(var-m3,w)/(len(var)-4)
    xb, yb, wyb, sigyb = binData(mu,  var - m2, gind, w)
    plt.errorbar(xb, yb, yerr=sigyb, marker='.', color='r', ls='none')
    plt.plot([0,mu.max()], [0,0], ls='--', color= 'k')
    plt.setp(ax2.get_xticklabels(), visible=False)
    ax2.text(0.1, 0.85, 'Quadratic fit',
        verticalalignment='top', horizontalalignment='left',
             transform=ax2.transAxes, fontsize=15)
    
    # fit with b=0
    ax3 = plt.subplot(gs[3], sharex = ax0, sharey=ax1)
    fit_nob = fit.copy()
    fit_nob.params['c'].fix(val=0)
    fit_nob.fit()
    mu,var,model,w = fit_nob.getNormalizedFitData(0, 0, divideByMu=False)
    
    xb, yb, wyb, sigyb = binData(mu, var - model, gind, w)
    plt.errorbar(xb, yb, yerr=sigyb, marker='.', color='g', ls='none')
    plt.plot([0, mu.max()], [0,0], ls='--', color= 'k')
    ax3.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    #plt.xlabel('$\mu$ ($10^3$ ADU)',fontsize=fontsize)
    plt.xlabel('$\mu$ (el)',fontsize=fontsize)
    ax3.text(0.1, 0.85, 'b=0',
        verticalalignment='top', horizontalalignment='left',
             transform=ax3.transAxes, fontsize=15)

    plt.tight_layout()
    # remove the 'largest' y label (unelegant overwritings occur)
    for ax in [ax1,ax2,ax3]:
        plt.setp(ax.get_yticklabels()[-1], visible = False)
    
    fig.suptitle("PTC from covariances and model residuals for amplifier %s" %ampName, fontsize=11)
    pdfPages.savefig(fig) 


def ab_vs_dist(covFits, pdfPages, brange=4):
    a = np.array([f.getA() for f in covFits.values()])
    y = a.mean(axis = 0)
    sy = a.std(axis = 0)/np.sqrt(len(covFits))
    i, j = np.indices(y.shape)
    upper = (i>=j).ravel()
    r = np.sqrt(i**2+j**2).ravel()
    y = y.ravel()
    sy = sy.ravel()
    fig = plt.figure(figsize=(6,9))
    ax = fig.add_subplot(211)
    ax.set_xlim([0.5, r.max()+1])
    ax.errorbar(r[upper], y[upper], yerr=sy[upper], marker='o', linestyle='none', color='b', label='$i>=j$')
    ax.errorbar(r[~upper], y[~upper], yerr=sy[~upper], marker='o', linestyle='none', color='r', label='$i<j$')
    ax.legend(loc='upper center', fontsize = 'x-large')
    ax.set_xlabel('$\sqrt{i^2+j^2}$',fontsize='x-large')
    ax.set_ylabel('$a_{ij}$',fontsize='x-large')
    ax.set_yscale('log')
    ax.tick_params(axis='both', labelsize='x-large')

    #axb.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    #
    axb = fig.add_subplot(212)
    b = np.array([f.getB() for f in covFits.values()])
    yb = b.mean(axis = 0)
    syb = b.std(axis = 0)/np.sqrt(len(covFits))
    ib, jb = np.indices(yb.shape)
    upper = (ib>jb).ravel()
    rb = np.sqrt(i**2+j**2).ravel()
    yb = yb.ravel()
    syb = syb.ravel()
    xmin = -0.2
    xmax = brange
    axb.set_xlim([xmin, xmax+0.2])
    cutu = (r>xmin) & (r<xmax) & (upper)
    cutl = (r>xmin) & (r<xmax) & (~upper)
    axb.errorbar(rb[cutu], yb[cutu], yerr=syb[cutu], marker='o', linestyle='none', color='b', label='$i>=j$')
    axb.errorbar(rb[cutl], yb[cutl], yerr=syb[cutl], marker='o', linestyle='none', color='r', label='$i<j$')
    plt.legend(loc='upper center', fontsize='x-large')
    axb.set_xlabel('$\sqrt{i^2+j^2}$',fontsize='x-large')
    axb.set_ylabel('$b_{ij}$',fontsize='x-large')
    axb.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    axb.tick_params(axis='both', labelsize='x-large')
    plt.tight_layout()
    pdfPages.savefig(fig)

from mpl_toolkits.mplot3d import Axes3D       

def make_noise_plot(covFits):
    size = covFits[0].r
    n = np.array([c.params['noise'].full.reshape(size,size) for c in covFits]).mean(axis=0)
    fig = plt.figure(figsize=(15,15))
    ax = fig.add_subplot(111, projection='3d')
    #ax.view_init(elev=20, azim=45)

    x,y = np.meshgrid(range(size),range(size))
    x = x.flatten()
    y = y.flatten()
    n = n.flatten()
    ax.bar3d(x, y, np.zeros(size**2),1,1,n, color='r')
    ax.set_ylabel('i', fontsize='x-large')
    ax.set_xlabel('j', fontsize='x-large')
    ax.set_zlabel('noise (el$^2$)', fontsize='x-large')
    #ax.invert_yaxis() # shows a different figure (!?)
    plt.savefig('noise.png')
        

def eval_nonlin_draw(tuple, knots=20, verbose= False):
    res, ccd, clap = eval_nonlin(tuple, knots, verbose, fullOutput = True)
    plt.figure(figsize=(9,14))
    gs = gridspec.GridSpec(len(ccd),1)
    gs.update(hspace=0) # stack subplots
    for amp in range(len(ccd)):
        x = ccd[amp]
        if x is None:
            continue
        y = clap[amp]
        spl = res[amp]
        model = interp.splev(x, spl)
        ax = plt.subplot(gs[len(ccd)-1-amp])
        binplot(x, y-model, nbins=50, data=False)
    plt.tight_layout()
    plt.show()
    return res

    
def make_distant_cov_plot(covFits, tupleName, pdfPages):
    # need the covFits to get the gains, and the tuple to get the distant
    # covariances
    
    # convert all inputs to electrons
    
    gain_amp = np.array([covFits[i].getGain() if covFits[i] != None else 1.0 for i in list(covFits.keys())])

    gain = gain_amp[tupleName['ext'].astype(int)]
    
    #gain = gain_amp

    mu = 0.5*(tupleName['mu1'] + tupleName['mu2'])*gain

    cov = 0.5*tupleName['cov']*(gain**2) 
    npix = (tupleName['npix'])
    fig = plt.figure(figsize=(8,16))
    # cov vs mu
    ax = plt.subplot(3,1,1)
    #idx = (tupleName['i']**2+tupleName['j']**2 >= 225) & (mu>2.5e4) & (mu<1e5)  
    idx = (tupleName['i']**2+tupleName['j']**2 >= 225) & (mu<2e5) # & (tupleName['sp1']<4) & (tupleName['sp2']<4)
    
    binplot(mu[idx], cov[idx],nbins=20, data=False)
    ax.set_xlabel('$\mu$ (el)',fontsize='x-large')
    ax.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    ax.set_ylabel('$<C_{ij}>$ (el$^2$)',fontsize='x-large')
    ax.text(0.05, 0.8, 'cut: $15 \leqslant \sqrt{i^2+j^2} <29 $' , horizontalalignment='left', transform=ax.transAxes, fontsize='x-large')
    # cov vs angle
    ax=plt.subplot(3,1,2)
    idx = (tupleName['i']**2+tupleName['j']**2 >= 225) & (mu>50000) & (mu<1e5) 
    binplot(np.arctan2(tupleName[idx]['j'],tupleName[idx]['i']), cov[idx],nbins=20, data=False)
    ax.set_xlabel('polar angle (radians)',fontsize='x-large')
    ax.set_ylabel('$<C_{ij}>$ (el$^2$)',fontsize='x-large')
    ax.text(0.15, 0.7, 'cuts: $15 \leqslant \sqrt{i^2+j^2} <29$ & $50000<\mu<100000$', horizontalalignment='left', transform=ax.transAxes, fontsize='x-large')
    #
    ax = plt.subplot(3,1,3)
    idx = (tupleName['j']==0) & (tupleName['i']>4) & (mu>50000) & (mu<1e5) 
    ivalues = np.unique((tupleName[idx]['i']).astype(int))
    bins = np.arange(ivalues.min()-0.5, ivalues.max()+0.55,1)
    binplot(tupleName[idx]['i'], cov[idx],bins=bins, data=False)
    ax.set_xlabel('$i$',fontsize='x-large')
    ax.set_ylabel('$<C_{i0}>$ (el$^2$)',fontsize='x-large')
    ax.text(0.2, 0.85, 'cuts: $i>4$ & $j=0$ & $50000<\mu<100000$', horizontalalignment='left', transform=ax.transAxes, fontsize='x-large')
    # big fonts on axes in all plots: 
    for ax in fig.get_axes():
        ax.tick_params(axis='both', labelsize='x-large')
    plt.tight_layout()
    #plt.show()
    #plt.savefig('distant_cov_plot.pdf')
    pdfPages.savefig(fig)

    
def makeFigSixPlot(tupleName, ampName, pdfPages):
    """Figure 6 in Astier+19"""
    # need the covFits to get the gains, and the tuple to get the distant
    # covariances
   
    # convert all inputs to electrons
    nt0 = tupleName[tupleName['ampName'] == ampName]

    fig = plt.figure(figsize=(6,8))
    gs = gridspec.GridSpec(3,1)
    gs.update(hspace=0.0)
    
    axes =[]
    texts = ['Variance','Nearest parallel \nneighbor covariance','Nearest serial \nneighbor covariance']
    # var vs mu, cov01 vs mu, cov10 vs mu
    for k, indices in enumerate([(0,0), (0,1), (1,0)]):
        if k == 0:
            ax = plt.subplot(gs[k])
            ax0 = ax
            ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
        else:
            ax = plt.subplot(gs[k], sharex = ax0)
        axes.append(ax)
        if k == 1: 
            ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
        i, j = indices
        nt = nt0[(nt0['i'] == i) & (nt0['j'] == j)]
        mu = 0.5*(nt['mu1'] + nt['mu2'])
        cov = 0.5*nt['cov']
        ax.plot(mu, cov,'.b')
        ax.set_ylabel(u'$C_{%d%d}$ (ADU$^2$)'%(i,j), fontsize='x-large')
        ax.text(0.1, 0.7, texts[k], fontsize='x-large', transform=ax.transAxes)
        
        if k != 2:
            plt.setp(ax.get_xticklabels(), visible=False)
            ax.xaxis.offsetText.set_visible(False)
        else:
            ax.set_xlabel('$\mu$ (ADU)', fontsize='x-large')
            ax.ticklabel_format(style='sci', axis='x', scilimits=(0,0))

    gs.tight_layout(fig)
    fig.suptitle(r"$C_{00}$, $C_{01}$, $C_{10}$ for amplifier %s" %ampName, fontsize=12) 
    pdfPages.savefig(fig)


def avoid_overlapping_y_labels(figure):
    axes = figure.get_axes()
    figure.canvas.draw() # make sure the labels are instanciated 
    # suppress the bottom labels, but removes
    # any decorator such as offset or multiplicator !
    for ax in axes:
        labels = [item.get_text() for item in ax.get_yticklabels()]
        labels[0] = ''
        ax.set_yticklabels(labels)

    
def plotRelativeBiasACoeffs(covFits, covFitsNoB, mu_el, pdfPages, maxr=None):
    """Illustrates systematic bias from estimating 'a'
    coefficients from the slope of correlations as opposed to the
    full model in Astier+19.

    Corresponds to Fig. 15 in Astier+19.
    """
    
    fig = plt.figure(figsize=(7,11))
    title = ["'a' relative bias", "'a' relative bias (b=0)"]
    data = [covFits, covFitsNoB]
    
    for k in range(2):
        diffs=[]
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
        diff=diff/amean
        diff[0,0] = 0
        if maxr is None: 
            maxr=diff.shape[0]
        diff = diff[:maxr,:maxr]
        ax0 = fig.add_subplot(2,1,k+1)
        im0 = ax0.imshow(diff.transpose(), origin='lower')
        ax0.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax0.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax0.tick_params(axis='both', labelsize='x-large')
        plt.colorbar(im0)
        ax0.set_title(title[k])
    
    plt.tight_layout()
    pdfPages.savefig(fig)

def eval_a_unweighted_quadratic_fit(fit):
    model = fit.evalCovModel()
    adm = np.zeros_like(fit.getA())
    for i in range(adm.shape[0]):
        for j in range(adm.shape[1]):
            # unweighted fit on purpose: this is what DM does (says Craig )
            p = np.polyfit(fit.mu, model[:,i,j],2)
            # no powers of gain involved for the quadratic term:
            adm[i,j] = p[0]
    return adm
    

def plot_da_dm(covFits, covFitsNoB, maxr=None, figname=None):
    """
    same as above, but consider now that the a are extracted from 
    a quadratic fit to Cov vs mu (above it was Cov/C_00 vs mu)
    """
    fig = plt.figure(figsize=(7,11))
    title = ['a relative bias', 'a relative bias (b=0)']
    data = [covFits,covFitsNoB]
    #
    for k in range(2):
        diffs=[]
        amean = []
        for fit in data[k]:
            if fit is None: continue
            adm = eval_a_unweighted_quadratic_fit(fit)
            a = fit.getA()
            amean.append(a)
            diffs.append((adm-a))
        amean = np.array(amean).mean(axis=0)
        diff = np.array(diffs).mean(axis=0)
        diff=diff/amean
        diff[0,0] = 0
        if maxr is None: maxr=diff.shape[0]
        diff = diff[:maxr, :maxr]
        ax0 = fig.add_subplot(2,1,k+1)
        im0 = ax0.imshow(diff.transpose(), origin='lower')
        ax0.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax0.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax0.tick_params(axis='both', labelsize='x-large')
        plt.colorbar(im0)
        ax0.set_title(title[k])
    #
    plt.tight_layout()
    if figname is not None: plt.savefig(figname)    

# borrowed from Marc Betoule.
def binplot(x, y, nbins=10, robust=False, data=True,
            scale=True, bins=None, weights=None, ls='none',
            dotkeys={'color': 'k'}, xerr=True, **keys):
    """ Bin the y data into n bins of x and plot the average and
    dispersion of each bins.

    Arguments:
    ----------
    nbins: int
      Number of bins

    robust: bool
      If True, use median and nmad as estimators of the bin average
      and bin dispersion.

    data: bool
      If True, add data points on the plot

    scale: bool
      Whether the error bars should present the error on the mean or
      the dispersion in the bin

    bins: list
      The bin definition

    weights: array(len(x))
      If not None, use weights in the computation of the mean.
      Provide 1/sigma**2 for optimal weighting with Gaussian noise

    dotkeys: dict
      To keys to pass to plot when drawing data points

    **keys:
      The keys to pass to plot when drawing bins

    Exemples:
    ---------
    >>> x = np.arange(1000); y = np.random.rand(1000);
    >>> binplot(x,y)
    """
    ind = ~np.isnan(x) & ~np.isnan(y)
    x = x[ind]
    y = y[ind]
    if weights is not None:
        weights = weights[ind]
    if bins is None:
        bins = np.linspace(x.min(), x.max() + abs(x.max() * 1e-7), nbins + 1)
    ind = (x < bins.max()) & (x >= bins.min())
    x = x[ind]
    y = y[ind]
    if weights is not None:
        weights = weights[ind]
    yd = np.digitize(x, bins)
    index = make_index(yd)
    ybinned = [y[e] for e in index]
    xbinned = 0.5 * (bins[:-1] + bins[1:])
    usedbins = np.array(np.sort(list(set(yd)))) - 1
    xbinned = xbinned[usedbins]
    bins = bins[usedbins + 1]
    if data and not 'noplot' in keys:
        plt.plot(x, y, ',', **dotkeys)

    if robust is True:
        yplot = [np.median(e) for e in ybinned]
        yerr = np.array([mad(e) for e in ybinned])
    elif robust:
        yres = [robust_average(e, sigma=None, clip=robust, mad=False, axis=0)
                for e in ybinned]
        yplot = [e[0] for e in yres]
        yerr = [np.sqrt(e[3]) for e in yres]
    elif weights is not None:
        wbinned = [weights[e] for e in index]
        yplot = [np.average(e, weights=w) for e, w in zip(ybinned, wbinned)]
        if not scale:
            #yerr = np.array([np.std((e - a) * np.sqrt(w))
            #                 for e, w, a in zip(ybinned, wbinned, yplot)])
            yerr = np.array([np.sqrt(np.std((e - a) * np.sqrt(w)) ** 2 / sum(w))
                             for e, w, a in zip(ybinned, wbinned, yplot)])
        else:
            yerr = np.array([np.sqrt(1 / sum(w))
                             for e, w, a in zip(ybinned, wbinned, yplot)])
        scale = False
    else:
        yplot = [np.mean(e) for e in ybinned]
        yerr = np.array([np.std(e) for e in ybinned])

    if scale:
        yerr /= np.sqrt(np.bincount(yd)[usedbins + 1])

    if xerr:
        xerr = np.array([bins, bins]) - np.array([xbinned, xbinned])
    else:
        xerr = None
    if not 'noplot' in keys:
        plt.errorbar(xbinned, yplot, yerr=yerr,
                     xerr=xerr,
                     ls=ls, **keys)
    return xbinned, yplot, yerr

    
