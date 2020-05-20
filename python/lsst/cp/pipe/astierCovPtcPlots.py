from __future__ import print_function
#from .ptcfit import *
from .astierCovPtcUtils import *

#import croaks

def covAstierMakeAllPlots (fits, fits_nb, pdfPages, maxmu = 1.4e5, maxmu_el = 1e5,r=8):
#def makeAllPlotsCovsAstier (fits, fits_nb, maxmu = 1.4e5, maxmu_el = 1e5,r=8):
    #try : 
    #    fits, fits_nb = load_fits('fits.pkl')
    #except :
    #    fits, fits_nb = fit_data(tuple_name,maxmu,maxmu_el,r)
    #    save_fits(fits, fits_nb, 'fits.pkl')

    #all_channels_nonlin_plot(figname = 'nonlin_all_channels.pdf')
    do_cov_exposure_plot(fits[10], pdfPages)
    #make_satur_plot(tuple_name = tuple_name, channel=1)
    plot_ptc(fits[0], pdfPages)
    #ptc_table(fits, fits_nb, pdfPages)
    plot_cov_2(fits, fits_nb, 0, 0, pdfPages, offset=0.01, top_plot=True)
    plot_cov_2(fits, fits_nb, 0, 1, pdfPages)
    plot_cov_2(fits, fits_nb, 1, 0, pdfPages)
    plot_a_b(fits, pdfPages)
    ab_vs_dist(fits, pdfPages, brange=4) #figname='ab_vs_dist.pdf')
    #make_distant_cov_plot(fits, tuple_name=tuple_name)
    plot_a_sum(fits, pdfPages)

def make_all_plots_itl(tuple_name='dc1-tuple-corr.npy',maxmu = 1.4e5, maxmu_el = 1e5,r=8) :
    try : 
        fits, fits_nb = load_fits('fits.pkl')
    except :
        fits, fits_nb = fit_data(tuple_name,maxmu,maxmu_el,r)
        save_fits(fits, fits_nb, 'fits.pkl')
    
    do_cov_exposure_plot(fits[10], profile_plot=False)
    make_satur_plot(tuple_name = tuple_name, channel=1)
    plot_ptc(fits[1])
    ptc_table(fits, fits_nb)
    plot_cov_2(fits,fits_nb, 0, 1)
    plot_a_b(fits,figname='a_and_b.png')
    ab_vs_dist(fits, brange=4, figname='a_b_dist.pdf')
    make_distant_cov_plot(fits, tuple_name=tuple_name)


#from .group import *
def CHI2(res,wy):
    wres = res*wy
    return (wres*wres).sum()
    


import matplotlib.pyplot as pl
from matplotlib import gridspec

def plot_cov(fits, i, j, offset=0.004):
    lchi2, la, lb, lcov = [],[], [], []
    pl.figure(figsize=(8,10))
    gs = gridspec.GridSpec(2,1, height_ratios=[3, 1])
    gs.update(hspace=0)
    ax0=pl.subplot(gs[0])
    pl.setp(ax0.get_xticklabels(), visible=False)
    mue, rese, wce = [], [], []
    for amp,fit in fits.items() :
        mu,c, model, wc = fit.getNormalizedFitData(i, j, divideByMu = True)
        chi2 = CHI2(c-model,wc)/(len(mu)-3)
        chi2bin= 0
        mue += list(mu)
        rese += list(c - model)
        wce += list(wc)

        # plot the data and fit
        points, = pl.plot(mu,c+amp*offset,'.')
        # the corresponding fit
        pl.plot(mu,model+amp*offset,'-', color="k",linewidth=4.0)
        # bin plot 
        gind = index_for_bins(mu, 25) #from group
        xb, yb, wyb, sigyb = bin_data(mu,c,gind, wc)  # from group
        chi2bin = (sigyb*wyb).mean() # chi2 of enforcing the same value in each bin
        z = pl.errorbar(xb,yb+amp*offset,yerr=sigyb, marker = 'o', linestyle='none', markersize = 7, color=points.get_color(), label="ch %d"%amp)
        aij = fit.getA()[i,j]
        bij = fit.getB()[i,j]
        la.append(aij)
        lb.append(bij)
        lcov.append(fit.getACov()[i,j,i,j])
        lchi2.append(chi2)
        print('%i : slope %g b %g  chi2 %f chi2bin %f'%(amp, aij , bij, chi2, chi2bin))
    # end loop on amps
    la = np.array(la)
    lb = np.array(lb)
    lcov = np.array(lcov)
    lchi2 = np.array(lchi2)
    mue = np.array(mue)
    rese = np.array(rese)
    wce = np.array(wce)
    
    #pl.legend(loc='upper left')
    pl.xlabel("$\mu (el)$",fontsize='x-large')
    pl.ylabel("$C_{%d%d}/\mu + Cst (el)$"%(i,j),fontsize='x-large')
    #gind = group.find_groups(mue, 2000.)
    gind = index_for_bins(mue, 25) # group
    xb, yb, wyb, sigyb = bin_data(mue,rese , gind, wce) # group 
    #pl.errorbar(xb,yb,yerr=sigyb, fmt='o', label='data')

    ax1 = pl.subplot(gs[1], sharex = ax0)
    pl.errorbar(xb,yb, yerr=sigyb, marker='o', linestyle='none')
    pl.plot(xb,[0]*len(xb),'--')
    #pl.plot(xb,model,'--')
    pl.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    pl.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    pl.xlabel('$\mu (el)$',fontsize='x-large')
    pl.ylabel('$C_{%d%d}/\mu$ -model (el)'%(i,j),fontsize='x-large')
    pl.tight_layout()
    pl.show()

    print('      & $a_{%d%d}$ & '%(j,i),' & $b_{%d%d}$ & '%(j,i),' & $\chi^2/N_{dof}$ \\\\ \ hline')
    print('value & %9.3g & %9.3g & %5.2f\\\\'%(la.mean(), lb.mean(), lchi2.mean()))
    print('scatter & %8.3g & %8.3g & %5.2f\\\\ \hline'%(la.std(), lb.std(), lchi2.std()))

    a_expected_rms = np.sqrt(lcov.mean())
    print("expected rms of a_%1d%1d = %g"%(j,i, a_expected_rms))
    #print('i,j = %d %d r2 = %8.3g +/- %8.3g (exp: %8.3g)'%(i,j,r2,r2_rms,r2_rms_expected))
    #print('%8.3g +/- %8.3g %8.3g +/- %8.3g %8.2g +/- %7.2g'%(lp0.mean(), np.sqrt(lcov[:,0,0].mean()), lp1.mean(), np.sqrt(lcov[:,1,1].mean()), lp2.mean(), np.sqrt(lcov[:,2,2].mean()) ))
    #print('offset : %g +r/- %g, expected rms = %g'%(lp2.mean(), lp2.std(), np.sqrt(lcov[:,2,2].mean())))

def plot_cov_2(fits, fits_nb, i, j, pdfPages, offset=0.004, figname=None, plot_data = True, top_plot=False):
    lchi2, la, lb, lcov = [],[], [], []

    if (not top_plot) :
        fig = pl.figure(figsize=(8,10))
        gs = gridspec.GridSpec(2,1, height_ratios=[3, 1])
        gs.update(hspace=0)
        ax0=pl.subplot(gs[0])
        pl.setp(ax0.get_xticklabels(), visible=False)
    else :
        fig = pl.figure(figsize=(8,8))
        ax0 = pl.subplot(111)
        ax0.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    ax0.tick_params(axis='both', labelsize='x-large')
    mue, rese, wce = [], [], []
    mue_nb, rese_nb, wce_nb = [], [], []
    for amp,fit in fits.items() :
        mu, c, model, wc = fit.getNormalizedFitData(i, j, divideByMu = True)
        chi2 = CHI2(c-model,wc)/(len(mu)-3)
        chi2bin= 0
        mue += list(mu)
        rese += list(c - model)
        wce += list(wc)

        fit_nb = fits_nb[amp]
        mu_nb, c_nb, model_nb, wc_nb = fit_nb.getNormalizedFitData(i, j, divideByMu = True)
        mue_nb += list(mu_nb)
        rese_nb += list(c_nb - model_nb)
        wce_nb += list(wc_nb)

        
        # the corresponding fit
        fit_curve, = pl.plot(mu,model+amp*offset,'-',linewidth=4.0)
        # bin plot 
        gind = index_for_bins(mu, 25) #group
        xb, yb, wyb, sigyb = bin_data(mu,c,gind, wc) # group 
        chi2bin = (sigyb*wyb).mean() # chi2 of enforcing the same value in each bin
        z = pl.errorbar(xb,yb+amp*offset,yerr=sigyb, marker = 'o', linestyle='none', markersize = 7, color=fit_curve.get_color(), label="ch %d"%amp)
        # plot the data
        if plot_data :
            points, = pl.plot(mu,c+amp*offset, '.', color = fit_curve.get_color())

        aij = fit.getA()[i,j]
        bij = fit.getB()[i,j]
        la.append(aij)
        lb.append(bij)
        lcov.append(fit.getACov()[i,j,i,j])
        lchi2.append(chi2)
        print('%i : slope %g b %g  chi2 %f chi2bin %f'%(amp, aij , bij, chi2, chi2bin))
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

    
    pl.xlabel("$\mu (el)$",fontsize='x-large')
    pl.ylabel("$C_{%d%d}/\mu + Cst (el)$"%(i,j),fontsize='x-large')
    if (not top_plot):
        #gind = group.find_groups(mue, 2000.)
        gind = index_for_bins(mue, 25)
        xb, yb, wyb, sigyb = bin_data(mue,rese , gind, wce)
        #pl.errorbar(xb,yb,yerr=sigyb, fmt='o', label='data')
        print('yb0 %g'%yb[0])
    
        ax1 = pl.subplot(gs[1], sharex = ax0)
        ax1.errorbar(xb,yb, yerr=sigyb, marker='o', linestyle='none', label='full fit')
        gind_nb = index_for_bins(mue_nb, 25)
        xb2, yb2, wyb2, sigyb2 = bin_data(mue_nb,rese_nb , gind_nb, wce_nb)
        print('yb0 %g %g'%(yb[0],yb2[0]))
    
        ax1.errorbar(xb2,yb2, yerr=sigyb2, marker='o', linestyle='none', label='b = 0')
        ax1.tick_params(axis='both', labelsize='x-large')
        pl.legend(loc='upper left', fontsize='large')    
        # horizontal line at zero
        pl.plot(xb,[0]*len(xb),'--', color = 'k')
        #pl.plot(xb,model,'--')
        pl.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
        pl.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
        pl.xlabel('$\mu (el)$',fontsize='x-large')
        pl.ylabel('$C_{%d%d}/\mu$ -model (el)'%(i,j),fontsize='x-large')
    pl.tight_layout()
    #pl.show()

    print('      & $a_{%d%d}$ & '%(i,j),' & $b_{%d%d}$ & '%(i,j),' & $\chi^2/N_{dof}$ \\\\ \hline')
    print('value & %9.3g & %9.3g & %5.2f\\\\'%(la.mean(), lb.mean(), lchi2.mean()))
    print('scatter & %8.3g & %8.3g & %5.2f\\\\ \hline'%(la.std(), lb.std(), lchi2.std()))

    a_expected_rms = np.sqrt(lcov.mean())
    print("expected rms of a_%1d%1d = %g"%(j,i, a_expected_rms))
    #print('i,j = %d %d r2 = %8.3g +/- %8.3g (exp: %8.3g)'%(i,j,r2,r2_rms,r2_rms_expected))
    # overlapping y labels:
    fig.canvas.draw()
    labels0 = [item.get_text() for item in ax0.get_yticklabels()]
    labels0[0] = u''
    ax0.set_yticklabels(labels0)
    #
    #if figname is not None:
    #    pl.savefig(figname)
    pdfPages.savefig(fig)

def plot_chi2_diff(fits, fits_nob) :
    chi2_diff = []
    for amp in fits.keys() :
        dchi2 =  ((fits_nob[amp].wres())**2).sum(axis=0)-((fits[amp].wres())**2).sum(axis=0)
        print(dchi2.sum())
        chi2_diff.append(dchi2)
    chi2_diff = np.array(chi2_diff).mean(axis=0)
    fig = pl.figure(figsize=(8,8))
    ax = fig.add_subplot(111)
    im = ax.imshow(chi2_diff.transpose(), origin='lower', norm = mpl.colors.LogNorm(), interpolation='none')
    pl.colorbar(im )
    ax.set_title(r'$\delta \chi^2$ for $b \neq 0$',fontsize = 'x-large') 
    
import matplotlib as mpl
from matplotlib.ticker import MaxNLocator

def plot_a_sum(fits, pdfPages) :
    a, b = [],[]
    for amp,fit in fits.items() :
        a.append(fit.getA())
        b.append(fit.getB())
    a = np.array(a).mean(axis=0)
    b = np.array(b).mean(axis=0)
    fig = pl.figure(figsize=(7,6))
    w = 4*np.ones_like(a)
    w[0,1:] = 2
    w[1:,0] = 2
    w[0,0] = 1
    wa = w*a
    indices = range(1,a.shape[0]+1)
    sums = [wa[0:n,0:n].sum() for n in indices]
    ax = pl.subplot(111)
    ax.plot(indices,sums/sums[0],'o',color='b')
    ax.set_yscale('log')
    ax.set_xlim(indices[0]-0.5, indices[-1]+0.5)
    ax.set_ylim(None, 1.2)
    ax.set_ylabel('$[\sum_{|i|<n\  &\  |j|<n} a_{ij}] / |a_{00}|$',fontsize='x-large')
    ax.set_xlabel('n',fontsize='x-large')
    ax.tick_params(axis='both', labelsize='x-large')
    pl.tight_layout()
    pdfPages.savefig(fig)
    

def plot_a_b(fits, pdfPages, brange=3) :
    a, b = [],[]
    for amp,fit in fits.items() :
        a.append(fit.getA())
        b.append(fit.getB())
    a = np.array(a).mean(axis=0)
    b = np.array(b).mean(axis=0)
    fig = pl.figure(figsize=(7,11))
    ax0 = fig.add_subplot(2,1,1)
    im0 = ax0.imshow(np.abs(a.transpose()), origin='lower', norm = mpl.colors.LogNorm(), interpolation='none')
    ax0.tick_params(axis='both', labelsize='x-large')
    ax0.set_title('$|a|$', fontsize='x-large')
    ax0.xaxis.set_ticks_position('bottom')
    cb0 = pl.colorbar(im0)
    cb0.ax.tick_params(labelsize='x-large')
    #
    ax1 = fig.add_subplot(2,1,2)
    ax1.tick_params(axis='both', labelsize='x-large')
    ax1.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
    im1 = ax1.imshow(1e6*b[:brange,:brange].transpose(), origin='lower', interpolation='none')
    cb1 = pl.colorbar(im1)
    cb1.ax.tick_params(labelsize='x-large')
    ax1.set_title(r'$b \times 10^6$', fontsize='x-large')
    ax1.xaxis.set_ticks_position('bottom')
    pl.tight_layout()
    pdfPages.savefig(fig)
    

import copy


def ptc_table(fits, fits_nb, i=0, j=0):
    amps = fits.keys()
    # collect arrays of everything, for stats 
    chi2_tot = np.array([fits[amp].chi2()/fits[amp].ndof() for amp in amps])
    a_00 = np.array([fits[amp].getA()[i,j] for amp in amps])
    sa_00 = np.array([fits[amp].getASig()[i,j] for amp in amps])
    b_00 = np.array([fits[amp].getB()[i,j] for amp in amps])
    n = np.sqrt(np.array([fits[amp].getNoise()[i,j] for amp in amps]))
    gains = np.array([fits[amp].getGain() for amp in amps])
    chi2_2 = []
    chi2_3 = []
    chi2 = []
    chi2_nb = []
    ndof = []
    for amp in amps :
        mu,var,model,w = fits[amp].getNormalizedFitData(i,j, divideByMu=False)
        par2 = np.polyfit(mu, var, 2, w = w)
        m2 = np.polyval(par2, mu)
        chi2_2.append(CHI2(var-m2,w)/(len(var)-3))
        par3 = np.polyfit(mu, var, 3, w = w)
        m3 = np.polyval(par3, mu)
        chi2_3.append(CHI2(var-m3,w)/(len(var)-4))
        chi2.append(((fits[amp].wres()[: ,i,j])**2).sum())
        chi2_nb.append(((fits_nb[amp].wres()[: ,i,j])**2).sum())
        ndof.append(len(fits[amp].mu-4))
        
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
    for x,n in zip(stuff, names) :
        print('%s : %g %g'%(n,x.mean(), x.std()))
        
def do_cov_exposure_plot(fit, pdfPages, profile_plot=True):
    # Argument is expected to be a covFit
    li = [0,1,1, 0]
    lj = [1,1,0, 0]
    fig=pl.figure(figsize=(8,8))
    for (i,j) in zip(li,lj) :
        mu,var,model,w = fit.getNormalizedFitData(i,j, divideByMu=True)
        if profile_plot : 
            gind = find_groups(mu, 1000.)
            xb, yb, wyb, sigyb = bin_data(mu, var, gind, w)
        else :
            xb,yb,wyb,sigyb = mu, var, mu/np.sqrt(var), np.sqrt(var)/mu
        ax = pl.subplot(2,2,i-2*j+3)

        ax.errorbar(xb,yb,yerr=sigyb, marker = 'o', linestyle='none', markersize = 7)
        ax.plot(mu,var,'.', alpha=0.5)
        ax.set_xlabel('$\mu$ (el)', fontsize='large')
        ax.set_ylabel('$C_{%d%d}/\mu$ (el)'%(i,j), fontsize='large')
        ax.ticklabel_format(style='sci', axis='x', scilimits=(0,0))

    pl.tight_layout()
    pdfPages.savefig(fig)


def plot_clap_data(nt) :
    amps = set(nt['ext'].astype(int))
    pl.figure(figsize=(15,15))
    pl.suptitle('clap-ccd vs clap',fontsize='x-large')
    for k,amp in enumerate(amps):
        ax = pl.subplot(4,4,k+1)
        cut = (nt['i']==0)&(nt['j']==0) & (nt['ext'] == amp)
        nt_amp = nt[cut]
        nt_amp = nt_amp[np.isfinite(nt_amp['c1'])]
        x = nt_amp['c1']
        if (x.mean()<0) : x= -x
        y = nt_amp['mu1']
        pars = np.polyfit(x,y,1)
        x *= pars[0]
        ax.plot(x, y-x, '.')
        ax.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
        ax.text(0.15, 0.85, 'amp %d'%amp, verticalalignment='top', horizontalalignment='left',transform=ax.transAxes, fontsize=15)
    pl.tight_layout()
    pl.show()

    
def plot_ptc_data(nt, i=0, j=0) :
    amps = set(nt['ext'].astype(int))
    pl.figure(figsize=(10,10))
    for k,amp in enumerate(amps):
        ax = pl.subplot(4,4,k+1)
        cut = (nt['i']==i)&(nt['j']==j) & (nt['ext'] == amp)
        nt_amp = nt[cut]
        ax.plot(nt_amp['mu1'], nt_amp['cov']/nt_amp['mu1'], '.')
        ax.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
        ax.text(0.15, 0.85, 'amp %d'%amp, verticalalignment='top', horizontalalignment='left',transform=ax.transAxes, fontsize=15)
    pl.tight_layout()
    pl.show()


def plot_ptc(fit, pdfPages) :
    # Firsdt Argument is expected to be a covFit
    fig=pl.figure(figsize=(6,12))
    gs = gridspec.GridSpec(4,1, height_ratios=[3, 1, 1, 1])
    gs.update(hspace=0) # stack subplots
    fontsize = 'x-large'
    # extract the data and model
    mu,var,model,w = fit.getNormalizedFitData(0,0, divideByMu=False)

    # var vs mu
    ax0 = pl.subplot(gs[0])
    # allows factors of 10 on the scale
    ax0.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    ax0.set_ylabel("$C_{00}$ (el$^2$)",fontsize=fontsize)
    pl.plot(mu, var, '.', label='data')
    pl.plot(mu, model, '-', label='full model')
    pl.setp(ax0.get_xticklabels(), visible=False)
    # pl.xlabel('$\mu$',fontsize=fontsize)
    pl.legend(loc='upper left',fontsize='large')
    #
    # residuals
    gind = index_for_bins(mu, 50)

    ax1 = pl.subplot(gs[1], sharex = ax0)
    xb, yb, wyb, sigyb = bin_data(mu, var - model, gind, w)
    pl.errorbar(xb, yb, yerr=sigyb, marker='.', ls='none')
    # draw a line at y=0 : 
    pl.plot([0, mu.max()], [0,0], ls='--', color= 'k')
    pl.setp(ax1.get_xticklabels(), visible=False)
    ax1.text(0.1, 0.85, 'Residuals to full fit',
        verticalalignment='top', horizontalalignment='left',
             transform=ax1.transAxes, fontsize=15)
    #
    #  quadratic fit
    ax2 = pl.subplot(gs[2], sharex = ax0, sharey=ax1)
    par2 = np.polyfit(mu, var, 2, w = w)
    m2 = np.polyval(par2, mu)
    chi2_2 = CHI2(var-m2,w)/(len(var)-3)
    par3 = np.polyfit(mu, var, 3, w = w)
    m3 = np.polyval(par3, mu)
    chi2_3 = CHI2(var-m3,w)/(len(var)-4)
    xb, yb, wyb, sigyb = bin_data(mu,  var - m2, gind, w)
    pl.errorbar(xb, yb, yerr=sigyb, marker='.', color='r', ls='none')
    pl.plot([0,mu.max()], [0,0], ls='--', color= 'k')
    pl.setp(ax2.get_xticklabels(), visible=False)
    ax2.text(0.1, 0.85, 'Quadratic fit',
        verticalalignment='top', horizontalalignment='left',
             transform=ax2.transAxes, fontsize=15)
    #
    # fit with b=0
    ax3 = pl.subplot(gs[3], sharex = ax0, sharey=ax1)
    fit_nob = fit.copy()
    fit_nob.params['c'].fix(val=0)
    fit_nob.fit()
    mu,var,model,w = fit_nob.getNormalizedFitData(0, 0, divideByMu=False)
    
    xb, yb, wyb, sigyb = bin_data(mu, var - model, gind, w)
    pl.errorbar(xb, yb, yerr=sigyb, marker='.', color='g', ls='none')
    pl.plot([0, mu.max()], [0,0], ls='--', color= 'k')
    ax3.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    #pl.xlabel('$\mu$ ($10^3$ ADU)',fontsize=fontsize)
    pl.xlabel('$\mu$ (el)',fontsize=fontsize)
    ax3.text(0.1, 0.85, 'b=0',
        verticalalignment='top', horizontalalignment='left',
             transform=ax3.transAxes, fontsize=15)

    pl.tight_layout()
#    pl.show()
    # remove the 'largest' y label (unelegant overwritings occur)
    for ax in [ax1,ax2,ax3] :
        pl.setp(ax.get_yticklabels()[-1], visible = False)
    
    pdfPages.savefig(fig) 
    #pl.savefig('ptc_fit_plot.pdf')

def ab_vs_dist(fits, pdfPages, brange=4) :
    a = np.array([f.getA() for f in fits.values()])
    y = a.mean(axis = 0)
    sy = a.std(axis = 0)/np.sqrt(len(fits))
    i, j = np.indices(y.shape)
    upper = (i>=j).ravel()
    r = np.sqrt(i**2+j**2).ravel()
    y = y.ravel()
    sy = sy.ravel()
    fig = pl.figure(figsize=(6,9))
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
    b = np.array([f.getB() for f in fits.values()])
    yb = b.mean(axis = 0)
    syb = b.std(axis = 0)/np.sqrt(len(fits))
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
    pl.legend(loc='upper center', fontsize='x-large')
    axb.set_xlabel('$\sqrt{i^2+j^2}$',fontsize='x-large')
    axb.set_ylabel('$b_{ij}$',fontsize='x-large')
    axb.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    axb.tick_params(axis='both', labelsize='x-large')
    #pl.show()
    pl.tight_layout()
    pdfPages.savefig(fig)
    #if figname is not None:
    #    pl.savefig(figname)

from mpl_toolkits.mplot3d import Axes3D       

def make_noise_plot(fits) :
    size = fits[0].r
    n = np.array([c.params['noise'].full.reshape(size,size) for c in fits]).mean(axis=0)
    fig = pl.figure(figsize=(15,15))
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
    pl.savefig('noise.png')
        

#import pickle
#def save_fits(fits, fits_nob, filename) :
#    file = open(filename,'wb')
#    pickle.dump(fits,file)
#    pickle.dump(fits_nob,file)
#    file.close()

#def load_fits(filename):
#    file = open(filename)
#    fits = pickle.load(file)
#    fits_nb = pickle.load(file)
#    file.close()
#    return fits,fits_nb

def eval_nonlin_draw(tuple, knots=20, verbose= False):
    res, ccd, clap = eval_nonlin(tuple, knots, verbose, fullOutput = True)
    pl.figure(figsize=(9,14))
    gs = gridspec.GridSpec(len(ccd),1)
    gs.update(hspace=0) # stack subplots
    for amp in range(len(ccd)) :
        x = ccd[amp]
        if x is None:
            continue
        y = clap[amp]
        spl = res[amp]
        model = interp.splev(x, spl)
        ax = pl.subplot(gs[len(ccd)-1-amp])
        binplot(x, y-model, nbins=50, data=False)
    pl.tight_layout()
    pl.show()
    return res

    
def make_distant_cov_plot(fits, tuple_name='v12/dc5-tuple.npy'):
    # need the fits to get the gains, and the tuple to get the distant
    # covariances
    ntuple = croaks.NTuple.fromfile(tuple_name)
    # convert all inputs to electrons
    gain_amp = np.array([fits[i].getGain() if fits[i] != None else 0 for i in range(len(fits))])
    gain = gain_amp[ntuple['ext'].astype(int)]
    
    mu = 0.5*(ntuple['mu1'] + ntuple['mu2'])*gain
    cov = 0.5*ntuple['cov']*(gain**2) 
    npix = (ntuple['npix'])
    fig = pl.figure(figsize=(8,16))
    # cov vs mu
    ax = pl.subplot(3,1,1)
    #idx = (ntuple['i']**2+ntuple['j']**2 >= 225) & (mu>2.5e4) & (mu<1e5)  
    idx = (ntuple['i']**2+ntuple['j']**2 >= 225) & (mu<1e5) & (ntuple['sp1']<4) & (ntuple['sp2']<4)
    binplot(mu[idx], cov[idx],nbins=20, data=False)
    ax.set_xlabel('$\mu$ (el)',fontsize='x-large')
    ax.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    ax.set_ylabel('$<C_{ij}>$ (el$^2$)',fontsize='x-large')
    ax.text(0.05, 0.8, 'cut: $15 \leqslant \sqrt{i^2+j^2} <29 $' , horizontalalignment='left', transform=ax.transAxes, fontsize='x-large')
    # cov vs angle
    ax=pl.subplot(3,1,2)
    idx = (ntuple['i']**2+ntuple['j']**2 >= 225) & (mu>50000) & (mu<1e5) 
    binplot(np.arctan2(ntuple[idx]['j'],ntuple[idx]['i']), cov[idx],nbins=20, data=False)
    ax.set_xlabel('polar angle (radians)',fontsize='x-large')
    ax.set_ylabel('$<C_{ij}>$ (el$^2$)',fontsize='x-large')
    ax.text(0.15, 0.7, 'cuts: $15 \leqslant \sqrt{i^2+j^2} <29$ & $50000<\mu<100000$', horizontalalignment='left', transform=ax.transAxes, fontsize='x-large')
    #
    ax = pl.subplot(3,1,3)
    idx = (ntuple['j']==0) & (ntuple['i']>4) & (mu>50000) & (mu<1e5) 
    ivalues = np.unique((ntuple[idx]['i']).astype(int))
    bins = np.arange(ivalues.min()-0.5, ivalues.max()+0.55,1)
    binplot(ntuple[idx]['i'], cov[idx],bins=bins, data=False)
    ax.set_xlabel('$i$',fontsize='x-large')
    ax.set_ylabel('$<C_{i0}>$ (el$^2$)',fontsize='x-large')
    ax.text(0.2, 0.85, 'cuts: $i>4$ & $j=0$ & $50000<\mu<100000$', horizontalalignment='left', transform=ax.transAxes, fontsize='x-large')
    # big fonts on axes in all plots: 
    for ax in fig.get_axes() :
        ax.tick_params(axis='both', labelsize='x-large')
    pl.tight_layout()
    pl.show()
    pl.savefig('distant_cov_plot.pdf')

#from mpl_toolkits.axes_grid1 import AxesGrid
    
def make_satur_plot(tuple_name='v12/dc4-tuple.npy', channel=0, figname=None):
    # need the fits to get the gains, and the tuple to get the distant
    # covariances
    ntuple = croaks.NTuple.fromfile(tuple_name)
    # convert all inputs to electrons
    nt0 = ntuple[ntuple['ext'] == channel]

    mu_el_cut = 1e5
    gain = 0.733
    mu_cut = mu_el_cut/gain
    fig = pl.figure(figsize=(6,8))
    gs = gridspec.GridSpec(3,1)
    gs.update(hspace=0.0)
    #g = AxesGrid(fig, 111, (3,1), label_mode='L', aspect=False)

    #gs.update(hspace=0.05)
    axes =[]
    texts = ['Variance','Nearest parallel \nneighbor covariance','Nearest serial \nneighbor covariance']
    # var vs mu, cov01 vs mu, cov10 vs mu
    for k, indices in enumerate([(0,0), (0,1), (1,0)]) :
        if k == 0:
            ax = pl.subplot(gs[k])
            ax0 = ax
            ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
        else :
            ax = pl.subplot(gs[k], sharex = ax0)
        axes.append(ax)
        if k == 1 : ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
        i,j= indices
        nt = nt0[(nt0['i'] == i) & (nt0['j'] == j)]
        mu = 0.5*(nt['mu1'] + nt['mu2'])
        cov = 0.5*nt['cov']
        ax.plot(mu,cov,'.b')
        # vertical line
        ax.plot([mu_cut, mu_cut],ax.get_ylim(),'--')
        ax.set_ylabel(u'$C_{%d%d}$ (ADU$^2$)'%(i,j), fontsize='x-large')
        ax.text(0.1, 0.7, texts[k], fontsize='x-large', transform=ax.transAxes)
        
        if k != 2 :
            pl.setp(ax.get_xticklabels(), visible=False)
            ax.xaxis.offsetText.set_visible(False)
        else :
            ax.set_xlabel('$\mu$ (ADU)', fontsize='x-large')
            ax.ticklabel_format(style='sci', axis='x', scilimits=(0,0))

    gs.tight_layout(fig)
    if figname is not None : pl.savefig(figname)
    #avoid_overlapping_y_labels(fig)

    return ax

def avoid_overlapping_y_labels(figure):
    axes = figure.get_axes()
    figure.canvas.draw() # make sure the labels are instanciated 
    # suppress the bottom labels, but removes
    # any decorator such as offset or multiplicator !
    for ax in axes :
        labels = [item.get_text() for item in ax.get_yticklabels()]
        labels[0] = ''
        ax.set_yticklabels(labels)


import scipy.interpolate as interp
        
#it has to be an uncorrected tuple
def make_nonlin_plot(filename, knots=20, channel=8) : 
    nt = croaks.NTuple.fromfile(filename)
    cut = (nt['mu1']<1.3e5) & (nt['sp1']<3.5) & (nt['sp2']<3.5)
    t8 = nt[cut & (nt['ext'] == channel)]
    dict = {}
    if knots is not None:
        dict['knots'] = knots
    s,x,yclap = fit_nonlin_corr(t8['mu1'],t8['c1'], fullOutput=True, **dict)
    model = interp.splev(x,s)
    fig=pl.figure(figsize=(8,14))
    axes = fig.add_subplot(2,1,1)

    pl.plot(x, x/yclap-1, '.', label= 'data')
    pl.plot(x, x/model -1, '-r', label='model')
    pl.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    my_fontsize='xx-large'
    pl.xlabel("$\mu$(ADU)", fontsize=my_fontsize)
    pl.xticks(fontsize=my_fontsize)
    pl.yticks(fontsize=my_fontsize)
    pl.ylabel("$\mu$/diode-1", fontsize=my_fontsize)
    pl.legend(loc='upper right', fontsize=my_fontsize)
    
    fig.add_subplot(2,1,2)
    mu = t8['mu1']
    mu_cor = interp.splev(mu,s)
    dd = interp.splder(s)
    der= interp.splev(mu,dd)
    var = 0.5*t8['var']
    var_cor = var*(der**2)
    pl.plot(mu, var/mu,'b.', label='before correction')
    pl.plot(mu_cor, var_cor/mu_cor,'r.', label='after correction')
    pl.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    pl.xlabel("$\mu$ (ADU)",fontsize=my_fontsize)
    pl.ylabel("$C_{00}/\mu$",fontsize=my_fontsize)
    pl.xticks(fontsize=my_fontsize)
    pl.yticks(fontsize=my_fontsize)
    pl.legend(loc='upper right', fontsize=my_fontsize)
    pl.tight_layout()
    fig.show()
    pl.savefig("nonlin_plot.pdf")

def all_channels_nonlin_plot(figname = None):
    pl.figure(figsize=(7,7))
    ax = pl.subplot(111)
    splines =  pickle.load(open('v12/nonlin.pkl','rb'))
    mu = np.linspace(1000,1.3e5,200)
    for spline in splines:
        mu_cor = interp.splev(mu,spline)
        ax.plot(mu, mu/mu_cor-1)
    ax.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    ax.set_xlabel("$\mu$ (ADU)", fontsize='x-large')
    ax.set_ylabel("$\mu$/diode-1", fontsize='x-large')
    pl.tight_layout()
    if (figname is not None) : pl.savefig(figname)


    
def do_spikes_plot() :
    nt_cti=croaks.NTuple.fromfile('v12/tuple-cti.npy')
    nt_before=croaks.NTuple.fromfile('v12/corr-tuple.npy')
    nt_after=croaks.NTuple.fromfile('v12/corr-dc2-tuple.npy')
    spikes_plot(nt_cti, nt_before, nt_after)

def spikes_plot(nt_cti, nt_before, nt_after, channel=10) :
    import pickle
    dc_corr = pickle.load(open('cte.pkl','rb'))
    #delta = dc_corr[channel](image[:,1:])
    pl.figure(figsize=(6,12))

    #
    ax0 = pl.subplot(3,1,1)
    max_mu = 1.4e5
    n = nt_cti[nt_cti['f0']<max_mu]
    ax0.plot(n['f0'], n['f1'], '.')
    max_mu = 1.4e5
    mu = np.linspace(0, max_mu , 140)    
    corr_model = dc_corr[channel](mu)
    ax0.plot(mu, corr_model,'r-')
    pl.xlabel('$\mu$ (ADU)',fontsize='x-large')
    pl.ylabel('next pixel (ADU)',fontsize='x-large')
    pl.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    #
    ax1 = pl.subplot(3,1,2, sharex = ax0)
    n = nt_before[(nt_before['i'] == 1) & (nt_before['j'] == 0) & (nt_before['ext'] == channel) ]
    ax1.plot(n['mu1'], 0.5*n['cov']/n['mu1'], '.', label='before', alpha=0.4)
    n = nt_after[(nt_after['i'] == 1) & (nt_after['j'] == 0) & (nt_after['ext'] == channel)]
    ax1.plot(n['mu1'], 0.5*n['cov']/n['mu1'], '.', label='after', alpha=0.4)
    pl.xlabel('$\mu$ (ADU)',fontsize='x-large')
    pl.ylabel('$C_{10}/\mu (ADU)$',fontsize='x-large')
    pl.legend(loc='upper left')
    pl.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    #
    ax1 = pl.subplot(3,1,3, sharex = ax0)
    n = nt_before[(nt_before['i'] == 0) & (nt_before['j'] == 0) & (nt_before['ext'] == channel) ]
    ax1.plot(n['mu1'], 0.5*n['var']/n['mu1'], '.', label='before', alpha=0.4)
    n = nt_after[(nt_after['i'] == 0) & (nt_after['j'] == 0) & (nt_after['ext'] == channel)]
    ax1.plot(n['mu1'], 0.5*n['var']/n['mu1'], '.', label='after', alpha=0.4)
    pl.xlabel('$\mu$ (ADU)',fontsize='x-large')
    pl.ylabel('$C_{00}/\mu (ADU)$',fontsize='x-large')
    pl.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    pl.legend(loc='upper right')

    pl.tight_layout()
    pl.show()
    pl.savefig('cti_plot.png')

    
def plot_da(fits, fitsnb, mu_el, maxr=None, figname=None):
    fig = pl.figure(figsize=(7,11))
    title = ['a relative bias', 'a relative bias (b=0)']
    data = [fits,fitsnb]
    #
    for k in range(2):
        diffs=[]
        amean = []
        for fit in data[k]:
            if fit is None: continue
            aOld = aCoeffsComputeOldFashion(fit, mu_el)
            a = fit.getA()
            amean.append(a)
            diffs.append((aOld-a))
        amean = np.array(amean).mean(axis=0)
        diff = np.array(diffs).mean(axis=0)
        diff=diff/amean
        diff[0,0] = 0
        if maxr is None: maxr=diff.shape[0]
        diff = diff[:maxr, :maxr]
        ax0 = fig.add_subplot(2,1,k+1)
        im0 = ax0.imshow(diff.transpose(), origin='lower', interpolation='none')
        ax0.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax0.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax0.tick_params(axis='both', labelsize='x-large')
        pl.colorbar(im0)
        ax0.set_title(title[k])
    #
    pl.tight_layout()
    if figname is not None: pl.savefig(figname)

def eval_a_unweighted_quadratic_fit(fit) :
    model = fit.evalCovModel()
    adm = np.zeros_like(fit.getA())
    for i in range(adm.shape[0]):
        for j in range(adm.shape[1]):
            # unweighted fit on purpose: this is what DM does (says Craig )
            p = np.polyfit(fit.mu, model[:,i,j],2)
            # no powers of gain involved for the quadratic term:
            adm[i,j] = p[0]
    return adm
    

def plot_da_dm(fits, fitsnb, maxr=None, figname=None):
    """
    same as above, but consider now that the a are extracted from 
    a quadratic fit to Cov vs mu (above it was Cov/C_00 vs mu)
    """
    fig = pl.figure(figsize=(7,11))
    title = ['a relative bias', 'a relative bias (b=0)']
    data = [fits,fitsnb]
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
        im0 = ax0.imshow(diff.transpose(), origin='lower', interpolation='none')
        ax0.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax0.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax0.tick_params(axis='both', labelsize='x-large')
        pl.colorbar(im0)
        ax0.set_title(title[k])
    #
    pl.tight_layout()
    if figname is not None: pl.savefig(figname)    

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
        pl.plot(x, y, ',', **dotkeys)

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
        print (yplot)
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
        pl.errorbar(xbinned, yplot, yerr=yerr,
                     xerr=xerr,
                     ls=ls, **keys)
    return xbinned, yplot, yerr

    
