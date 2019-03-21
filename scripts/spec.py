from __future__ import print_function, division

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.backends.backend_pdf import PdfPages
from mpl_toolkits.axes_grid1 import make_axes_locatable

from astropy.table import Table
import astropy.units as u
import astropy.coordinates as coord
from astropy.io import fits
from astropy.constants import c

import pickle
from Payne.fitting.genmod import GenMod
from dynesty import plotting as dyplot


def inspect_nights(itarget=27):
    """"""
    
    hdu_1 = fits.open('../data/spHect-gd1_catalog_5.2284-0100.fits')
    hdu_2 = fits.open('../data/spHect-gd1_catalog_5.2352-0100.fits')
    
    w_1 = hdu_1[0].data[itarget]
    f_1 = hdu_1[1].data[itarget]
    ivar_1 = hdu_1[2].data[itarget]
    
    w_2 = hdu_2[0].data[itarget]
    f_2 = hdu_2[1].data[itarget]
    ivar_2 = hdu_2[2].data[itarget]
    
    # combine
    w = w_1
    f_2interp = np.interp(w, w_2, f_2)
    
    f = f_1 + f_2interp
    c = ivar_1 + ivar_2
    e = c**-0.5
    
    deg = 5
    p = np.polynomial.chebyshev.chebfit(w, f, deg)
    p[0] += 0.15 * np.nanmedian(f)
    cont = np.polynomial.chebyshev.chebval(w, p)
    f_norm = f / cont
    e_norm = e / cont


    plt.close()
    fig, ax = plt.subplots(2,1,figsize=(12,10))
    
    plt.sca(ax[0])
    #plt.plot(w_1, f_1, 'r-', lw=0.5, alpha=0.5)
    #plt.plot(w_2, f_2, 'b-', lw=0.5, alpha=0.5)
    #plt.plot(w, f, 'k-', lw=0.5)
    plt.plot(w, f, '-', color='k', lw=0.5)
    plt.fill_between(w, f+e, f-e, color='k', alpha=0.2)
    plt.plot(w, cont, 'r-', lw=0.2)
    
    plt.sca(ax[1])
    plt.plot(w, f_norm, 'k-', lw=0.5)
    
    for i in range(2):
        plt.sca(ax[i])
        # check if Mgb at the right spot
        mgb = [5167.321, 5172.684, 5183.604]
        for m in mgb:
            plt.axvline(m, color='navy', ls='-', zorder=0, lw=4, alpha=0.1)
        
        plt.xlabel('Wavelength ($\AA$)')
        plt.ylabel('Flux (a.u.)')
    
    plt.tight_layout()

    hdu_1.close()
    hdu_2.close()

def get_target_index(target, hdr):
    """"""
    Nfiber = np.size(hdr)
    targets = []
    
    for i in range(Nfiber):
        targets += [hdr[i][1]]
    
    targets = np.array(targets)
    indices = np.arange(Nfiber, dtype='int')
    ind = np.where(targets==target)[0][0]

    return indices[ind]

def get_target_name(index, hdr):
    """"""
    Nfiber = np.size(hdr)
    targets = []
    
    for i in range(Nfiber):
        targets += [hdr[i][1]]
    
    targets = np.array(targets)

    return targets[index]

def payne_input(target=30842):
    """"""
    
    starget = '{:d}'.format(target)
    hdu_1 = fits.open('../data/spHect-gd1_catalog_5.2284-0100.fits')
    hdu_2 = fits.open('../data/spHect-gd1_catalog_5.2352-0100.fits')
    
    #hdu_2.info()
    itarget = get_target_index(starget, hdu_2[5].data)
    
    w_1 = hdu_1[0].data[itarget]
    f_1 = hdu_1[1].data[itarget]
    ivar_1 = hdu_1[2].data[itarget]
    
    w_2 = hdu_2[0].data[itarget]
    f_2 = hdu_2[1].data[itarget]
    ivar_2 = hdu_2[2].data[itarget]
    
    # combine
    w = w_1
    f_2interp = np.interp(w, w_2, f_2)
    
    f = f_1 + f_2interp
    c = ivar_1 + ivar_2
    e = c**-0.5
    
    # keep only pixels with a positive flux
    keep = (f>0) & (w>5150) & (w<5295) #& (w>5145) & (w<5307.39)
    f = f[keep]
    w = w[keep]
    e = e[keep]
    
    deg = 5
    p = np.polynomial.chebyshev.chebfit(w, f, deg)
    p[0] += 0.15 * np.nanmedian(f)
    cont = np.polynomial.chebyshev.chebval(w, p)
    f_norm = f / cont
    e_norm = e / cont

    hdu_1.close()
    hdu_2.close()
    
    plt.close()
    
    plt.plot(w, f_norm, 'k-', lw=0.5)
    
    t = Table.read('../data/gd1_input_catalog.fits')

    inputdict = {}
    inputdict['spec'] = {}
    inputdict['specANNpath'] = 'nn_spec.h5'
    inputdict['spec']['obs_wave'] = w
    inputdict['spec']['obs_flux'] = f_norm
    inputdict['spec']['obs_eflux'] = e_norm
    inputdict['spec']['normspec'] = False

    inputdict['phot'] = {}
    inputdict['photANNpath'] = 'SED/'
    inputdict['phot']['PS_g'] = [t['g'][target] - t['A_g'][target], t['g_error'][target]]
    inputdict['phot']['PS_r'] = [t['r'][target] - t['A_r'][target], t['r_error'][target]]
    inputdict['phot']['PS_i'] = [t['i'][target] - t['A_i'][target], t['i_error'][target]]
    inputdict['phot']['PS_z'] = [t['z'][target] - t['A_z'][target], t['z_error'][target]]
    inputdict['phot']['PS_y'] = [t['y'][target] - t['A_y'][target], t['y_error'][target]]

    inputdict['sampler'] = {}
    inputdict['sampler']['samplemethod'] = 'rwalk'
    inputdict['sampler']['npoints'] = 100
    inputdict['sampler']['samplertype'] = 'single'
    inputdict['sampler']['flushnum'] = 100
    #inputdict['sampler']['maxiter'] = 1000
    
    #npoints = samplerdict.get('npoints',200)
    #samplertype = samplerdict.get('samplertype','multi')
    #bootstrap = samplerdict.get('bootstrap',0)
    #update_interval = samplerdict.get('update_interval',0.6)
    #samplemethod = samplerdict.get('samplemethod','unif')
    #delta_logz_final = samplerdict.get('delta_logz_final',0.01)
    #flushnum = samplerdict.get('flushnum',10)
    #maxiter = samplerdict.get('maxiter',sys.maxint)

    inputdict['priordict'] = {}
    inputdict['priordict']['Teff']   = {'uniform': [3000.0,10000.0]}
    inputdict['priordict']['log(g)'] = {'uniform': [1.0,5.0]}
    inputdict['priordict']['[Fe/H]'] = {'uniform': [-2.,0.]}
    inputdict['priordict']['Dist'] = {'uniform': [1000.0,20000.0]}
    inputdict['priordict']['Inst_R'] = {'gaussian': [32000.,1000.]}
    inputdict['priordict']['Av'] = {'uniform': [0,0.2]}
    #inputdict['priordict']['blaze_coeff'] = ([
    #[1.14599359e+05, 0.5e-3],
    #[-6.92901664e+01, 1e-7],
    #[6.94323467e-03, 1e-7],
    #[-2.30598613e-07, 1e-12]
    #])
    #inputdict['priordict']['blaze_coeff'] = ([
    #[1.14599359e+05, 0.5],
    #[-6.92901664e+01, 0.1],
    #[6.94323467e-03, 0.1],
    #[-2.30598613e-07, 0.1],
    #[0, 0.001], [0, 0.001], [0, 0.001], [0, 0.001], [0, 0.001], [0, 0.001]
    #])
    #allparams: 'Teff', 'logg', 'FeH', 'aFe', 'Vrad', 'Vrot', 'Inst_R', 'logR', 'Dist', 'Av'
    
    inputdict['output'] = '../data/results/gd1_5_{}.dat'.format(target)
    inputdict['out'] = {}
    inputdict['out']['results'] = '../data/results/gd1_5_{}.pkl'.format(target)
    
    pickle.dump(inputdict, open('../data/inputs/gd1_5_{}.input'.format(target), 'wb'))

def modspectrum(target=30842):
    """"""
    
    inpars = pickle.load(open('../data/inputs/gd1_5_{}.input'.format(target), 'rb'))
    w = inpars['spec']['obs_wave']
    f = inpars['spec']['obs_flux']
    e = inpars['spec']['obs_eflux']
    
    plt.close()
    
    plt.figure(figsize=(15,5))
    plt.plot(w, f, '-', color='navy')
    plt.fill_between(w, f+e, f-e, color='navy', alpha=0.2)
    
    plt.xlabel('Wavelength ($\AA$)')
    plt.ylabel('Flux (a.u.)')
    
    results = pickle.load(open('../data/results/gd1_5_{}.pkl'.format(target), 'rb'))
    
    imax = np.argmax(results['logl'])
    specpars = results['samples'][imax]
    normspec_bool = False
    
    GM = GenMod()
    GM._initspecnn(nnpath='../data/nn_spec.h5')
    mw, mf = GM.genspec(specpars, outwave=w, normspec_bool=normspec_bool)
    
    plt.plot(mw-1.2, mf, 'r-')
    
    plt.tight_layout()
    #plt.savefig('../plots/spectrum_styx_6_{}.png'.format(ind), dpi=200)

def pdf(target=30842):
    """"""
    results = pickle.load(open('../data/results/gd1_5_{}.pkl'.format(target), 'rb'))
    labels = ['Teff', 'logg', 'FeH', 'aFe', 'Vrad', 'Vrot', 'Inst_R', 'logR', 'Dist', 'Av']
    #print(np.shape(results['samples']))

    # Plot the 2-D marginalized posteriors.
    plt.close()
    fig, ax = plt.subplots(10,10,figsize=(12,12))
    dyplot.cornerplot(results, show_titles=True, fig=(fig, ax), labels=labels, verbose=True)


## data taken during dark time ##

def show_spectra(itarget=27):
    """"""
    
    hdu = fits.open('/home/ana/data/hectochelle/ingest/gd1_catalog_5/spHect-gd1_catalog_5.3050-0100.fits')
    
    #lambda_heliocentric = lambda_orig/(1. + HELIO_RV/c)
    
    w = hdu[0].data[itarget] / (1 + hdu[0].header['HELIO_RV']/c.to(u.km/u.s).value)
    f = hdu[1].data[itarget]
    ivar = hdu[2].data[itarget]
    
    hdu_1 = fits.open('../data/spHect-gd1_catalog_5.2284-0100.fits')
    hdu_2 = fits.open('../data/spHect-gd1_catalog_5.2352-0100.fits')
    
    w_1 = hdu_1[0].data[itarget] / (1 + hdu_1[0].header['HELIO_RV']/c.to(u.km/u.s).value)
    f_1 = hdu_1[1].data[itarget]
    ivar_1 = hdu_1[2].data[itarget]
    
    w_2 = hdu_2[0].data[itarget] / (1 + hdu_2[0].header['HELIO_RV']/c.to(u.km/u.s).value)
    f_2 = hdu_2[1].data[itarget]
    ivar_2 = hdu_2[2].data[itarget]
    
    f_1interp = np.interp(w, w_1, f_1)
    f_2interp = np.interp(w, w_2, f_2)
    ivar_1interp = np.interp(w, w_1, ivar_1)
    ivar_2interp = np.interp(w, w_2, ivar_2)
    
    f_tot = f + f_1interp + f_2interp
    ivar_tot = ivar + ivar_1interp + ivar_2interp
    
    print(np.nanmedian(ivar_tot**0.5 * f_tot))
    
    plt.close()
    plt.figure(figsize=(12,6))
    
    plt.plot(w, f_tot, '-')
    
    mgb = [5167.321, 5172.684, 5183.604]
    for m in mgb:
        plt.axvline(m, color='navy', ls='-', zorder=0, lw=4, alpha=0.1)
    
    hdu.close()
    hdu_1.close()
    hdu_2.close()

def fiber_number():
    """Find fiber number for each star"""
    
    hdu = fits.open('/home/ana/data/hectochelle/ingest/gd1_catalog_5/spHect-gd1_catalog_5.3050-0100.fits')
    
    hdr = hdu[5].data
    Nfiber = np.size(hdr)
    targets = []
    indices = []
    
    for i in range(Nfiber):
        target = hdr[i][1]
        if (target!='UNUSED') & (target!='SKY') & ('REJECT' not in target):
            targets += [int(target)]
            indices += [i]
    
    indices = np.array(indices)
    targets = np.array(targets)
    
    print(indices[targets==30778])

def assemble_payne_inputs(field=5):
    """"""
    
    if field==5:
        hdu = fits.open('/home/ana/data/hectochelle/ingest/gd1_catalog_5/spHect-gd1_catalog_5.3050-0100.fits')
        hdu_1 = fits.open('../data/spHect-gd1_catalog_5.2284-0100.fits')
        hdu_2 = fits.open('../data/spHect-gd1_catalog_5.2352-0100.fits')
    elif field==4:
        hdu = fits.open('/home/ana/data/hectochelle/ingest/gd1_catalog_4/spHect-gd1_catalog_4.2961-0100.fits')
    elif field==3:
        hdu = fits.open('/home/ana/data/hectochelle/ingest/gd1_catalog_3/spHect-gd1_catalog_3.3153-0100.fits')

    t = Table.read('../data/gd1_input_catalog.fits')
    
    hdr = hdu[5].data
    Nfiber = np.size(hdr)
    targets = []
    indices = []
    
    for i in range(Nfiber):
        target = hdr[i][1]
        if (target!='UNUSED') & (target!='SKY') & ('REJECT' not in target):
            targets += [int(target)]
            indices += [i]
    
    indices = np.array(indices)
    targets = np.array(targets)
    t = t[targets]
    
    N = len(t)
    snr = np.zeros(N)
    for e, i in enumerate(indices):
        if field==5:
            w = hdu[0].data[i] / (1 + hdu[0].header['HELIO_RV']/c.to(u.km/u.s).value)
            f = hdu[1].data[i]
            ivar = hdu[2].data[i]
            
            w_1 = hdu_1[0].data[i] / (1 + hdu_1[0].header['HELIO_RV']/c.to(u.km/u.s).value)
            f_1 = hdu_1[1].data[i]
            ivar_1 = hdu_1[2].data[i]
            
            w_2 = hdu_2[0].data[i] / (1 + hdu_2[0].header['HELIO_RV']/c.to(u.km/u.s).value)
            f_2 = hdu_2[1].data[i]
            ivar_2 = hdu_2[2].data[i]
            
            # interpolate
            f_1interp = np.interp(w, w_1, f_1)
            f_2interp = np.interp(w, w_2, f_2)
            ivar_1interp = np.interp(w, w_1, ivar_1)
            ivar_2interp = np.interp(w, w_2, ivar_2)
            
            # coadd
            w_tot = w[:]
            f_tot = f + f_1interp + f_2interp
            ivar_tot = ivar + ivar_1interp + ivar_2interp
        else:
            w_tot = hdu[0].data[i] / (1 + hdu[0].header['HELIO_RV']/c.to(u.km/u.s).value)
            f_tot = hdu[1].data[i]
            ivar_tot = hdu[2].data[i]
        
        mask = (f_tot>0) & (ivar_tot>0)
        w_tot = w_tot[mask]
        f_tot = f_tot[mask]
        ivar_tot = ivar_tot[mask]
        e_tot = ivar_tot**-0.5
        
        snr[e] = np.nanmedian(ivar_tot**0.5 * f_tot)
        
        # create payne input
        inputdict = {}
        inputdict['spec'] = {}
        inputdict['specANNpath'] = 'nn_spec.h5'
        inputdict['spec']['obs_wave'] = w_tot
        inputdict['spec']['obs_flux'] = f_tot
        inputdict['spec']['obs_eflux'] = e_tot
        inputdict['spec']['normspec'] = True

        inputdict['phot'] = {}
        inputdict['photANNpath'] = 'SED/'
        inputdict['phot']['PS_g'] = [t['g'][e] - t['A_g'][e], t['g_error'][e]]
        inputdict['phot']['PS_r'] = [t['r'][e] - t['A_r'][e], t['r_error'][e]]
        inputdict['phot']['PS_i'] = [t['i'][e] - t['A_i'][e], t['i_error'][e]]
        inputdict['phot']['PS_z'] = [t['z'][e] - t['A_z'][e], t['z_error'][e]]
        inputdict['phot']['PS_y'] = [t['y'][e] - t['A_y'][e], t['y_error'][e]]

        inputdict['sampler'] = {}
        inputdict['sampler']['samplemethod'] = 'rwalk'
        inputdict['sampler']['npoints'] = 100
        inputdict['sampler']['samplertype'] = 'single'
        inputdict['sampler']['flushnum'] = 100

        inputdict['priordict'] = {}
        inputdict['priordict']['Teff']   = {'uniform': [3000.0,10000.0]}
        inputdict['priordict']['log(g)'] = {'uniform': [1.0,5.0]}
        inputdict['priordict']['[Fe/H]'] = {'uniform': [-2.,0.]}
        inputdict['priordict']['Dist'] = {'uniform': [1000.0,20000.0]}
        inputdict['priordict']['Inst_R'] = {'gaussian': [32000.,1000.]}
        inputdict['priordict']['Av'] = {'uniform': [0,0.2]}
        
        inputdict['misc'] = {}
        inputdict['misc']['fiber'] = i + 1
        
        inputdict['input_table'] = {}
        for k in t.colnames:
            inputdict['input_table'][k] = t[k]
        
        inputdict['output'] = '../data/results/gd1_{}_rank.{}_object.{}.dat'.format(field, t['priority'][e], t['name'][e])
        inputdict['out'] = {}
        inputdict['out']['results'] = '../data/results/gd1_{}_rank.{}_object.{}.pkl'.format(field, t['priority'][e], t['name'][e])
        
        pickle.dump(inputdict, open('../data/inputs/gd1_{}_rank.{}_object.{}.input'.format(field, t['priority'][e], t['name'][e]), 'wb'))
    
    for i in [5, 10, 15]:
        print(i, np.sum(snr<i), len(t))
    print(np.median(snr))
    
    plt.close()
    plt.figure(figsize=(6,5))
    
    cmap = mpl.cm.get_cmap('viridis', 5)
    im = plt.scatter(t['r'], snr, c=t['priority'], cmap=cmap)
    plt.axhline(5, ls='-', color='r', lw=3, alpha=0.2)
    
    plt.gca().set_yscale('log')
    plt.ylim(1, 1000)
    plt.xlabel('r')
    plt.ylabel('SNR')
    
    divider = make_axes_locatable(plt.gca())
    cax = divider.append_axes("right", size="3%", pad=0.1)
    plt.colorbar(im, cax=cax, ticks=np.arange(1,6,1))
    plt.ylabel('Rank')
    
    plt.tight_layout()
    plt.savefig('../plots/gd1_{}_snr.png'.format(field))

def check_dist(field=3, rank=4, star=23017):
    """Print fiber number stored in the pickle file
    Default should be 31"""
    
    pkl = pickle.load(open('../data/inputs/gd1_{}_rank.{}_object.{}.input'.format(field, rank, star), 'rb'))
    print(pkl['misc']['fiber'])

