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
import glob
import scipy.interpolate
from multiprocessing import Pool
import emcee

#from Payne.fitting.genmod import GenMod
#from dynesty import plotting as dyplot


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


# 2019 data

def plot_sky(n=5, ex=1, mem=False):
    """"""
    
    date = glob.glob('../data/tiles/gd1_{:1d}/d*'.format(n))[0].split('/')[-1]
    dr = '../data/tiles/gd1_{:d}/{:s}/reduced/v3.0/'.format(n, date)
    fname = 'specptg_gd1_{:d}_cluster_{:s}.ex{:1d}.fits'.format(n, date, ex)
    
    hdu = fits.open(dr+fname)
    
    w = hdu[0].data / (1 + hdu[0].header['HELIO_RV']/c.to(u.km/u.s).value)
    f = hdu[1].data
    sky = hdu[4].data
    
    # science fibers
    t = Table.read('../data/master_catalog.fits')
    t = t[t['field']==n]
    if mem:
        ind = (t['priority']<=3) & (t['delta_Vrad']>-20) & (t['delta_Vrad']<0)
        t = t[ind]
    
    plt.close()
    plt.figure(figsize=(12,6))
    for i in range(240):
        if i in t['fibID']:
            color = 'r'
            alpha = 0.02
            if mem:
                alpha = 0.6
        else:
            color = 'k'
            alpha = 0.07
            if mem:
                alpha = 0.02
        plt.plot(w[i], sky[i], '-', color=color, alpha=alpha)
    
    w_test = np.linspace(8000,8200,2)
    s_test = np.ones(2) * 20
    if mem:
        plt.plot(w_test, s_test, 'r-', label='Members')
        plt.plot(w_test, s_test, 'k-', label='Other fibers')
    else:
        plt.plot(w_test, s_test, 'r-', label='Target')
        plt.plot(w_test, s_test, 'k-', label='Sky')
    
    plt.legend(loc=3, ncol=2)
    plt.xlabel('Wavelength [$\AA$]')
    plt.ylabel('Flux')
    
    #ylims = plt.gca().get_ylim()
    #ylims[0] = 0
    plt.gca().set_ylim(bottom=-10)
    plt.xlim(5197, 5201)
    #plt.xlim(5197, 5198)
    
    plt.tight_layout()
    #plt.savefig('../plots/sky_gd1_{:d}_ex{:d}_mem{:d}.png'.format(n, ex, mem))
    
def plot_sky_ptgs(mem=False):
    """Plot sky spectra in all fibers for all gd-1 exposures"""
    
    for i in range(8):
        for j in range(3):
            plot_sky(n=i+1, ex=j+1, mem=mem)

def find_skyline(n=5, ex=1, itarget=0):
    """"""
    date = glob.glob('../data/tiles/gd1_{:1d}/d*'.format(n))[0].split('/')[-1]
    dr = '../data/tiles/gd1_{:d}/{:s}/reduced/v3.0/'.format(n, date)
    fname = 'specptg_gd1_{:d}_cluster_{:s}.ex{:1d}.fits'.format(n, date, ex)
    hdu = fits.open(dr+fname)
    
    w0 = hdu[0].data
    #w0 = hdu[0].data / (1 + hdu[0].header['HELIO_RV']/c.to(u.km/u.s).value)
    w = hdu[0].data / (1 + hdu[0].header['HELIO_RV']/c.to(u.km/u.s).value)
    f = hdu[1].data
    sky = hdu[4].data
    spec = f + sky
    spec = sky
    
    # sky line catalog
    tl = Table.read('../data/gident_580L.dat', format='ascii.fixed_width', header_start=1, data_start=3, data_end=91, delimiter=' ')
    
    # targets
    t = Table.read('../data/master_catalog.fits')
    t = t[t['field']==n]
    fibid = t['fibID'] - 1
    
    pp = PdfPages('../plots/sky_lines.f{:}.e{:}.pdf'.format(n, ex))
    
    for itarget in range(len(t)):
        plt.close()
        fig = plt.figure(figsize=(15,6))
        
        plt.plot(w0[fibid][itarget], spec[fibid][itarget], 'k-', lw=0.5)
        for i in range(len(tl)):
            plt.axvline(tl['CENTER'][i], lw=1, alpha=0.5, color='r')
        
        plt.xlim(5140, 5310)
        plt.xlabel('Wavelength [$\AA$]')
        plt.ylabel('Flux [count]')
        
        plt.tight_layout()
        pp.savefig(fig)
    
    pp.close()


def plot_tellurics():
    """"""
    t = Table.read('../data/hitran_top7.hdf5')
    tnames = Table.read('../docs/hitran_molecules.txt', format='ascii.no_header', delimiter='\t')
    
    w = (t['nu']**-1*u.cm).to(u.angstrom)
    wind = (w>5000*u.angstrom) & (w<9000*u.angstrom)
    t = t[wind]
    w = w[wind]
    mol_id = np.unique(t['molec_id'])

    plt.close()
    plt.figure(figsize=(15,6))
    
    for mid in mol_id:
        ind = t['molec_id']==mid
        plt.plot(w[ind], t['sw'][ind], 'o', ms=2, zorder=0, label=tnames['col3'][mid-1])
    plt.axvspan(5150, 5300, color='tab:red', alpha=0.3, zorder=1, label='RV31')
    
    plt.legend(markerscale=3, fontsize='small', ncol=6)
    plt.xlabel('Wavelength [$\AA$]')
    plt.ylabel('Intensity [cm$^{-1}$/(molec.cm$^{-2}$)]')
    
    plt.xlim(5000,9000)
    plt.ylim(1e-29, 1e-20)
    plt.gca().set_yscale('log')
    plt.tight_layout()
    plt.savefig('../plots/telluric_lines.png', dpi=200)


def get_date(n):
    """Get date when the field was observed (assumes no repeats)"""
    
    return glob.glob('/home/ana/data/hectochelle/tiles/gd1_{:d}/d*'.format(n))[0].split('/')[-1]

def ccd_temp(verbose=False):
    """Print CCD temperature during individual exposures"""
    
    fields = np.arange(1,9,1,dtype=int)
    dates = [get_date(n_) for n_ in fields]
    if verbose:
        print(fields)
        print(dates)
    
    for e, n in enumerate(fields):
        for i in range(3):
            hdu = fits.open('/home/ana/data/hectochelle/tiles/gd1_{0:d}/{1:s}/reduced/v3.0/specptg_gd1_{0:d}_cluster_{1:s}.ex{2:1d}.fits'.format(n, dates[e], i+1))
            print(n, i, hdu[0].header['CCDTEMP'], hdu[0].header['ROTANGLE'], hdu[0].header['POSANGLE'], hdu[0].header['HA'], hdu[0].header['PARANGLE'])

def hdr():
    """"""
    n = 6
    date = get_date(6)
    i = 0
    hdu = fits.open('/home/ana/data/hectochelle/tiles/gd1_{0:d}/{1:s}/reduced/v3.0/specptg_gd1_{0:d}_cluster_{1:s}.ex{2:1d}.fits'.format(n, date, i+1))
    print(hdu[0].header.keys)


##################
# field variations

def choose_sky(n=1, exp=1, graph=False):
    """Find sky fiber closest to the field center for a given exposure"""
    
    date = get_date(n)
    fname = '/home/ana/data/hectochelle/tiles/gd1_{0:d}/{1:s}/reduced/v3.0/specptg_gd1_{0:d}_cluster_{1:s}.ex{2:1d}.fits'.format(n, date, exp)
    hdu = fits.open(fname)

    # find sky fiber closest to the center
    ind_sky = hdu[5].data['OBJTYPE']=='SKY'
    rfocal = np.sqrt(hdu[5].data['XFOCAL']**2 + hdu[5].data['YFOCAL']**2)
    ind = np.argmin(rfocal[ind_sky])

    # extract wavelength and total flux
    w = hdu[0].data[ind_sky][ind]
    flux = hdu[1].data[ind_sky][ind]
    sky = hdu[4].data[ind_sky][ind]
    allflux = flux + sky
    
    if graph:
        plt.close()
        plt.figure(figsize=(12,6))
        plt.plot(w, allflux, 'k-', lw=0.5, alpha=1)
        plt.plot(w, sky, 'r-', lw=0.5, alpha=1)
        
        mgb = [5167.321, 5172.684, 5183.604]
        for m in mgb:
            plt.axvline(m, color='navy', ls='-', zorder=0, lw=4, alpha=0.1)
    
    return (w, allflux)

def run_sky_xcorr():
    """"""
    for i in range(8):
        for exp in range(3):
            sky_cross_correlation(n=i+1, exp=exp+1)

def sky_cross_correlation(n=1, exp=1, graph=False):
    """"""
    
    w_sky, f_sky = choose_sky(n=n, exp=exp)
    
    # data grid
    date = get_date(n)
    fname = '/home/ana/data/hectochelle/tiles/gd1_{0:d}/{1:s}/reduced/v3.0/specptg_gd1_{0:d}_cluster_{1:s}.ex{2:1d}.fits'.format(n, date, exp)
    hdu = fits.open(fname)
    
    w = hdu[0].data
    flux = hdu[1].data
    sky = hdu[4].data
    f = flux + sky
    
    # available spectra
    objtype = hdu[5].data['OBJTYPE']
    ind_spec = np.array([False if ('UNUSED' in x_) or ('REJECT' in x_) else True for x_ in objtype])
    #ind_spec = objtype[ind_spec]=='SKY'
    xfocal = hdu[5].data['XFOCAL'][ind_spec]
    yfocal = hdu[5].data['YFOCAL'][ind_spec]
    ind_sky = objtype[ind_spec]=='SKY'
    
    # rv list
    rv = np.arange(-300,300,3)*u.km/u.s
    rv = np.arange(-15,15,0.2)*u.km/u.s
    cc = np.zeros(np.size(rv))
    
    nspec = np.sum(ind_spec)
    delta_rv = np.zeros(nspec)
    
    # pick fiber
    for i in range(nspec):
        w_data = w[ind_spec][i]
        f_data = f[ind_spec][i]
        cc = np.zeros(np.size(rv))
        
        # clip to 5155 to 5295 AA
        ind_clip = (w_data>5155) & (w_data<5295)
        w_data = w_data[ind_clip]
        f_data = f_data[ind_clip]
        
        for e, rv_ in enumerate(rv):
            # shift wavelength
            w_sky_shift = w_sky / (1 + (rv_/c).decompose())
        
            # interpolate sky flux
            f_interp = scipy.interpolate.interp1d(w_sky_shift, f_sky)
            f_sky_interp = f_interp(w_data)
            f_sky_interp -= np.median(f_sky_interp)
            
            # cross-correlate
            cc[e] = np.dot(f_data, f_sky_interp) / np.sqrt(np.dot(f_sky_interp, f_sky_interp))
        
        ind_max = np.argmax(cc)
        delta_rv[i] = rv[ind_max].value
        
        np.savez('../data/cache/cc_perstar/field.{:02d}_exp.{:1d}_obj.{:03d}'.format(n, exp, i), rv=rv, cc=cc)

    np.savez('../data/cache/cc_field.{:02d}_exp.{:1d}'.format(n,exp), x=xfocal, y=yfocal, drv=delta_rv, isky=ind_sky)

    if graph:
        plt.close()
        plt.hist(delta_rv[~ind_sky], bins=np.arange(-300,300,3), histtype='step')
        plt.hist(delta_rv[ind_sky], bins=np.arange(-300,300,3), histtype='step')

def sky_dvr_histogram(zoom=False):
    """"""
    
    if zoom:
        vrbins = np.arange(-20,20,3)
        vrbins = np.arange(-15,15,0.5)
    else:
        vrbins = np.arange(-300,300,3)
    
    plt.close()
    fig, ax = plt.subplots(2,4,figsize=(14,7), sharex=True, sharey=True)
    
    for i in range(8):
        irow = int(i/4)
        icol = i%4
        plt.sca(ax[irow][icol])
        
        for j in range(3):
            din = np.load('../data/cache/cc_field.{:02d}_exp.{:1d}.npz'.format(i+1, j+1))
            #plt.hist(din['drv'][din['isky']], bins=vrbins, histtype='step', color=mpl.cm.Blues_r(j/4), lw=2)
            plt.hist(din['drv'], bins=vrbins, histtype='step', color=mpl.cm.Blues_r(j/4), lw=2)
        
        if irow==1:
            plt.xlabel('$\Delta$ $V_r$ [km s$^{-1}$]')
        if icol==0:
            plt.ylabel('N')
    
    plt.tight_layout(h_pad=0, w_pad=0)
    plt.savefig('../plots/sky_dvr_zoom.{:d}.png'.format(zoom), dpi=200)



    #plt.scatter(xfocal[:30], yfocal[:30], c=delta_rv[:30], vmin=-50, vmax=50, cmap='RdBu', ec='k')

def sky_xcorr_diag(n=1, i=0, exp=1):
    """"""
    f = np.load('../data/cache/cc_perstar/field.{:02d}_exp.{:d}_obj.{:03d}.npz'.format(n, exp, i))
    
    plt.close()
    plt.plot(f['rv'], f['cc'], 'ko')
    
    plt.xlim(-20,20)
    
    plt.tight_layout()


def normal(x, mu, std):
    return (2*np.pi*std**2)**-0.5 * np.exp(-0.5 * (x-mu)**2 / std**2)

def lnnormal(x, mu, std):
    return -0.5 * (x-mu)**2 / std**2 - 0.5*np.log(2*np.pi) - np.log(std)

def lnprior(p, w0):
    abg, a, mu, std = p
    if (np.abs(mu - w0)>1) | (std>0.6) | (std<0) | (abg<0) | (a<0):
        return -np.inf
    else:
        return 0

def gen_model(p, x):
    abg, a, mu, std = p
    ym = a * normal(x, mu, std) + abg
    
    return ym
    
def lnlike(p, x, y, yerr):
    ym = gen_model(p, x)
    lnl = lnnormal(y, ym, yerr)
    
    return lnl

def lnprob(p, x, y, yerr, w0):
    lp = lnprior(p, w0)
    if not np.all(np.isfinite(lp)):
        return -np.inf
    
    ll = lnlike(p, x, y, yerr)
    if not np.all(np.isfinite(ll)):
        return -np.inf
    
    return ll.sum() + lp

def fit_line(w_, f_, v_, sky, n, exp, i, iline, nwalkers, nsteps, pool, outdir):
    """"""
    # initialize walkers
    p0s = np.array([np.median(f_), 5, sky[iline], 0.05])
    p0 = emcee.utils.sample_ball(p0s, [1e-3, 1e-3, 1e-3, 1e-3], nwalkers)
    p0[:,:2] = np.abs(p0[:,:2])

    sampler = emcee.EnsembleSampler(nwalkers, p0.shape[1], pool=pool,
                                    log_prob_fn=lnprob, args=(w_, f_, v_, sky[iline]))
    _ = sampler.run_mcmc(p0, nsteps)

    # save sampler
    out_dict = {'lnprobability': sampler.lnprobability, 'chain': sampler.chain, 'dim': sampler.ndim}
    fname = 'sky_field.{:d}.{:d}_spec.{:03d}_line.{:d}'.format(n, exp, i, iline)
    pickle.dump(out_dict, open('{:s}/{:s}.pkl'.format(outdir, fname), 'wb'))


def plot_diagnostics(w_, f_, v_, sky, n, exp, i, iline):
    """"""
    fname = 'sky_field.{:d}.{:d}_spec.{:03d}_line.{:d}'.format(n, exp, i, iline)
    pkl = pickle.load(open('../data/cache/chains/{:s}.pkl'.format(fname), 'rb'))
    
    # plot chains
    names = [r'$\alpha_{bg}$', r'$\alpha$', r'$\mu$', r'$\sigma$']
    plt.close()
    fig, ax = plt.subplots(pkl['dim'], figsize=(10,10), sharex=True)

    for k in range(pkl['dim']):
        for walker in pkl['chain'][..., k]:
            ax[k].plot(walker, marker='', drawstyle='steps-mid', alpha=0.2)
        ax[k].set_ylabel(names[k])

    plt.sca(ax[pkl['dim']-1])
    plt.xlabel('Step')

    plt.tight_layout(h_pad=0)
    plt.savefig('../plots/diag/chain_{:s}.png'.format(fname), dpi=80)
    
    # plot best-fit
    med = np.median(pkl['chain'][:,1024:,:].reshape(-1,pkl['dim']), axis=0)
    dlambda = med[2] - sky[iline]
    dvr = (c*sky[iline]*(1/sky[iline] - 1/med[2])).to(u.km/u.s)

    xm = np.linspace(np.min(w_), np.max(w_), 200)
    ym = gen_model(med, xm)
    ym_fid = gen_model(med - dlambda, xm)

    plt.close()
    plt.figure(figsize=(10,8))
    plt.errorbar(w_, f_, yerr=v_, color='k', fmt='o', label='Sky spectrum')
    plt.plot(xm, ym, '-', color='tab:orange', label='Best-fit')
    plt.plot(xm, ym_fid, '--', color='tab:orange', label='Fiducial')
    plt.axvline(sky[iline], ls='-', lw=5, color='tab:blue', alpha=0.5, label='Sky line')

    plt.legend(loc=2, fontsize='small', handlelength=1, frameon=False)
    plt.xlabel('Wavelength [$\AA$]')
    plt.ylabel('Flux [a.u.]')
    plt.text(0.95, 0.95, '$\Delta$ $V_r$ = {:.2f}'.format(dvr),
             transform=plt.gca().transAxes, ha='right', va='top', fontsize='small')

    plt.tight_layout()
    plt.savefig('../plots/diag/fit_{:s}.png'.format(fname), dpi=80)
    
def sky_diag_field(n=5, exp=3, coadd=False, i0=0):
    """"""
    # read in hdu data
    date = get_date(n)
    if coadd:
        exp_label = 'sum'
    else:
        exp_label = 'ex{:d}'.format(exp)

    fname = '../data/tiles/gd1_{0:d}/{1:s}/reduced/v3.0/specptg_gd1_{0:d}_cluster_{1:s}.{2:s}.fits'.format(n, date, exp_label)
    hdu = fits.open(fname)
    
    # available spectra
    objtype = hdu[5].data['OBJTYPE']
    ind_spec = np.array([False if ('UNUSED' in x_) or ('REJECT' in x_) else True for x_ in objtype])
    
    # cutout sky lines, from: https://www.eso.org/observing/dfo/quality/UVES/uvessky/sky_5800L_2.html
    sky = [5197.928223, 5202.979004, 5224.145020, 5238.751953]
    dw = 1
    
    # sampling setup
    #nwalkers = 64
    #nsteps = 4096
    #nsteps = 2048
    #np.random.seed(94629)
    outdir = '../data/cache/chains'
    
    # sample
    ids = np.arange(np.size(ind_spec), dtype=int)
    for i in ids[ind_spec][i0:]:
        w = np.array(hdu[0].data[i], dtype=float)
        fsky = np.array(hdu[4].data[i], dtype=float)
        vsky = np.sqrt(np.abs(fsky))
        
        for iline, wline in enumerate(sky):
            # select input data
            ind = np.abs(w - wline)<dw
            w_ = np.array(w[ind], dtype=float)
            f_ = np.array(fsky[ind], dtype=float)
            v_ = np.array(vsky[ind], dtype=float)
            
            plot_diagnostics(w_, f_, v_, sky, n, exp, i, iline)


def sky_fit_field(n=5, exp=3, coadd=False, i0=0, nth=3):
    """"""
    # read in hdu data
    date = get_date(n)
    if coadd:
        exp_label = 'sum'
    else:
        exp_label = 'ex{:d}'.format(exp)

    fname = '../data/tiles/gd1_{0:d}/{1:s}/reduced/v3.0/specptg_gd1_{0:d}_cluster_{1:s}.{2:s}.fits'.format(n, date, exp_label)
    hdu = fits.open(fname)
    
    # available spectra
    objtype = hdu[5].data['OBJTYPE']
    ind_spec = np.array([False if ('UNUSED' in x_) or ('REJECT' in x_) else True for x_ in objtype])
    
    # cutout sky lines, from: https://www.eso.org/observing/dfo/quality/UVES/uvessky/sky_5800L_2.html
    sky = [5197.928223, 5202.979004, 5224.145020, 5238.751953]
    dw = 1
    
    # sampling setup
    nwalkers = 64
    nsteps = 4096
    nsteps = 2048
    np.random.seed(94629)
    outdir = '../data/cache/chains'
    
    # sample
    ids = np.arange(np.size(ind_spec), dtype=int)
    for i in ids[ind_spec][i0:]:
        w = np.array(hdu[0].data[i], dtype=float)
        fsky = np.array(hdu[4].data[i], dtype=float)
        vsky = np.sqrt(np.abs(fsky))
        
        for iline, wline in enumerate(sky):
            # select input data
            ind = np.abs(w - wline)<dw
            w_ = np.array(w[ind], dtype=float)
            f_ = np.array(fsky[ind], dtype=float)
            v_ = np.array(vsky[ind], dtype=float)
            pool = Pool(processes=nth)
            
            fit_line(w_, f_, v_, sky, n, exp, i, iline, nwalkers, nsteps, pool, outdir)
            
            pool.close()
            pool.terminate()
    
def sky_extract_offsets(n=5, exp=3, coadd=False):
    """"""
    # read in hdu data
    date = get_date(n)
    if coadd:
        exp_label = 'sum'
    else:
        exp_label = 'ex{:d}'.format(exp)

    fname = '/home/ana/data/hectochelle/tiles/gd1_{0:d}/{1:s}/reduced/v3.0/specptg_gd1_{0:d}_cluster_{1:s}.{2:s}.fits'.format(n, date, exp_label)
    hdu = fits.open(fname)
    
    # available spectra
    objtype = hdu[5].data['OBJTYPE']
    ind_spec = np.array([False if ('UNUSED' in x_) or ('REJECT' in x_) else True for x_ in objtype])
    
    # sampling setup
    nwalkers = 64
    nsteps = 2048
    outdir = '../data/cache/chains'
    ids = np.arange(np.size(ind_spec), dtype=int)
    sky = [5197.928223, 5202.979004, 5224.145020, 5238.751953]
    
    # output table
    t = Table(np.zeros((np.sum(ind_spec),7), dtype=float), names=('fib', 'x', 'y', 'dvr0', 'dvr1', 'dvr2', 'dvr3'))
    t['fib'].dtype = int
    
    for ii, i in enumerate(ids[ind_spec]):
        xfocal = hdu[5].data['XFOCAL'][i]
        yfocal = hdu[5].data['YFOCAL'][i]
        
        t['fib'][ii] = i
        t['x'][ii] = xfocal
        t['y'][ii] = yfocal
        
        for iline in range(4):
            #iline = 1
            fname = 'sky_field.{:d}.{:d}_spec.{:03d}_line.{:d}'.format(n, exp, i, iline)
            sampler = pickle.load(open('{:s}/{:s}.pkl'.format(outdir, fname), 'rb'))
            
            med = np.median(sampler['chain'][:,1024:,:].reshape(-1,sampler['dim']), axis=0)
            t['dvr{:1d}'.format(iline)][ii] = (c*sky[iline]*(1/sky[iline] - 1/med[2])).to(u.km/u.s).value
    
    t.write('../data/sky_offsets_field.{:d}.{:d}.fits'.format(n, exp), overwrite=True)
    
    plt.close()
    fig, ax = plt.subplots(1,4,figsize=(16,4), sharex=True, sharey=True)
    
    for i in range(4):
        plt.sca(ax[i])
        plt.scatter(t['x'], t['y'], c=t['dvr{:1d}'.format(i)], cmap='RdBu_r')
    
    plt.tight_layout()

def sky_offsets_plot(n=5, exp=3):
    """"""
    t = Table.read('../data/sky_offsets_field.{:d}.{:d}.fits'.format(n, exp))
    #t.pprint()
    
    plt.close()
    fig, ax = plt.subplots(1,4,figsize=(16,5.5), sharex=True, sharey=True)
    
    for i in range(4):
        plt.sca(ax[i])
        ind = t['fib']<120
        med1 = np.median(t['dvr{:1d}'.format(i)][ind])
        med2 = np.median(t['dvr{:1d}'.format(i)][~ind])
        med = 0.5 * (med1 + med2)
        print(i, med1, med2, np.abs(med1 - med2))
        im = plt.scatter(t['x'][ind], t['y'][ind], c=t['dvr{:1d}'.format(i)][ind], cmap='RdBu_r', vmin=med-0.5, vmax=med+0.5, marker='o')
        im = plt.scatter(t['x'][~ind], t['y'][~ind], c=t['dvr{:1d}'.format(i)][~ind], cmap='RdBu_r', vmin=med-0.5, vmax=med+0.5, marker='s')
        
        plt.xlabel('X')
        plt.gca().set_aspect('equal')
        
        divider = make_axes_locatable(plt.gca())
        cax = divider.append_axes('top', size='3%', pad=0.1)
        plt.colorbar(im, cax=cax, orientation='horizontal') #, ticks=np.arange(0,51,25))
        plt.xlabel('$\Delta$ $V_r$ [km s$^{-1}$]')
        plt.gca().xaxis.set_label_position('top')
        plt.gca().xaxis.tick_top()

    
    plt.sca(ax[0])
    plt.ylabel('Y')
    
    plt.tight_layout()
    plt.savefig('../plots/sky_offsets_field.{:1d}_exp.{:1d}.png'.format(n, exp))

def sky_offset_summary():
    """"""
    
    #alphas = [0.75, 0.5, 0.25]
    colors = ['tab:blue', 'tab:red']
    
    pp = PdfPages('../plots/sky_offset_summary.pdf')
    
    for n in range(1,9):
        plt.close()
        fig, ax = plt.subplots(3,1,figsize=(8,10), sharex=True, sharey=True)
        
        for e in range(3):
            t = Table.read('../data/sky_offsets_field.{:d}.{:d}.fits'.format(n, e+1))
            ind_ = t['fib']<120
            
            w0 = np.array([5197.928223, 5202.979004, 5224.145020, 5238.751953])
            dv_med = np.zeros((2, 4))
            dv_std = np.zeros((2, 4))
            
            for i in range(4):
                for j, ind in enumerate([ind_, ~ind_]):
                    dv_med[j][i] = np.median(t['dvr{:1d}'.format(i)][ind])
                    dv_std[j][i] = np.std(t['dvr{:1d}'.format(i)][ind])
            
            plt.sca(ax[e])
            for i in range(2):
                plt.errorbar(w0, dv_med[i], yerr=dv_std[i], color=colors[i], fmt='o', label='')
                plt.fill_between(w0, dv_med[i] - dv_std[i], dv_med[i] + dv_std[i], alpha=0.3, color=colors[i], label='Chip {:1d}'.format(i+1))
            
            plt.ylabel('$\Delta$ $V_r$ [km s$^{-1}$]')
            plt.axhline(0, color='k', lw=2, alpha=0.5)
            plt.text(0.05,0.85, 'Exposure {:1d}'.format(e+1), transform=plt.gca().transAxes, fontsize='small')
            
            plt.minorticks_on()
            plt.gca().yaxis.set_minor_locator(mpl.ticker.MultipleLocator(1))
        
        plt.xlabel('Wavelength [$\AA$]')
        plt.legend(fontsize='small')
        
        plt.sca(ax[0])
        plt.text(0.95, 0.85, 'Field {:1d}'.format(n), transform=plt.gca().transAxes, fontsize='medium', ha='right')
        
        plt.tight_layout(h_pad=0)
        pp.savefig(fig)
    
    pp.close()


def plot_all_sky(n=5, exp=3, coadd=False):
    """"""
    
    # read in hdu data
    date = get_date(n)
    if coadd:
        exp_label = 'sum'
    else:
        exp_label = 'ex{:d}'.format(exp)

    fname = '../data/tiles/gd1_{0:d}/{1:s}/reduced/v3.0/specptg_gd1_{0:d}_cluster_{1:s}.{2:s}.fits'.format(n, date, exp_label)
    hdu = fits.open(fname)
    
    # available spectra
    objtype = hdu[5].data['OBJTYPE']
    ind_spec = np.array([False if ('UNUSED' in x_) or ('REJECT' in x_) else True for x_ in objtype])
    ids = np.arange(np.size(ind_spec), dtype=int)
    
    plt.close()
    plt.figure(figsize=(15,8))
    
    for ii, i in enumerate(ids[ind_spec]):
        w = np.array(hdu[0].data[i], dtype=float)
        fsky = np.array(hdu[4].data[i], dtype=float)
        if i<120:
            plt.plot(w, fsky + ii*5, 'k-', lw=0.5, alpha=0.8)
        else:
            plt.plot(w, fsky + ii*5, 'b-', lw=0.5, alpha=0.8)
    
    plt.xlim(5150, 5250)
    plt.ylim(0,1150)
    plt.xlabel('Wavelength [$\AA$]')
    plt.ylabel('Sky flux [a.u.]')
    plt.title('Field {:1d} | Exposure {:1d}'.format(n, exp), fontsize='medium')
    plt.tight_layout()
    plt.savefig('../plots/diag/skyflux_field.{:1d}_exp.{:1d}.png'.format(n, exp), dpi=100)


#######
# Paper
from vel import get_members

def print_exptimes():
    """"""
    fields = np.arange(1,9,dtype=int)
    dates = [get_date(n_) for n_ in fields]
    
    exptimes = np.zeros(8)
    for e, n in enumerate(fields):
        for i in range(3):
            fname = '/home/ana/data/hectochelle/tiles/gd1_{0:d}/{1:s}/reduced/v3.0/specptg_gd1_{0:d}_cluster_{1:s}.ex{2:1d}.fits'.format(n, dates[e], i+1)
            hdu = fits.open(fname)
            exptimes[e] += hdu[0].header['exptime']
    
    print((exptimes*u.s).to(u.hr))

def print_snr():
    """"""
    t = Table.read('../data/master_catalog.fits')
    p = np.percentile(t['SNR'], [5,10,50])
    
    print(' '.join('{:.1f}'.format(p_) for p_ in p))
    
    plt.close()
    plt.figure(figsize=(12,9))
    
    plt.plot(t['g'], t['SNR'], 'ko')
    plt.axhline(2, color='r', lw=2)

def print_params():
    """"""
    t = Table.read('../data/master_catalog.fits')
    keys = t.colnames
    ind0 = keys.index('Teff')
    ind1 = keys.index('Para')
    keys = keys[ind0:ind1+1]
    keys = [x_ for x_ in keys if ('lerr' not in x_) and ('uerr' not in x_) and ('std' not in x_)]
    print(keys)

def print_precision():
    """"""
    t = Table.read('../data/master_catalog.fits')
    
    for k in ['Vrad', 'FeH', 'aFe']:
        m = t['std_{:s}'.format(k)]
        print(k, np.percentile(m, [10,50,90]))

def exhibit_spectra(verbose=False):
    """Plot spectra at different SNR"""
    
    t = Table.read('../data/master_catalog.fits')
    g = Table(fits.getdata('/home/ana/projects/legacy/GD1-DR2/output/gd1_members.fits'))
    
    # find indices of percentile spectra among the members
    mem = get_members(t)
    tmem = t[mem]
    p = np.percentile(tmem['SNR'], [90,50,10])
    if verbose: print(' '.join('{:.1f}'.format(p_) for p_ in p))

    pind = np.array([np.argmin(np.abs(tmem['SNR'] - p_)) for p_ in p])
    if verbose:
        print(pind, tmem['SNR'][pind])
        print(tmem['std_Vrad'][pind])
        print(tmem['std_FeH'][pind])
        print(tmem['std_aFe'][pind])
    
    plt.close()
    fig = plt.figure(figsize=(11,9))
    gs1 = mpl.gridspec.GridSpec(1,1)
    gs1.update(left=0.08, right=0.975, top=0.98, bottom=0.75)

    gs2 = mpl.gridspec.GridSpec(3,1)
    gs2.update(left=0.08, right=0.975, top=0.65, bottom=0.08, hspace=0.1)

    ax0 = fig.add_subplot(gs1[0])
    ax1 = fig.add_subplot(gs2[0])
    ax2 = fig.add_subplot(gs2[1], sharex=ax1)
    ax3 = fig.add_subplot(gs2[2], sharex=ax1)
    plt.setp(ax1.get_xticklabels(), visible=False)
    plt.setp(ax2.get_xticklabels(), visible=False)
    ax = [ax0, ax1, ax2, ax3]
    
    # positions on the sky
    plt.sca(ax[0])
    plt.scatter(g['phi1'], g['phi2'], s=g['pmem']*2, c=g['pmem'], cmap='binary', vmin=0.5, vmax=1.1, label='Likely GD-1 members', zorder=0)
    plt.plot(t['phi1'], t['phi2'], 'o', color='darkorange', ms=2, mec='none', label='Hectochelle targets', zorder=1)
    
    for i in range(8):
        t_ = t[t['field']==i+1]
        phi_off = 0.3
        plt.text(np.median(t_['phi1'])+phi_off, np.median(t_['phi2'])+phi_off, '{:d}'.format(i+1), fontsize='small')
    
    plt.xlim(-53,-27)
    plt.ylim(-3,3)
    plt.gca().set_aspect('equal', adjustable='datalim')
    plt.xlabel('$\phi_1$ [deg]')
    plt.ylabel('$\phi_2$ [deg]')
    
    #customize the order of legend entries
    handles, labels = plt.gca().get_legend_handles_labels()
    order = [1,0]
    handles = [handles[x] for x in order]
    labels = [labels[x] for x in order]
    plt.legend(handles, labels, loc=3, scatterpoints=1, frameon=False, fontsize='small', handlelength=0.5, markerscale=3)

    for i in range(3):
        plt.sca(ax[i+1])
        plt.ylabel('Flux')
        
        # read in spectrum
        n = tmem['field'][pind[i]]
        date = get_date(n)
        fname = '/home/ana/data/hectochelle/tiles/gd1_{0:d}/{1:s}/reduced/v3.0/specptg_gd1_{0:d}_cluster_{1:s}.sum.fits'.format(n, date)
        hdu = fits.open(fname)
        
        fib = tmem['fibID'][pind[i]] - 1
        w = hdu[0].data[fib] / (1 + (hdu[0].header['HELIO_RV'] + tmem['Vrad'][pind[i]])/c.to(u.km/u.s).value)
        flux = hdu[1].data[fib]
        
        plt.plot(w, flux, 'k-', lw=0.5, label='Observed')
        
        mgb = [5167.321, 5172.684, 5183.604]
        labels = ['Mg b', '', '']
        for e, m in enumerate(mgb):
            plt.axvline(m, color='navy', ls='-', zorder=0, lw=4, alpha=0.1, label=labels[e])
        
        plt.text(0.025, 0.9, 'S/N = {:.1f}'.format(tmem['SNR'][pind[i]]), fontsize='small', transform=plt.gca().transAxes, va='top')
    
    plt.legend(loc=4, fontsize='small')
    plt.xlim(5150,5300)
    plt.xlabel('Wavelength [$\AA$]')
    
    plt.savefig('../paper/spectra.pdf')


def mem_fnames():
    """"""
    
    t = Table.read('../data/master_catalog.fits')
    mem = get_members(t)
    t = t[mem]
    
    print(t.colnames, len(t))
    #print(t['starname'], t['fibID'], t['field'])

    spurfields = [2,4,5,6]
    dates = [get_date(n_) for n_ in spurfields]
    
    plt.close()
    #plt.figure(figsize=(16,8))
    fig, ax = plt.subplots(4,1,figsize=(16,16), sharex=True)
    
    for e, n in enumerate(spurfields[:]):
        ind = (t['field']==n) #& (t['SNR']>3)
        t_ = t[ind]
        #print(n, np.array(t_['fibID']),  np.median(t_['phi1']), np.median(t_['phi2']))
        #print(np.array(t_['delta_Vrad']))
        fname = '/home/ana/data/hectochelle/tiles/gd1_{0:d}/{1:s}/reduced/v3.0/specptg_gd1_{0:d}_cluster_{1:s}.sum.fits'.format(n, dates[e])
        #print(fname)
        hdu = fits.open(fname)
        hdu.info()
        #print(hdu[5].header.keys)
        #print(hdu[5].data['FIBERID'])
        
        isort = np.argsort(t_['SNR'])
        
        plt.sca(ax[e])
        for ef, fib in enumerate(np.array(t_['fibID'])):
            #w = hdu[0].data[:,fib]
            fib -= 1
            w = hdu[0].data[fib] / (1 + (hdu[0].header['HELIO_RV'] + t_['Vrad'][ef])/c.to(u.km/u.s).value)
            print(fib-1, hdu[5].data['OBJTYPE'][fib], hdu[5].data['FIBERID'][fib], hdu[0].header['HELIO_RV'], t_['Vrad'][ef],  t_['delta_Vrad'][ef])
            #w = w[fib]
            flux = hdu[1].data[fib]
            sky = hdu[4].data[fib]
            allflux = flux + sky
            
            plt.plot(w, allflux, '-', color=mpl.cm.gray(ef/(len(t_)+1)), lw=0.5)
            
        #plt.errorbar(t_['phi1'], t_['Vrad'], yerr=t_['std_Vrad'], fmt='o')
    
    #plt.ylim(-100,-50)
    #plt.xlim(5200,5225)
    plt.tight_layout(h_pad=0)
    plt.savefig('../plots/spur_spectra.png', dpi=200)

def spur_member_spectra(spur=True, exp=1, lobs=True, coadd=False):
    """"""
    t = Table.read('../data/master_catalog.fits')
    mem = get_members(t)
    t = t[mem]
    
    if spur:
        spurfields = [2,4,5,6]
        field_label = 'spur'
    else:
        spurfields = [1,3,7,8]
        field_label = 'stream'
    #spurfields = np.arange(8,dtype=int) + 1
    dates = [get_date(n_) for n_ in spurfields]
    
    for e, n in enumerate(spurfields[:]):
    #for e in [2,]:
        n = spurfields[e]
        plt.close()
        fig, ax = plt.subplots(2,1,figsize=(16,12))
    
        ind = (t['field']==n) & (t['SNR']>3)
        t_ = t[ind]
        #print(np.array(t_['Vrot']))
        if coadd:
            exp_label = 'sum'
        else:
            exp_label = 'ex{:d}'.format(exp)
        
        fname = '/home/ana/data/hectochelle/tiles/gd1_{0:d}/{1:s}/reduced/v3.0/specptg_gd1_{0:d}_cluster_{1:s}.{2:s}.fits'.format(n, dates[e], exp_label)
        hdu = fits.open(fname)
        
        for ef, fib in enumerate(np.array(t_['fibID'])):
            fib -= 1
            if lobs:
                w = hdu[0].data[fib]
            else:
                w = hdu[0].data[fib] / (1 + (hdu[0].header['HELIO_RV'] + t_['Vrad'][ef])/c.to(u.km/u.s).value)
                #w = hdu[0].data[fib] / (1 + (t_['Vrad'][ef])/c.to(u.km/u.s).value)
            flux = hdu[1].data[fib]
            sky = hdu[4].data[fib]
            allflux = flux + sky
            
            #print(hdu[0].header['HELIO_RV'] + t_['Vrad'][ef])
            
            for ea in range(2):
                plt.sca(ax[ea])
                plt.plot(w, allflux + ef*50, '-', color=mpl.cm.gray(ef/(len(t_)+1)), lw=1)
                plt.plot(w, sky, ':', color=mpl.cm.gray(ef/(len(t_)+1)), lw=1)
        
        plt.sca(ax[0])
        mgb = [5167.321, 5172.684, 5183.604]
        for m in mgb:
            plt.axvline(m, color='navy', ls='-', zorder=0, lw=4, alpha=0.1)
        
        dv_array = np.array([1,5,10,20])*u.km/u.s
        
        for i, dv in enumerate(dv_array):
            y_ = np.ones(2) * (-20*i - 30)
            w_ = np.array([mgb[0]/(1 - (0.5*dv/c).decompose().value), mgb[0]/(1 + (0.5*dv/c).decompose().value)])
        
            plt.plot(w_, y_, 'k-')
        
        plt.text(0.05, 0.9, 'Mgb zoom', transform=plt.gca().transAxes)
        plt.ylabel('Flux')
        plt.xlim(5160,5190)
        
        plt.sca(ax[1])
        sky = [5197.928223, 5200.285645, 5202.979004]
        for s in sky:
            plt.axvline(s, color='navy', ls='-', zorder=0, lw=4, alpha=0.1)

        for i, dv in enumerate(dv_array):
            y_ = np.ones(2) * (-20*i - 30)
            w_ = np.array([sky[0]/(1 - (0.5*dv/c).decompose().value), sky[0]/(1 + (0.5*dv/c).decompose().value)])
        
            plt.plot(w_, y_, 'k-')

        plt.text(0.05, 0.9, 'Sky zoom', transform=plt.gca().transAxes)
        lobs_label = ['RV shifted', 'Observed']
        plt.xlabel('{:s} wavelength [$\AA$]'.format(lobs_label[lobs]))
        plt.ylabel('Flux')
        plt.xlim(5195,5205)
        
        plt.tight_layout()
        plt.savefig('../plots/{:s}_{:d}_spectra_{:s}_lobs{:d}.png'.format(field_label, n, exp_label, lobs), dpi=150)

def spur_member_spectra_exposures():
    """"""
    t = Table.read('../data/master_catalog.fits')
    mem = get_members(t)
    t = t[mem]
    
    spurfields = [2,4,5,6]
    spurfields = [1,]
    dates = [get_date(n_) for n_ in spurfields]
    
    for e, n in enumerate(spurfields[:]):
        ind = (t['field']==n) #& (t['SNR']>3)
        t_ = t[ind]
        
        plt.close()
        fig, ax = plt.subplots(len(t_), 1, figsize=(10,10), sharex=True)
        
        for ex in ['ex1', 'ex2', 'ex3', 'sum']:
            fname = '/home/ana/data/hectochelle/tiles/gd1_{0:d}/{1:s}/reduced/v3.0/specptg_gd1_{0:d}_cluster_{1:s}.{2:s}.fits'.format(n, dates[e], ex)
            hdu = fits.open(fname)
            
            for ef, fib in enumerate(np.array(t_['fibID'])):
                plt.sca(ax[ef])
                fib -= 1
                w = hdu[0].data[fib] / (1 + (hdu[0].header['HELIO_RV'] + t['Vrad'][ef])/c.to(u.km/u.s).value)
                flux = hdu[1].data[fib]
                sky = hdu[4].data[fib]
                allflux = flux + sky
                
                if ex=='sum':
                    plt.ylabel('Flux')
                    lw = 2
                    color = 'k'
                else:
                    lw = 0.5
                    color = '0.5'
                    plt.plot(w, sky, '-', color='tab:blue', lw=0.5)
                
                plt.plot(w, allflux, '-', color=color, lw=0.5)

        plt.xlabel('Wavelength [$\AA$]')
        
        plt.tight_layout(h_pad=0)
        plt.savefig('../plots/spur_{:d}_spectra_exposures.png'.format(n), dpi=150)

def dvr_focal(selection='spur'):
    """"""
    t = Table.read('../data/master_catalog.fits')
    mem = get_members(t)
    t = t[mem]
    
    #print(t.colnames, len(t))
    #print(t['starname'], t['fibID'], t['field'])

    if selection=='spur':
        fields = [2,4,5,6]
    elif selection=='stream':
        fields = [1,3,7,8]
    else:
        selection = 'all'
        fields = np.arange(1,9,1)
    
    dates = [get_date(n_) for n_ in fields]
    
    p = np.polyfit(t['phi1'], t['Vrad'], 1)
    poly = np.poly1d(p)
    
    plt.close()
    fig, ax = plt.subplots(3,1,figsize=(12,9), sharex=True)
    
    for e, n in enumerate(fields[:]):
        ind = (t['field']==n)
        t_ = t[ind]
        fname = '/home/ana/data/hectochelle/tiles/gd1_{0:d}/{1:s}/reduced/v3.0/specptg_gd1_{0:d}_cluster_{1:s}.sum.fits'.format(n, dates[e])
        hdu = fits.open(fname)
        xfocal = hdu[5].data['XFOCAL'][t_['fibID']-1]
        yfocal = hdu[5].data['YFOCAL'][t_['fibID']-1]
        
        x = np.zeros_like(xfocal)
        y = np.zeros_like(xfocal)
        angle = (float(hdu[0].header['ROTANGLE']) + float(hdu[0].header['POSANGLE'])) * u.deg
        angle = 0*u.deg
        R = np.array([[np.cos(angle), -np.sin(angle)],[np.sin(angle), np.cos(angle)]])
        posin = np.vstack([xfocal, yfocal])
        posout = np.matmul(R,posin)
        x, y = posout
        
        p = np.polyfit(t_['phi1'], t_['Vrad'], 1)
        poly = np.poly1d(p)
        
        plt.sca(ax[0])
        plt.scatter(t_['phi1'], t_['Vrad'], c=np.arctan2(y,x), cmap='magma', s=0.5*np.sqrt(x**2 + y**2), vmin=-np.pi, vmax=np.pi)
        plt.ylabel('$V_r$ [km s$^{-1}$]')
        
        plt.sca(ax[1])
        #plt.scatter(t_['phi1'], t_['Vrad'] - np.median(t_['Vrad']), c=np.arctan2(y,x), cmap='magma', s=0.5*np.sqrt(x**2 + y**2), vmin=-np.pi, vmax=np.pi)
        plt.scatter(t_['phi1'], t_['Vrad'] - poly(t_['phi1']), c=np.arctan2(y,x), cmap='magma', s=0.5*np.sqrt(x**2 + y**2), vmin=-np.pi, vmax=np.pi)
        plt.ylabel('$\Delta V_r$ [km s$^{-1}$]')
        plt.ylim(-10,10)
        
        plt.sca(ax[2])
        plt.scatter(t_['phi1'], t_['delta_Vrad'], c=np.arctan2(y,x), cmap='magma', s=0.5*np.sqrt(x**2 + y**2), vmin=-np.pi, vmax=np.pi)
        plt.ylabel('$\Delta V_r$ [km s$^{-1}$]')
        
    plt.xlabel('$\phi_1$ [deg]')
    
    plt.tight_layout()
    plt.savefig('../plots/dvr_focal_angle_{:s}.png'.format(selection))








