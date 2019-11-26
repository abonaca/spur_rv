import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from astropy.table import Table
import astropy.units as u
import astropy.coordinates as coord
from astropy.io import ascii, fits
import astropy.table

#from pyia import GaiaData
import gala.coordinates as gc
import glob
import pickle
import myutils
wangle = 180*u.deg

def build_payne_catalog():
    """"""
    
    par_list = glob.glob('/home/ana/data/hectochelle/tiles/m13_1/d20170504/pars/*pars')
    
    t = Table.read(par_list[0], format='ascii')
    tout = t[:0].copy()
    
    #for tdir in tile_dirs[:]:
        #print('entering {}'.format(tdir))
    #par_list = glob.glob('{}/results/V1.2/pars/*pars'.format(tdir))
    
    for pfile in par_list[:]:
        t = Table.read(pfile, format='ascii')
        tout.add_row(t[0])
    
    tout.pprint()
    #print(tout[0])
    tout.write('../data/m13_payne_catalog.fits', overwrite=True)

def finalize_catalog():
    """"""

    tout = Table.read('../data/m13_payne_catalog.fits')
    tphot = Table.read('/home/ana/data/hectochelle/tiles/m13_1/data/speccatfile.fits')
    
    ind = myutils.wherein(tphot['FIBERID'], tout['fibID'])
    tobs = tphot[ind]
    
    # output table
    key_pairs = {'ra': 'RA', 'dec': 'DEC', 'xfoc': 'XFOCAL', 'yfoc': 'YFOCAL'}
    for k in key_pairs:
        tout[k] = tobs[key_pairs[k]]
    tout.write('../data/m13_catalog.fits', overwrite=True)


    #plt.close()
    #plt.figure()
    
    #plt.plot(tobs['YFOCAL'], t['Vrad'], 'ko')
    
    #plt.ylim(-267, -227)
    #plt.tight_layout()
    

def vr_angle():
    """"""
    t = Table.read('../data/m13_catalog.fits')
    ind = (t['Vrad']>-267) & (t['Vrad']<-227) & (-t['lnL'] < 2.5E3+t['SNR']**2.4) & (t['SNR']>2)
    t = t[ind]
    t['Vrad'] -= np.median(t['Vrad'])
    t['ra'] -= np.median(t['ra'])
    t['dec'] -= np.median(t['dec'])
    
    r = np.array(np.sqrt(t['xfoc']**2 +t['yfoc']**2))
    theta = np.array(np.arctan2(t['yfoc'], t['xfoc']))*u.rad
    
    #r = np.array(np.sqrt(t['ra']**2 +t['dec']**2))
    #theta = np.array(np.arctan2(t['dec'], t['ra']))*u.rad

    nrow = 4
    ncol = 5
    angles = np.linspace(0,180,nrow*ncol)*u.deg
    
    da = 2.5
    plt.close()
    fig, ax = plt.subplots(nrow, ncol, figsize=(da*ncol, da*nrow), sharex=True, sharey=True)
    
    for i in range(nrow*ncol):
        irow = int(i/ncol)
        icol = i%ncol
        plt.sca(ax[irow][icol])
        
        x_ = r * np.cos(angles[i] - theta)
        plt.plot(x_, t['Vrad'], 'k.')
        plt.errorbar(x_, t['Vrad'], yerr=(t['lerr_Vrad'], t['uerr_Vrad']), fmt='none', color='k')
        
        plt.text(0.1,0.9, '{:.0f}$^\circ$'.format(angles[i].value), fontsize='small', transform=plt.gca().transAxes, va='top')
    
    for i in range(nrow):
        plt.sca(ax[i][0])
        plt.ylabel('$V_r$ [km s$^{-1}$]')
        
    for i in range(ncol):
        plt.sca(ax[nrow-1][i])
        plt.xlabel('$x_{focal}$')
    
    plt.sca(ax[0][int(ncol/2)])
    plt.title('M 13', fontsize='medium')
    plt.xlim(-199,199)
    
    plt.tight_layout(h_pad=0, w_pad=0)
    plt.savefig('../plots/m13_dvr_angles.png')

def dvr_focalplane():
    """"""
    
    t = Table.read('../data/m13_catalog.fits')
    ind = (t['Vrad']>-267) & (t['Vrad']<-227) & (-t['lnL'] < 2.5E3+t['SNR']**2.4) & (t['SNR']>2)
    t = t[ind]
    t['Vrad'] -= np.median(t['Vrad'])
    t['ra'] -= np.median(t['ra'])
    t['dec'] -= np.median(t['dec'])
    
    ind_vminus = t['Vrad']<0
    ind_vplus = t['Vrad']>0
    
    plt.close()
    fig, ax = plt.subplots(1,3,figsize=(15,5.5), sharey=True, sharex=True)
    
    plt.sca(ax[0])
    plt.scatter(t['xfoc'], t['yfoc'], c=t['Vrad'], cmap='RdBu', vmin=-8, vmax=8)
    plt.xlim(-190,190)
    plt.ylim(-190,190)
    plt.xlabel('$x_{focal}$')
    plt.ylabel('$y_{focal}$')
    
    plt.sca(ax[1])
    plt.scatter(t['xfoc'][ind_vminus], t['yfoc'][ind_vminus], c=t['Vrad'][ind_vminus], cmap='RdBu', vmin=-8, vmax=8)
    plt.xlabel('$x_{focal}$')
    
    plt.sca(ax[2])
    plt.scatter(t['xfoc'][ind_vplus], t['yfoc'][ind_vplus], c=t['Vrad'][ind_vplus], cmap='RdBu', vmin=-8, vmax=8)
    plt.xlabel('$x_{focal}$')
    
    plt.tight_layout()
    plt.savefig('../plots/m13_dvr_focalplane.png')

def get_names():
    """Get names of H3 clusters"""
    fnames = glob.glob('/home/ana/data/hectochelle/clusters/*MSG*fits')
    names = []

    for f in fnames:
        names += [f.split('_')[0].split('/')[-1]]
    
    return names
    
def vr_hist_all():
    """Plot velocity histograms for all H3 clusters, decide on velocity ranges"""
    
    names = get_names()
    N = len(names)
    vlims = np.zeros((N,2))
    voff = np.array([[-13,5],[-20,5],[-15,10],[-4,4],[-2,2],[-4,4]])
    
    nrow = int(np.sqrt(N))
    ncol = int(np.ceil(N/nrow))
    
    da = 2.5
    bv = np.linspace(-400,400,100)
    plt.close()
    fig, ax = plt.subplots(nrow, ncol, figsize=(ncol*da, nrow*da), sharex=False, sharey=False)
    
    for i in range(N):
        irow = int(i/ncol)
        icol = i%ncol
        plt.sca(ax[irow][icol])
        
        t = Table.read('/home/ana/data/hectochelle/clusters/{:s}_MSG_V1.3.fits'.format(names[i]))
        ind = (-t['lnL'] < 2.5E3+t['SNR']**2.4) & (t['SNR']>2)
        t = t[ind]
        vlims[i] = np.median(t['Vrad'])+voff[i,0], np.median(t['Vrad'])+voff[i,1]
        np.save('../data/cluster_vlims', vlims)
        
        plt.hist(t['Vrad'], bins=100, zorder=0)
        plt.axvspan(vlims[i][0], vlims[i][1], color='r', alpha=0.3)
        
        plt.text(0.9,0.9, names[i], transform=plt.gca().transAxes, va='top', ha='right')
    
    plt.tight_layout()

def dvr_angle_all():
    """Create figures for all H3 clusters of delta radial velocity as a function of angle"""
    
    names = get_names()
    N = len(names)
    vlims = np.load('../data/cluster_vlims.npy')
    
    for n in range(N):
        print(names[n])
        t = Table.read('/home/ana/data/hectochelle/clusters/{:s}_MSG_V1.3.fits'.format(names[n]))
        ind = (-t['lnL'] < 2.5E3+t['SNR']**2.4) & (t['SNR']>2) & (t['Vrad']>vlims[n][0]) & (t['Vrad']<vlims[n][1])
        t = t[ind]
        
        # center
        t['Vrad'] -= np.median(t['Vrad'])
        t['ra'] -= np.median(t['ra'])
        t['dec'] -= np.median(t['dec'])
        
        # polar coordinates
        r = np.array(np.sqrt(t['ra']**2 +t['dec']**2))
        theta = np.array(np.arctan2(t['dec'], t['ra']))*u.rad
        
        nrow = 4
        ncol = 5
        angles = np.linspace(0,180,nrow*ncol)*u.deg
        
        da = 2.5
        plt.close()
        fig, ax = plt.subplots(nrow, ncol, figsize=(da*ncol, da*nrow), sharex=True, sharey=True)
        
        for i in range(nrow*ncol):
            irow = int(i/ncol)
            icol = i%ncol
            plt.sca(ax[irow][icol])
            
            x_ = r * np.cos(angles[i] - theta)
            plt.plot(x_, t['Vrad'], 'k.')
            #plt.scatter(x_, t['Vrad'], c=t['PS_R'], vmin=14, vmax=16, zorder=1)
            #print(np.min(t['PS_R'][np.isfinite(t['PS_R'])]), np.max(t['PS_R'][np.isfinite(t['PS_R'])]))
            plt.errorbar(x_, t['Vrad'], yerr=(t['lerr_Vrad'], t['uerr_Vrad']), fmt='none', color='k', zorder=0)
            
            plt.text(0.1,0.9, '{:.0f}$^\circ$'.format(angles[i].value), fontsize='small', transform=plt.gca().transAxes, va='top')
        
        for i in range(nrow):
            plt.sca(ax[i][0])
            plt.ylabel('$V_r$ [km s$^{-1}$]')
            
        for i in range(ncol):
            plt.sca(ax[nrow-1][i])
            plt.xlabel('RA')
        
        plt.sca(ax[0][int(ncol/2)])
        plt.title(names[n], fontsize='medium')
        #plt.xlim(-199,199)
        
        plt.tight_layout(h_pad=0, w_pad=0)
        plt.savefig('../plots/{:s}_dvr_angles.png'.format(names[n]))

def dvr_angle_m67():
    """"""
    t = Table.read('../data/M67_geller.fits')
    #ind = (t['Vrad']>-267) & (t['Vrad']<-227) & (-t['lnL'] < 2.5E3+t['SNR']**2.4) & (t['SNR']>2)
    ind = (t['Mm']=='SM')
    t = t[ind]
    
    print(t.colnames)
    t['Vrad'] = t['__RV_'] - np.median(t['__RV_'])
    t['ra'] = t['_RA'] - np.median(t['_RA'])
    t['dec'] = t['_DE'] - np.median(t['_DE'])
    
    #r = np.array(np.sqrt(t['xfoc']**2 +t['yfoc']**2))
    #theta = np.array(np.arctan2(t['yfoc'], t['xfoc']))*u.rad
    
    r = np.array(np.sqrt(t['ra']**2 +t['dec']**2))
    theta = np.array(np.arctan2(t['dec'], t['ra']))*u.rad

    nrow = 4
    ncol = 5
    angles = np.linspace(0,180,nrow*ncol)*u.deg
    
    da = 2.5
    plt.close()
    fig, ax = plt.subplots(nrow, ncol, figsize=(da*ncol, da*nrow), sharex=True, sharey=True)
    
    for i in range(nrow*ncol):
        irow = int(i/ncol)
        icol = i%ncol
        plt.sca(ax[irow][icol])
        
        x_ = r * np.cos(angles[i] - theta)
        plt.plot(x_, t['Vrad'], 'k.')
        #plt.errorbar(x_, t['Vrad'], yerr=(t['lerr_Vrad'], t['uerr_Vrad']), fmt='none', color='k')
        
        plt.text(0.1,0.9, '{:.0f}$^\circ$'.format(angles[i].value), fontsize='small', transform=plt.gca().transAxes, va='top')
    
    for i in range(nrow):
        plt.sca(ax[i][0])
        plt.ylabel('$V_r$ [km s$^{-1}$]')
        
    for i in range(ncol):
        plt.sca(ax[nrow-1][i])
        plt.xlabel('RA')
    
    plt.sca(ax[0][int(ncol/2)])
    plt.title('M 67 (Geller+2015)', fontsize='medium')
    #plt.xlim(-199,199)
    
    plt.tight_layout(h_pad=0, w_pad=0)
    plt.savefig('../plots/m67geller_dvr_angles.png')

def save_cluster_members():
    """Save cluster members in separate files"""
    
    names = get_names()
    N = len(names)
    vlims = np.load('../data/cluster_vlims.npy')
    
    for n in range(N):
        print(names[n])
        t = Table.read('/home/ana/data/hectochelle/clusters/{:s}_MSG_V1.3.fits'.format(names[n]))
        ind = (-t['lnL'] < 2.5E3+t['SNR']**2.4) & (t['SNR']>2) & (t['Vrad']>vlims[n][0]) & (t['Vrad']<vlims[n][1])
        t = t[ind]
        
        t.write('../data/{:s}_members.fits'.format(names[n]), overwrite=True)












