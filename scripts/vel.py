import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.backends.backend_pdf import PdfPages

from astropy.table import Table
import astropy.units as u
import astropy.coordinates as coord
from astropy.io import ascii, fits
import astropy.table

#from pyia import GaiaData
import gala.coordinates as gc
import glob
import pickle

from spec import *
wangle = 180*u.deg

def gd1_dist(phi1):
    # 0, 10
    # -60, 7
    m = (10-7) / (60)
    return (m*phi1.wrap_at(180*u.deg).value + 10) * u.kpc

def show_vel():
    """"""
    t4 = Table.read('../data/gd1_4_vels.tab', format='ascii.commented_header', delimiter='\t')
    t5 = Table.read('../data/gd1_5_vels.tab', format='ascii.commented_header', delimiter='\t')
    
    r55 = t5['rank']==5
    r45 = t4['rank']==5
    print(np.median(t5['VELOCITY'][r55]))
    print(np.median(t4['VELOCITY'][r45]))
    
    deltav = np.median(t4['VELOCITY'][r45]) - np.median(t5['VELOCITY'][r55])
    t4['VELOCITY'] -= deltav
    print(np.median(t5['VELOCITY'][r55]))
    print(np.median(t4['VELOCITY'][r45]))
    
    r51 = t5['rank']==1
    r41 = t4['rank']==1
    
    print(np.median(t5['VELOCITY'][r51]))
    print(np.median(t4['VELOCITY'][r41]))
    
    vbins = np.linspace(-200,200,50)
    
    plt.close()
    plt.figure()
    
    plt.hist(t4['VELOCITY'], bins=vbins, histtype='step', color='navy')
    plt.hist(t5['VELOCITY'], bins=vbins, histtype='step', color='orange')
    
    plt.hist(t4['VELOCITY'][r41], bins=vbins, histtype='stepfilled', alpha=0.2, color='navy')
    plt.hist(t5['VELOCITY'][r51], bins=vbins, histtype='stepfilled', alpha=0.2, color='orange')
    
    
    plt.tight_layout()

def spur_vel():
    """"""

    trank = Table.read('/home/ana/observing/Hectochelle/2019A/xfitfibs/gd1_catalog.cat', format='ascii.fixed_width_two_line', delimiter='\t')
    t1 = Table.read('../data/gd1_1_vels.tab', format='ascii.commented_header', delimiter='\t')
    good = t1['CZXCR']>3.5
    t1 = t1[good]
    rank1 = trank['rank'][t1['object']]
    r11 = rank1==1
    r1s = rank1<4

    t2 = Table.read('../data/gd1_2_vels.tab', format='ascii.commented_header', delimiter='\t')
    good = t2['CZXCR']>3.5
    t2 = t2[good]
    rank2 = trank['rank'][t2['object']]
    r21 = rank2==1
    r2s = rank2<4

    t4 = Table.read('../data/gd1_4_vels.tab', format='ascii.commented_header', delimiter='\t')
    good = t4['CZXCR']>3.5
    t4 = t4[good]
    r41 = t4['rank']==1
    r4s = t4['rank']<4

    vbins = np.linspace(-200,200,100)

    plt.close()
    plt.figure()
    
    plt.hist(t4['VELOCITY'][r4s], bins=vbins, histtype='step', label='GD-1')
    plt.hist(t1['VELOCITY'][r1s], bins=vbins, histtype='step', label='Spur')
    plt.hist(t2['VELOCITY'][r2s], bins=vbins, histtype='step', label='Spur 2')
    
    plt.xlabel('Radial velocity [km s$^{-1}$]')
    plt.ylabel('Number')
    
    plt.legend()
    plt.tight_layout()

def rv_ra(flag=True, verbose=False):
    """"""
    
    trank = Table.read('/home/ana/observing/Hectochelle/2019A/xfitfibs/gd1_catalog.cat', format='ascii.fixed_width_two_line', delimiter='\t')
    t1 = Table.read('../data/gd1_1_vels.tab', format='ascii.commented_header', delimiter='\t')
    if flag:
        good = t1['CZXCR']>3.5
        t1 = t1[good]
    rank1 = trank['rank'][t1['object']]
    r11 = rank1==1
    r1s = rank1<4

    t2 = Table.read('../data/gd1_2_vels.tab', format='ascii.commented_header', delimiter='\t')
    if flag:
        good = t2['CZXCR']>3.5
        t2 = t2[good]
    rank2 = trank['rank'][t2['object']]
    r21 = rank2==1
    r2s = rank2<4

    t4 = Table.read('../data/gd1_4_vels.tab', format='ascii.commented_header', delimiter='\t')
    if flag:
        good = t4['CZXCR']>3.5
        t4 = t4[good]
    r41 = t4['rank']==1
    r4s = t4['rank']<4
    
    kop_vr = ascii.read("""phi1 phi2 vr err
-45.23 -0.04 28.8 6.9
-43.17 -0.09 29.3 10.2
-39.54 -0.07 2.9  8.7
-39.25 -0.22 -5.2 6.5
-37.95 0.00 1.1   5.6
-37.96 -0.00 -11.7 11.2
-35.49 -0.05 -50.4 5.2
-35.27 -0.02 -30.9 12.8
-34.92 -0.15 -35.3 7.5
-34.74 -0.08 -30.9 9.2
-33.74 -0.18 -74.3 9.8
-32.90 -0.15 -71.5 9.6
-32.25 -0.17 -71.5 9.2
-29.95 -0.00 -92.7 8.7
-26.61 -0.11 -114.2 7.3
-25.45 -0.14 -67.8 7.1
-24.86 0.01 -111.2 17.8
-21.21 -0.02 -144.4 10.5
-14.47 -0.15 -179.0 10.0
-13.73 -0.28 -191.4 7.5
-13.02 -0.21 -162.9 9.6
-12.68 -0.26 -217.2 10.7
-12.55 -0.23 -172.2 6.6""")
    
    cg = gc.GD1(kop_vr['phi1']*u.deg, kop_vr['phi2']*u.deg)
    ceq = cg.transform_to(coord.ICRS)
    
    # model
    pkl = pickle.load(open('/home/ana/projects/gd1_spur/data/fiducial.pkl', 'rb'))
    cmg = pkl['cg']
    cmeq = cmg.transform_to(coord.ICRS)
    
    plt.close()
    plt.figure(figsize=(8,6))
    
    #plt.plot(t1['rad'][r1s], t1['VELOCITY'][r1s], 'ko')
    #plt.plot(t2['rad'][r2s], t2['VELOCITY'][r2s], 'ko')
    #plt.plot(t4['rad'][r4s], t4['VELOCITY'][r4s], 'ko')
    
    if verbose:
        print(np.median(t4['VELOCITY'][r41]), np.median(t1['VELOCITY'][r11]), np.median(t2['VELOCITY'][r21]))
        print(np.std(t4['VELOCITY'][r41]), np.std(t1['VELOCITY'][r11]), np.std(t2['VELOCITY'][r21]))
        print(t1['VELOCITY'][r11])

    plt.plot(ceq.ra, kop_vr['vr'], 'o', color='0.5', label='GD-1 (Koposov)')
    plt.plot(t4['rad'][r41], t4['VELOCITY'][r41], 'wo', mec='k', label='GD-1 (hecto)')

    plt.plot(t1['rad'][r11], t1['VELOCITY'][r11], 'ko', label='Spur (hecto)')
    plt.plot(t2['rad'][r21], t2['VELOCITY'][r21], 'ko', label='')
    
    plt.plot(cmeq.ra, cmeq.radial_velocity, 'ro', ms=1, label='Model (Bonaca+2018)')
    
    plt.xlabel('R.A. [deg]')
    plt.ylabel('Radial velocity [km s$^{-1}$]')
    plt.legend(frameon=False, fontsize='small')
    
    plt.ylim(-250,100)
    plt.xlim(140,180)
    
    plt.tight_layout()
    plt.savefig('../plots/gd1_rv_ra_{:d}.png'.format(flag))


def payne_info():
    """Explore Payne results"""
    
    t = Table.read('../data/GD1_TPv1.0.dat', format='ascii')
    #t.pprint()
    
    id_done = np.int64(t['star'])
    
    tin = Table.read('../data/gd1_input_catalog.fits')
    tin = tin[id_done]
    
    #print(tin.colnames, t.colnames)
    verr = 0.5*(t['Vrad_lerr'] + t['Vrad_uerr'])
    
    plt.close()
    plt.figure(figsize=(10,6))
    
    plt.plot(tin['g'], verr, 'ko')

    plt.xlabel('g [mag]')
    plt.ylabel('$V_{err}$ [km s$^{-1}$]')
    
    plt.ylim(0.001,10)
    plt.gca().set_yscale('log')
    
    plt.tight_layout()
    #plt.savefig('../plots/payne_verr.png')

def verr_corr():
    """Correlations of radial velocity uncertainty with other stellar parameters"""
    
    t = Table.read('../data/GD1_TPv1.0.dat', format='ascii')
    
    id_done = np.int64(t['star'])
    
    tin = Table.read('../data/gd1_input_catalog.fits')
    tin = tin[id_done]
    
    verr = 0.5*(t['Vrad_lerr'] + t['Vrad_uerr'])
    
    plt.close()
    plt.figure(figsize=(10,6))
    
    #plt.plot(t['Vrot'], verr, 'ko')
    im = plt.scatter(tin['g'], verr, c=t['Vrot'], s=60, vmin=0, vmax=50, cmap=mpl.cm.viridis)

    plt.xlabel('g [mag]')
    plt.ylabel('$\sigma_{V_{rad}}$ [km s$^{-1}$]')
    
    plt.ylim(0.001,20)
    plt.gca().set_yscale('log')
    
    divider = make_axes_locatable(plt.gca())
    cax = divider.append_axes("right", size="3%", pad=0.1)
    plt.colorbar(im, cax=cax) #, ticks=np.arange(0,51,25))
    plt.ylabel('$V_{rot}$ [km s$^{-1}$]')
    
    plt.tight_layout()
    plt.savefig('../plots/payne_verr.png')

def rv_catalog():
    """Generate a radial velocity catalog of likely GD-1 members"""
    
    wangle = 180*u.deg
    
    trank = Table.read('/home/ana/observing/Hectochelle/2019A/xfitfibs/gd1_catalog.cat', format='ascii.fixed_width_two_line', delimiter='\t')

    tlit = Table.read('/home/ana/projects/vision/data/koposov_vr.dat', format='ascii.commented_header')
    cg = gc.GD1(tlit['phi1']*u.deg, tlit['phi2']*u.deg)
    ceq = cg.transform_to(coord.ICRS)
    
    ra = ceq.ra
    dec = ceq.dec
    phi1 = cg.phi1.wrap_at(wangle)
    phi2 = cg.phi2
    vr = tlit['vr']*u.km/u.s
    vre = tlit['err']*u.km/u.s
    ref = np.zeros(len(tlit))
    
    for e, ind in enumerate([1,2,4]):
        tin = Table.read('../data/gd1_{:1d}_vels.tab'.format(ind), format='ascii.commented_header', delimiter='\t')
        #if ind>=4:
            #rank = tin['rank']
            #xcr = 3
        #else:
        rank = trank['rank'][tin['object']]
        xcr = 3.5
        
        keep = (tin['CZXCR']>xcr) & (rank<4)
        #print(ind, np.sum(keep))
        #if ind==5:
            #tin[rank<4].pprint()
        tin = tin[keep]
        c_ = coord.ICRS(ra=tin['rad']*u.deg, dec=tin['decd']*u.deg)
        cg_ = c_.transform_to(gc.GD1)
        
        ra = np.concatenate([ra, tin['rad']])
        dec = np.concatenate([dec, tin['decd']])
        phi1 = np.concatenate([phi1, cg_.phi1.wrap_at(wangle)])
        phi2 = np.concatenate([phi2, cg_.phi2])
        vr = np.concatenate([vr, tin['VELOCITY']])
        if ind!=5:
            vre = np.concatenate([vre, tin['CZXCERR']])
        else:
            vre = np.concatenate([vre, np.ones(len(tin))])
        ref = np.concatenate([ref, np.ones(len(tin))*ind])
    
    tout = Table([ra, dec, phi1, phi2, vr, vre, ref], names=('ra', 'dec', 'phi1', 'phi2', 'vr', 'vre', 'ref'))
    tout.pprint()
    
    tout.write('../data/gd1_vr_2019.fits', overwrite=True)

def rv_map():
    """"""
    wangle = 180*u.deg
    cmodel = mpl.cm.Blues(0.9)
    cmodel = '0.5'
    
    pkl = pickle.load(open('/home/ana/projects/gd1_spur/data/fiducial_perturb_python3.pkl', 'rb'))
    cmg = pkl['cg']
    c1 = pkl['cg']
    
    pkl0 = pickle.load(open('/home/ana/projects/gd1_spur/data/fiducial_noperturb_python3.pkl', 'rb'))
    cmg0 = pkl0['cg']
    c0 = pkl0['cg']
    
    # polynomial fit to the track
    ind = (cmg0.phi1.wrap_at(wangle)<0*u.deg) & (cmg0.phi1.wrap_at(wangle)>-80*u.deg)
    prv = np.polyfit(cmg0.phi1.wrap_at(wangle)[ind], cmg0.radial_velocity[ind], 3)
    polyrv = np.poly1d(prv)
    
    pmu1 = np.polyfit(c0.phi1.wrap_at(180*u.deg)[ind], c0.pm_phi1_cosphi2[ind].to(u.mas/u.yr), 3)
    polymu1 = np.poly1d(pmu1)
    
    pmu2 = np.polyfit(c0.phi1.wrap_at(180*u.deg)[ind], c0.pm_phi2[ind].to(u.mas/u.yr), 3)
    polymu2 = np.poly1d(pmu2)
    
    # members
    tmem = Table.read('/home/ana/data/gd1-better-selection.fits')
    #print(tmem.colnames)
    
    # rv observations
    #trv = Table.read('/home/ana/projects/vision/data/gd1_vr.fits')
    trv = Table.read('../data/gd1_vr_2019.fits')
    #print(np.array(trv['phi2']))
    
    plt.close()
    fig, ax = plt.subplots(2,1,figsize=(10,5.5), sharex=True)
    
    plt.sca(ax[0])
    #plt.plot(cmg.phi1.wrap_at(wangle), cmg.phi2, '.', color=cmodel, ms=2)
    plt.plot(tmem['phi1'], tmem['phi2'], 'k.', ms=2.5, label='Observed GD-1')
    #plt.scatter(tmem['phi1'], tmem['phi2'], s=tmem['stream_prob']*2, c=tmem['stream_prob'], cmap=mpl.cm.binary, vmin=0.5, vmax=1.1, rasterized=True)
    #plt.plot(trv['phi1'], trv['phi2'], 'wo', mec='k')
    
    plt.xlim(-70, -10)
    plt.ylim(-6,6)
    plt.ylabel('$\phi_2$ [deg]')
    
    
    plt.sca(ax[1])
    plt.plot(cmg.phi1.wrap_at(wangle), cmg.radial_velocity - polyrv(cmg.phi1.wrap_at(wangle))*u.km/u.s, '.', color=cmodel, ms=2, label='GD-1 model')
    #plt.plot(trv['phi1'], trv['vr'] - polyrv(trv['phi1']), 'wo', mec='k')
    #plt.errorbar(trv['phi1'], trv['vr'] - polyrv(trv['phi1']), yerr=trv['vre'], fmt='none', color='k', mec='k')
    
    plt.ylim(-30,30)
    plt.xlabel('$\phi_1$ [deg]')
    plt.ylabel('$\Delta$ $V_r$ [km s$^{-1}$]')
    
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']
    colors = ['steelblue', mpl.cm.Oranges(0.4), mpl.cm.Oranges(0.8), mpl.cm.Oranges(0.6)]
    msize = [5, 7, 7, 7]
    markers = ['o', 'D', 'D', 'D']
    labels = ['Koposov et al. (2010)', '', '', 'Hectochelle 2019A']
    
    for e,i in enumerate(np.unique(trv['ref'])[:]):
        ind = trv['ref']==i
        color = colors[e]
        ms = msize[e]
        marker = markers[e]
        
        plt.sca(ax[0])
        plt.plot(trv['phi1'][ind], trv['phi2'][ind], marker, ms=ms, color=color, label='')
        
        plt.sca(ax[1])
        plt.plot(trv['phi1'][ind], trv['vr'][ind] - polyrv(trv['phi1'][ind]), marker, ms=ms, color=color, label=labels[e])
        plt.errorbar(trv['phi1'][ind], trv['vr'][ind] - polyrv(trv['phi1'][ind]), yerr=trv['vre'][ind], fmt='none', color=color, label='')
    
    plt.sca(ax[0])
    plt.legend(loc=2, fontsize='small', handlelength=0.4)
    
    plt.sca(ax[1])
    plt.legend(loc=2, fontsize='small', handlelength=0.4)
    
    plt.tight_layout(h_pad=0)
    #plt.savefig('/home/ana/proposals/CfA2019B/hecto_gd1spur/plots/vr_map.pdf')
    plt.savefig('../plots/vr_map.pdf')


## field diagnostics

def field_stats():
    """"""
    t = Table.read('../data/gd1_chelle.fits')
    ind = (t['rank']<4) & (np.abs(t['dvr'])<20) & (t['dvr']<2)
    t = t[ind]
    
    for e, f in enumerate(np.unique(t['field'])):
        ind = t['field'] ==f
        t_ = t[ind]
        print('{} {:.1f} {:.1f}'.format(f, np.median(t_['dvr']), np.std(t_['dvr'])))
    
    ind = (t['field']>=2) & (t['field']<=6)
    t_ = t[ind]
    print('spur {:.1f} {:.1f}'.format(np.median(t_['dvr']), np.std(t_['dvr'])))
    
    t_ = t[~ind]
    print('stream {:.1f} {:.1f}'.format(np.median(t_['dvr']), np.std(t_['dvr'])))
    
    # bootstrap uncertainties
    np.random.seed(477428)
    N = len(t)
    Nsample = 1000
    dvr = np.random.randn(N*Nsample).reshape(N, Nsample) + t['dvr'][:,np.newaxis]
    
    dvr_ = dvr[ind]
    med_spur = np.median(dvr_, axis=0)
    std_spur = np.std(dvr_, axis=0)
    print('spur bootstrap {:.1f} +- {:.1f}, {:.2f} +- {:.2f}'.format(np.median(med_spur), np.std(med_spur), np.median(std_spur), np.std(std_spur)))
    
    dvr_ = dvr[~ind]
    med_stream = np.median(dvr_, axis=0)
    std_stream = np.std(dvr_, axis=0)
    print('stream bootstrap {:.1f} +- {:.1f}, {:.2f} +- {:.2f}'.format(np.median(med_stream), np.std(med_stream), np.median(std_stream), np.std(std_stream)))
    
    bins_med = np.linspace(-12,-3,30)
    bins_std = np.linspace(0,6,30)
    
    plt.close()
    fig, ax = plt.subplots(1,2,figsize=(10,5))
    
    plt.sca(ax[0])
    plt.hist(med_spur, bins=bins_med, alpha=0.3, density=True)
    plt.hist(med_stream, bins=bins_med, alpha=0.3, density=True)
    
    plt.xlabel('Median $\Delta V_r$ [km s$^{-1}$]')
    plt.ylabel('Density')
    
    plt.sca(ax[1])
    plt.hist(std_spur, bins=bins_std, alpha=0.3, density=True, label='Spur')
    plt.hist(std_stream, bins=bins_std, alpha=0.3, density=True, label='Stream')
    
    plt.xlabel('STD $\Delta V_r$ [km s$^{-1}$]')
    plt.ylabel('Density')
    plt.legend(frameon=False, fontsize='small', loc=2, handlelength=1.5)
    
    plt.tight_layout()
    plt.savefig('../plots/stream_spur_dvr_moments.png', dpi=200)

def priorities():
    """Plot a CMD color-coded by rank"""
    
    t = Table.read('../data/gd1_chelle.fits')
    for r in np.unique(t['rank']):
        print(r, np.sum(t['rank']==r))
    
    ind = t['rank']<5
    t5 = t[~ind]
    t = t[ind]
    
    plt.close()
    plt.figure(figsize=(5,10))
    
    plt.plot(t5['g0'] - t5['i0'], t5['g0'], 'k.', ms=3, mew=0, alpha=0.5, zorder=0)
    plt.scatter(t['g0'] - t['i0'], t['g0'], c=t['rank'], s=15, vmin=1, vmax=5, cmap='Oranges_r')
    
    plt.xlabel('g - i')
    plt.ylabel('g')
    plt.xlim(-1,2)
    plt.ylim(20.5,13.5)
    plt.gca().set_aspect('equal')
    plt.savefig('../plots/diag_cmd_rank.png')

def cmd_vr():
    """"""
    
    t = Table.read('../data/gd1_chelle.fits')
    ind = (np.abs(t['vr'])>500) & (t['rank']<4)
    tb = t[ind]
    
    ind = (t['rank']<4) & (np.abs(t['dvr'])<20)
    t = t[ind]
    
    plt.close()
    fig, ax = plt.subplots(2,4,figsize=(10,8), sharex=True, sharey=True)
    
    plt.sca(ax[1][2])
    plt.plot(tb['g0'] - tb['i0'], tb['g0'], 'ko', ms=2, mew=0, alpha=0.5)
    plt.scatter(t['g0'] - t['i0'], t['g0'], c=t['dvr'], s=20, vmin=-20, vmax=20, cmap='magma')
    
    plt.xlim(0,0.9)
    plt.ylim(20.5,16)
    plt.xlabel('g - i')
    
    plt.sca(ax[1][3])
    plt.axis('off')
    
    for e, f in enumerate(np.unique(t['field'])):
        irow = np.int64(e/4)
        icol = e%4
        plt.sca(ax[irow][icol])
        
        ind = t['field'] == f
        t_ = t[ind]
        plt.scatter(t_['g0'] - t_['i0'], t_['g0'], c=t_['dvr'], s=40, vmin=-20, vmax=20, cmap='magma')
        
        ind = tb['field'] == f
        tb_ = tb[ind]
        plt.plot(tb_['g0'] - tb_['i0'], tb_['g0'], 'ko', ms=4, mew=0, alpha=0.5)
        
        plt.text(0.05,0.9, '{:.1f}, {:.1f}'.format(np.median(t_['phi1']), np.median(t_['phi2'])), transform=plt.gca().transAxes, fontsize='small')
        
        if icol==0:
            plt.ylabel('g')
        if irow==1:
            plt.xlabel('g - i')
        
    plt.tight_layout(h_pad=0.1, w_pad=0.01)
    plt.savefig('../plots/fields_cmd_vr.png')

## misc

def vr_gradient():
    """Plot radial velocity along the stream in the fiducial, non-perturbed model"""
    
    pkl0 = pickle.load(open('/home/ana/projects/gd1_spur/data/fiducial_noperturb_python3.pkl', 'rb'))
    c0 = pkl0['cg']
    
    pkl = pickle.load(open('/home/ana/projects/gd1_spur/data/fiducial_perturb_python3.pkl', 'rb'))
    c = pkl['cg']

    ind_fit = (c0.phi1.wrap_at(wangle)>-100*u.deg) & (c0.phi1.wrap_at(wangle)<20*u.deg)
    pfit = np.polyfit(c0.phi1.wrap_at(wangle).value[ind_fit], c0.radial_velocity.value[ind_fit], 6)
    poly = np.poly1d(pfit)
    
    plt.close()
    fig, ax = plt.subplots(2,1,figsize=(10,10), sharex=True)
    
    plt.sca(ax[0])
    plt.plot(c0.phi1.wrap_at(wangle), c0.radial_velocity, 'k.', ms=1)
    plt.plot(c.phi1.wrap_at(wangle), c.radial_velocity, 'r.', ms=1)

    plt.ylabel('$V_r$ [km s$^{-1}$]')
    plt.ylim(-500,500)
    
    plt.sca(ax[1])
    plt.plot(c0.phi1.wrap_at(wangle), c0.radial_velocity.value - poly(c0.phi1.wrap_at(wangle).value), 'ko', ms=1)
    plt.plot(c.phi1.wrap_at(wangle), c.radial_velocity.value - poly(c.phi1.wrap_at(wangle).value), 'ro', ms=1)
    
    plt.xlabel('$\phi_1$ [deg]')
    plt.ylabel('$\Delta$ $V_r$ [km s$^{-1}$]')
    plt.xlim(-100,20)
    plt.ylim(-20,20)
    plt.tight_layout()
    
    plt.savefig('../plots/model_vr_gradient.png')

def h3():
    """Observed H3 stars in GD-1 coordinates"""
    
    t = Table.read('/home/ana/data/rcat_V1.4_MSG.fits')
    c = coord.ICRS(ra=t['GaiaDR2_ra']*u.deg, dec=t['GaiaDR2_dec']*u.deg)
    cg = c.transform_to(gc.GD1)
    #print(t.colnames)
    
    tmem = Table.read('/home/ana/data/gd1-better-selection.fits')
    
    plt.close()
    plt.figure(figsize=(10,5))
    
    plt.plot(cg.phi1.wrap_at(wangle), cg.phi2, '.', color='tab:blue', ms=1)
    plt.plot(tmem['phi1'], tmem['phi2'], 'k.', ms=2.5, label='Observed GD-1')
    
    plt.xlim(-80,0)
    plt.ylim(-10,10)
    
    plt.tight_layout()


## 2019 summary

def check_input_catalogs():
    """"""
    #t1 = Table.read('/home/ana/observing/Hectochelle/2019A/data/gd1_input_catalog.fits')
    t1 = Table.read('../data/gd1_input_catalog.fits')
    print(t1.colnames, len(t1))
    
    t2 = Table.read('/home/ana/observing/Hectochelle/2019A/xfitfibs/gd1_catalog.cat', format='ascii.fixed_width_two_line', delimiter='\t', fill_values='')
    targets = t2['rank']>0
    t2 = t2[targets]
    print(t2.colnames, len(t2))
    
    ra = coord.Angle(t2['ra'][::100], unit=u.hour).deg
    dec = coord.Angle(t2['dec'][::100], unit=u.degree).deg
    
    plt.close()
    fig, ax = plt.subplots(2,1,figsize=(10,10), sharex=True, sharey=True)
    
    plt.sca(ax[0])
    plt.plot(t1['ra'][::100], t1['dec'][::100], 'k.', rasterized=True)
    
    plt.sca(ax[1])
    plt.plot(ra, dec, 'k.', rasterized=True)
    
    plt.tight_layout()

def chelle_catalog():
    """"""
    wangle = 180*u.deg
    
    trank = Table.read('/home/ana/observing/Hectochelle/2019A/xfitfibs/gd1_catalog.cat', format='ascii.fixed_width_two_line', delimiter='\t', fill_values='')
    tphot = Table.read('../data/gd1_input_catalog.fits')

    tlit = Table.read('/home/ana/projects/vision/data/koposov_vr.dat', format='ascii.commented_header')
    cg = gc.GD1(tlit['phi1']*u.deg, tlit['phi2']*u.deg)
    ceq = cg.transform_to(coord.ICRS)
    
    ra = ceq.ra
    dec = ceq.dec
    phi1 = cg.phi1.wrap_at(wangle)
    phi2 = cg.phi2
    vr = tlit['vr']*u.km/u.s
    vre = tlit['err']*u.km/u.s
    ref = np.zeros(len(tlit))
    
    tin = Table.read('../data/gd1_2019_all.tab', format='ascii.commented_header', delimiter='\t')
    rank = trank['rank'][tin['object']]
    xcr = 0
    keep = (tin['CZXCR']>xcr) #& (rank<5)
    
    # construct the catalog
    tin = tin[keep]
    
    # positions
    c = coord.ICRS(ra=tin['rad']*u.deg, dec=tin['decd']*u.deg)
    cg = c.transform_to(gc.GD1)
    
    # velocity differential
    # polynomial fit to the track
    pkl0 = pickle.load(open('/home/ana/projects/gd1_spur/data/fiducial_noperturb_python3.pkl', 'rb'))
    cmg0 = pkl0['cg']
    c0 = pkl0['cg']
    
    ind = (cmg0.phi1.wrap_at(wangle)<0*u.deg) & (cmg0.phi1.wrap_at(wangle)>-80*u.deg)
    prv = np.polyfit(cmg0.phi1.wrap_at(wangle)[ind], cmg0.radial_velocity[ind], 3)
    polyrv = np.poly1d(prv)
    
    drv = tin['VELOCITY'] - polyrv(cg.phi1.wrap_at(wangle))
    
    # photometry
    g = tphot['g'][tin['object']] - tphot['A_g'][tin['object']]
    r = tphot['r'][tin['object']] - tphot['A_r'][tin['object']]
    i = tphot['i'][tin['object']] - tphot['A_i'][tin['object']]
    
    tout = Table([tin['rad'], tin['decd'], cg.phi1.wrap_at(wangle), cg.phi2, tin['VELOCITY'], tin['CZXCERR'], drv, tin['field'], rank[keep], g, r, i], names=('ra', 'dec', 'phi1', 'phi2', 'vr', 'vre', 'dvr', 'field', 'rank', 'g0', 'r0', 'i0'))
    tout.pprint()
    tout.write('../data/gd1_chelle.fits', overwrite=True)

def chelle_map():
    wangle = 180*u.deg
    cmodel = mpl.cm.Blues(0.9)
    cmodel = '0.5'
    
    pkl = pickle.load(open('/home/ana/projects/gd1_spur/data/fiducial_perturb_python3.pkl', 'rb'))
    cmg = pkl['cg']
    c1 = pkl['cg']
    
    pkl0 = pickle.load(open('/home/ana/projects/gd1_spur/data/fiducial_noperturb_python3.pkl', 'rb'))
    cmg0 = pkl0['cg']
    c0 = pkl0['cg']
    
    # polynomial fit to the track
    ind = (cmg0.phi1.wrap_at(wangle)<0*u.deg) & (cmg0.phi1.wrap_at(wangle)>-80*u.deg)
    prv = np.polyfit(cmg0.phi1.wrap_at(wangle)[ind], cmg0.radial_velocity[ind], 3)
    polyrv = np.poly1d(prv)
    
    pmu1 = np.polyfit(c0.phi1.wrap_at(180*u.deg)[ind], c0.pm_phi1_cosphi2[ind].to(u.mas/u.yr), 3)
    polymu1 = np.poly1d(pmu1)
    
    pmu2 = np.polyfit(c0.phi1.wrap_at(180*u.deg)[ind], c0.pm_phi2[ind].to(u.mas/u.yr), 3)
    polymu2 = np.poly1d(pmu2)
    
    # members
    #tmem = Table.read('/home/ana/data/gd1-better-selection.fits')
    #tmem = Table.read('/home/ana/projects/gd1_spur/data/members.fits')
    g = Table(fits.getdata('/home/ana/projects/GD1-DR2/output/gd1_members.fits'))
    
    # rv observations
    #trv = Table.read('/home/ana/projects/vision/data/gd1_vr.fits')
    trv = Table.read('../data/gd1_chelle.fits')
    #print(np.array(trv['phi2']))
    
    plt.close()
    fig, ax = plt.subplots(2,1,figsize=(10,5.5), sharex=True)
    
    plt.sca(ax[0])
    #plt.plot(cmg.phi1.wrap_at(wangle), cmg.phi2, '.', color=cmodel, ms=2)
    #plt.plot(tmem['phi1'], tmem['phi2'], 'k.', ms=2.5, label='Observed GD-1')
    plt.scatter(g['phi1'], g['phi2'], s=g['pmem']*2, c=g['pmem'], cmap=mpl.cm.binary, vmin=0.5, vmax=1.1, label='Observed GD-1')

    #plt.scatter(tmem['phi1'], tmem['phi2'], s=tmem['stream_prob']*2, c=tmem['stream_prob'], cmap=mpl.cm.binary, vmin=0.5, vmax=1.1, rasterized=True)
    #plt.plot(trv['phi1'], trv['phi2'], 'wo', mec='k')
    
    #plt.xlim(-50, -20)
    #plt.xlim(-40, -25)
    plt.xlim(-60, -20)
    plt.ylim(-6,6)
    plt.ylabel('$\phi_2$ [deg]')
    
    
    plt.sca(ax[1])
    plt.plot(cmg.phi1.wrap_at(wangle), cmg.radial_velocity - polyrv(cmg.phi1.wrap_at(wangle))*u.km/u.s, '.', color=cmodel, ms=2, label='GD-1 model')
    #plt.plot(trv['phi1'], trv['vr'] - polyrv(trv['phi1']), 'wo', mec='k')
    #plt.errorbar(trv['phi1'], trv['vr'] - polyrv(trv['phi1']), yerr=trv['vre'], fmt='none', color='k', mec='k')
    
    plt.ylim(-20,20)
    plt.xlabel('$\phi_1$ [deg]')
    plt.ylabel('$\Delta$ $V_r$ [km s$^{-1}$]')
    
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']
    colors = ['steelblue', mpl.cm.Oranges(0.4), mpl.cm.Oranges(0.8), mpl.cm.Oranges(0.6)]
    msize = [5, 7, 7, 7]
    markers = ['o', 'D', 'D', 'D']
    labels = ['Koposov et al. (2010)', '', '', 'Hectochelle 2019A']
    
    for e,i in enumerate(np.unique(trv['field'])[:]):
        ind = (trv['field']==i) & (trv['rank']<4)
        #color = colors[e]
        #ms = msize[e]
        #print(e)
        color = mpl.cm.magma(e/7)
        ms = 6
        marker = 'o'
        #marker = markers[e]
        
        plt.sca(ax[0])
        #plt.plot(trv['phi1'][ind], trv['phi2'][ind], marker, ms=ms, color=color, label='')
        if e==0:
            plt.scatter(trv['phi1'][ind], trv['phi2'][ind], c=trv['phi2'][ind], vmin=-0.5, vmax=1.5, cmap='viridis', label='Spectroscopic targets', s=50)
        else:
            plt.scatter(trv['phi1'][ind], trv['phi2'][ind], c=trv['phi2'][ind], vmin=-0.5, vmax=1.5, cmap='viridis', label='', s=50)
        
        plt.sca(ax[1])
        #plt.plot(trv['phi1'][ind], trv['vr'][ind] - polyrv(trv['phi1'][ind]), marker, ms=ms, color=color, label='')
        #plt.errorbar(trv['phi1'][ind], trv['vr'][ind] - polyrv(trv['phi1'][ind]), yerr=trv['vre'][ind], fmt='none', color=color, label='')
        
        plt.errorbar(trv['phi1'][ind], trv['vr'][ind] - polyrv(trv['phi1'][ind]), yerr=trv['vre'][ind], fmt='none', color='k', label='', zorder=0)
        plt.scatter(trv['phi1'][ind], trv['vr'][ind] - polyrv(trv['phi1'][ind]),  c=trv['phi2'][ind], vmin=-0.5, vmax=1.5, cmap='viridis', label='', s=50)
    
    
    plt.sca(ax[0])
    plt.legend(loc=2, fontsize='small', handlelength=0.4)
    
    plt.sca(ax[1])
    plt.legend(loc=2, fontsize='small', handlelength=0.4)
    
    plt.tight_layout(h_pad=0)
    #plt.savefig('/home/ana/proposals/CfA2019B/hecto_gd1spur/plots/vr_map.pdf')
    plt.savefig('../plots/vr_map_chelle.pdf')
    plt.savefig('../plots/vr_map_chelle.png', dpi=200)
    
    
# The Payne & Minesweeper reduction

def read_msg_catalog():
    """"""
    tp = Table.read('../data/payne_catalog.fits')
    keys = tp.colnames
    
    t = Table.read('../data/GD1_MSG_V1.3.fits')
    t.keep_columns(keys)
    t.pprint()
    t.write('../data/msg_catalog.fits', overwrite=True)

def build_payne_catalog():
    """"""
    
    tile_dirs = glob.glob('../data/tiles/gd1*')
    par_list = glob.glob('{}/results/V1.2/pars/*pars'.format(tile_dirs[0]))
    
    t = Table.read(par_list[0], format='ascii')
    tout = t[:0].copy()
    
    for tdir in tile_dirs[:]:
        print('entering {}'.format(tdir))
        par_list = glob.glob('{}/results/V1.2/pars/*pars'.format(tdir))
        
        for pfile in par_list[:]:
            t = Table.read(pfile, format='ascii')
            tout.add_row(t[0])
    
    tout.pprint()
    tout.write('../data/payne_catalog.fits', overwrite=True)

def build_master_catalog():
    """"""
    
    t = Table.read('../data/msg_catalog.fits')
    N = len(t)
    starid = np.zeros(N, dtype='int')
    
    tile_dirs = glob.glob('../data/tiles/gd1*')
    field = np.zeros(N, dtype='int')
    
    #for tl in tile_dirs:
    tile_dates = [glob.glob('{:s}/d*'.format(tl))[0].split('/')[-1] for tl in tile_dirs]
    fields = [int(tl.split('_')[-1]) for tl in tile_dirs]

    for i in range(N):
        name_elements = t['starname'][i].split('_')
        starid[i] = int(name_elements[0][3:])
        
        ind = tile_dates.index(name_elements[-1])
        #print(tile_dates, name_elements[-1], ind, fields[ind])
        #print(fields[ind])
        field[i] = fields[ind]
        
        #tile = [s for s in tile_dirs if name_elements[-1] in s][0]
        #tile_elements = tile.split('_')
        #field[i] = int(tile_elements[1])
    
    tphot = Table.read('../data/gd1_input_catalog.fits')
    tphot = tphot[starid]
    tphot['field'] = field
    
    tout = astropy.table.hstack([tphot, t])
    tout['field'][(tout['field']==7) & (tout['ra']>150)] = 5
    tout['field'][(tout['field']==7) & (tout['ra']<150)] = 7
    #tout['field'][tout['ra']<150] = 7
    
    # positions
    #c = coord.ICRS(ra=tout['ra']*u.deg, dec=tout['dec']*u.deg)
    #cg = c.transform_to(gc.GD1)
    
    # heliocentric vr correction
    for f in fields:
        fname = '/home/ana/data/hectochelle/tiles/gd1_{0:d}/{1:s}/reduced/v3.0/specptg_gd1_{0:d}_cluster_{1:s}.sum.fits'.format(f, get_date(f))
        hdu = fits.open(fname)
        dvr = hdu[0].header['HELIO_RV']
        ind = field==f
        #tout['Vrad'][ind] += dvr
        
    c = coord.SkyCoord(ra=tout['ra']*u.deg, dec=tout['dec']*u.deg, pm_ra_cosdec=tout['pmra']*u.mas/u.yr, pm_dec=tout['pmdec']*u.mas/u.yr, radial_velocity=tout['Vrad']*u.km/u.s)
    #c = coord.SkyCoord(ra=g.ra, dec=g.dec, pm_ra_cosdec=g.pmra, pm_dec=g.pmdec)
    cg = c.transform_to(gc.GD1)

    gd1_c_dist = gc.GD1(phi1=cg.phi1, phi2=cg.phi2,
                        #distance=gd1_dist(cg.phi1),
                        distance=gd1_dist(cg.phi1),
                        pm_phi1_cosphi2=cg.pm_phi1_cosphi2,
                        pm_phi2=cg.pm_phi2,
                        radial_velocity=cg.radial_velocity)

    # Correct for reflex motion
    v_sun = coord.Galactocentric.galcen_v_sun
    observed = gd1_c_dist.transform_to(coord.Galactic)
    rep = observed.cartesian.without_differentials()
    rep = rep.with_differentials(observed.cartesian.differentials['s'] + v_sun)
    # gd1_c_nosunv = coord.Galactic(rep).transform_to(gc.GD1)
    cg = coord.Galactic(rep).transform_to(gc.GD1)
    
    # velocity differential
    # polynomial fit to the track
    pkl0 = pickle.load(open('/home/ana/projects/gd1_spur/data/fiducial_noperturb_python3.pkl', 'rb'))
    cmg0 = pkl0['cg']
    #c0 = pkl0['cg']
    
    ind = (cmg0.phi1.wrap_at(wangle)<0*u.deg) & (cmg0.phi1.wrap_at(wangle)>-80*u.deg)
    prv = np.polyfit(cmg0.phi1.wrap_at(wangle)[ind], cmg0.radial_velocity[ind], 3)
    polyrv = np.poly1d(prv)
    drv = tout['Vrad'] - polyrv(cg.phi1.wrap_at(wangle).deg)
    
    #ind = (tout['field']==1) | (tout['field']==3) | (tout['field']==7) | (tout['field']==8)
    #mem = (tout['priority']<=3) & (-tout['lnL'] < 2.5E3+tout['SNR']**2.4) & (tout['SNR']>2) & (drv>-20) & (drv<-1)
    #prv = np.polyfit(cg.phi1.wrap_at(wangle)[ind & mem], cg.radial_velocity[ind & mem], 3)
    
    stream = (tout['field']==1) | (tout['field']==3) | (tout['field']==7) | (tout['field']==8)
    mem = (drv>-20) & (drv<-1) & (t['FeH']<-2) & (-tout['lnL'] < 2.5E3+tout['SNR']**2.4) & (tout['SNR']>2)
    prv_ = np.polyfit(cg.phi1.wrap_at(wangle)[stream & mem], tout['Vrad'][stream & mem], 1)
    
    polyrv_ = np.poly1d(prv_)
    drv = tout['Vrad'] - polyrv_(cg.phi1.wrap_at(wangle).deg)
    
    x_ = np.linspace(-47,-29,50)
    y_ = polyrv_(x_)
    
    plt.close()
    #plt.plot(cg.phi1.wrap_at(wangle)[mem], tout['Vrad'][mem], 'ko')
    plt.plot(cg.phi1.wrap_at(wangle)[mem], drv[mem], 'ro')
    plt.plot(x_, y_, '-')
    
    tout['phi1'] = cg.phi1.wrap_at(wangle)
    tout['phi2'] = cg.phi2
    tout['delta_Vrad'] = drv
    tout['pm_phi1_cosphi2'] = cg.pm_phi1_cosphi2.to(u.mas/u.yr)
    tout['pm_phi2'] = cg.pm_phi2.to(u.mas/u.yr)
    
    dates = [get_date(n_) for n_ in fields]
    
    tout['delta_Vrad_field'] = np.zeros_like(drv)
    tout['xfocal'] = np.zeros_like(drv)
    tout['yfocal'] = np.zeros_like(drv)
    
    # per field properties
    for e, f in enumerate(fields):
        # select field
        ind_field = tout['field'] == f
        t_ = t[ind_field]
        
        # relative delta vrad
        tout['delta_Vrad_field'][ind_field] = drv[ind_field] - np.median(drv[ind_field])
        
        # focal plane
        fname = '/home/ana/data/hectochelle/tiles/gd1_{0:d}/{1:s}/reduced/v3.0/specptg_gd1_{0:d}_cluster_{1:s}.sum.fits'.format(f, dates[e])
        hdu = fits.open(fname)
        tout['xfocal'][ind_field] = hdu[5].data['XFOCAL'][t_['fibID']-1]
        tout['yfocal'][ind_field] = hdu[5].data['YFOCAL'][t_['fibID']-1]
    
    tout.pprint()
    tout.write('../data/master_catalog.fits', overwrite=True)

def test_master():
    """"""
    
    t = Table.read('../data/master_catalog.fits')
    t1 = Table.read('../data/master_catalog_v1.fits')
    t2 = Table.read('../data/master_catalog_v2.fits')
    
    print(len(t), len(t1), len(t2))


def gd1_dist(phi1):
    m = (10-7) / (60)
    return (m*phi1.wrap_at(180*u.deg).value + 10) * u.kpc


def ra_vr():
    """"""
    
    t = Table.read('../data/master_catalog.fits')
    ind = t['priority']<4
    t = t[ind]
    print(t.colnames)
    
    plt.close()
    for f in np.unique(t['field']):
        t_ = t[t['field']==f]
        plt.plot(t_['ra'], t_['Vrad'], 'o')
        
    #plt.ylim(-120,-20)
    
    plt.tight_layout()
    
def phi1_vr():
    """"""
    tall = Table.read('/home/ana/data/gd1-better-selection.fits')
    t = Table.read('../data/master_catalog.fits')
    ind = t['priority']<4
    t = t[ind]
    
    # fit polynomial
    stream = (t['field']==1) | (t['field']==3) | (t['field']==7) | (t['field']==8)
    mem = (t['delta_Vrad']>-20) & (t['delta_Vrad']<-1) & (t['FeH']<-2) & (-t['lnL'] < 2.5E3+t['SNR']**2.4) & (t['SNR']>2)
    prv = np.polyfit(t['phi1'][stream & mem], t['Vrad'][stream & mem], 1)
    
    polyrv = np.poly1d(prv)
    phi1_ = np.linspace(-47,-29,50)
    vr_ = polyrv(phi1_)
    
    pkl0 = pickle.load(open('/home/ana/projects/gd1_spur/data/fiducial_noperturb_python3.pkl', 'rb'))
    cmg0 = pkl0['cg']
    pkl = pickle.load(open('/home/ana/projects/gd1_spur/data/fiducial_perturb_python3.pkl', 'rb'))
    cmg = pkl['cg']
    
    plt.close()
    fig, ax = plt.subplots(2,1,figsize=(9,9), sharex=True, gridspec_kw={'height_ratios': (1,2)})
    
    for e, f in enumerate(np.unique(t['field'])):
        t_ = t[t['field']==f]
        #print(f, np.median(t_['phi1']))
        #ind = (t_['delta_Vrad']>-20) & (t_['FeH']<-2) & (-t_['lnL'] < 2.5E3+t_['SNR']**2.4) & (t_['SNR']>2) # & (t_['aFe']<0.3)
        ind = (t_['delta_Vrad']>-20) & (t_['delta_Vrad']<20) & (-t_['lnL'] < 2.5E3+t_['SNR']**2.4) & (t_['SNR']>2) # & (t_['aFe']<0.3)
        print(e, np.sum(ind))

        plt.sca(ax[0])
        plt.plot(t_['phi1'], t_['phi2'], 'o', alpha=0.2, color='C{:1d}'.format(e))
        plt.plot(t_['phi1'][ind], t_['phi2'][ind], 'o', color='C{:1d}'.format(e))
        
        plt.sca(ax[1])
        plt.plot(t_['phi1'], t_['Vrad'] - polyrv(t_['phi1']), 'o', alpha=0.2, color='C{:1d}'.format(e))
        plt.plot(t_['phi1'][ind], t_['Vrad'][ind] - polyrv(t_['phi1'][ind]), 'o', color='C{:1d}'.format(e))
        plt.scatter(t_['phi1'][ind], t_['Vrad'][ind] - polyrv(t_['phi1'][ind]), c=t_['phi2'][ind], s=20, vmin=-1, vmax=2)
        
    plt.sca(ax[0])
    plt.plot(tall['phi1'], tall['phi2'], 'k.', ms=2.5, label='Observed GD-1', zorder=0)
    plt.scatter(cmg.phi1.wrap_at(wangle), cmg.phi2, c=cmg.phi2.value, vmin=-1, vmax=2, s=10)
    #plt.plot(cmg0.phi1.wrap_at(wangle), cmg0.phi2, 'b.', ms=1)
    plt.ylim(-3,3)
    plt.xlim(-48,-20)
    plt.ylabel('$\phi_2$ [deg]')
    
    plt.sca(ax[1])
    #plt.plot(t['phi1'][stream & mem], t['Vrad'][stream & mem], 'ko')
    #plt.plot(phi1_, vr_, '-')
    #plt.plot(cmg0.phi1.wrap_at(wangle), cmg0.radial_velocity-10*u.km/u.s - polyrv(cmg0.phi1.wrap_at(wangle).deg)*u.km/u.s, 'b.', ms=1)
    #plt.plot(cmg.phi1.wrap_at(wangle), cmg.radial_velocity-11*u.km/u.s - polyrv(cmg.phi1.wrap_at(wangle).deg)*u.km/u.s, 'r.', ms=1)
    plt.scatter(cmg.phi1.wrap_at(wangle), cmg.radial_velocity-11*u.km/u.s - polyrv(cmg.phi1.wrap_at(wangle).deg)*u.km/u.s, c=cmg.phi2.value, vmin=-1, vmax=2, s=10)
    plt.ylim(-40,40)
    plt.ylabel('$V_r$ [km s$^{-1}$]')
    plt.xlabel('$\phi_1$ [deg]')
    
    plt.tight_layout(h_pad=0)
    plt.savefig('../plots/phi1_vr.png', dpi=150)

def vr_perfield(color=''):
    """"""
    t = Table.read('../data/master_catalog.fits')
    ind = (t['priority']<=3) & (t['delta_Vrad']<-1) & (t['delta_Vrad']>-20) & (t['FeH']<-2) & (-t['lnL'] < 2.5E3+t['SNR']**2.4) & (t['SNR']>2)
    t = t[ind]
    #print(t.colnames)
    
    print(np.min(t['Teff']), np.max(t['Teff']))
    
    fields = np.unique(t['field'])
    Nf = np.size(fields)
    nrow = int(np.ceil(np.sqrt(Nf)))
    ncol = int(np.ceil(Nf/nrow))
    
    da = 2.5
    plt.close()
    fig, ax = plt.subplots(nrow, ncol, figsize=(ncol*da, nrow*da), sharex=True, sharey=True)
    for i in range(Nf):
        t_ = t[t['field']==fields[i]]

        irow = i%ncol
        icol = int(i/ncol)
        axes = [ax[irow][icol], ax[nrow-1][ncol-1]]

        for ax_ in axes:
            plt.sca(ax_)
            
            if len(color)==0:
                plt.plot(t_['phi1'] - np.median(t_['phi1']), t_['Vrad'] - np.median(t_['Vrad']), 'o', color='C{:1d}'.format(i))
                plt.errorbar(t_['phi1'] - np.median(t_['phi1']), t_['Vrad'] - np.median(t_['Vrad']), yerr=(t_['lerr_Vrad'], t_['uerr_Vrad']), fmt='none', color='C{:1d}'.format(i))
            elif color=='phi2':
                plt.scatter(t_['phi1'] - np.median(t_['phi1']), t_['Vrad'] - np.median(t_['Vrad']), c=t_['phi2'] - np.median(t_['phi2']), vmin=-0.3, vmax=0.3, cmap='RdBu', zorder=1)
                plt.errorbar(t_['phi1'] - np.median(t_['phi1']), t_['Vrad'] - np.median(t_['Vrad']), yerr=(t_['lerr_Vrad'], t_['uerr_Vrad']), fmt='none', color='0.5', zorder=0)
            elif color=='afe':
                plt.scatter(t_['phi1'] - np.median(t_['phi1']), t_['Vrad'] - np.median(t_['Vrad']), c=t_['aFe'], vmin=-0.2, vmax=0.6, cmap='magma', zorder=1)
                plt.errorbar(t_['phi1'] - np.median(t_['phi1']), t_['Vrad'] - np.median(t_['Vrad']), yerr=(t_['lerr_Vrad'], t_['uerr_Vrad']), fmt='none', color='0.5', zorder=0)
            elif color=='feh':
                plt.scatter(t_['phi1'] - np.median(t_['phi1']), t_['Vrad'] - np.median(t_['Vrad']), c=t_['FeH'], vmin=-3, vmax=-2, cmap='magma', zorder=1)
                plt.errorbar(t_['phi1'] - np.median(t_['phi1']), t_['Vrad'] - np.median(t_['Vrad']), yerr=(t_['lerr_Vrad'], t_['uerr_Vrad']), fmt='none', color='0.5', zorder=0)
            elif color=='teff':
                plt.scatter(t_['phi1'] - np.median(t_['phi1']), t_['Vrad'] - np.median(t_['Vrad']), c=t_['Teff'], cmap='magma', vmin=5200, vmax=6900, zorder=1)
                plt.errorbar(t_['phi1'] - np.median(t_['phi1']), t_['Vrad'] - np.median(t_['Vrad']), yerr=(t_['lerr_Vrad'], t_['uerr_Vrad']), fmt='none', color='0.5', zorder=0)
        
        plt.xlim(-0.6, 0.6)
        plt.ylim(-7, 7)
    
    for i in range(ncol):
        plt.sca(ax[nrow-1][i])
        plt.xlabel('$\Delta$ $\phi_1$ [deg]')
    
    for i in range(nrow):
        plt.sca(ax[i][0])
        plt.ylabel('$\Delta$ $V_{rad}$ [km s$^{-1}$]')
    
    plt.tight_layout(h_pad=0.4, w_pad=0.1)
    plt.savefig('../plots/phi1_vr_perfield_{:s}.png'.format(color), dpi=150)
    

def phi1_dvr():
    """"""
    
    tall = Table.read('/home/ana/data/gd1-better-selection.fits')

    t = Table.read('../data/master_catalog.fits')
    #ind = (t['priority']<4) #& (-t['lnL'] < 2.5E3+t['SNR']**2.4) & (t['SNR']>2)
    ind = (t['priority']<=3) & (-t['lnL'] < 2.5E3+t['SNR']**2.4) & (t['SNR']>2) #& (t['delta_Vrad']<-1) & (t['delta_Vrad']>-20) & (t['FeH']<-2)
    
    ind = get_members(t)
    t = t[ind]
    
    plt.close()
    fig, ax = plt.subplots(4,1,figsize=(10,10), sharex=True, sharey=False)
    
    for f in np.unique(t['field']):
        t_ = t[t['field']==f]

        plt.sca(ax[0])
        plt.plot(t_['phi1'], t_['phi2'], 'o', label='{}'.format(f))
        
        plt.sca(ax[1])
        plt.plot(t_['phi1'], t_['delta_Vrad'], 'o')
        plt.errorbar(t_['phi1'], t_['delta_Vrad'], yerr=(t_['lerr_Vrad'], t_['uerr_Vrad']), fmt='none', color='0.2', zorder=0)
    
    plt.sca(ax[0])
    plt.plot(tall['phi1'], tall['phi2'], 'k.', ms=2.5, label='', zorder=0)
    plt.ylim(-3,3)
    plt.ylabel('$\phi_2$ [deg]')
    #plt.legend(loc=2, bbox_to_anchor=(1,1), handlelength=0.5)
    
    plt.sca(ax[1])
    plt.ylim(-10,10)
    plt.xlim(-48,-28)
    plt.ylabel('$\Delta V_r$ [km s$^{-1}$]')
    
    plt.sca(ax[2])
    im2 = plt.scatter(t['phi1'], t['delta_Vrad'], c=t['FeH'], vmin=-3, vmax=-1.5)
    plt.ylim(-10,10)
    plt.ylabel('$\Delta V_r$ [km s$^{-1}$]')
    
    plt.sca(ax[3])
    im3 = plt.scatter(t['phi1'], t['delta_Vrad'], c=t['aFe'], vmin=-0.2, vmax=0.6)
    plt.ylim(-10,10)
    plt.ylabel('$\Delta V_r$ [km s$^{-1}$]')
    plt.xlabel('$\phi_1$ [deg]')
    
    plt.tight_layout(h_pad=0)
    
    plt.sca(ax[2])
    [[x00,y10],[x11,y01]] = plt.gca().get_position().get_points()
    pad = 0.01; width = 0.02
    cbar_ax = fig.add_axes([x11+pad, y10, width, y01-y10])
    plt.colorbar(im2, cax=cbar_ax)
    plt.ylabel('[Fe/H]')
    
    plt.sca(ax[3])
    [[x00,y10],[x11,y01]] = plt.gca().get_position().get_points()
    pad = 0.01; width = 0.02
    cbar_ax = fig.add_axes([x11+pad, y10, width, y01-y10])
    plt.colorbar(im3, cax=cbar_ax)
    plt.ylabel('[$\\alpha$/Fe]')
    
    plt.savefig('../plots/phi1_dvr.png')

def phi2_dvr():
    """"""
    t = Table.read('../data/master_catalog.fits')
    ind = (t['priority']<4) #& (-t['lnL'] < 2.5E3+t['SNR']**2.4) & (t['SNR']>2)
    t = t[ind]
    
    plt.close()
    for f in np.unique(t['field']):
        t_ = t[t['field']==f]
        plt.plot(t_['phi2'], t_['delta_Vrad'], 'o')
    plt.ylim(-20,20)
    
    plt.tight_layout()

def phi1_dvr_phi2():
    """Plot delta vr vs phi1, colored by phi2"""
    t = Table.read('../data/master_catalog.fits')
    ind = (t['priority']<=3) & (t['delta_Vrad']<-1) & (t['delta_Vrad']>-20) & (t['FeH']<-2) & (-t['lnL'] < 2.5E3+t['SNR']**2.4) & (t['SNR']>2)
    t = t[ind]
    
    plt.close()
    plt.figure(figsize=(10,10))
    
    plt.scatter(t['phi1'], t['Vrad'], c=t['phi2'], vmin=-0.5, vmax=1.5, s=40)
    #plt.ylim(-20,0)
    plt.xlabel('$\phi_1$ [deg]')
    plt.ylabel('$V_r$ [km s$^{-1}$]')
    
    plt.tight_layout()

    

def afeh():
    """"""
    t = Table.read('../data/master_catalog.fits')
    ind = (t['priority']<4) & (np.abs(t['delta_Vrad'])<20)
    tmem = t[ind]
    
    bins_feh = np.linspace(-3,0,40)
    bins_afe = np.linspace(-0.2,0.6,15)
    
    plt.close()
    fig, ax = plt.subplots(2,2,figsize=(11,5), sharex='col', sharey='row', gridspec_kw = {'width_ratios':[5, 1], 'height_ratios':[1,4], 'hspace':0, 'wspace':0})
    
    plt.sca(ax[0][1])
    plt.axis('off')
    
    plt.sca(ax[1][0])
    plt.plot(t['FeH'], t['aFe'], 'ko', ms=3, zorder=0, label='All targeted')
    plt.plot(tmem['FeH'], tmem['aFe'], 'ro', label='PM, CMD, |$\Delta V_r$|<20 km s$^{-1}$')
    plt.errorbar(tmem['FeH'], tmem['aFe'], xerr=(tmem['lerr_FeH'], tmem['uerr_FeH']), yerr=(tmem['lerr_aFe'], tmem['uerr_aFe']), fmt='none', color='r', lw=0.5, label='')
    #plt.errorbar(tmem['FeH'], tmem['aFe'], xerr=tmem['std_FeH'], yerr=tmem['std_aFe'], fmt='none', color='r', lw=0.5, label='')
    
    plt.legend(loc=4, fontsize='small')
    plt.xlabel('[Fe/H]')
    plt.ylabel('[$\\alpha$/Fe]')
    
    plt.xlim(-3,0)
    plt.ylim(-0.2, 0.6)
    #plt.gca().set_aspect('equal')
    
    plt.sca(ax[0][0])
    plt.hist(t['FeH'], bins=bins_feh, color='k', alpha=0.3, density=True)
    plt.hist(tmem['FeH'], bins=bins_feh, color='r', alpha=0.3, density=True)
    plt.axis('off')
    
    plt.sca(ax[1][1])
    plt.hist(t['aFe'], bins=bins_afe, color='k', alpha=0.3, density=True, orientation='horizontal')
    plt.hist(tmem['aFe'], bins=bins_afe, color='r', alpha=0.3, density=True, orientation='horizontal')
    plt.axis('off')
    
    plt.tight_layout(h_pad=0, w_pad=0)
    plt.savefig('../plots/afeh.png')

def dvr_feh():
    """"""
    t = Table.read('../data/master_catalog.fits')
    ind = t['priority']<4
    t = t[ind]
    
    plt.close()
    fig, ax = plt.subplots(1,2,figsize=(10,5), sharey=True)
    
    plt.sca(ax[0])
    plt.plot(t['FeH'], t['delta_Vrad'], 'ko')
    plt.ylim(-20,20)
    
    plt.sca(ax[1])
    plt.plot(t['aFe'], t['delta_Vrad'], 'ko')
    
    
    plt.tight_layout()


# velocity statistics

def brute_membership():
    """A very crude determination of the stream members"""
    
    t = Table.read('../data/master_catalog.fits')
    ind = (t['priority']<=3) & (t['delta_Vrad']>-20) & (t['delta_Vrad']<-1)
    t = t[ind]
    t.write('../data/members_catalog.fits', overwrite=True)
    

def vr_profile(deg=1):
    """"""
    
    t = Table.read('../data/members_catalog.fits')
    
    spur = (t['field']==2) | (t['field']==4) | (t['field']==6) | (t['field']==5)
    stream = ~spur
    stream2 = (t['field']==1) | (t['field']==3)
    labels = ['Stream', 'Spur']
    p = []
    
    plt.close()
    fig, ax = plt.subplots(1,2,figsize=(10,5), sharex=True, sharey=True)
    
    for e, ind in enumerate([stream, spur]):
        # polyfit
        p += [np.polyfit(t['phi1'][ind], t['Vrad'][ind], deg, w=t['std_Vrad'][ind]**-1)]
        poly = np.poly1d(p[e])
        x = np.linspace(-48, -29.2, 100)
        y = poly(x)
        
        # plot
        plt.sca(ax[e])
        plt.plot(t['phi1'][ind], t['Vrad'][ind], 'o', label='')
        plt.errorbar(t['phi1'][ind], t['Vrad'][ind], yerr=(t['lerr_Vrad'][ind], t['uerr_Vrad'][ind]), fmt='none', color='tab:blue', zorder=0, lw=1.5, label='')
        plt.plot(x, y, '-', lw=0.75, zorder=0, color='navy', label='$V_r$ = {:.1f}$\\times\phi_1$ {:+.0f}'.format(p[e][0], p[e][1]))
        plt.title(labels[e], fontsize='medium')
        
        plt.sca(ax[(e+1)%2])
        plt.plot(x, y, '--', lw=0.75, zorder=0, color='navy', label='')
        
        plt.xlabel('$\phi_1$ [deg]')
    
    outdict = {'p_stream': p[0], 'p_spur': p[1]}
    pkl = pickle.dump(outdict, open('../data/polyfit_rv_{:d}.pkl'.format(deg), 'wb'))
    
    plt.sca(ax[0])
    plt.ylabel('$V_r$ [km s$^{-1}$]')
    plt.legend(frameon=False, loc=1)
    
    plt.sca(ax[1])
    plt.legend(frameon=False, loc=1)
    
    plt.tight_layout()
    plt.savefig('../plots/vr_profile.png')

def relative_vr():
    """"""

    t = Table.read('../data/members_catalog.fits')
    spur = (t['field']==2) | (t['field']==4) | (t['field']==6) | (t['field']==5)
    stream = ~spur
    
    pkl = pickle.load(open('../data/polyfit_rv_2.pkl', 'rb'))
    #p = pkl['p_spur']
    p = pkl['p_stream']
    print(p)
    poly = np.poly1d(p)
    dvr = t['Vrad'] - poly(t['phi1'])
    
    indices = [stream, spur]
    colors = ['darkorange', 'navy', ]
    labels = ['Stream', 'Spur']

    plt.close()
    plt.figure(figsize=(10,5))
    
    plt.axhline(0, color='k', alpha=0.5, lw=2)
    
    for e, ind in enumerate(indices):
        plt.plot(t['phi1'][ind], dvr[ind], 'o', color=colors[e], label=labels[e])
        plt.errorbar(t['phi1'][ind], dvr[ind], yerr=(t['lerr_Vrad'][ind], t['uerr_Vrad'][ind]), fmt='none', color=colors[e], zorder=0, lw=1.5, label='')
    
    plt.legend(loc=1, frameon=False)
    plt.xlabel('$\phi_1$ [deg]')
    plt.ylabel('$V_r$ - $V_{r, stream\;fit}$ [km s$^{-1}$]')
    plt.tight_layout()
    plt.savefig('../plots/relative_vr.png')


def afeh_comparison(priority=1):
    """"""
    t = Table.read('../data/master_catalog.fits')
    ind = (t['priority']<=priority) & (t['delta_Vrad']<0) & (t['delta_Vrad']>-20)
    
    spur = (t['field']==2) | (t['field']==4) | (t['field']==5) | (t['field']==6)
    stream = ~spur
    
    bins_feh = np.linspace(-3,0,40)
    bins_afe = np.linspace(-0.2,0.6,15)
    
    plt.close()
    fig, ax = plt.subplots(2,2,figsize=(11,5), sharex='col', sharey='row', gridspec_kw = {'width_ratios':[5, 1], 'height_ratios':[1,4], 'hspace':0, 'wspace':0})
    
    plt.sca(ax[0][1])
    plt.axis('off')
    
    plt.sca(ax[1][0])
    plt.plot(t['FeH'], t['aFe'], 'ko', ms=3, zorder=0, label='All targeted')

    tmem = t[ind & stream]
    plt.plot(tmem['FeH'], tmem['aFe'], 'ro', label='Stream')
    plt.errorbar(tmem['FeH'], tmem['aFe'], xerr=(tmem['lerr_FeH'], tmem['uerr_FeH']), yerr=(tmem['lerr_aFe'], tmem['uerr_aFe']), fmt='none', color='r', lw=0.5, label='')

    tmem = t[ind & spur]
    plt.plot(tmem['FeH'], tmem['aFe'], 'bo', label='Spur')
    plt.errorbar(tmem['FeH'], tmem['aFe'], xerr=(tmem['lerr_FeH'], tmem['uerr_FeH']), yerr=(tmem['lerr_aFe'], tmem['uerr_aFe']), fmt='none', color='b', lw=0.5, label='')

    
    plt.legend(loc=4, fontsize='small')
    plt.xlabel('[Fe/H]')
    plt.ylabel('[$\\alpha$/Fe]')
    
    plt.xlim(-3,0)
    plt.ylim(-0.2, 0.6)
    #plt.gca().set_aspect('equal')
    
    plt.sca(ax[0][0])
    plt.hist(t['FeH'], bins=bins_feh, color='k', alpha=0.3, density=True)

    tmem = t[ind & stream]
    plt.hist(tmem['FeH'], bins=bins_feh, color='r', alpha=0.3, density=True)

    tmem = t[ind & spur]
    plt.hist(tmem['FeH'], bins=bins_feh, color='b', alpha=0.3, density=True)
    plt.axis('off')
    
    plt.sca(ax[1][1])
    plt.hist(t['aFe'], bins=bins_afe, color='k', alpha=0.3, density=True, orientation='horizontal')
    
    tmem = t[ind & stream]
    plt.hist(tmem['aFe'], bins=bins_afe, color='r', alpha=0.3, density=True, orientation='horizontal')
    
    tmem = t[ind & spur]
    plt.hist(tmem['aFe'], bins=bins_afe, color='b', alpha=0.3, density=True, orientation='horizontal')
    plt.axis('off')
    
    plt.tight_layout(h_pad=0, w_pad=0)
    plt.savefig('../plots/afeh_comparison.{:d}.png'.format(priority))


def vr_median():
    """"""
    
    t = Table.read('../data/master_catalog.fits')
    ind = (t['priority']<=3) & (t['delta_Vrad']<-1) & (t['delta_Vrad']>-20) & (t['FeH']<-2)
    t = t[ind]
    
    M = 1000
    fields = np.unique(t['field'])
    N = np.size(fields)
    phi1 = np.zeros(N)
    phi2 = np.zeros(N)
    sigma = np.zeros(N)

    v_med = np.zeros(N)
    v_std = np.zeros(N)
    sigma_med = np.zeros(N)
    sigma_std = np.zeros(N)
    
    for e in range(N):
        ind = t['field']==fields[e]
        t_ = t[ind]
        Nstar = len(t_)
        phi1[e] = np.median(t_['phi1'])
        phi2[e] = np.median(t_['phi2'])
        
        # subtract local gradient
        p = np.polyfit(t_['phi1'], t_['Vrad'], 1, w=t_['std_Vrad']**-1)
        poly = np.poly1d(p)
        
        dvr = t_['Vrad'] - poly(t_['phi1'])
        sigma[e] = np.std(dvr)

        vrad = np.random.randn(Nstar*M).reshape(Nstar,M) + t_['Vrad'][:,np.newaxis]
        dvr = vrad - poly(t_['phi1'])[:,np.newaxis]
        
        vs = np.median(vrad, axis=0)
        v_med[e] = np.median(vs)
        v_std[e] = np.std(vs)
        
        sigmas = np.std(dvr, axis=0)
        sigma_med[e] = np.median(sigmas)
        sigma_std[e] = np.std(sigmas)
    
    q = np.polyfit(phi1, v_med, 2, w=v_std**-1)
    qpoly = np.poly1d(q)
    dv_med = v_med - qpoly(phi1)
    np.save('../data/poly_vr_median', q)
    
    spur = (fields==2) | (fields==4) | (fields==5) | (fields==6)
    stream = ~spur
    
    np.savez('../data/field_vr_summary', phi1=phi1, phi2=phi2, dv_med=dv_med, v_std=v_std, sigma_med=sigma_med, sigma_std=sigma_std, fields=fields, spur=spur, stream=stream)
    
    spur_global = (t['field']==2) | (t['field']==4) | (t['field']==5) | (t['field']==6)
    stream_global = ~spur_global
    labels = ['Stream', 'Spur']
    colors = ['deepskyblue', 'midnightblue']
    colors = [mpl.cm.Blues(0.6), mpl.cm.Blues(0.85)]
    
    plt.close()
    fig, ax = plt.subplots(2,1,figsize=(6,6), sharex=True)
    
    plt.sca(ax[0])
    for e, ind in enumerate([stream_global, spur_global]):
        plt.plot(t['phi1'][ind], t['Vrad'][ind], 'o', label=labels[e], color=colors[e])
        plt.errorbar(t['phi1'][ind], t['Vrad'][ind], yerr=t['std_Vrad'][ind], fmt='none', color=colors[e], label='')
    
    x0, x1 = plt.gca().get_xlim()
    y0, y1 = plt.gca().get_ylim()
    x_ = np.linspace(x0, x1, 100)
    y_ = qpoly(x_)
    plt.plot(x_, y_, 'k-', alpha=0.3, lw=2, zorder=0)
    
    plt.xlim(x0, x1)
    plt.ylim(y0, y1)
    plt.legend(frameon=False, loc=1)
    plt.ylabel('$V_r$ [km s$^{-1}$]')
    plt.text(0.05, 0.1, 'Individual members', transform=plt.gca().transAxes)
    
    plt.sca(ax[1])
    for e, ind in enumerate([stream, spur]):
        plt.plot(phi1[ind], dv_med[ind], 'o', color=colors[e], label=labels[e])
        plt.errorbar(phi1[ind], dv_med[ind], yerr=v_std[ind], fmt='none', color=colors[e], label='')
    
    plt.axhline(0, color='k', alpha=0.3, lw=2, zorder=0)
    plt.legend(frameon=False, loc=3)
    plt.ylabel('$\Delta V_r$ [km s$^{-1}$]')
    plt.ylim(-3,3)
    #plt.text(0.05, 0.1, 'Field median', transform=plt.gca().transAxes)
    
    #plt.sca(ax[2])
    #for e, ind in enumerate([stream, spur]):
        ##plt.plot(phi1[ind], sigma[ind], 'o')
        #plt.plot(phi1[ind], sigma_med[ind], 'o', color=colors[e])
        #plt.errorbar(phi1[ind], sigma_med[ind], yerr=sigma_std[ind], fmt='none', color=colors[e])
    
    plt.xlabel('$\phi_1$ [deg]')
    #plt.ylabel('$\sigma_{V_r}$ [km s$^{-1}$]')
    #plt.text(0.05, 0.1, 'Field dispersion', transform=plt.gca().transAxes)
    plt.tight_layout(h_pad=0.2)
    plt.savefig('../plots/kinematic_profile_median.png', dpi=200)
    
def vr_dispersion():
    """"""
    
    t = Table.read('../data/master_catalog.fits')
    #ind = (t['priority']<=3) & (t['delta_Vrad']<-1) & (t['delta_Vrad']>-20) & (t['FeH']<-2)
    ind = (t['priority']<=3) & (t['delta_Vrad']>-20) & (t['delta_Vrad']<-1) & (t['FeH']<-2) & (-t['lnL'] < 2.5E3+t['SNR']**2.4) & (t['SNR']>2)
    t = t[ind]
    
    M = 1000
    fields = np.unique(t['field'])
    N = np.size(fields)
    phi1 = np.zeros(N)
    sigma = np.zeros(N)

    v_med = np.zeros(N)
    v_std = np.zeros(N)
    sigma_med = np.zeros(N)
    sigma_std = np.zeros(N)
    
    for e in range(N):
        ind = t['field']==fields[e]
        t_ = t[ind]
        Nstar = len(t_)
        phi1[e] = np.median(t_['phi1'])
        
        # subtract local gradient
        p = np.polyfit(t_['phi1'], t_['Vrad'], 1, w=t_['std_Vrad']**-1)
        poly = np.poly1d(p)
        
        dvr = t_['Vrad'] - poly(t_['phi1'])
        sigma[e] = np.std(dvr)

        vrad = np.random.randn(Nstar*M).reshape(Nstar,M) + t_['Vrad'][:,np.newaxis]
        dvr = vrad - poly(t_['phi1'])[:,np.newaxis]
        
        vs = np.median(vrad, axis=0)
        v_med[e] = np.median(vs)
        v_std[e] = np.std(vs)
        
        sigmas = np.std(dvr, axis=0)
        sigma_med[e] = np.median(sigmas)
        sigma_std[e] = np.std(sigmas)
    
    q = np.polyfit(phi1, v_med, 2, w=v_std**-1)
    qpoly = np.poly1d(q)
    dv_med = v_med - qpoly(phi1)
    
    spur = (fields==2) | (fields==4) | (fields==5) | (fields==6)
    stream = ~spur
    
    spur_global = (t['field']==2) | (t['field']==4) | (t['field']==5) | (t['field']==6)
    stream_global = ~spur_global
    labels = ['Stream', 'Spur']
    colors = ['deepskyblue', 'midnightblue']
    colors = [mpl.cm.Blues(0.6), mpl.cm.Blues(0.85)]
    
    plt.close()
    fig, ax = plt.subplots(3,1,figsize=(10,9), sharex=True)
    
    plt.sca(ax[0])
    for e, ind in enumerate([stream_global, spur_global]):
        plt.plot(t['phi1'][ind], t['Vrad'][ind], 'o', label=labels[e], color=colors[e])
        plt.errorbar(t['phi1'][ind], t['Vrad'][ind], yerr=t['std_Vrad'][ind], fmt='none', color=colors[e], label='')
    
    x0, x1 = plt.gca().get_xlim()
    y0, y1 = plt.gca().get_ylim()
    x_ = np.linspace(x0, x1, 100)
    y_ = qpoly(x_)
    plt.plot(x_, y_, 'k-', alpha=0.3, lw=2, zorder=0)
    
    plt.xlim(x0, x1)
    plt.ylim(y0, y1)
    plt.legend(frameon=False, loc=1)
    plt.ylabel('$V_r$ [km s$^{-1}$]')
    plt.text(0.05, 0.1, 'Individual members', transform=plt.gca().transAxes)
    plt.ylim(-100,-10)
    
    plt.sca(ax[1])
    for e, ind in enumerate([stream, spur]):
        plt.plot(phi1[ind], dv_med[ind], 'o', color=colors[e])
        plt.errorbar(phi1[ind], dv_med[ind], yerr=v_std[ind], fmt='none', color=colors[e])
    
    plt.axhline(0, color='k', alpha=0.3, lw=2, zorder=0)
    plt.ylabel('$\Delta V_r$ [km s$^{-1}$]')
    plt.ylim(-3,3)
    plt.text(0.05, 0.1, 'Field median', transform=plt.gca().transAxes)
    
    plt.sca(ax[2])
    for e, ind in enumerate([stream, spur]):
        #plt.plot(phi1[ind], sigma[ind], 'o')
        plt.plot(phi1[ind], sigma_med[ind], 'o', color=colors[e])
        plt.errorbar(phi1[ind], sigma_med[ind], yerr=sigma_std[ind], fmt='none', color=colors[e])
    
    plt.xlabel('$\phi_1$ [deg]')
    plt.ylabel('$\sigma_{V_r}$ [km s$^{-1}$]')
    plt.ylim(0,5)
    plt.minorticks_on()
    plt.text(0.05, 0.1, 'Field dispersion', transform=plt.gca().transAxes)
    plt.tight_layout(h_pad=0.2)
    plt.savefig('../plots/kinematic_profile.png')

def cmd_members():
    """"""
    t = Table.read('../data/master_catalog.fits')
    ind = (t['priority']<=3) & (t['delta_Vrad']<-1) & (t['delta_Vrad']>-20) & (t['FeH']<-2)
    t = t[ind]
    
    spur = (t['field']==2) | (t['field']==4) | (t['field']==5) | (t['field']==6)
    stream = ~spur
    labels = ['Stream', 'Spur']
    colors = [mpl.cm.Blues(0.6), mpl.cm.Blues(0.85)]
    
    plt.close()
    plt.figure(figsize=(5,8))
    
    for e, ind in enumerate([stream, spur]):
        plt.plot(t['g'][ind] - t['i'][ind], t['g'][ind], 'o', color=colors[e])
    
    plt.gca().invert_yaxis()
    plt.tight_layout()

def snr_members(verbose=False):
    """"""
    t = Table.read('../data/master_catalog.fits')
    ind = (t['priority']<=3) & (t['delta_Vrad']<-1) & (t['delta_Vrad']>-20) & (t['FeH']<-2)
    t = t[ind]
    
    spur = (t['field']==2) | (t['field']==4) | (t['field']==5) | (t['field']==6)
    stream = ~spur
    labels = ['Stream', 'Spur']
    colors = [mpl.cm.Blues(0.6), mpl.cm.Blues(0.85)]
    
    if verbose:
        print(t.colnames)
        ind = t['SNR']>10
        print(t['starname'][ind])
        print(t['field'][ind])
    
    plt.close()
    plt.figure(figsize=(8,6))
    
    for e, ind in enumerate([stream, spur]):
        plt.plot(t['g'][ind], t['SNR'][ind], 'o', color=colors[e], label=labels[e])
    
    plt.legend(loc=1, frameon=False)
    plt.xlabel('g [mag]')
    plt.ylabel('S/N')
    
    plt.tight_layout()
    plt.savefig('../plots/snr_members.png')

def afeh_members():
    """"""
    t = Table.read('../data/master_catalog.fits')
    ind = (t['priority']<=3) & (t['delta_Vrad']<-1) & (t['delta_Vrad']>-20) & (t['FeH']<-2)
    t = t[ind]
    
    spur = (t['field']==2) | (t['field']==4) | (t['field']==5) | (t['field']==6)
    stream = ~spur
    labels = ['Stream', 'Spur']
    colors = [mpl.cm.Blues(0.6), mpl.cm.Blues(0.85)]
    
    plt.close()
    fig, ax = plt.subplots(1,2,figsize=(11,6), sharey=True)
    
    for e, ind in enumerate([stream, spur]):
        plt.sca(ax[0])
        plt.plot(t['FeH'][ind], t['aFe'][ind], 'o', color=colors[e], label=labels[e])
        plt.errorbar(t['FeH'][ind], t['aFe'][ind], xerr=t['std_FeH'][ind], yerr=t['std_aFe'][ind], fmt='none', color=colors[e], label='', lw=0.5)
        
        plt.sca(ax[1])
        isnr = t['SNR']>10
        plt.plot(t['FeH'][ind & isnr], t['aFe'][ind & isnr], 'o', color=colors[e], label=labels[e])
        plt.errorbar(t['FeH'][ind & isnr], t['aFe'][ind & isnr], xerr=t['std_FeH'][ind & isnr], yerr=t['std_aFe'][ind & isnr], fmt='none', color=colors[e], label='', lw=0.5)
    
    plt.sca(ax[0])
    plt.xlabel('[Fe/H]')
    plt.ylabel('[$\\alpha$/Fe]')
    
    plt.sca(ax[1])
    plt.legend(loc=1, frameon=False)
    plt.xlabel('[Fe/H]')
    plt.title('S/N>10', fontsize='medium')
    
    plt.tight_layout()
    plt.savefig('../plots/afeh_members.png')

def distance_members():
    """"""
    
    t = Table.read('../data/master_catalog.fits')
    ind = (t['priority']<=3) & (t['delta_Vrad']<-1) & (t['delta_Vrad']>-20) & (t['FeH']<-2)
    t = t[ind]
    
    spur = (t['field']==2) | (t['field']==4) | (t['field']==5) | (t['field']==6)
    stream = ~spur
    labels = ['Stream', 'Spur']
    colors = [mpl.cm.Blues(0.6), mpl.cm.Blues(0.85)]
    
    plt.close()
    plt.figure(figsize=(10, 5))
    
    for e, ind in enumerate([stream, spur]):
        plt.plot(t['phi1'][ind], t['Dist'][ind], 'o', color=colors[e])
        plt.errorbar(t['phi1'][ind], t['Dist'][ind], yerr=t['std_Dist'][ind], fmt='none', color=colors[e])
    
    plt.xlabel('$\phi_1$ [deg]')
    plt.ylabel('Distance [kpc]')
    plt.ylim(0,13)
    
    plt.tight_layout()
    plt.savefig('../plots/distance_members.png')


steelblue = '#a2b3d2'
navyblue = '#294882'
fuchsia = '#ff3643'

def plot_membership():
    """Plot likely members and their selection in the CMD, radial velocity and chemical space"""
    
    t = Table.read('../data/master_catalog.fits')

    mem_dict = get_members(t, full=True)
    cmdmem = mem_dict['cmdmem']
    pmmem = mem_dict['pmmem']
    vrmem = mem_dict['vrmem']
    fehmem = mem_dict['fehmem']
    vrlims = mem_dict['vrlims']
    fehlims = mem_dict['fehlims']
    
    print(np.sum(pmmem & cmdmem), np.sum(cmdmem & vrmem), np.sum(mem_dict['mem']))
    
    bvr = np.linspace(-50,50,50)
    
    plt.close()
    fig, ax = plt.subplots(1, 3, figsize=(15,5), gridspec_kw={'width_ratios': [1,1.7,3.2]})
    
    plt.sca(ax[0])
    plt.plot(t['g'] - t['i'], t['g'], 'o', color=steelblue, mec='none', ms=5, alpha=0.5)
    plt.plot(t['g'][pmmem] - t['i'][pmmem], t['g'][pmmem], 'o', color=navyblue, mec='none', ms=5)
    pm = mpl.patches.Polygon(mem_dict['cmdbox'], facecolor='none', edgecolor=fuchsia, lw=3, ls='--', zorder=2)
    plt.gca().add_artist(pm)
    
    plt.xlim(-0.5,1.5)
    plt.ylim(20.6,14.5)
    plt.xlabel('(g - i)$_0$ [mag]')
    plt.ylabel('g$_0$ [mag]')
    plt.title('Proper motion', fontsize='medium')
    plt.text(0.9, 0.9, '{:2d}'.format(np.sum(cmdmem & pmmem)), transform=plt.gca().transAxes, ha='right')
    
    plt.sca(ax[1])
    plt.hist(t['delta_Vrad'][~cmdmem], bins=bvr, histtype='stepfilled', color=steelblue, alpha=0.5, density=False)
    plt.hist(t['delta_Vrad'][cmdmem], bins=bvr, histtype='stepfilled', color=navyblue, density=False)
    
    for vrlim in vrlims:
        plt.axvline(vrlim, ls='--', lw=3, color=fuchsia)
    
    plt.xlim(-50,50)
    plt.ylabel('Number')
    plt.xlabel('$V_r$ - $V_{r,orbit}$ [km s$^{-1}$]')
    plt.title('+ Color - magnitude', fontsize='medium')
    plt.text(0.94, 0.9, '{:2d}'.format(np.sum(cmdmem & vrmem)), transform=plt.gca().transAxes, ha='right')
    
    plt.sca(ax[2])
    plt.plot(t['FeH'][cmdmem & vrmem], t['aFe'][cmdmem & vrmem], 'o', color=navyblue, mec='none', ms=6, label='GD-1 members', zorder=1)
    plt.plot(t['FeH'][~(cmdmem & vrmem)], t['aFe'][~(cmdmem & vrmem)], 'o', color=steelblue, mec='none', alpha=0.5, ms=6, label='Field stars', zorder=0)
    
    for fehlim in fehlims:
        plt.axvline(fehlim, ls='--', lw=3, color=fuchsia, label='', zorder=2)
    
    plt.text(0.97, 0.9, '{:2d}'.format(np.sum(cmdmem & vrmem & fehmem)), transform=plt.gca().transAxes, ha='right')
    plt.legend(loc=4, frameon=True, handlelength=1, fontsize='small', markerscale=1.3)
    
    plt.xlim(-3,0)
    plt.ylim(-0.2,0.6)
    plt.ylabel('[$\\alpha$/Fe]')
    plt.xlabel('[Fe/H]')
    plt.title('+ Radial velocity selection', fontsize='medium')

    plt.tight_layout()
    plt.savefig('../paper/members.pdf')

def get_members(t, full=False):
    """Return indices of GD-1 members"""
    
    # cmd selection
    cmdlim = 3
    cmdmem = t['priority']<=cmdlim
    
    iso = Table.read('/home/ana/data/isochrones/panstarrs/mist_12.6_-1.50.cmd', format='ascii.commented_header', header_start=12)
    phasecut = (iso['phase']>=0) & (iso['phase']<3)
    iso = iso[phasecut]

    # distance modulus
    distance_app = 7.8*u.kpc
    dm = 5*np.log10((distance_app.to(u.pc)).value)-5

    # main sequence + rgb
    i_gi = iso['PS_g']-iso['PS_i']
    i_g = iso['PS_g']+dm

    i_left_narrow = i_gi - 0.4*(i_g/28)**5
    i_right_narrow = i_gi + 0.5*(i_g/28)**5
    poly_narrow = np.hstack([np.array([i_left_narrow, i_g]), np.array([i_right_narrow[::-1], i_g[::-1]])]).T

    i_left_wide = i_gi - 0.6*(i_g/28)**3
    i_right_wide = i_gi + 0.7*(i_g/28)**3
    poly_wide = np.hstack([np.array([i_left_wide, i_g]), np.array([i_right_wide[::-1], i_g[::-1]])]).T

    ind = (poly_wide[:,1]<18.3) & (poly_wide[:,1]>14)
    poly_low = poly_wide[ind]

    ind = (poly_narrow[:,1]<20.5) & (poly_narrow[:,1]>18.3)
    poly_med = poly_narrow[ind]
    
    # manual union
    nhalf_low = int(np.shape(poly_low)[0]/2)
    nhalf_med = int(np.shape(poly_med)[0]/2)
    cmdbox = np.vstack([poly_low[nhalf_low:,:], poly_med[nhalf_med:,:], poly_med[:nhalf_med,:], poly_low[:nhalf_low,:]])
    
    # radial velocity selection
    #vrlims = np.array([-20, -1])
    vrlims = np.array([-5,5])
    vrmem = (t['delta_Vrad']>vrlims[0]) & (t['delta_Vrad']<vrlims[1])
    
    # feh selection
    fehlims = np.array([-2.8, -1.9])
    #fehlims = np.array([-2.1, -2.8])
    fehmem = (t['FeH']>fehlims[0]) & (t['FeH']<fehlims[1])
    
    # pm selection
    pm1lims = np.array([-9,-4.5])
    pm2lims = np.array([-1.7,1])
    #pm2lims = np.array([-2,2])
    pmmem = (t['pm_phi1_cosphi2']>pm1lims[0]) & (t['pm_phi1_cosphi2']<pm1lims[1]) & (t['pm_phi2']>pm2lims[0]) & (t['pm_phi2']<pm2lims[1])
    
    members = cmdmem & pmmem & vrmem & fehmem
    
    if full:
        return_dict = {'mem': members, 'cmdmem': cmdmem, 'pmmem': pmmem, 'vrmem': vrmem, 'fehmem': fehmem, 'cmdlim': cmdlim, 'cmdbox': cmdbox, 'vrlims': vrlims, 'fehlims': fehlims}
        return return_dict
    else:
        return members

def print_members_field():
    """"""
    t = Table.read('../data/master_catalog.fits')
    mem = get_members(t)
    t = t[mem]
    
    print(t.colnames)
    
    for i in range(8):
        ind = t['field']==i+1
        t_ = t[ind]
        print(i+1, np.array(t_['name']))
    


def spur_dvr_dep():
    """Plot how delta radial velocity in the spur depends on various properties"""
    t = Table.read('../data/master_catalog.fits')
    mem = get_members(t)
    t = t[mem]
    
    #print(t.colnames, len(t))
    spurfields = [2,4,5,6]
    ind = np.array([t['field']==x for x in spurfields])
    ind_spur = np.sum(ind, axis=0, dtype=bool)
    #t = t[ind_spur]
    fields = np.unique(t['field'])
    
    params = ['logg', 'Teff', 'FeH', 'aFe']
    
    plt.close()
    fig, ax = plt.subplots(4,1,figsize=(8,12), sharex=True, sharey=True)
    
    #for f in fields:
        #t_ = t[t['field']==f]
        ##plt.scatter(t_['phi1'] - np.median(t_['phi1']), t_['Vrad'] - np.median(t_['Vrad']), c=t_['logg'], s=20, cmap='magma', vmin=3, vmax=5)
        #plt.scatter(t_['phi1'] - np.median(t_['phi1']), t_['Vrad'] - np.median(t_['Vrad']), c=t_['Teff'], s=20, cmap='magma')
    #plt.scatter(t['phi1'], t['delta_Vrad'], c=t['logg'], s=40, cmap='magma', vmin=3, vmax=5)
    for i in range(4):
        plt.sca(ax[i])
        plt.scatter(t['phi1'], t['Vrad'], c=t[params[i]], s=t['SNR']*8, cmap='magma') #, vmin=3, vmax=5)
    
    #plt.sca(ax[1])
    #plt.scatter(t['phi1'], t['Vrad'], c=t['Teff'], s=t['SNR']*5, cmap='magma') #, vmin=3, vmax=5)
    
    plt.tight_layout(h_pad=0)

# proper motions

def pm_offset():
    """"""
    t = Table.read('../data/master_catalog.fits')
    ind = (t['priority']<=5) & (t['delta_Vrad']<-1) & (t['delta_Vrad']>-20) & (t['FeH']<-2) #& (np.abs(t['pmra_error']/t['pmra'])<0.07)
    t = t[ind]
    
    spur = (t['field']==2) | (t['field']==4) | (t['field']==5) | (t['field']==6)
    stream = ~spur
    labels = ['Stream', 'Spur']
    colors = [mpl.cm.Blues(0.6), mpl.cm.Blues(0.85)]
    
    #print(np.median(np.abs(t['pmra_error']/t['pmra'])))
    #print(np.median(np.abs(t['pmdec_error']/t['pmdec'])))
    
    qra = np.polyfit(t['phi1'][stream], t['pm_phi1_cosphi2'][stream], 2, w=t['pmra_error'][stream]**-1)
    polyra = np.poly1d(qra)
    dra = t['pm_phi1_cosphi2'] #- polyra(t['phi1'])
    #dra = t['pmra']
    
    qdec = np.polyfit(t['phi1'][stream], t['pm_phi2'][stream], 2, w=t['pmdec_error'][stream]**-1)
    polydec = np.poly1d(qdec)
    ddec = t['pm_phi2'] #- polydec(t['phi1'])
    #ddec = t['pmdec']
    
    plt.close()
    fig, ax = plt.subplots(2,1,figsize=(12,8), sharex=True)
    
    for e, s in enumerate([stream, spur]):
        plt.sca(ax[0])
        plt.plot(t['phi1'][s], dra[s], 'o', color=colors[e])
        plt.errorbar(t['phi1'][s], dra[s], yerr=t['pmra_error'][s], color=colors[e], fmt='none')
        #plt.plot(t['phi1'][s], t['pmra'][s], 'o', color=colors[e])
        #plt.errorbar(t['phi1'][s], t['pmra'][s], yerr=t['pmra_error'][s], color=colors[e], fmt='none')
        
        #plt.axhline(0, color='k', alpha=0.5, lw=0.5)
        #plt.ylim(-2.5,2.5)
        #plt.xlabel('$\phi_1$ [deg]')
        plt.ylabel('$\Delta\mu_{\phi_1}$ [mas yr$^{-1}$]')
        
        plt.sca(ax[1])
        plt.plot(t['phi1'][s], ddec[s], 'o', color=colors[e])
        plt.errorbar(t['phi1'][s], ddec[s], yerr=t['pmdec_error'][s], color=colors[e], fmt='none')
        #plt.plot(t['phi1'][s], t['pmdec'][s], 'o', color=colors[e])
        #plt.errorbar(t['phi1'][s], t['pmdec'][s], yerr=t['pmdec_error'][s], color=colors[e], fmt='none')
        
        #plt.axhline(0, color='k', alpha=0.5, lw=0.5)
        #plt.ylim(-2.5,2.5)
        plt.xlabel('$\phi_1$ [deg]')
        plt.ylabel('$\Delta\mu_{\phi_2}$ [mas yr$^{-1}$]')
    
    plt.tight_layout()
    plt.savefig('../plots/pm_offset.png')


# payne version comparison

def version_match():
    """Match stars ran through both versions of the pipeline"""
    
    t0 = Table.read('../data/master_catalog_v1.fits')
    t1 = Table.read('../data/master_catalog.fits')
    
    c0 = coord.SkyCoord(ra=t0['ra']*u.deg, dec=t0['dec']*u.deg)
    c1 = coord.SkyCoord(ra=t1['ra']*u.deg, dec=t1['dec']*u.deg)
    idx, d2d, d3d = c0.match_to_catalog_sky(c1)
    
def version_dvr(members=False):
    """"""
    
    t0 = Table.read('../data/master_catalog_v1.fits')
    t1 = Table.read('../data/master_catalog.fits')
    
    if members:
        ind = t0['priority']<=2
        t0 = t0[ind]
    
    c0 = coord.SkyCoord(ra=t0['ra']*u.deg, dec=t0['dec']*u.deg)
    c1 = coord.SkyCoord(ra=t1['ra']*u.deg, dec=t1['dec']*u.deg)
    idx, d2d, d3d = c0.match_to_catalog_sky(c1)
    
    dvr = t0['Vrad'] - t1[idx]['Vrad']
    
    plt.close()
    fig, ax = plt.subplots(2,2, figsize=(16,8), sharex='col', sharey='row')
    
    # along the stream
    plt.sca(ax[0][0])
    plt.plot(t0['phi1'], dvr, 'ko')
    
    plt.ylabel('$\Delta V_{r}$ [km s$^{-1}$]')
    
    plt.sca(ax[1][0])
    plt.plot(t0['phi1'], dvr, 'ko')
    plt.errorbar(t0['phi1'], dvr, yerr=t0['std_Vrad'], fmt='none', color='k', alpha=0.1)
    plt.axhline(0, color='r', lw=0.5, zorder=0)
    
    plt.ylim(-3,3)
    plt.xlabel('$\phi_1$ [deg]')
    plt.ylabel('$\Delta V_{r}$ [km s$^{-1}$]')
    
    # as a function of SNR
    plt.sca(ax[0][1])
    plt.plot(t0['SNR'], dvr, 'ko')
    
    plt.sca(ax[1][1])
    plt.plot(t0['SNR'], dvr, 'ko')
    plt.errorbar(t0['SNR'], dvr, yerr=t0['std_Vrad'], fmt='none', color='k', alpha=0.1)
    plt.axhline(0, color='r', lw=0.5, zorder=0)
    
    plt.ylim(-3,3)
    plt.xlabel('S/N')

    plt.tight_layout()
    plt.savefig('../plots/version_dvr.png')

def version_phi1_dvr():
    """"""
    
    t0 = Table.read('../data/master_catalog_v1.fits')
    ind = t0['priority']<=2
    t0 = t0[ind]
    t1 = Table.read('../data/master_catalog.fits')
    
    c0 = coord.SkyCoord(ra=t0['ra']*u.deg, dec=t0['dec']*u.deg)
    c1 = coord.SkyCoord(ra=t1['ra']*u.deg, dec=t1['dec']*u.deg)
    idx, d2d, d3d = c0.match_to_catalog_sky(c1)
    
    t = t1[idx]
    
    #t = Table.read('../data/members_catalog.fits')
    spur = (t['field']==2) | (t['field']==4) | (t['field']==6) | (t['field']==5)
    stream = ~spur
    
    pkl = pickle.load(open('../data/polyfit_rv_2.pkl', 'rb'))
    #p = pkl['p_spur']
    p = pkl['p_stream']
    print(p)
    poly = np.poly1d(p)
    dvr = t['Vrad'] - poly(t['phi1'])
    dvr0 = t0['Vrad'] - poly(t0['phi1'])
    
    print(np.shape(dvr), np.shape(dvr0))
    
    indices = [stream, spur]
    colors = ['darkorange', 'navy', ]
    labels = ['Stream', 'Spur']

    plt.close()
    plt.figure(figsize=(10,5))
    
    plt.axhline(0, color='k', alpha=0.5, lw=2)
    
    for e, ind in enumerate(indices):
        plt.plot(t['phi1'][ind], dvr[ind], 'o', color=colors[e], label=labels[e]+' new')
        plt.errorbar(t['phi1'][ind], dvr[ind], yerr=(t['lerr_Vrad'][ind], t['uerr_Vrad'][ind]), fmt='none', color=colors[e], zorder=0, lw=1.5, label='')
        
        plt.errorbar(t0['phi1'][ind], dvr0[ind], yerr=(t0['lerr_Vrad'][ind], t0['uerr_Vrad'][ind]), fmt='none', color=colors[e], zorder=0, lw=1.5, label='')
        plt.plot(t0['phi1'][ind], dvr0[ind], 'wo', mec=colors[e], label=labels[e]+' old')

    plt.ylim(-20,20)
    plt.legend(loc=1, frameon=False, ncol=2)
    plt.xlabel('$\phi_1$ [deg]')
    plt.ylabel('$V_r$ - $V_{r, stream\;fit}$ [km s$^{-1}$]')
    
    plt.tight_layout()
    plt.savefig('../plots/version_phi_dvr.png')

def fiber_vr(expand=1):
    """"""
    
    t = Table.read('../data/master_catalog.fits')
    ind = (t['priority']<=3) & (t['delta_Vrad']>-20) & (t['delta_Vrad']<0)
    t = t[ind]
    
    fields = np.unique(t['field'])
    #fields = [2,4,5]
    #fields = [5,7]
    
    plt.close()
    plt.figure(figsize=(8,10))
    
    for e, f in enumerate(fields):
        ind = t['field']==f
        t_ = t[ind]
        
        #plt.plot(t_['fibID'], t_['delta_Vrad'] + e*30, 'o', ms=10, label='{}'.format(f))
        plt.plot(t_['fibID'], t_['Vrad'] - np.median(t_['Vrad']) + e*expand*30, 'o', ms=10, label='{}'.format(f))

    plt.xlabel('Fiber ID')
    plt.ylabel('$V_r$ - $\overline{V_r}$ [km s$^{-1}$]')
    
    #plt.legend()
    #plt.ylim(-30,30)
    #plt.ylim(-10,10)
    plt.tight_layout()
    plt.savefig('../plots/fiber_vr_{}.png'.format(expand))
    
def fiber_vr_april():
    """"""
    
    t = Table.read('../data/master_catalog.fits')
    ind = (t['priority']<=3) & (t['delta_Vrad']>-20) & (t['delta_Vrad']<0)
    t = t[ind]
    print(t.colnames)
    print(t['name'])

    fields = [5,7]
    
    plt.close()
    plt.figure(figsize=(8,8))
    
    for e, f in enumerate(fields):
        ind = t['field']==f
        t_ = t[ind]
        print(np.array(t_['name']))
        #print(np.array(t_['']))
        
        #plt.plot(t_['fibID'], t_['delta_Vrad'] + e*30, 'o', ms=10, label='{}'.format(f))
        plt.plot(t_['fibID'], t_['Vrad'] - 0*np.median(t_['Vrad']) + e*0, 'o', ms=10, label='{}'.format(f))

    #plt.legend()
    #plt.ylim(-30,30)
    #plt.ylim(-10,10)
    plt.xlabel('Fiber ID')
    plt.ylabel('$V_r$ - $\overline{V_r}$ [km s$^{-1}$]')
    
    plt.tight_layout()
    plt.savefig('../plots/fiber_vr_april.png')

def field_snr():
    """"""
    
    t = Table.read('../data/master_catalog.fits')
    #print(t.colnames)
    
    fields = np.unique(t['field'])
    #fields = [2,4,5]
    
    plt.close()
    plt.figure(figsize=(8,8))
    
    for e, f in enumerate(fields):
        ind = t['field']==f
        t_ = t[ind]
        #print(t_['SNR'])
        #print(t_['phot_g_mean_mag'])
        
        plt.plot(t_['phot_g_mean_mag'], t_['SNR'], 'o', ms=4, label='{}'.format(f))
    
    plt.gca().set_yscale('log')
    plt.legend(markerscale=2)
    plt.xlabel('G')
    plt.ylabel('S/N')

    plt.tight_layout()
    plt.savefig('../plots/field_snr.png')

def phi1_vr_fiber():
    """"""
    
    t = Table.read('../data/master_catalog.fits')
    ind = (t['priority']<=3) & (t['delta_Vrad']>-20) & (t['delta_Vrad']<0)
    t = t[ind]
    
    fields = np.unique(t['field'])
    
    plt.close()
    fig, ax = plt.subplots(8,1,figsize=(7,10))
    
    for e, f in enumerate(fields):
        ind = t['field']==f
        t_ = t[ind]
        
        plt.sca(ax[e])
        plt.scatter(t_['phi1'], t_['Vrad'], c=t_['fibID'], vmin=0, vmax=255,)
        
        plt.xticks([])

    plt.xlabel('$\phi_1$ [deg]')
    fig.text(0.01, 0.5, '$V_r$ [km s$^{-1}$]', va='center', rotation='vertical')
    
    plt.tight_layout(rect=[0.03,0,1,1], h_pad=0)
    plt.savefig('../plots/phi1_vr_field.png')


# single field analysis
def field_vr(field=1):
    """"""
    
    t = Table.read('../data/master_catalog.fits')
    ind = (t['priority']<4) & (t['field']==field)
    t = t[ind]
    
    #print(t.colnames)
    
    #print(np.median(t['Vrad']))
    dvr = t['Vrad'] - np.median(t['Vrad'])
    ind = np.abs(dvr)<10
    t = t[ind]
    dvr = dvr[ind]
    #print(t['starname'])
    
    plt.close()
    fig, ax = plt.subplots(1,2,sharey=True)
    
    plt.sca(ax[0])
    plt.scatter(t['phi1'], t['phi2'], c=dvr, vmin=-1, vmax=4, s=40)
    #plt.gca().set_aspect('equal')
    
    plt.sca(ax[1])
    plt.plot(dvr, t['phi2'], 'ko')
    
    
    plt.tight_layout()

def diagnostic_summary(f=1, color_by='phi1', cmd_mem=True, return_figure=False):
    """Summary plot per field"""
    tin = Table.read('../data/master_catalog.fits')
    #print(tin.colnames)
    
#def br():
    if cmd_mem:
        md = get_members(tin, full=True)
        mem = md['vrmem'] & md['fehmem'] & md['pmmem']
    else:
        mem = get_members(tin)
    
    ind = tin['field']==f
    print(f, np.sum(ind), len(tin))
    
    tin['gr'] = tin['g'] - tin['r']
    t = tin[ind & mem]
    
    alphas = [0.5,1]
    pairs = [[['phi1', 'phi2'], ['xfocal', 'yfocal']], [['phi1', 'Vrad'], ['FeH', 'aFe']], [['phi1', 'delta_Vrad'], ['gr', 'g']]]
    labels = [[['$\phi_1$ [deg]', '$\phi_2$ [deg]'], ['$x_{focal}$', '$y_{focal}$']], [['$\phi_1$ [deg]', '$V_{rad}$ [km s$^{-1}$]'], ['[Fe/H]', '[$\\alpha$/Fe]']], [['$\phi_1$ [deg]', '$\Delta V_{rad}$ [km s$^{-1}$]'], ['(g - r)$_0$ [mag]', 'g$_0$ [mag]']]]
    xlims = [[[-47,-29], [-300,300]], [[-47,-29], [-2.8,-1.8]], [[-47,-29], [0.1,0.6]]]
    ylims = [[[-0.5,1.7], [-300,300]], [[-100,50], [-0.1,0.6]], [[-10,10], [21,15.5]]]
    
    
    plt.close()
    fig, ax = plt.subplots(3,2,figsize=(15,8), gridspec_kw={'width_ratios': (3,1)})
    
    ticks_minor = [0.25, 2, 1]
    ticks_major = [0.5, 10, 5]
    
    for irow in range(3):
        for icol in range(2):
            plt.sca(ax[irow][icol])
            #for e, t_ in enumerate([tin[mem], t]):
            plt.plot(tin[mem][pairs[irow][icol][0]], tin[mem][pairs[irow][icol][1]], 'ko', alpha=0.2, mec='none', ms=6, zorder=0)
            if color_by=='order':
                plt.scatter(t[pairs[irow][icol][0]], t[pairs[irow][icol][1]], c=np.arange(len(tin[mem])), zorder=1, cmap='plasma', s=50, ec='k')
            else:
                plt.scatter(t[pairs[irow][icol][0]], t[pairs[irow][icol][1]], c=t[color_by], zorder=1, cmap='plasma', s=50, ec='k')
            
            plt.xlabel(labels[irow][icol][0])
            plt.ylabel(labels[irow][icol][1])
            plt.xlim(xlims[irow][icol])
            plt.ylim(ylims[irow][icol])
    
        icol = 0
        plt.sca(ax[irow][icol])
        axins = plt.gca().inset_axes([0.2, 0.1, 0.3, 0.8])
        axins.scatter(t[pairs[irow][icol][0]], t[pairs[irow][icol][1]], c=t[color_by], zorder=5, cmap='plasma', s=50, ec='k')
        axins.tick_params(labelleft=False, labelbottom=False)
        axins.yaxis.set_major_locator(mpl.ticker.MultipleLocator(ticks_major[irow]))
        axins.yaxis.set_minor_locator(mpl.ticker.MultipleLocator(ticks_minor[irow]))
        axins.minorticks_on()
        axins.grid(b=True, which='both', axis='both', zorder=0)
    
    irow = 2
    icol = 0
    plt.sca(ax[irow][icol])
    #axins = plt.gca().inset_axes([0.2, 0.1, 0.3, 0.8])
    #axins.scatter(t[pairs[irow][icol][0]], t[pairs[irow][icol][1]], c=t[color_by], zorder=5, cmap='plasma', s=50, ec='k')
    #axins.tick_params(labelleft=False, labelbottom=False)
    #axins.minorticks_on()
    #axins.grid(b=True, which='both', axis='y', zorder=0)
    plt.text(0.1,0.1,'{:.1f}'.format(np.std(t[pairs[irow][icol][1]])), transform=plt.gca().transAxes)
    #plt.xlim(xlims[irow][icol])
    #plt.ylim(ylims[irow][icol])
    plt.ylim(-5,5)
    
    plt.tight_layout(h_pad=0, w_pad=0)
    
    if return_figure:
        return fig

def diagnostic_summary_all(cmd_mem=True, color_by='phi1'):
    """"""
    
    pp = PdfPages('../plots/spectroscopic_summary_cmd{:d}_color{:s}.pdf'.format(cmd_mem, color_by))
    
    for f in range(1,9):
        fig = diagnostic_summary(f=f, color_by=color_by, cmd_mem=cmd_mem, return_figure=True)
        pp.savefig(fig)
    
    pp.close()


def afeh_kinematics():
    """"""
    t = Table.read('../data/master_catalog.fits')
    
    md = get_members(t, full=True)
    mem = md['vrmem'] & md['fehmem'] & md['pmmem']
    t = t[mem]
    
    p1 = [-2.6,0.5]
    p2 = [-2, 0]
    p1 = [-2.7,0.4]
    p2 = [-1.9, 0.2]
    
    k = (p2[1] - p1[1])/(p2[0] - p1[0])
    l = p1[1] - k*p1[0]
    ind = t['aFe'] < k*t['FeH'] + l

    plt.close()
    fig, ax = plt.subplots(2,1,figsize=(10,8))
    plt.sca(ax[0])
    plt.plot(t['FeH'], t['aFe'], 'ko')
    plt.plot(t['FeH'][ind], t['aFe'][ind], 'ro')
    
    plt.sca(ax[1])
    #plt.plot(t['phi1'], t['delta_Vrad'], 'ko')
    #plt.plot(t['phi1'][ind], t['delta_Vrad'][ind], 'ro')
    plt.plot(t['phi1'][~ind], t['delta_Vrad'][~ind], 'ro')

    plt.tight_layout()
