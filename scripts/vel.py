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
wangle = 180*u.deg

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
    cmg0 = pkl0['cg']
    c0 = pkl0['cg']
    
    plt.close()
    plt.figure(figsize=(10,5))
    
    plt.plot(c0.phi1.wrap_at(wangle), c0.radial_velocity, 'k.', ms=1)
    
    plt.xlabel('$\phi_1$ [deg]')
    plt.ylabel('$V_r$ [km s$^{-1}$]')
    plt.xlim(-100,20)
    plt.ylim(-500,500)
    plt.tight_layout()
    
    plt.savefig('../plots/model_vr_gradient.png')

def h3():
    """Observed H3 stars in GD-1 coordinates"""
    
    t = Table.read('/home/ana/data/rcat_V1.2_MSG.fits')
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
    
    t = Table.read('../data/payne_catalog.fits')
    N = len(t)
    starid = np.zeros(N, dtype='int')
    
    tile_dirs = glob.glob('../data/tiles/gd1*')
    field = np.zeros(N, dtype='int')
    
    for i in range(N):
        name_elements = t['starname'][i].split('_')
        starid[i] = int(name_elements[0][3:])
        
        tile = [s for s in tile_dirs if name_elements[-1] in s][0]
        tile_elements = tile.split('_')
        field[i] = int(tile_elements[1])
    
    tphot = Table.read('../data/gd1_input_catalog.fits')
    tphot = tphot[starid]
    tphot['field'] = field
    
    tout = astropy.table.hstack([tphot, t])
    
    # positions
    c = coord.ICRS(ra=tout['ra']*u.deg, dec=tout['dec']*u.deg)
    cg = c.transform_to(gc.GD1)
    
    # velocity differential
    # polynomial fit to the track
    pkl0 = pickle.load(open('/home/ana/projects/gd1_spur/data/fiducial_noperturb_python3.pkl', 'rb'))
    cmg0 = pkl0['cg']
    c0 = pkl0['cg']
    
    ind = (cmg0.phi1.wrap_at(wangle)<0*u.deg) & (cmg0.phi1.wrap_at(wangle)>-80*u.deg)
    prv = np.polyfit(cmg0.phi1.wrap_at(wangle)[ind], cmg0.radial_velocity[ind], 3)
    polyrv = np.poly1d(prv)
    
    drv = tout['Vrad'] - polyrv(cg.phi1.wrap_at(wangle))
    
    tout['phi1'] = cg.phi1.wrap_at(wangle)
    tout['phi2'] = cg.phi2
    tout['delta_Vrad'] = drv
    
    tout.pprint()
    tout.write('../data/master_catalog.fits', overwrite=True)

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
        
    plt.ylim(-120,-20)
    
    plt.tight_layout()
    
def phi1_vr():
    """"""
    tall = Table.read('/home/ana/data/gd1-better-selection.fits')
    t = Table.read('../data/master_catalog.fits')
    ind = t['priority']<4
    t = t[ind]
    
    plt.close()
    fig, ax = plt.subplots(2,1,figsize=(9,6), sharex=True)
    
    for e, f in enumerate(np.unique(t['field'])):
        t_ = t[t['field']==f]
        plt.sca(ax[0])
        plt.plot(t_['phi1'], t_['phi2'], 'o', color='C{:1d}'.format(e))
        
        plt.sca(ax[1])
        ind = (t_['delta_Vrad']>-20) & (t_['delta_Vrad']<-1)
        plt.plot(t_['phi1'], t_['Vrad'], 'o', alpha=0.2, color='C{:1d}'.format(e))
        plt.plot(t_['phi1'][ind], t_['Vrad'][ind], 'o', color='C{:1d}'.format(e))
        plt.errorbar(t_['phi1'][ind], t_['Vrad'][ind], yerr=(t_['lerr_Vrad'][ind], t_['uerr_Vrad'][ind]), fmt='none', color='0.2', zorder=0)
        
    plt.sca(ax[0])
    plt.plot(tall['phi1'], tall['phi2'], 'k.', ms=2.5, label='Observed GD-1', zorder=0)
    plt.ylim(-3,3)
    plt.xlim(-38,-28)
    plt.ylabel('$\phi_2$ [deg]')
    
    plt.sca(ax[1])
    plt.ylim(-90,-30)
    plt.ylabel('$V_r$ [km s$^{-1}$]')
    plt.xlabel('$\phi_1$ [deg]')
    
    plt.tight_layout(h_pad=0)
    plt.savefig('../plots/phi1_vr.png')

def phi1_dvr():
    """"""
    
    tall = Table.read('/home/ana/data/gd1-better-selection.fits')

    t = Table.read('../data/master_catalog.fits')
    ind = (t['priority']<4) #& (-t['lnL'] < 2.5E3+t['SNR']**2.4) & (t['SNR']>2)
    t = t[ind]
    
    plt.close()
    fig, ax = plt.subplots(4,1,figsize=(10,10), sharex=True, sharey=False)
    
    for f in np.unique(t['field']):
        t_ = t[t['field']==f]

        plt.sca(ax[0])
        plt.plot(t_['phi1'], t_['phi2'], 'o')
        
        plt.sca(ax[1])
        plt.plot(t_['phi1'], t_['delta_Vrad'], 'o')
        plt.errorbar(t_['phi1'], t_['delta_Vrad'], yerr=(t_['lerr_Vrad'], t_['uerr_Vrad']), fmt='none', color='0.2', zorder=0)
    
    plt.sca(ax[0])
    plt.plot(tall['phi1'], tall['phi2'], 'k.', ms=2.5, label='Observed GD-1', zorder=0)
    plt.ylim(-3,3)
    plt.ylabel('$\phi_2$ [deg]')
    
    plt.sca(ax[1])
    plt.ylim(-20,20)
    plt.xlim(-38,-28)
    plt.ylabel('$\Delta V_r$ [km s$^{-1}$]')
    
    plt.sca(ax[2])
    im2 = plt.scatter(t['phi1'], t['delta_Vrad'], c=t['FeH'], vmin=-3, vmax=-1.5)
    plt.ylim(-20,20)
    plt.ylabel('$\Delta V_r$ [km s$^{-1}$]')
    
    plt.sca(ax[3])
    im3 = plt.scatter(t['phi1'], t['delta_Vrad'], c=t['aFe'], vmin=-0.2, vmax=0.6)
    plt.ylim(-20,20)
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






