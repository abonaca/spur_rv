from model import *
from vel import get_members
from matplotlib.legend_handler import HandlerLine2D
import healpy as hp

wangle = 180*u.deg
lightsteelblue = '#dde3ef'
steelblue = '#a2b3d2'
navyblue = '#294882'
fuchsia = '#ff3643'

#########
# Figures

def plot_membership():
    """Plot likely members and their selection in the CMD, radial velocity and chemical space"""
    
    t = Table.read('../data/master_catalog.fits')
    ind = (-t['lnL'] < 2.5E3+t['SNR']**2.4) & (t['SNR']>3) & np.isfinite(t['aFe'])
    t = t[ind]
    
    spur = (t['field']==2) | (t['field']==4) | (t['field']==5) | (t['field']==6)
    stream = ~spur

    mem_dict = get_members(t, full=True)
    cmdmem = mem_dict['cmdmem']
    pmmem = mem_dict['pmmem']
    vrmem = mem_dict['vrmem']
    fehmem = mem_dict['fehmem']
    vrlims = mem_dict['vrlims']
    fehlims = mem_dict['fehlims']
    mem = mem_dict['mem']
    #mem = pmmem & vrmem
    
    print(np.sum(pmmem & cmdmem), np.sum(pmmem & cmdmem & vrmem), np.sum(mem_dict['mem']))
    
    bvr = np.linspace(-50,50,50)
    
    plt.close()
    
    fig = plt.figure(figsize=(11.25,8.1))
    gs1 = mpl.gridspec.GridSpec(1,3)
    gs1.update(left=0.08, right=0.975, top=0.95, bottom=0.6, wspace=0.25)
    
    gs2 = mpl.gridspec.GridSpec(1,1)
    gs2.update(left=0.08, right=0.975, top=0.47, bottom=0.08)

    ax0 = fig.add_subplot(gs1[0])
    ax1 = fig.add_subplot(gs1[1])
    ax2 = fig.add_subplot(gs1[2])
    ax3 = fig.add_subplot(gs2[0])
    ax = [ax0, ax1, ax2, ax3]
    
    #fig, ax = plt.subplots(1, 3, figsize=(15,5.5)) #, gridspec_kw={'width_ratios': [1,1.7,3.2]})
    
    plt.sca(ax[0])
    prelim_mem = pmmem & ~mem
    plt.plot(t['pm_phi1_cosphi2'], t['pm_phi2'], 'o', color=lightsteelblue, mec='none', ms=3, alpha=1, label='Field stars')
    plt.plot(t['pm_phi1_cosphi2'][prelim_mem], t['pm_phi2'][prelim_mem], 'o', color=steelblue, mec='none', ms=6, alpha=1, label='Preliminary\nGD-1 members')
    plt.plot(t['pm_phi1_cosphi2'][mem & stream], t['pm_phi2'][mem & stream], 'o', color=navyblue, mec='none', ms=6, label='GD-1 stream\nmembers')
    plt.plot(t['pm_phi1_cosphi2'][mem & spur], t['pm_phi2'][mem & spur], '*', color=navyblue, mec='none', ms=10, label='GD-1 spur\nmembers')
    
    pm = mpl.patches.Polygon(mem_dict['pmbox'], facecolor='none', edgecolor=fuchsia, lw=3, ls='--', zorder=2)
    plt.gca().add_artist(pm)
    
    #plt.legend(fontsize='small', loc=4, handlelength=0.75)
    plt.xlim(-12,2)
    plt.ylim(-5,5)
    plt.xlabel('$\mu_{\phi_1}$ [mas yr$^{-1}$]')
    plt.ylabel('$\mu_{\phi_2}$ [mas yr$^{-1}$]')
    plt.title('Proper motion', fontsize='medium')
    plt.text(0.1, 0.9, '{:2d}'.format(np.sum(pmmem)), transform=plt.gca().transAxes, ha='left')
    
    plt.sca(ax[1])
    prelim_mem = pmmem & ~mem
    plt.plot(t['g'] - t['i'], t['g'], 'o', color=lightsteelblue, mec='none', ms=3, alpha=1)
    plt.plot(t['g'][prelim_mem] - t['i'][prelim_mem], t['g'][prelim_mem], 'o', color=steelblue, mec='none', ms=6, alpha=1)
    #plt.plot(t['g'][pmmem & stream] - t['i'][pmmem & stream], t['g'][pmmem & stream], 'o', color=navyblue, mec='none', ms=5)
    #plt.plot(t['g'][pmmem & spur] - t['i'][pmmem & spur], t['g'][pmmem & spur], '*', color=navyblue, mec='none', ms=9)
    
    plt.plot(t['g'][mem & stream] - t['i'][mem & stream], t['g'][mem & stream], 'o', color=navyblue, mec='none', ms=6)
    plt.plot(t['g'][mem & spur] - t['i'][mem & spur], t['g'][mem & spur], '*', color=navyblue, mec='none', ms=10)
    #plt.plot(t['g'][mem] - t['i'][mem], t['g'][mem], 'o', color=navyblue, mec='none', ms=5)
    pm = mpl.patches.Polygon(mem_dict['cmdbox'], facecolor='none', edgecolor=fuchsia, lw=3, ls='--', zorder=2)
    plt.gca().add_artist(pm)
    
    plt.xlim(-0.5,1.5)
    plt.xlim(-0.1,1.1)
    plt.ylim(20.6,14.5)
    plt.xlabel('(g - i)$_0$ [mag]')
    plt.ylabel('g$_0$ [mag]')
    plt.title('+ Isochrone', fontsize='medium')
    plt.text(0.1, 0.9, '{:2d}'.format(np.sum(cmdmem & pmmem)), transform=plt.gca().transAxes, ha='left')
    
    plt.sca(ax[2])
    prelim_mem = pmmem & cmdmem & ~mem
    plt.hist(t['delta_Vrad'][~cmdmem & ~pmmem], bins=bvr, histtype='stepfilled', color=lightsteelblue, alpha=1, density=False)
    plt.hist(t['delta_Vrad'][prelim_mem], bins=bvr, histtype='stepfilled', color=steelblue, density=False)
    #plt.hist(t['delta_Vrad'][pmmem & cmdmem], bins=bvr, histtype='stepfilled', color=navyblue, density=False)
    plt.hist(t['delta_Vrad'][mem], bins=bvr, histtype='stepfilled', color=navyblue, density=False)
    
    for vrlim in vrlims:
        plt.axvline(vrlim, ls='--', lw=3, color=fuchsia)
    
    plt.xlim(-50,50)
    plt.ylabel('Number')
    plt.xlabel('$V_r$ - $V_{r,orbit}$ [km s$^{-1}$]')
    plt.title('+ Radial velocity', fontsize='medium')
    plt.text(0.1, 0.9, '{:2d}'.format(np.sum(pmmem & cmdmem & vrmem)), transform=plt.gca().transAxes, ha='left')
    
    plt.sca(ax[3])
    prelim_mem = pmmem & cmdmem & vrmem & ~mem
    #plt.plot(t['FeH'][pmmem & cmdmem & vrmem], t['aFe'][pmmem & cmdmem & vrmem], 'o', color=navyblue, mec='none', ms=6, label='GD-1 members', zorder=1)
    #plt.plot(t['FeH'][mem], t['aFe'][mem], 'o', color=navyblue, mec='none', ms=6, label='GD-1 members', zorder=1)

    plt.plot(t['init_FeH'][~(cmdmem & vrmem)], t['aFe'][~(cmdmem & vrmem)], 'o', color=lightsteelblue, mec='none', alpha=1, ms=4, label='Field stars', zorder=0)
    plt.plot(t['init_FeH'][prelim_mem], t['aFe'][prelim_mem], 'o', color=steelblue, mec='none', alpha=1, ms=7, zorder=0, label='Preliminary GD-1 members')
    plt.plot(t['init_FeH'][mem & stream], t['aFe'][mem & stream], 'o', color=navyblue, mec='none', ms=7, label='GD-1 stream members', zorder=1)
    plt.plot(t['init_FeH'][mem & spur], t['aFe'][mem & spur], '*', color=navyblue, mec='none', ms=12, label='GD-1 spur members', zorder=1)
    plt.errorbar(t['init_FeH'][mem], t['aFe'][mem], yerr=t['std_init_FeH'][mem], xerr=t['std_aFe'][mem], fmt='none', color=navyblue, label='', zorder=0, alpha=0.5, lw=0.7)

    
    for fehlim in fehlims:
        plt.axvline(fehlim, ls='--', lw=3, color=fuchsia, label='', zorder=2)
    
    #plt.text(0.97, 0.9, '{:2d}'.format(np.sum(pmmem & cmdmem & vrmem & fehmem)), transform=plt.gca().transAxes, ha='right')
    plt.text(0.03, 0.9, '{:2d}'.format(np.sum(pmmem & cmdmem & vrmem & fehmem)), transform=plt.gca().transAxes, ha='left')
    plt.legend(loc=1, frameon=True, handlelength=1, fontsize='medium', markerscale=1.2)
    
    plt.xlim(-3.2,0.1)
    plt.ylim(-0.2,0.6)
    plt.ylabel('[$\\alpha$/Fe]')
    plt.xlabel('[Fe/H]$_{init}$')
    plt.title('+ Metallicity selection', fontsize='medium')

    #plt.tight_layout(w_pad=0.1)
    plt.savefig('../paper/members.pdf')

def dvr():
    """"""

    t = Table.read('../data/master_catalog.fits')
    ind = (-t['lnL'] < 2.5E3+t['SNR']**2.4) & (t['SNR']>3)
    t = t[ind]
    mem = get_members(t)
    t = t[mem]
    
    spur = (t['field']==2) | (t['field']==4) | (t['field']==5) | (t['field']==6)
    stream = ~spur
    
    cspur = mpl.cm.Blues_r(0.15)
    cstream = mpl.cm.Blues_r(0.4)
    colors = [cstream, cspur, 'darkorange']
    colors = ['darkorange', 'orangered', 'navy']
    
    labels = ['Bonaca et al. (2020)', '', 'Koposov et al. (2010)']
    colors = ['darkorange', 'orangered', '0.8']
    colors = ['dodgerblue', 'orangered', '0.7']
    ecolors = ['navy', 'red', '0.7']
    
    colors = ['#3dd0e8', '#ff8d3e', '0.7']
    #colors = ['#3dd0e8', '#f84600', '0.7']
    ecolors = ['#3d7be8', '#f82b00', '0.7']
    ecolors = ['#3d7be8', '#f84600', '0.7']
    markers = ['o', '*', 'o']
    sizes = [10, 18, 8]
    msizes = [6.5, 10, 4]
    #msizes = [10, 18, 4]
    ms = 4
    mew = 1.5
    
    tk = Table.read('../data/koposov_vr.dat', format='ascii.commented_header')

    g = Table(fits.getdata('/home/ana/projects/legacy/GD1-DR2/output/gd1_members.fits'))

    pkl = pickle.load(open('../data/orbit_vr_interp.pkl', 'rb'))
    qpoly = pkl['f']
    xphi = np.linspace(-50,-10,100)
    yvr = qpoly(xphi)

    plt.close()
    fig, ax = plt.subplots(3,1,figsize=(10,7), sharex=True, gridspec_kw=dict(height_ratios=[1,1.6,1.4]))
    
    plt.sca(ax[0])
    p1, = plt.plot(tk['phi1'], tk['phi2'], 'o', color=colors[2], alpha=1, label=labels[2], ms=ms+1.5)
    
    plt.sca(ax[1])
    plt.plot(xphi, yvr, '-', color='k', lw=2, alpha=0.7)
    plt.errorbar(tk['phi1'], tk['vr'], yerr=tk['err'], fmt='o', color=colors[2], lw=1.5, alpha=1, label='', ms=ms)
    
    kvr = qpoly(tk['phi1'])
    
    plt.sca(ax[2])
    plt.axhline(0, color='k', lw=2, alpha=0.7, zorder=0)
    plt.errorbar(tk['phi1'], tk['vr'] - kvr, yerr=tk['err'], fmt='o', color=colors[2], lw=1.5, alpha=1, label='', zorder=0, ms=ms)
    
    p2 = []
    
    for e, ind in enumerate([stream, spur]):
        plt.sca(ax[0])
        p_, = plt.plot(t['phi1'][ind], t['phi2'][ind], marker=markers[e], ls='none', color=colors[e], label=labels[e], ms=msizes[e], mec=ecolors[e], mew=mew)
        p2 += [p_]
        
        plt.sca(ax[1])
        plt.errorbar(t['phi1'][ind], t['Vrad'][ind], yerr=(t['lerr_Vrad'][ind], t['uerr_Vrad'][ind]), color=ecolors[e], mfc=colors[e], fmt=markers[e], label='', ms=msizes[e], mec=ecolors[e], mew=mew)
        
        plt.sca(ax[2])
        vr = qpoly(t['phi1'][ind])
        
        plt.errorbar(t['phi1'][ind], t['Vrad'][ind] - vr, yerr=(t['lerr_Vrad'][ind], t['uerr_Vrad'][ind]), fmt=markers[e], color=ecolors[e], mfc=colors[e], zorder=0, lw=2, ms=msizes[e], mec=ecolors[e], mew=mew)
    
    # medians
    phi1_med = np.zeros(8)
    vr_med = np.zeros(8)
    vr_sig = np.zeros(8)
    vr_std = np.zeros(8)
    
    for e, ind in enumerate([stream, spur]):
        plt.sca(ax[2])
        fields = np.unique(t['field'][ind])
        
        for ee, f in enumerate(fields):
            ifield = t['field']==f
            vr = qpoly(t['phi1'][ind & ifield])
            plt.errorbar(np.median(t['phi1'][ind & ifield]), np.median(t['Vrad'][ind & ifield] - vr), yerr=np.std(t['Vrad'][ind & ifield]), fmt='none', color='k', lw=2, zorder=ee+2)
            plt.plot(np.median(t['phi1'][ind & ifield]), np.median(t['Vrad'][ind & ifield] - vr), marker=markers[e], color=colors[e], ms=sizes[e], mec='k', mew=2, zorder=ee+3+e)
            
            phi1_med[f-1] = np.median(t['phi1'][ind & ifield])
            vr_med[f-1] = np.median(t['Vrad'][ind & ifield] - vr)
            vr_sig[f-1] = np.std(t['Vrad'][ind & ifield])
            vr_std[f-1] = np.median(t['std_Vrad'][ind & ifield])
    
    print(vr_med[0] - vr_med[5], vr_std[0], vr_std[5])
    print(vr_med[2] - vr_med[1], vr_std[2], vr_std[1])

    plt.sca(ax[0])
    plt.scatter(g['phi1'], g['phi2'], s=g['pmem']*4, c=g['pmem'], cmap=mpl.cm.binary, vmin=0.5, vmax=1.1, zorder=0, label='')
    plt.ylim(-4,4)
    plt.xlim(-48, -26)
    plt.ylabel('$\phi_2$ [deg]')
    
    plt.legend([p1, (p2[0], p2[1])],[labels[2], labels[0]], handler_map={p2[0]:HandlerLine2D(numpoints=2), p2[1]:HandlerLine2D(numpoints=1)}, ncol=2, frameon=False, handlelength=2.5, loc=3, fontsize='small', numpoints=1)
    
    plt.sca(ax[1])
    plt.ylim(-140,49)
    plt.ylabel('$V_r$ [km s$^{-1}$]')
    
    plt.sca(ax[2])
    plt.ylim(-9,9)
    plt.ylabel('$\Delta V_r$ [km s$^{-1}$]')
    plt.xlabel('$\phi_1$ [deg]')
    
    plt.tight_layout(h_pad=0)
    plt.savefig('../paper/gd1_kinematics.pdf')

def skybox(label='v500w200', N=99856, step=0, colorby='dvr1', dvrcut=False):
    """"""
    
    t = Table.read('../data/perturber_now_{:s}_r{:06d}.fits'.format(label, N))
    
    c = coord.Galactocentric(x=t['x']*u.kpc, y=t['y']*u.kpc, z=t['z']*u.kpc, v_x=t['vx']*u.km/u.s, v_y=t['vy']*u.km/u.s, v_z=t['vz']*u.km/u.s, **gc_frame_dict)
    ceq = c.transform_to(coord.ICRS)
    cgal = c.transform_to(coord.Galactic)
    
    w0 = gd.PhaseSpacePosition(c.transform_to(gc_frame).cartesian)
    orbit = ham.integrate_orbit(w0, dt=1., n_steps=1)
    epot = orbit.potential_energy()[0,:]
    ekin = orbit.kinetic_energy()[0,:]
    energy = orbit.energy()[0,:]
    
    label = 'GD-1 perturber now'
    #ind = (np.abs(t['dvr1'])<0.5) & (np.abs(t['dvr2'])<0.5)
    
    cplane = coord.Galactic(l=np.linspace(0,360,100)*u.deg, b=np.zeros(100)*u.deg)
    cplane_eq = cplane.transform_to(coord.ICRS)
    
    g = Table(fits.getdata('/home/ana/projects/legacy/GD1-DR2/output/gd1_members.fits'))
    cgd1 = gc.GD1(phi1=g['phi1']*u.deg, phi2=g['phi2']*u.deg)
    cgd1_eq = cgd1.transform_to(coord.ICRS)
    
    
    # sgr models
    
    # law & majewski (2010)
    tsgr = Table.read('/home/ana/projects/h3/data/SgrTriax_DYN.dat.gz', format='ascii')
    tsgr = tsgr[::10]
    c_sgr = coord.ICRS(ra=tsgr['ra']*u.deg, dec=tsgr['dec']*u.deg, distance=tsgr['dist']*u.kpc, pm_ra_cosdec=tsgr['mua']*u.mas/u.yr, pm_dec=tsgr['mud']*u.mas/u.yr)
    vr = gc.vgsr_to_vhel(c_sgr, tsgr['vgsr']*u.km/u.s)
    c_sgr = coord.ICRS(ra=tsgr['ra']*u.deg, dec=tsgr['dec']*u.deg, distance=tsgr['dist']*u.kpc, pm_ra_cosdec=tsgr['mua']*u.mas/u.yr, pm_dec=tsgr['mud']*u.mas/u.yr, radial_velocity=vr)
    
    # dierickx & loeb (2017)
    tdm = Table.read('../data/DL17_DM.fits')
    cdm = coord.SkyCoord(ra=tdm['ra']*u.deg, dec=tdm['dec']*u.deg, frame='icrs')
    ts = Table.read('../data/DL17_Stars.fits')
    cs = coord.SkyCoord(ra=ts['ra']*u.deg, dec=ts['dec']*u.deg, frame='icrs')
    
    # color-coding
    clr = t['dvr1']
    clabel0 = '$\Delta V_{r,stream-spur}$ [km s$^-1$]'
    clabel0 = '$\Delta V_{r}$ [km s$^-1$]'
    cmap = 'twilight'
    vmin = -5
    vmax = 5

    plt.close()
    fig, ax = plt.subplots(2,2,figsize=(12,6.5), subplot_kw=dict(projection='mollweide'))
    
    plt.sca(ax[0][0])
    isort_clr = np.argsort(clr)[::-1]
    im0 = plt.scatter(ceq.ra.wrap_at(wangle).radian, ceq.dec.radian, rasterized=True, c=clr, zorder=0, s=2, ec='none', cmap=cmap, vmin=vmin, vmax=vmax, label=label)
    
    font_tick = 'x-small'
    plt.xticks(fontsize=font_tick)
    plt.yticks(fontsize=font_tick)
    
    plt.xlabel('R.A. [deg]', fontsize='small')
    plt.ylabel('Dec [deg]', fontsize='small')
    plt.title('Objects producing a GD-1 spur', fontsize=17, pad=15)

    plt.sca(ax[0][1])
    
    ind = (np.abs(t['dvr1'])<1) & (np.abs(t['dvr2'])<1)
    #ind = (t['dvr1']>0.7) & (t['dvr1']<2.7) & (np.abs(t['dvr2'])<1)
    ind_bound = ekin<epot
    vsub = np.sqrt(t['vxsub']**2 + t['vysub']**2)
    t = t[ind & ind_bound]
    
    # color by: perturber velocity
    clr = np.sqrt(t['vxsub']**2 + t['vysub']**2)
    clabel = 'Perturber velocity [km s$^{-1}$]'
    cmap = 'magma'
    vmin = 0
    vmax = 500
    
    # color by: proper motion projection
    clr = t['dmu22']
    clabel = '$\Delta \mu_{\phi_2,stream-spur}$ [mas yr$^-1$]'
    clabel = '$\Delta \mu_{\phi_2}$ [mas yr$^-1$]'
    cmap = 'magma_r'
    vmin = -0.4
    vmax = -0.1

    c = coord.Galactocentric(x=t['x']*u.kpc, y=t['y']*u.kpc, z=t['z']*u.kpc, v_x=t['vx']*u.km/u.s, v_y=t['vy']*u.km/u.s, v_z=t['vz']*u.km/u.s, **gc_frame_dict)
    ceq = c.transform_to(coord.ICRS)
    cgal = c.transform_to(coord.Galactic)
    
    w0 = gd.PhaseSpacePosition(c.transform_to(gc_frame).cartesian)
    orbit = ham.integrate_orbit(w0, dt=1., n_steps=1)
    epot = orbit.potential_energy()[0,:]
    ekin = orbit.kinetic_energy()[0,:]
    energy = orbit.energy()[0,:]
    label += '\n$|\Delta$ $V_r$| < 1 km s$^{-1}$'
    
    isort_clr = np.argsort(clr)[::-1]
    im = plt.scatter(ceq.ra.wrap_at(wangle).radian[isort_clr], ceq.dec.radian[isort_clr], rasterized=True, c=clr[isort_clr], zorder=0, s=2, ec='none', cmap=cmap, vmin=vmin, vmax=vmax, label=label)
    
    plt.xticks(fontsize=font_tick)
    plt.yticks(fontsize=font_tick)
    plt.xlabel('R.A. [deg]', fontsize='small')
    plt.ylabel('Dec [deg]', fontsize='small')
    plt.title('Objects producing a comoving GD-1 spur', fontsize=17, pad=15)
    
    
    plt.sca(ax[1][0])
    #plt.plot(cdm.ra.wrap_at(wangle).radian, cdm.dec.radian, 'o', color='0.2', ms=2, mew=0, alpha=0.3, rasterized=True, label='Sagittarius dark matter\n(Dierickx & Loeb 2017)')
    plt.plot(cdm.ra.wrap_at(wangle).radian, cdm.dec.radian, 'o', color='0.2', ms=2, mew=0, alpha=0.3, rasterized=True, label='Dierickx & Loeb (2017)')
    
    # running median
    Nbin = 30
    ra_ed = np.linspace(-180,180,Nbin+1)*u.deg
    ra_med = 0.5*(ra_ed[1:] + ra_ed[:-1])
    dec_med = np.zeros(Nbin)*u.deg
    dec_up = np.zeros(Nbin)*u.deg
    dec_dn = np.zeros(Nbin)*u.deg

    # in sgr coordinates
    csgr = cdm.transform_to(gc.Sagittarius)
    for i in range(Nbin):
        ind = (csgr.Lambda.wrap_at(wangle)>ra_ed[i]) & (csgr.Lambda.wrap_at(wangle)<ra_ed[i+1])
        dec_med[i] = np.median(csgr.Beta[ind])
        dec_dn[i], dec_up[i] = np.percentile(csgr.Beta[ind], [25,75])*u.deg
    
    cmed_sgr = gc.Sagittarius(Lambda=ra_med, Beta=dec_med)
    cup_sgr = gc.Sagittarius(Lambda=ra_med, Beta=dec_up)
    cdn_sgr = gc.Sagittarius(Lambda=ra_med, Beta=dec_dn)
    
    cmed = cmed_sgr.transform_to(coord.ICRS)
    cup = cup_sgr.transform_to(coord.ICRS)
    cdn = cdn_sgr.transform_to(coord.ICRS)
    
    isort = np.argsort(cmed.ra.wrap_at(wangle).radian)
    cmed = cmed[isort]
    isort = np.argsort(cup.ra.wrap_at(wangle).radian)
    cup = cup[isort]
    isort = np.argsort(cdn.ra.wrap_at(wangle).radian)
    cdn = cdn[isort]
    
    plt.plot(cmed.ra.wrap_at(wangle).radian, cmed.dec.radian, 'k-', lw=2, alpha=0.8, label='Median')
    plt.plot(cup.ra.wrap_at(wangle).radian, cup.dec.radian, 'k-', lw=2, alpha=0.5, label='Interquartile range')
    plt.plot(cdn.ra.wrap_at(wangle).radian, cdn.dec.radian, 'k-', lw=2, alpha=0.5, label='')
    #plt.fill_between(cdn.ra.wrap_at(wangle).radian, cdn.dec.radian, cup.dec.radian, color='k', alpha=0.3)
    #plt.plot(ra_med, dec_med, 'k-', lw=2)
    
    plt.legend(loc=4, fontsize='x-small', markerscale=2, handlelength=0.5)
    plt.xticks(fontsize=font_tick)
    plt.yticks(fontsize=font_tick)
    plt.xlabel('R.A. [deg]', fontsize='small')
    plt.ylabel('Dec [deg]', fontsize='small')
    plt.title('Simulated Sagittarius dark matter debris', fontsize=17, pad=15)
    
    plt.sca(ax[1][1])
    isort_clr = np.argsort(clr)[::-1]
    im = plt.scatter(ceq.ra.wrap_at(wangle).radian[isort_clr], ceq.dec.radian[isort_clr], rasterized=True, c=clr[isort_clr], zorder=0, s=2, ec='none', cmap=cmap, vmin=vmin, vmax=vmax, label='')
    plt.plot(cmed.ra.wrap_at(wangle).radian, cmed.dec.radian, 'k-', lw=2, alpha=0.8)
    plt.plot(cup.ra.wrap_at(wangle).radian, cup.dec.radian, 'k-', lw=2, alpha=0.5)
    plt.plot(cdn.ra.wrap_at(wangle).radian, cdn.dec.radian, 'k-', lw=2, alpha=0.5)
    #plt.plot(cdm.ra.wrap_at(wangle).radian, cdm.dec.radian, 'o', color='0.2', ms=2, mew=0, alpha=0.3, rasterized=True, label='Sagittarius dark matter\n(Dierickx & Loeb 2017)')
    
    ra0 = coord.Angle(65*u.deg)
    dec0 = coord.Angle(-10*u.deg)
    plt.text(ra0.radian, dec0.radian, 'Sagittarius', rotation=-17, fontsize=15, bbox=dict(facecolor='w', ec='none', alpha=0.7, boxstyle='round', pad=0.2))
    
    #plt.legend(loc=4, fontsize='x-small', markerscale=2, handlelength=0.5)
    plt.xticks(fontsize=font_tick)
    plt.yticks(fontsize=font_tick)
    plt.xlabel('R.A. [deg]', fontsize='small')
    plt.ylabel('Dec [deg]', fontsize='small')
    plt.title('Objects producing a comoving GD-1 spur + Sagittarius', fontsize=17, pad=15)
    
    plt.tight_layout(w_pad=5)
    
    # colorbars
    plt.sca(ax[0][0])
    pos = plt.gca().get_position()
    cax = plt.axes([pos.x1+0.007,pos.y0,0.013,pos.y1 - pos.y0])
    plt.colorbar(im0, cax=cax, ticks=[-5,-2.5,0,2.5,5])
    plt.yticks(fontsize=font_tick)
    plt.ylabel(clabel0, fontsize='small')
    
    plt.sca(ax[0][1])
    pos = plt.gca().get_position()
    cax = plt.axes([pos.x1+0.007,pos.y0,0.013,pos.y1 - pos.y0])
    cb = plt.colorbar(im, cax=cax, ticks=np.linspace(-0.4,-0.1,4))
    plt.yticks(fontsize=font_tick)
    cb.set_label(clabel, fontsize='small')
    
    plt.sca(ax[1][1])
    pos = plt.gca().get_position()
    cax = plt.axes([pos.x1+0.007,pos.y0,0.013,pos.y1 - pos.y0])
    plt.colorbar(im, cax=cax, ticks=np.linspace(-0.4,-0.1,4))
    plt.yticks(fontsize=font_tick)
    plt.ylabel(clabel, fontsize='small')

    #plt.savefig('../paper/skybox.png')
    plt.savefig('../paper/skybox.pdf')


##########
# For text

def spectra():
    """"""
    t = Table.read('../data/master_catalog.fits')
    print(np.sum(-t['lnL'] < 2.5E3+t['SNR']**2.4), np.sum(t['SNR']>3), np.sum(np.isfinite(t['aFe'])))
    ind = (-t['lnL'] < 2.5E3+t['SNR']**2.4) & (t['SNR']>3) & np.isfinite(t['aFe'])
    t = t[ind]
    print('{:d} good spectra'.format(len(t)))
    
    for k in ['Vrad', 'FeH', 'aFe']:
        print(k, np.median(t['std_{:s}'.format(k)]), np.percentile(t['std_{:s}'.format(k)], [90]))

def members(snr=3):
    """"""
    t = Table.read('../data/master_catalog.fits')
    ind = (-t['lnL'] < 2.5E3+t['SNR']**2.4) & (t['SNR']>snr) & np.isfinite(t['aFe'])
    t = t[ind]

    mem = get_members(t)
    t = t[mem]
    
    print('chemistry')
    for k in ['FeH', 'init_FeH', 'aFe']:
        print(k, np.median(t['{:s}'.format(k)]), np.std(t['{:s}'.format(k)]))
    
    print('uncertainties')
    for k in ['Vrad', 'FeH', 'init_FeH', 'aFe']:
        print(k, np.median(t['std_{:s}'.format(k)]), np.percentile(t['std_{:s}'.format(k)], [90]))

def publish_catalog():
    """Produce catalog with good spectra, membership columns"""
    
    t = Table.read('../data/master_catalog.fits')
    ind = (-t['lnL'] < 2.5E3+t['SNR']**2.4) & (t['SNR']>3) & np.isfinite(t['aFe'])
    t = t[ind]

    mem_dict = get_members(t, full=True)
    t['mem'] = mem_dict['mem']
    t['cmdmem'] = mem_dict['cmdmem']
    t['pmmem'] = mem_dict['pmmem']
    t['vrmem'] = mem_dict['vrmem']
    t['fehmem'] = mem_dict['fehmem']
    
    t.pprint()
    t.write('../data/catalog.fits', overwrite=True)

def occupation(nside=64):
    """"""
    label = 'v500w200'
    N = 99856
    t = Table.read('../data/perturber_now_{:s}_r{:06d}.fits'.format(label, N))
    c = coord.Galactocentric(x=t['x']*u.kpc, y=t['y']*u.kpc, z=t['z']*u.kpc, v_x=t['vx']*u.km/u.s, v_y=t['vy']*u.km/u.s, v_z=t['vz']*u.km/u.s, **gc_frame_dict)
    ceq = c.transform_to(coord.ICRS)
    
    cgal = c.transform_to(coord.Galactic)
    w0 = gd.PhaseSpacePosition(c.transform_to(gc_frame).cartesian)
    orbit = ham.integrate_orbit(w0, dt=1., n_steps=1)
    epot = orbit.potential_energy()[0,:]
    ekin = orbit.kinetic_energy()[0,:]
    
    ind = (np.abs(t['dvr1'])<1) & (np.abs(t['dvr2'])<1)
    ind_bound = ekin<epot
    ceq2 = ceq[ind & ind_bound]
    #vsub = np.sqrt(t['vxsub']**2 + t['vysub']**2)
    #t = t[ind & ind_bound]
    
    res = (hp.nside2resol(nside, arcmin=True)*u.arcmin).to(u.deg)
    area = hp.nside2pixarea(nside, degrees=True)
    ntot = hp.nside2npix(nside)
    print('{:.1f} deg^2'.format(area))
    
    ipix = hp.ang2pix(nside, ceq.ra.degree, ceq.dec.degree, lonlat=True)
    n0 = np.size(np.unique(ipix))
    
    ipix_dvr = hp.ang2pix(nside, ceq2.ra.degree, ceq2.dec.degree, lonlat=True)
    ndvr = np.size(np.unique(ipix_dvr))
    
    print(n0/ntot, ndvr/ntot, ndvr/n0)
    print(n0*area, ndvr*area)

def afe_spread():
    """"""
    t = Table.read('../data/master_catalog.fits')
    ind = (-t['lnL'] < 2.5E3+t['SNR']**2.4) & (t['SNR']>3) & np.isfinite(t['aFe'])
    t = t[ind]

    mem = get_members(t)
    t = t[mem]
    
    plt.close()
    fig, ax = plt.subplots(1,3,figsize=(15,5))
    
    plt.sca(ax[0])
    plt.plot(t['logg'], t['aFe'], 'ko')
    plt.xlabel('log g')
    plt.ylabel('[$\\alpha$/Fe]')
    
    plt.sca(ax[1])
    plt.plot(t['Teff'], t['aFe'], 'ko')
    plt.xlabel('$T_{eff}$ [K]')
    plt.ylabel('[$\\alpha$/Fe]')
    
    plt.sca(ax[2])
    plt.scatter(t['g'] - t['i'], t['g'], c=t['aFe'], cmap='gray', vmax=0.7)
    #plt.scatter(t['g'] - t['i'], t['g'], c=t['FeH'], cmap='gray')
    
    plt.xlim(0.1,0.7)
    plt.ylim(21,16)
    plt.xlabel('g - i')
    plt.ylabel('g')
    
    plt.tight_layout()
    plt.savefig('../plots/afe_correlations.png')

def init_abundances():
    """"""
    t = Table.read('../data/master_catalog.fits')
    ind = (-t['lnL'] < 2.5E3+t['SNR']**2.4) & (t['SNR']>3) & np.isfinite(t['aFe'])
    t = t[ind]

    mem = get_members(t)
    t = t[mem]
    
    plt.close()
    fig, ax = plt.subplots(1,1,figsize=(10,5))
    
    plt.plot(t['FeH'], t['aFe'], 'ko')
    plt.plot(t['init_FeH'], t['aFe'], 'wo', mec='k')
    print(t.colnames)
    
    dfeh = t['init_FeH'] - t['FeH']
    print(dfeh, np.median(dfeh), np.median(t['FeH']), np.median(t['init_FeH']), np.std(t['FeH']), np.std(t['init_FeH']))
    
    plt.tight_layout()


##########
# Response

def reformat_lamost():
    """"""
    
    tl = Table.read('../data/lamost_gd1.txt', format='ascii')
    tl.pprint()
    
    coord_list = [tl['ID'][i][1:12] + ' ' + tl['ID'][i][12:] for i in range(len(tl))]

    c = coord.SkyCoord(coord_list, unit=(u.hourangle, u.deg))
    cg = c.transform_to(gc.GD1)
    
    plt.close()
    plt.figure()
    
    plt.plot(cg.phi1, tl['vLOS'], 'ko')
    
    tout = Table([cg.phi1, cg.phi2, tl['vLOS'], tl['e_vLOS']], names=('phi1', 'phi2', 'vr', 'err'))
    tout.pprint()
    tout.write('../data/lamost_vr.fits', overwrite=True)

def dvr_lamost():
    """"""

    t = Table.read('../data/master_catalog.fits')
    ind = (-t['lnL'] < 2.5E3+t['SNR']**2.4) & (t['SNR']>3)
    t = t[ind]
    mem = get_members(t)
    t = t[mem]
    
    spur = (t['field']==2) | (t['field']==4) | (t['field']==5) | (t['field']==6)
    stream = ~spur
    
    cspur = mpl.cm.Blues_r(0.15)
    cstream = mpl.cm.Blues_r(0.4)
    colors = [cstream, cspur, 'darkorange']
    colors = ['darkorange', 'orangered', 'navy']
    
    labels = ['Bonaca et al. (2020)', 'Huang et al. (2019)', 'Koposov et al. (2010)']
    colors = ['darkorange', 'orangered', '0.8']
    colors = ['dodgerblue', 'orangered', '0.7']
    ecolors = ['navy', 'red', '0.7']
    
    colors = ['#3dd0e8', '#ff8d3e', '0.7']
    #colors = ['#3dd0e8', '#f84600', '0.7']
    ecolors = ['#3d7be8', '#f82b00', '0.7']
    ecolors = ['#3d7be8', '#f84600', '0.7']
    markers = ['o', '*', 'o']
    sizes = [10, 18, 8]
    msizes = [6.5, 10, 4]
    #msizes = [10, 18, 4]
    ms = 4
    mew = 1.5
    
    tk = Table.read('../data/koposov_vr.dat', format='ascii.commented_header')
    tl = Table.read('../data/lamost_vr.fits')

    g = Table(fits.getdata('/home/ana/projects/legacy/GD1-DR2/output/gd1_members.fits'))

    pkl = pickle.load(open('../data/orbit_vr_interp.pkl', 'rb'))
    qpoly = pkl['f']
    xphi = np.linspace(-50,-10,100)
    yvr = qpoly(xphi)

    plt.close()
    fig, ax = plt.subplots(3,1,figsize=(10,7), sharex=True, gridspec_kw=dict(height_ratios=[1,1.6,1.4]))
    
    plt.sca(ax[0])
    p1, = plt.plot(tk['phi1'], tk['phi2'], 'o', color=colors[2], alpha=1, label=labels[2], ms=ms+1.5)
    p1b, = plt.plot(tl['phi1'], tl['phi2'], 's', color=colors[2], alpha=1, label=labels[2], ms=ms+1.5)
    
    plt.sca(ax[1])
    plt.plot(xphi, yvr, '-', color='k', lw=2, alpha=0.7)
    plt.errorbar(tk['phi1'], tk['vr'], yerr=tk['err'], fmt='o', color=colors[2], lw=1.5, alpha=1, label='', ms=ms)
    plt.errorbar(tl['phi1'], tl['vr'], yerr=tl['err'], fmt='s', color=colors[2], lw=1.5, alpha=1, label='', ms=ms)
    
    kvr = qpoly(tk['phi1'])
    lvr = qpoly(tl['phi1'])
    
    plt.sca(ax[2])
    plt.axhline(0, color='k', lw=2, alpha=0.7, zorder=0)
    plt.errorbar(tk['phi1'], tk['vr'] - kvr, yerr=tk['err'], fmt='o', color=colors[2], lw=1.5, alpha=1, label='', zorder=0, ms=ms)
    plt.errorbar(tl['phi1'], tl['vr'] - lvr, yerr=tl['err'], fmt='s', color=colors[2], lw=1.5, alpha=1, label='', zorder=0, ms=ms)
    
    p2 = []
    
    for e, ind in enumerate([stream, spur]):
        plt.sca(ax[0])
        p_, = plt.plot(t['phi1'][ind], t['phi2'][ind], marker=markers[e], ls='none', color=colors[e], label=labels[e], ms=msizes[e], mec=ecolors[e], mew=mew)
        p2 += [p_]
        
        plt.sca(ax[1])
        plt.errorbar(t['phi1'][ind], t['Vrad'][ind], yerr=(t['lerr_Vrad'][ind], t['uerr_Vrad'][ind]), color=ecolors[e], mfc=colors[e], fmt=markers[e], label='', ms=msizes[e], mec=ecolors[e], mew=mew)
        
        plt.sca(ax[2])
        vr = qpoly(t['phi1'][ind])
        
        plt.errorbar(t['phi1'][ind], t['Vrad'][ind] - vr, yerr=(t['lerr_Vrad'][ind], t['uerr_Vrad'][ind]), fmt=markers[e], color=ecolors[e], mfc=colors[e], zorder=0, lw=2, ms=msizes[e], mec=ecolors[e], mew=mew)
    
    # medians
    phi1_med = np.zeros(8)
    vr_med = np.zeros(8)
    vr_sig = np.zeros(8)
    vr_std = np.zeros(8)
    
    for e, ind in enumerate([stream, spur]):
        plt.sca(ax[2])
        fields = np.unique(t['field'][ind])
        
        for ee, f in enumerate(fields):
            ifield = t['field']==f
            vr = qpoly(t['phi1'][ind & ifield])
            plt.errorbar(np.median(t['phi1'][ind & ifield]), np.median(t['Vrad'][ind & ifield] - vr), yerr=np.std(t['Vrad'][ind & ifield]-vr), fmt='none', color='k', lw=2, zorder=ee+2)
            plt.plot(np.median(t['phi1'][ind & ifield]), np.median(t['Vrad'][ind & ifield] - vr), marker=markers[e], color=colors[e], ms=sizes[e], mec='k', mew=2, zorder=ee+3+e)
            
            print(np.median(t['phi1'][ind & ifield]), np.median(t['Vrad'][ind & ifield] - vr), np.std(t['Vrad'][ind & ifield]-vr))
            #print(t['Vrad'][ind & ifield])
            
            phi1_med[f-1] = np.median(t['phi1'][ind & ifield])
            vr_med[f-1] = np.median(t['Vrad'][ind & ifield] - vr)
            vr_sig[f-1] = np.std(t['Vrad'][ind & ifield]-vr)
            vr_std[f-1] = np.median(t['std_Vrad'][ind & ifield])
    
    print(vr_med[0] - vr_med[5], vr_std[0], vr_std[5])
    print(vr_med[2] - vr_med[1], vr_std[2], vr_std[1])
    #print(vr_std)

    plt.sca(ax[0])
    plt.scatter(g['phi1'], g['phi2'], s=g['pmem']*4, c=g['pmem'], cmap=mpl.cm.binary, vmin=0.5, vmax=1.1, zorder=0, label='')
    plt.ylim(-4,4)
    plt.xlim(-48, -26)
    #plt.xlim(-58, -26)
    plt.ylabel('$\phi_2$ [deg]')
    
    plt.legend([p1, p1b, (p2[0], p2[1])],[labels[2], labels[1], labels[0]], handler_map={p2[0]:HandlerLine2D(numpoints=2), p2[1]:HandlerLine2D(numpoints=1)}, ncol=3, frameon=False, handlelength=2, loc=3, fontsize='small', numpoints=1)
    
    plt.sca(ax[1])
    plt.ylim(-140,49)
    #plt.ylim(-140,70)
    plt.ylabel('$V_r$ [km s$^{-1}$]')
    
    plt.sca(ax[2])
    plt.ylim(-9,9)
    plt.ylabel('$\Delta V_r$ [km s$^{-1}$]')
    plt.xlabel('$\phi_1$ [deg]')
    
    plt.tight_layout(h_pad=0)
    plt.savefig('../paper/gd1_kinematics.pdf')

def overlaps():
    """"""
    tk = Table.read('../data/koposov_vr.dat', format='ascii.commented_header')
    ck = gc.GD1(phi1=tk['phi1']*u.deg, phi2=tk['phi2']*u.deg)

    tk = Table.read('../data/lamost_vr.fits')
    ck = gc.GD1(phi1=tk['phi1'], phi2=tk['phi2'])
    ckeq = ck.transform_to(coord.ICRS)
    ckeq = coord.SkyCoord(ra=ckeq.ra, dec=ckeq.dec)
    
    t = Table.read('../data/master_catalog.fits')
    ind = (-t['lnL'] < 2.5E3+t['SNR']**2.4) & (t['SNR']>3)
    t = t[ind]
    mem = get_members(t)
    t = t[mem]
    c = gc.GD1(phi1=t['phi1'], phi2=t['phi2'])
    ceq = c.transform_to(coord.ICRS)
    ceq = coord.SkyCoord(ra=ceq.ra, dec=ceq.dec)
    
    idx, d2d, d3d = ceq.match_to_catalog_sky(ckeq)
    
    print(np.min(d2d))
    plt.close()
    #fig, ax = plt.subplots(1,2, figsize=(10,5))
    #plt.sca(ax[0])
    #plt.hist(d2d.to(u.arcsec).value, bins=np.linspace(0,2,20))
    
    #plt.sca(ax[1])
    
    fig, ax = plt.subplots(2,1,figsize=(10,7), sharex=True)
    
    plt.sca(ax[0])
    imatch = d2d<0.5*u.arcsec
    #print(np.shape(idx), len(t), len(tk))
    plt.errorbar(t['phi1'][imatch], t['Vrad'][imatch], yerr=t['std_Vrad'][imatch], fmt='o', label='Hectochelle (Bonaca et al. 2020)')
    plt.errorbar(tk['phi1'][idx[imatch]], tk['vr'][idx[imatch]], yerr=tk['err'][idx[imatch]], fmt='o', label='LAMOST (Huang et al. 2019)')
    
    plt.legend()
    plt.ylabel('$V_r$ [km s$^{-1}$]')

    dvr = t['Vrad'][imatch] - tk['vr'][idx[imatch]]
    dvr_err = np.sqrt(tk['err'][idx[imatch]]**2 + t['std_Vrad'][imatch]**2)
    #print(dvr, dvr_err, dvr/dvr_err)
    
    plt.sca(ax[1])
    plt.axhline(0, color='k')
    plt.errorbar(t['phi1'][imatch], dvr, yerr=dvr_err, fmt='o')
    
    plt.xlabel('$\phi_1$ [deg]')
    plt.ylabel('$\Delta$ $V_r$ [km s$^{-1}$]')
    
    plt.tight_layout()
    plt.savefig('../paper/response_dvr.png')

def perturber_properties(label='v500w200', N=99856, p=5):
    """Print the range of allowed perturber masses, sizes, impact parameters, impact times"""
    
    # spatial
    sampler = np.load('../../gd1_spur/data/unique_samples_v500w200.npz')
    
    models = np.unique(sampler['chain'], axis=0)
    models = sampler['chain']
    lnp = sampler['lnp']
    pp = np.percentile(lnp, p)
    
    ind = lnp>=pp
    models = models[ind]
    lnp = lnp[ind]
    
    print(np.shape(models))
    params = ['Timpact', 'bx', 'by', 'vxsub', 'vysub', 'M', 'rs', 'Tgap']
    tall = Table(models, names=params)
    tall['vsub'] = np.sqrt(tall['vxsub']**2 + tall['vysub']**2)
    tall['b'] = np.sqrt(tall['bx']**2 + tall['by']**2)
    
    # spatial + rv
    t = Table.read('../data/perturber_now_{:s}_r{:06d}.fits'.format(label, N))
    
    c = coord.Galactocentric(x=t['x']*u.kpc, y=t['y']*u.kpc, z=t['z']*u.kpc, v_x=t['vx']*u.km/u.s, v_y=t['vy']*u.km/u.s, v_z=t['vz']*u.km/u.s, **gc_frame_dict)
    ceq = c.transform_to(coord.ICRS)
    cgal = c.transform_to(coord.Galactic)
    
    w0 = gd.PhaseSpacePosition(c.transform_to(gc_frame).cartesian)
    orbit = ham.integrate_orbit(w0, dt=1., n_steps=1)
    epot = orbit.potential_energy()[0,:]
    ekin = orbit.kinetic_energy()[0,:]
    
    ind = (np.abs(t['dvr1'])<1) & (np.abs(t['dvr2'])<1)
    ind_bound = ekin<epot
    
    t['vsub'] = np.sqrt(t['vxsub']**2 + t['vysub']**2)
    t['b'] = np.sqrt(t['bx']**2 + t['by']**2)
    
    keys = ['M', 'rs', 'b', 'Timpact', 'vsub']
    percentiles = [0.001,50,99.999]
    print_fmt = ['{:.2g}'] * len(percentiles)
    print_fmt = ' '.join(print_fmt)
    
    for k in keys:
        p = np.percentile(t[k], percentiles)
        pp = '{:s} ' + print_fmt
        print(pp.format(k, *p))
        
        #p = np.percentile(t[k], percentiles)
        #pp = '  ' + print_fmt
        #print(pp.format(*p))
        
        p = np.percentile(t[k][ind & ind_bound], percentiles)
        pp = '  ' + print_fmt
        print(pp.format(*p))

def skybox_quivers(label='v500w200', N=99856, step=1, colorby='dvr1', dvrcut=False):
    """"""
    
    t = Table.read('../data/perturber_now_{:s}_r{:06d}.fits'.format(label, N))
    
    c = coord.Galactocentric(x=t['x']*u.kpc, y=t['y']*u.kpc, z=t['z']*u.kpc, v_x=t['vx']*u.km/u.s, v_y=t['vy']*u.km/u.s, v_z=t['vz']*u.km/u.s, **gc_frame_dict)
    ceq = c.transform_to(coord.ICRS)
    cgal = c.transform_to(coord.Galactic)
    
    w0 = gd.PhaseSpacePosition(c.transform_to(gc_frame).cartesian)
    orbit = ham.integrate_orbit(w0, dt=1., n_steps=1)
    epot = orbit.potential_energy()[0,:]
    ekin = orbit.kinetic_energy()[0,:]
    energy = orbit.energy()[0,:]
    
    label = 'GD-1 perturber now'
    
    ind = (np.abs(t['dvr1'])<0.5) & (np.abs(t['dvr2'])<0.5)
    
    if step>1:
        ind = (np.abs(t['dvr1'])<1) & (np.abs(t['dvr2'])<1)
        ind_bound = ekin<epot
        vsub = np.sqrt(t['vxsub']**2 + t['vysub']**2)
        t = t[ind & ind_bound]
    
        c = coord.Galactocentric(x=t['x']*u.kpc, y=t['y']*u.kpc, z=t['z']*u.kpc, v_x=t['vx']*u.km/u.s, v_y=t['vy']*u.km/u.s, v_z=t['vz']*u.km/u.s, **gc_frame_dict)
        ceq = c.transform_to(coord.ICRS)
        cgal = c.transform_to(coord.Galactic)
        
        w0 = gd.PhaseSpacePosition(c.transform_to(gc_frame).cartesian)
        orbit = ham.integrate_orbit(w0, dt=1., n_steps=1)
        epot = orbit.potential_energy()[0,:]
        ekin = orbit.kinetic_energy()[0,:]
        energy = orbit.energy()[0,:]
        
        label += '\n$|\Delta$ $V_r$| < 1 km s$^{-1}$'
    
    # reflex motion correction
    ceq = gc.reflex_correct(ceq, galactocentric_frame=gc_frame)
    
    cplane = coord.Galactic(l=np.linspace(0,360,100)*u.deg, b=np.zeros(100)*u.deg)
    cplane_eq = cplane.transform_to(coord.ICRS)
    
    tsgr = Table.read('/home/ana/projects/h3/data/SgrTriax_DYN.dat.gz', format='ascii')
    tsgr = tsgr[::10]
    c_sgr = coord.ICRS(ra=tsgr['ra']*u.deg, dec=tsgr['dec']*u.deg, distance=tsgr['dist']*u.kpc, pm_ra_cosdec=tsgr['mua']*u.mas/u.yr, pm_dec=tsgr['mud']*u.mas/u.yr)
    #vr = gc.vgsr_to_vhel(c_sgr, tsgr['vgsr']*u.km/u.s)
    #c_sgr = coord.ICRS(ra=tsgr['ra']*u.deg, dec=tsgr['dec']*u.deg, distance=tsgr['dist']*u.kpc, pm_ra_cosdec=tsgr['mua']*u.mas/u.yr, pm_dec=tsgr['mud']*u.mas/u.yr, radial_velocity=vr)
    
    g = Table(fits.getdata('/home/ana/projects/legacy/GD1-DR2/output/gd1_members.fits'))
    cgd1 = gc.GD1(phi1=g['phi1']*u.deg, phi2=g['phi2']*u.deg)
    cgd1_eq = cgd1.transform_to(coord.ICRS)
    
    tdm = Table.read('../data/DL17_DM.fits')
    #cdm = coord.SkyCoord(ra=tdm['ra']*u.deg, dec=tdm['dec']*u.deg, frame='icrs')
    cdm_gal = coord.Galactocentric(x=tdm['X_gal']*u.kpc, y=tdm['Y_gal']*u.kpc, z=tdm['Z_gal']*u.kpc, v_x=tdm['Vx_gal']*u.km/u.s, v_y=tdm['Vy_gal']*u.km/u.s, v_z=tdm['Vz_gal']*u.km/u.s, **gc_frame_dict)
    cdm = cdm_gal.transform_to(coord.ICRS)

    ts = Table.read('../data/DL17_Stars.fits')
    #cs = coord.SkyCoord(ra=ts['ra']*u.deg, dec=ts['dec']*u.deg, frame='icrs')
    cs_gal = coord.Galactocentric(x=ts['X_gal']*u.kpc, y=ts['Y_gal']*u.kpc, z=ts['Z_gal']*u.kpc, v_x=ts['Vx_gal']*u.km/u.s, v_y=ts['Vy_gal']*u.km/u.s, v_z=ts['Vz_gal']*u.km/u.s, **gc_frame_dict)
    cs = cs_gal.transform_to(coord.ICRS)
    
    #cs = c_sgr
    
    # coloring
    if colorby=='mass':
        clr = t['M']
        clabel = 'log M$_{perturb}$ / M$_\odot$'
        cmap = 'magma'
        vmin = 6.2
        vmax = 7.2
    elif colorby=='rs':
        clr = t['rs']
        clabel = 'Scale size [pc]'
        cmap = 'magma'
        vmin = 0
        vmax = 10
    elif colorby=='b':
        clr = np.sqrt(t['bx']**2 + t['by']**2)
        clabel = 'Impact parameter [pc]'
        cmap = 'magma'
        vmin = 0
        vmax = 60
    elif colorby=='vsub':
        clr = np.sqrt(t['vxsub']**2 + t['vysub']**2)
        clabel = 'Perturber velocity [km s$^{-1}$]'
        cmap = 'magma'
        vmin = 0
        vmax = 500
    elif colorby=='t':
        clr = t['Timpact']
        clabel = 'Impact time [Gyr]'
        cmap = 'magma'
        vmin = 0.35
        vmax = 0.55
    elif colorby=='tgap':
        clr = t['Tgap']
        clabel = 'Gap time [Myr]'
        cmap = 'magma'
        vmin = 28
        vmax = 30
    elif colorby=='dvr1':
        clr = t['dvr1']
        clabel = '$\Delta$ V$_r$($\phi_1$=-33.7) [km s$^-1$]'
        clabel = 'Relative radial velocity [km s$^-1$]'
        cmap = 'twilight'
        vmin = -5
        vmax = 5
    elif colorby=='dvr2':
        clr = t['dvr2']
        clabel = '$\Delta$ V$_r$($\phi_1$=-30) [km s$^-1$]'
        cmap = 'twilight'
        vmin = -5
        vmax = 5
    elif colorby=='energy':
        clr = energy.to(u.kpc**2/u.Myr**2)
        clabel = 'Energy [kpc$^2$ Myr$^{-2}$]'
        cmap = 'magma'
        vmin = 0.1
        vmax = 0.3
    elif colorby=='pm':
        clr = t['dmu22']
        clabel = '$\Delta \mu_{\phi_2,stream-spur}$ [mas yr$^-1$]'
        clabel = '$\Delta \mu_{\phi_2}$ [mas yr$^-1$]'
        cmap = 'magma_r'
        vmin = -0.4
        vmax = -0.1

    plt.close()
    fig = plt.figure(figsize=(12,5.2))
    ax = fig.add_subplot(111, projection='mollweide')
    
    if step==0:
        plt.plot(ceq.ra.wrap_at(wangle).radian, ceq.dec.radian, 'ko', rasterized=True, zorder=0, ms=3, mec='0.3', mew=0.5, label=label)
    
    if step>0:
        isort_clr = np.argsort(clr)[::-1]
        ceq = ceq[::5]
        clr = clr[::5]
        im = plt.scatter(ceq.ra.wrap_at(wangle).radian, ceq.dec.radian, rasterized=True, c=clr, zorder=0, s=1, cmap=cmap, vmin=vmin, vmax=vmax, label=label)
        plt.quiver(ceq.ra.wrap_at(wangle).radian, ceq.dec.radian, ceq.pm_ra_cosdec.value, ceq.pm_dec.value, color=mpl.cm.magma(0), width=1, units='dots', headlength=2, scale_units='inches', scale=4, label='', alpha=0.8, zorder=0)
    
    if step==3:
        cs = cs[::5]
        plt.plot(cs.ra.wrap_at(wangle).radian, cs.dec.radian, 'o', color=mpl.cm.magma(0.5), ms=2.5, mew=0, alpha=0.6, rasterized=True, label='Sagittarius stars\nDierickx & Loeb (2017)')
        plt.quiver(cs.ra.wrap_at(wangle).radian, cs.dec.radian, cs.pm_ra_cosdec.value, cs.pm_dec.value, color=mpl.cm.magma(0.5), width=1, units='dots', headlength=2, scale_units='inches', scale=4, label='', alpha=0.8)
    
    if step==4:
        cdm = cdm[::5]
        plt.plot(cdm.ra.wrap_at(wangle).radian, cdm.dec.radian, 'o', color=mpl.cm.magma(0.7), ms=0.5, mew=0, alpha=0.6, rasterized=True, label='Sagittarius dark matter\nDierickx & Loeb (2017)')
        plt.quiver(cdm.ra.wrap_at(wangle).radian, cdm.dec.radian, cdm.pm_ra_cosdec.value, cdm.pm_dec.value, color=mpl.cm.magma(0.7), width=1, units='dots', headlength=1, scale_units='inches', scale=4, label='', alpha=1)
    
    plt.legend(frameon=True, loc=4, handlelength=0.2, fontsize='small', markerscale=2)
    if step>0:
        legend = plt.gca().get_legend()
        legend.legendHandles[0].set_color(mpl.cm.twilight(0.5))
    if step>2:
        legend = plt.gca().get_legend()
        legend.legendHandles[1].set_color(mpl.cm.twilight(0.5))
    
    
    plt.xlabel('R.A. [deg]')
    plt.ylabel('Dec [deg]')
    plt.grid(True)
    
    plt.tight_layout()
    
    ## add custom colorbar
    ##sm = plt.cm.ScalarMappable(cmap=mpl.cm.viridis, norm=plt.Normalize(vmin=0, vmax=20))
    #sm = plt.cm.ScalarMappable(cmap=viriwarm, norm=plt.Normalize(vmin=0, vmax=20))
    ## fake up the array of the scalar mappable. Urgh...
    #sm._A = []
    
    if step>0:
        cb = fig.colorbar(im, ax=ax, pad=0.04, aspect=20)
        cb.set_label(clabel)
    
    plt.savefig('../plots/skybox_pm.png')
