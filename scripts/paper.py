from model import *
from vel import get_members

wangle = 180*u.deg
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

    mem_dict = get_members(t, full=True)
    cmdmem = mem_dict['cmdmem']
    pmmem = mem_dict['pmmem']
    vrmem = mem_dict['vrmem']
    fehmem = mem_dict['fehmem']
    vrlims = mem_dict['vrlims']
    fehlims = mem_dict['fehlims']
    
    print(np.sum(pmmem), np.sum(pmmem & cmdmem), np.sum(pmmem & cmdmem & vrmem), np.sum(pmmem & cmdmem & vrmem & fehmem), np.sum(mem_dict['mem']))
    #print(len(t['g'][pmmem & ~cmdmem]))
    
    bvr = np.linspace(-50,50,50)
    
    plt.close()
    fig, ax = plt.subplots(1, 3, figsize=(15,5), gridspec_kw={'width_ratios': [1,1.7,3.2]})
    
    plt.sca(ax[0])
    plt.plot(t['g'] - t['i'], t['g'], 'o', color=steelblue, mec='none', ms=5, alpha=0.5)
    plt.plot(t['g'][pmmem] - t['i'][pmmem], t['g'][pmmem], 'o', color=navyblue, mec='none', ms=5)
    #plt.plot(t['g'][pmmem & ~cmdmem] - t['i'][pmmem & ~cmdmem], t['g'][pmmem & ~cmdmem], 'o', color=navyblue, mec='none', ms=5)
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
    plt.title('+ Isochrone', fontsize='medium')
    plt.text(0.94, 0.9, '{:2d}'.format(np.sum(cmdmem & vrmem)), transform=plt.gca().transAxes, ha='right')
    
    plt.sca(ax[2])
    plt.plot(t['FeH'][cmdmem & vrmem], t['aFe'][cmdmem & vrmem], 'o', color=navyblue, mec='none', ms=6, label='GD-1 members', zorder=1)
    plt.plot(t['FeH'][~(cmdmem & vrmem)], t['aFe'][~(cmdmem & vrmem)], 'o', color=steelblue, mec='none', alpha=0.5, ms=6, label='Field stars', zorder=0)
    
    for fehlim in fehlims:
        plt.axvline(fehlim, ls='--', lw=3, color=fuchsia, label='', zorder=2)
    
    plt.text(0.97, 0.9, '{:2d}'.format(np.sum(cmdmem & vrmem & fehmem)), transform=plt.gca().transAxes, ha='right')
    plt.legend(loc=4, frameon=True, handlelength=1, fontsize='small', markerscale=1.3)
    
    plt.xlim(-3.2,0.1)
    plt.ylim(-0.2,0.6)
    plt.ylabel('[$\\alpha$/Fe]')
    plt.xlabel('[Fe/H]')
    plt.title('+ Radial velocity selection', fontsize='medium')

    plt.tight_layout()
    plt.savefig('../paper/members.pdf')

def dvr():
    """"""

    t = Table.read('../data/master_catalog.fits')
    ind = (-t['lnL'] < 2.5E3+t['SNR']**2.4) & (t['SNR']>3)
    t = t[ind]
    mem = get_members(t)
    t = t[mem]
    #ind = (t['priority']<3) & (t['delta_Vrad']>-20) & (t['delta_Vrad']<-1) #& (t['FeH']<-2)
    #t = t[ind]
    
    spur = (t['field']==2) | (t['field']==4) | (t['field']==5) | (t['field']==6)
    stream = ~spur
    
    cspur = mpl.cm.Blues_r(0.15)
    cstream = mpl.cm.Blues_r(0.4)
    colors = [cstream, cspur, 'darkorange']
    colors = ['darkorange', 'orangered', 'navy']
    colors = ['darkorange', 'orangered', '0.8']
    labels = ['Bonaca et al. (2020)', '', 'Koposov et al. (2010)']
    
    tk = Table.read('../data/koposov_vr.dat', format='ascii.commented_header')

    g = Table(fits.getdata('/home/ana/projects/legacy/GD1-DR2/output/gd1_members.fits'))
    
    #q = np.load('../data/poly_vr_median.npy')
    #qpoly = np.poly1d(q)
    
    pkl = pickle.load(open('../data/orbit_vr_interp.pkl', 'rb'))
    qpoly = pkl['f']
    xphi = np.linspace(-50,-10,100)
    yvr = qpoly(xphi)
    #print(qpoly)

    plt.close()
    fig, ax = plt.subplots(3,1,figsize=(10,10), sharex=True, gridspec_kw=dict(height_ratios=[1,2,1]))
    
    plt.sca(ax[0])
    #plt.plot(tk['phi1'], tk['phi2'], 'o', color='w', mec='none', ms=8, label='')
    plt.plot(tk['phi1'], tk['phi2'], 'o', color=colors[2], alpha=1, label=labels[2])
    
    plt.sca(ax[1])
    plt.plot(xphi, yvr, '-', color='teal', lw=2, alpha=0.7)
    plt.errorbar(tk['phi1'], tk['vr'], yerr=tk['err'], fmt='o', color=colors[2], lw=2, alpha=1, label='')
    #plt.plot(tk['phi1'], tk['vr'], 'o', color='w', ms=8, mec='none', label='')
    #plt.plot(tk['phi1'], tk['vr'], 'o', color=colors[2], ms=8, alpha=1, mec='none', label='')
    
    #plt.sca(ax[2])
    #plt.axhline(0, color='teal', lw=2, alpha=0.7, zorder=0)
    ##plt.fill_between(np.linspace(-60,-20,10), -5, 5, color='k', alpha=0.2)
    
    kvr = qpoly(tk['phi1'])
    #plt.errorbar(tk['phi1'], tk['vr'] - kvr, yerr=tk['err'], fmt='o', color=colors[2], lw=2, alpha=1, label='', zorder=0)
    ##plt.plot(tk['phi1'], tk['vr'] - kvr, 'o', color='w', ms=8, mec='none', label='')
    ##plt.plot(tk['phi1'], tk['vr'] - kvr, 'o', color=colors[2], ms=8, alpha=1, mec='none', label='')
    
    plt.sca(ax[2])
    plt.axhline(0, color='teal', lw=2, alpha=0.7, zorder=0)
    plt.errorbar(tk['phi1'], tk['vr'] - kvr, yerr=tk['err'], fmt='o', color=colors[2], lw=2, alpha=1, label='', zorder=0)
    #plt.fill_between(np.linspace(-60,-20,10), -5, 5, color='k', alpha=0.2)
    
    for e, ind in enumerate([stream, spur]):
        plt.sca(ax[0])
        plt.plot(t['phi1'][ind], t['phi2'][ind], 'o', color=colors[e], label=labels[e])
        
        plt.sca(ax[1])
        plt.errorbar(t['phi1'][ind], t['Vrad'][ind], yerr=(t['lerr_Vrad'][ind], t['uerr_Vrad'][ind]), color=colors[e], fmt='o', label='')
        #plt.plot(t['phi1'][ind], t['Vrad'][ind], 'o', color=colors[e], ms=8, label=labels[e])
        
        plt.sca(ax[2])
        vr = qpoly(t['phi1'][ind])
        
        #plt.plot(t['phi1'][ind], t['Vrad'][ind] - vr, 'o', color=colors[e], ms=8)
        plt.errorbar(t['phi1'][ind], t['Vrad'][ind] - vr, yerr=(t['lerr_Vrad'][ind], t['uerr_Vrad'][ind]), fmt='o', color=colors[e], zorder=0, lw=2)
        
        #plt.sca(ax[3])
        ##plt.plot(t['phi1'][ind], t['Vrad'][ind] - vr, 'o', color=colors[e], ms=3, alpha=0.3)
        #plt.errorbar(t['phi1'][ind], t['Vrad'][ind] - vr, yerr=(t['lerr_Vrad'][ind], t['uerr_Vrad'][ind]), fmt='o', color=colors[e], zorder=0, lw=2, alpha=1)
        
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
            plt.plot(np.median(t['phi1'][ind & ifield]), np.median(t['Vrad'][ind & ifield] - vr), 'o', color=colors[e], ms=10, mec='k', mew=2, zorder=ee+3)
            
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
    plt.legend(ncol=2, frameon=False, handlelength=0.6, loc=3, fontsize='small')
    
    plt.sca(ax[1])
    plt.ylim(-140,49)
    plt.ylabel('$V_r$ [km s$^{-1}$]')
    
    #plt.sca(ax[2])
    #plt.ylim(-35, 35)
    #plt.ylabel('$\Delta V_r$ [km s$^{-1}$]')

    plt.sca(ax[2])
    plt.ylim(-10,10)
    plt.ylabel('$\Delta V_r$ [km s$^{-1}$]')
    plt.xlabel('$\phi_1$ [deg]')
    
    
    plt.tight_layout(h_pad=0)
    #plt.savefig('../paper/gd1_kinematics.pdf')

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
    ind = (np.abs(t['dvr1'])<0.5) & (np.abs(t['dvr2'])<0.5)
    
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
    cmap = 'twilight'
    vmin = -5
    vmax = 5

    plt.close()
    fig, ax = plt.subplots(3,1,figsize=(6,10), subplot_kw=dict(projection='mollweide'))
    
    plt.sca(ax[0])
    isort_clr = np.argsort(clr)[::-1]
    im0 = plt.scatter(ceq.ra.wrap_at(wangle).radian, ceq.dec.radian, rasterized=True, c=clr, zorder=0, s=2, ec='none', cmap=cmap, vmin=vmin, vmax=vmax, label=label)
    
    font_tick = 'x-small'
    plt.xticks(fontsize=font_tick)
    plt.yticks(fontsize=font_tick)
    
    plt.xlabel('R.A. [deg]', fontsize='small')
    plt.ylabel('Dec [deg]', fontsize='small')
    plt.title('Objects producing a GD-1 spur', fontsize=17, pad=15)

    plt.sca(ax[1])
    
    ind = (np.abs(t['dvr1'])<1) & (np.abs(t['dvr2'])<1)
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
    #theta = 120*u.deg
    #rotmat = np.array([[np.cos(theta), -np.sin(theta)],[np.sin(theta), np.cos(theta)]])
    #vin = np.array([t['dmu21'], t['dmu22']])
    #v = np.matmul(rotmat, vin)
    #clr = v[0]
    #cmap = 'magma_r'
    #vmin = 0.1
    #vmax = 0.4
    clr = t['dmu22']
    clabel = '$\Delta \mu_{\phi_2,stream-spur}$ [mas yr$^-1$]'
    cmap = 'magma'
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
    
    
    plt.sca(ax[2])
    isort_clr = np.argsort(clr)[::-1]
    im = plt.scatter(ceq.ra.wrap_at(wangle).radian[isort_clr], ceq.dec.radian[isort_clr], rasterized=True, c=clr[isort_clr], zorder=0, s=2, ec='none', cmap=cmap, vmin=vmin, vmax=vmax, label='')
    plt.plot(cdm.ra.wrap_at(wangle).radian, cdm.dec.radian, 'o', color='0.2', ms=2, mew=0, alpha=0.3, rasterized=True, label='Sagittarius dark matter\n(Dierickx & Loeb 2017)')
    
    plt.legend(loc=4, fontsize='x-small', markerscale=2, handlelength=0.5)
    plt.xticks(fontsize=font_tick)
    plt.yticks(fontsize=font_tick)
    plt.xlabel('R.A. [deg]', fontsize='small')
    plt.ylabel('Dec [deg]', fontsize='small')
    plt.title('Objects producing a comoving GD-1 spur + Sagittarius', fontsize=17, pad=15)
    
    
    plt.tight_layout()
    
    # colorbar
    plt.sca(ax[0])
    pos = plt.gca().get_position()
    cax = plt.axes([0.95,pos.y0,0.025,pos.y1 - pos.y0])
    plt.colorbar(im0, cax=cax, ticks=[-5,-2.5,0,2.5,5])
    plt.yticks(fontsize=font_tick)
    plt.ylabel(clabel0, fontsize='small')

    for i in range(1,3):
        plt.sca(ax[i])
        pos = plt.gca().get_position()
        cax = plt.axes([0.95,pos.y0,0.025,pos.y1 - pos.y0])
        #cb = plt.colorbar(im, cax=cax, ticks=np.linspace(0,500,6))
        cb = plt.colorbar(im, cax=cax, ticks=np.linspace(-0.4,-0.1,4))
        plt.yticks(fontsize=font_tick)
        cb.set_label(clabel, fontsize='small')
    
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

def members():
    """"""
    t = Table.read('../data/master_catalog.fits')
    ind = (-t['lnL'] < 2.5E3+t['SNR']**2.4) & (t['SNR']>3) & np.isfinite(t['aFe'])
    t = t[ind]

    mem = get_members(t)
    t = t[mem]
    
    print('chemistry')
    for k in ['FeH', 'aFe']:
        print(k, np.median(t['{:s}'.format(k)]), np.std(t['{:s}'.format(k)]))
    
    print('uncertainties')
    for k in ['Vrad', 'FeH', 'aFe']:
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


