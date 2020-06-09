from model import *
from vel import get_members
from matplotlib.legend_handler import HandlerLine2D
import healpy as hp
from scipy.ndimage import gaussian_filter

wangle = 180*u.deg
lightsteelblue = '#dde3ef'
steelblue = '#a2b3d2'
navyblue = '#294882'
fuchsia = '#ff3643'

def poly_phi2_orbit():
    """"""
    pin = np.load('../data/new_orbit.npy')
    phi1, phi2, d, pm1, pm2, vr = pin
    ham = gp.Hamiltonian(gp.load('../data/mwpot.yml'))
    dt = 0.5 * u.Myr

    c = gc.GD1(phi1=phi1*u.deg, phi2=phi2*u.deg, distance=d*u.kpc, 
            pm_phi1_cosphi2=pm1*u.mas/u.yr,
            pm_phi2=pm2*u.mas/u.yr,
            radial_velocity=vr*u.km/u.s)
    w0 = gd.PhaseSpacePosition(c.transform_to(gc_frame).cartesian)
    fit_orbit = ham.integrate_orbit(w0, dt=dt, n_steps=120)

    model_gd1 = fit_orbit.to_coord_frame(gc.GD1, galactocentric_frame=gc_frame)
    model_x = model_gd1.phi1.wrap_at(180*u.deg).degree
    print(np.min(model_x), np.max(model_x))
    
    f = scipy.interpolate.interp1d(model_x, model_gd1.phi2, kind='cubic')
    xnew = np.linspace(-95,9,100)
    ynew = f(xnew)
    np.save('../data/orbit_phi2_interp.npy', f)
    pickle.dump(dict(f=f), open('../data/orbit_phi2_interp.pkl', 'wb'))
    
    
    plt.close()
    fig, ax = plt.subplots(2,1, figsize=(10,10), sharex=True)
    
    plt.sca(ax[0])
    plt.plot(model_gd1.phi1.wrap_at(wangle), model_gd1.phi2, 'ko')
    plt.plot(xnew, ynew, 'r-')
    
    plt.sca(ax[1])
    plt.plot(model_gd1.phi1.wrap_at(wangle), model_gd1.phi2.value - f(model_gd1.phi1.wrap_at(wangle).value), 'ko')
    
    plt.tight_layout()

def field_centers(t):
    """"""
    
    obs_fields = np.unique(t['field'])
    nobs = np.size(obs_fields)
    phi1 = np.empty(nobs)
    phi2 = np.empty(nobs)
    
    for i, f in enumerate(obs_fields):
        ind = t['field']==f
        phi1[i] = np.median(t['phi1'][ind])
        phi2[i] = np.median(t['phi2'][ind])
    
    return (phi1, phi2)

def dvr():
    """"""

    t = Table.read('../data/master_catalog.fits')
    obs_phi1, obs_phi2 = field_centers(t)
    
    ind = (-t['lnL'] < 2.5E3+t['SNR']**2.4) & (t['SNR']>3)
    t = t[ind]
    mem = get_members(t)
    t = t[mem]
    
    spur = (t['field']==2) | (t['field']==4) | (t['field']==5) | (t['field']==6)
    stream = ~spur
    
    labels = ['Bonaca et al. (2020) [Hectochelle]', 'Huang et al. (2019)', 'Koposov et al. (2010)']
    colors = ['#3dd0e8', '#ff8d3e', '0.7']
    ecolors = ['#3d7be8', '#f84600', '0.7']
    markers = ['o', '*', 'o']
    sizes = [10, 18, 8]
    msizes = [6.5, 10, 4]
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
    fig, ax = plt.subplots(3,1,figsize=(12,7), sharex=True, gridspec_kw=dict(height_ratios=[1,1,1.5]))
    
    plt.sca(ax[1])
    p1, = plt.plot(tk['phi1'], tk['phi2'], 'o', color=colors[2], alpha=1, label=labels[2], ms=ms+1.5)
    p1b, = plt.plot(tl['phi1'], tl['phi2'], 's', color=colors[2], alpha=1, label=labels[1], ms=ms+1.5)
    
    kvr = qpoly(tk['phi1'])
    lvr = qpoly(tl['phi1'])
    
    plt.sca(ax[2])
    plt.axhline(0, color='k', lw=2, alpha=0.7, zorder=0)
    plt.errorbar(tk['phi1'], tk['vr'] - kvr, yerr=tk['err'], fmt='o', color=colors[2], lw=1.5, alpha=1, label='', zorder=0, ms=ms)
    plt.errorbar(tl['phi1'], tl['vr'] - lvr, yerr=tl['err'], fmt='s', color=colors[2], lw=1.5, alpha=1, label='', zorder=0, ms=ms)
    
    p2 = []
    
    for e, ind in enumerate([stream, spur]):
        plt.sca(ax[1])
        p_, = plt.plot(t['phi1'][ind], t['phi2'][ind], marker=markers[e], ls='none', color=colors[e], label=labels[e], ms=msizes[e], mec=ecolors[e], mew=mew)
        p2 += [p_]
        
        plt.sca(ax[2])
        vr = qpoly(t['phi1'][ind])
        
        plt.errorbar(t['phi1'][ind], t['Vrad'][ind] - vr, yerr=(t['lerr_Vrad'][ind], t['uerr_Vrad'][ind]), fmt=markers[e], color=ecolors[e], mfc=colors[e], zorder=0, lw=2, ms=msizes[e], mec=ecolors[e], mew=mew)

    # fields
    #newfield = np.array([[8.5,-1.6], [0.4, -1.3], [-5.2, -0.74]])
    
    pkl_phi2 = pickle.load(open('../data/orbit_phi2_interp.pkl', 'rb'))
    phi2_poly = pkl_phi2['f']
    field_phi1 = np.concatenate([np.arange(-80,-45,5), np.array([-40]), np.arange(-25,10,5)])
    field_phi2 = phi2_poly(field_phi1)
    print(np.size(field_phi1))
    
    cf = gc.GD1(phi1=field_phi1*u.deg, phi2=field_phi2*u.deg)
    cf_eq = cf.transform_to(coord.ICRS)
    print(cf_eq.ra.deg[:7]/15) #11 to 14 for fall
    print(cf_eq.dec.deg[:7])

    plt.sca(ax[0])
    pm_, = plt.plot(g['phi1'], g['phi2'], 'k.', mew=0, ms=5, alpha=0.8, zorder=0, label='GD-1 members (Gaia $\\times$ PanSTARRS)')
    
    for x, y in zip(field_phi1, field_phi2):
        label = 'Proposed'
        c_ = mpl.patches.Circle((x,y), radius=0.7, edgecolor='r', facecolor='none', lw=2, zorder=1, label=label)
        plt.gca().add_artist(c_)
    
    for x, y in zip(obs_phi1, obs_phi2):
        label = 'Observed'
        co_ = mpl.patches.Circle((x,y), radius=0.7, edgecolor='orange', facecolor='none', lw=2, zorder=1, label=label)
        plt.gca().add_artist(co_)
    
    plt.ylim(-9,5)
    plt.xlim(-90, 10)
    plt.gca().set_aspect('equal', adjustable='datalim')
    plt.ylabel('$\phi_2$ [deg]')
    plt.legend(handles=[pm_, co_, c_], fontsize='small', ncol=3, handlelength=0.5, loc=3)
    
    plt.sca(ax[1])
    plt.plot(g['phi1'], g['phi2'], 'k.', mew=0, ms=5, alpha=0.6, zorder=0)
    plt.ylim(-9,5)
    plt.gca().set_aspect('equal', adjustable='datalim')
    plt.ylabel('$\phi_2$ [deg]')
    
    plt.legend([p1, p1b, (p2[0], p2[1])],[labels[2], labels[1], labels[0]], handler_map={p2[0]:HandlerLine2D(numpoints=2), p2[1]:HandlerLine2D(numpoints=1)}, ncol=3, frameon=True, handlelength=2, loc=3, fontsize='small', numpoints=1)
    
    #plt.sca(ax[1])
    #plt.ylim(-140,49)
    #plt.ylabel('$V_r$ [km s$^{-1}$]')
    
    plt.sca(ax[2])
    plt.ylim(-40,40)
    plt.ylabel('$\Delta V_r$ [km s$^{-1}$]')
    plt.xlabel('$\phi_1$ [deg]')
    
    plt.tight_layout(h_pad=0.)
    plt.savefig('../plots/prop20c_targeting.pdf')
    
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
    
    fig = plt.figure(figsize=(12,4.3))
    gs1 = mpl.gridspec.GridSpec(1,3)
    gs1.update(left=0.08, right=0.975, top=0.92, bottom=0.16, wspace=0.25)
    
    #gs2 = mpl.gridspec.GridSpec(1,1)
    #gs2.update(left=0.08, right=0.975, top=0.47, bottom=0.08)

    ax0 = fig.add_subplot(gs1[0])
    ax1 = fig.add_subplot(gs1[1])
    ax2 = fig.add_subplot(gs1[2])
    #ax3 = fig.add_subplot(gs2[0])
    ax = [ax0, ax1, ax2]
    #ax = [ax0, ax1, ax2, ax3]
    
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
    #plt.text(0.1, 0.9, '{:2d}'.format(np.sum(pmmem)), transform=plt.gca().transAxes, ha='left')
    
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
    #plt.text(0.1, 0.9, '{:2d}'.format(np.sum(cmdmem & pmmem)), transform=plt.gca().transAxes, ha='left')
    
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
    #plt.text(0.1, 0.9, '{:2d}'.format(np.sum(pmmem & cmdmem & vrmem)), transform=plt.gca().transAxes, ha='left')
    
    #plt.sca(ax[3])
    #prelim_mem = pmmem & cmdmem & vrmem & ~mem
    ##plt.plot(t['FeH'][pmmem & cmdmem & vrmem], t['aFe'][pmmem & cmdmem & vrmem], 'o', color=navyblue, mec='none', ms=6, label='GD-1 members', zorder=1)
    ##plt.plot(t['FeH'][mem], t['aFe'][mem], 'o', color=navyblue, mec='none', ms=6, label='GD-1 members', zorder=1)

    ##plt.plot(t['init_FeH'][~(cmdmem & vrmem)], t['init_aFe'][~(cmdmem & vrmem)], 'o', color=lightsteelblue, mec='none', alpha=1, ms=4, label='Field stars', zorder=0)
    ##plt.plot(t['init_FeH'][prelim_mem], t['init_aFe'][prelim_mem], 'o', color=steelblue, mec='none', alpha=1, ms=7, zorder=0, label='Preliminary GD-1 members')
    ##plt.plot(t['init_FeH'][mem & stream], t['init_aFe'][mem & stream], 'o', color=navyblue, mec='none', ms=7, label='GD-1 stream members', zorder=1)
    ##plt.plot(t['init_FeH'][mem & spur], t['init_aFe'][mem & spur], '*', color=navyblue, mec='none', ms=12, label='GD-1 spur members', zorder=1)
    ##plt.errorbar(t['init_FeH'][mem], t['init_aFe'][mem], yerr=t['std_init_FeH'][mem], xerr=t['std_init_aFe'][mem], fmt='none', color=navyblue, label='', zorder=0, alpha=0.5, lw=0.7)
    
    #plt.plot(t['init_FeH'][~(cmdmem & vrmem)], t['aFe'][~(cmdmem & vrmem)], 'o', color=lightsteelblue, mec='none', alpha=1, ms=4, label='Field stars', zorder=0)
    #plt.plot(t['init_FeH'][prelim_mem], t['aFe'][prelim_mem], 'o', color=steelblue, mec='none', alpha=1, ms=7, zorder=0, label='Preliminary GD-1 members')
    #plt.plot(t['init_FeH'][mem & stream], t['aFe'][mem & stream], 'o', color=navyblue, mec='none', ms=7, label='GD-1 stream members', zorder=1)
    #plt.plot(t['init_FeH'][mem & spur], t['aFe'][mem & spur], '*', color=navyblue, mec='none', ms=12, label='GD-1 spur members', zorder=1)
    #plt.errorbar(t['init_FeH'][mem], t['aFe'][mem], yerr=t['std_init_FeH'][mem], xerr=t['std_aFe'][mem], fmt='none', color=navyblue, label='', zorder=0, alpha=0.5, lw=0.7)

    ##plt.plot(t['FeH'][~(cmdmem & vrmem)], t['aFe'][~(cmdmem & vrmem)], 'o', color=lightsteelblue, mec='none', alpha=1, ms=4, label='Field stars', zorder=0)
    ##plt.plot(t['FeH'][prelim_mem], t['aFe'][prelim_mem], 'o', color=steelblue, mec='none', alpha=1, ms=7, zorder=0, label='Preliminary GD-1 members')
    ##plt.plot(t['FeH'][mem & stream], t['aFe'][mem & stream], 'o', color=navyblue, mec='none', ms=7, label='GD-1 stream members', zorder=1)
    ##plt.plot(t['FeH'][mem & spur], t['aFe'][mem & spur], '*', color=navyblue, mec='none', ms=12, label='GD-1 spur members', zorder=1)
    ##plt.errorbar(t['FeH'][mem], t['aFe'][mem], yerr=t['std_FeH'][mem], xerr=t['std_aFe'][mem], fmt='none', color=navyblue, label='', zorder=0, alpha=0.5, lw=0.7)

    
    #for fehlim in fehlims:
        #plt.axvline(fehlim, ls='--', lw=3, color=fuchsia, label='', zorder=2)
    
    #plt.text(0.97, 0.9, '{:2d}'.format(np.sum(pmmem & cmdmem & vrmem & fehmem)), transform=plt.gca().transAxes, ha='right')
    #plt.text(0.03, 0.9, '{:2d}'.format(np.sum(pmmem & cmdmem & vrmem & fehmem)), transform=plt.gca().transAxes, ha='left')
    #plt.legend(loc=1, frameon=True, handlelength=1, fontsize='medium', markerscale=1.2)
    
    #plt.xlim(-3.2,0.1)
    #plt.ylim(-0.2,0.6)
    #plt.ylabel('[$\\alpha$/Fe]')
    #plt.xlabel('[Fe/H]$_{init}$')
    #plt.title('+ Metallicity selection', fontsize='medium')

    #plt.tight_layout(w_pad=0.1)
    plt.savefig('../plots/prop20c_members.pdf')

def dispersion():
    """"""
    t = Table.read('../data/master_catalog.fits')
    obs_phi1, obs_phi2 = field_centers(t)
    
    ind = (-t['lnL'] < 2.5E3+t['SNR']**2.4) & (t['SNR']>3)
    t = t[ind]
    mem = get_members(t)
    t = t[mem]
    
    
    fields = np.unique(t['field'])
    n = np.size(fields)
    sigma = np.empty(n)
    
    pkl = pickle.load(open('../data/orbit_vr_interp.pkl', 'rb'))
    qpoly = pkl['f']
    vr = qpoly(t['phi1'])
    
    for i in range(n):
        ind = t['field']==fields[i]
        sigma[i] = np.std(t['Vrad'][ind] - vr[ind])
    
    spur = (fields==2) | (fields==4) | (fields==5) | (fields==6)
    stream = ~spur
        
    print(sigma[stream])
    print(sigma[spur])
    
    rescale = 75 + (170-75)/4 * sigma
    print(rescale)
    
    plt.close()
    plt.figure()
    
    plt.plot(obs_phi1[stream], sigma[stream], 'ko')
    plt.plot(obs_phi1[spur], sigma[spur], 'o', color='0.5')
    
    plt.tight_layout()
    
def uncertainties():
    """"""
    t = Table.read('../data/master_catalog.fits')
    ind = (-t['lnL'] < 2.5E3+t['SNR']**2.4) & (t['SNR']>3)
    t = t[ind]
    
    print(np.percentile(t['std_Vrad'], [50,90]))
    print(np.percentile(t['std_FeH'], [50,90]))
    print(np.percentile(t['std_aFe'], [50,90]))
    
    plt.close()
    plt.figure()
    
    plt.plot(t['g'], t['std_Vrad'], 'ko', ms=3)
    
    plt.ylim(0.01,5)
    plt.gca().set_yscale('log')
    
    
