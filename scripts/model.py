from __future__ import print_function, division

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
#from mpl_toolkits.mplot3d import Axes3D
from matplotlib.backends.backend_pdf import PdfPages

from astropy.table import Table
import astropy.units as u
import astropy.coordinates as coord
from astropy.constants import G, c as c_
from astropy.io import fits

import gala.coordinates as gc
import gala.dynamics as gd
import gala.potential as gp
from gala.units import galactic
from gala.dynamics import mockstream

#import scipy.optimize
#import scipy.spatial
import scipy.interpolate
import emcee
import corner

#from colossus.cosmology import cosmology
#from colossus.halo import concentration

import interact3 as interact
#import myutils

import time
import copy
import pickle
import h5py

gc_frame_dict = {'galcen_distance':8*u.kpc, 'z_sun':0*u.pc}
gc_frame = coord.Galactocentric(**gc_frame_dict)
ham = gp.Hamiltonian(gp.LogarithmicPotential(v_c=225*u.km/u.s, r_h=0*u.kpc, q1=1, q2=1, q3=1, units=galactic))
ham_log = gp.Hamiltonian(gp.LogarithmicPotential(v_c=225*u.km/u.s, r_h=0*u.kpc, q1=1, q2=1, q3=1, units=galactic))
ham_mw = gp.Hamiltonian(gp.load('../../gd1_spur/data/mwpot.yml'))

def check_vr():
    pkl = pickle.load(open('/home/ana/projects/gd1_spur/data/fiducial_perturb_python3.pkl', 'rb'))
    cg = pkl['cg']
    ceq = cg.transform_to(coord.ICRS)
    cgal = cg.transform_to(coord.Galactocentric)
    
    wangle = 180*u.deg
    
    plt.close()
    fig, ax = plt.subplots(3,1,figsize=(8,8), sharex='col')
    
    plt.sca(ax[0])
    plt.plot(cg.phi1.wrap_at(wangle), cg.phi2, 'ko', ms=1)
    plt.ylim(-5,5)
    
    plt.sca(ax[1])
    plt.plot(cg.phi1.wrap_at(wangle), cg.radial_velocity, 'ko', ms=1)
    plt.ylim(-200,150)
    
    plt.sca(ax[2])
    plt.plot(cg.phi1.wrap_at(wangle), cg.distance.to(u.kpc), 'ko', ms=1)
    
    plt.ylim(7,10)
    plt.xlim(-80,0)
    #plt.ylim(-250,100)

def cartesian():
    """"""
    pkl = pickle.load(open('/home/ana/projects/gd1_spur/data/fiducial_perturb_python3.pkl', 'rb'))
    cg = pkl['cg']
    ceq = cg.transform_to(coord.ICRS)
    cgal = cg.transform_to(coord.Galactocentric)
    
    plt.close()
    fig, ax = plt.subplots(2,2,figsize=(8,8))
    
    plt.sca(ax[0][0])
    plt.plot(cgal.x, cgal.y, 'k.', ms=1)

import os.path
def generate_streakline_fiducial(vnorm=250*u.km/u.s, theta=95.7*u.deg, t_impact=0.495*u.Gyr, bnorm=15*u.pc, bx=6*u.pc):
    """"""
    
    np.random.seed(143531)
    
    ##t_impact, M, rs, bnorm, bx, vnorm, vx = fiducial_params()
    #t_impact = 0.495*u.Gyr
    #M = 5e6*u.Msun
    #rs = 10*u.pc
    #bnorm = 15*u.pc
    #bx = 6*u.pc
    #vnorm = 250*u.km/u.s
    #vx = -25*u.km/u.s
    
    #print(np.arccos(vx/vnorm).to(u.deg))
    
#def br():
    
    #t_impact = 0.2*u.Gyr
    #bx = -7*u.pc
    #vx = -150*u.km/u.s
    
    ##bnorm = 10*u.pc
    ##bx = 10*u.pc
    M = 5e6*u.Msun
    ##t_impact = 0.5*u.Gyr
    rs = 10*u.pc
    ##vnorm = 300*u.km/u.s
    ##vx = -100*u.km/u.s
    vx = vnorm * np.cos(theta)

    # check if model at impact exists
    filepath = '../data/model_tenc.{:.4f}.pkl'.format(t_impact.to(u.Gyr).value)
    if os.path.isfile(filepath):
        pkl = pickle.load(open(filepath, 'rb'))
        model = pkl['model']
        xgap = pkl['xgap']
        vgap = pkl['vgap']
    else:
        # load one orbital point
        pos = np.load('/home/ana/projects/gd1_spur/data/log_orbit.npy')
        phi1, phi2, d, pm1, pm2, vr = pos

        c = gc.GD1(phi1=phi1*u.deg, phi2=phi2*u.deg, distance=d*u.kpc, pm_phi1_cosphi2=pm1*u.mas/u.yr, pm_phi2=pm2*u.mas/u.yr, radial_velocity=vr*u.km/u.s)
        w0 = gd.PhaseSpacePosition(c.transform_to(gc_frame).cartesian)
        
        # best-fitting orbit
        dt = 0.5*u.Myr
        n_steps = 120
        wangle = 180*u.deg

        # integrate back in time
        fit_orbit = ham.integrate_orbit(w0, dt=dt, n_steps=n_steps)
        
        prog_phi0 = -20*u.deg

        model_gd1 = fit_orbit.to_coord_frame(gc.GD1, galactocentric_frame=gc_frame)
        prog_i = np.abs(model_gd1.phi1.wrap_at(180*u.deg) - prog_phi0).argmin()
        prog_w0 = fit_orbit[prog_i]
        
        dt_orbit = 0.5*u.Myr
        nstep_impact = np.int64((t_impact / dt_orbit).decompose())
        #prog_orbit = ham.integrate_orbit(prog_w0, dt=-dt_orbit, t1=0*u.Myr, t2=-3*u.Gyr)
        prog_orbit = ham.integrate_orbit(prog_w0, dt=-dt_orbit, t1=0*u.Myr, t2=-3*u.Gyr)
        impact_orbit = prog_orbit[nstep_impact:]
        impact_orbit = impact_orbit[::-1]
        prog_orbit = prog_orbit[::-1]
        
        t_disrupt = -300*u.Myr
        minit = 7e4
        mfin = 1e3
        nrelease = 1
        n_times = (prog_orbit.t < t_disrupt).sum()
        prog_mass = np.linspace(minit, mfin, n_times)
        prog_mass = np.concatenate((prog_mass, np.zeros(len(prog_orbit.t) - n_times))) * u.Msun
        model_present = mockstream.dissolved_fardal_stream(ham, prog_orbit, prog_mass=prog_mass, t_disrupt=t_disrupt, release_every=nrelease)
        
        n_steps_disrupt = int(abs(t_disrupt / (prog_orbit.t[1]-prog_orbit.t[0])))
        model_present = model_present[:-2*n_steps_disrupt]
        
        model_gd1 = model_present.to_coord_frame(gc.GD1, galactocentric_frame=gc_frame)
        ind_gap = np.where((model_gd1.phi1.wrap_at(wangle)>-43*u.deg) & (model_gd1.phi1.wrap_at(wangle)<-33*u.deg))[0]

        n_times = (impact_orbit.t < t_disrupt).sum()
        prog_mass = np.linspace(minit, mfin, n_times)
        prog_mass = np.concatenate((prog_mass, np.zeros(len(impact_orbit.t) - n_times))) * u.Msun
        model = mockstream.dissolved_fardal_stream(ham, impact_orbit, prog_mass=prog_mass, t_disrupt=t_disrupt, release_every=nrelease)

        n_steps_disrupt = int(abs(t_disrupt / (impact_orbit.t[1]-impact_orbit.t[0])))
        model = model[:-2*n_steps_disrupt]

        Nstar = np.shape(model)[0]
        ivalid = ind_gap < Nstar
        ind_gap = ind_gap[ivalid]
        
        xgap = np.median(model.xyz[:,ind_gap], axis=1)
        vgap = np.median(model.v_xyz[:,ind_gap], axis=1)
        
        outdict = {'model': model, 'xgap': xgap, 'vgap': vgap}
        pickle.dump(outdict, open(filepath, 'wb'))
    
    
    ########################
    # Perturber at encounter
    
    i = np.array([1,0,0], dtype=float)
    j = np.array([0,1,0], dtype=float)
    k = np.array([0,0,1], dtype=float)
    
    # find positional plane
    bi = np.cross(j, vgap)
    bi = bi/np.linalg.norm(bi)
    
    bj = np.cross(vgap, bi)
    bj = bj/np.linalg.norm(bj)
    
    # pick b
    by = np.sqrt(bnorm**2 - bx**2)
    b = bx*bi + by*bj
    xsub = xgap + b
    
    # find velocity plane
    vi = np.cross(vgap, b)
    vi = vi/np.linalg.norm(vi)
    
    vj = np.cross(b, vi)
    vj = vj/np.linalg.norm(vj)
    
    # pick v
    vy = np.sqrt(vnorm**2 - vx**2)
    vsub = vx*vi + vy*vj
    
    # impact parameters
    Tenc = 0.01*u.Gyr
    dt = 0.05*u.Myr
    
    # potential parameters
    potential = 3
    Vh = 225*u.km/u.s
    q = 1*u.Unit(1)
    rhalo = 0*u.pc
    par_pot = np.array([Vh.si.value, q.value, rhalo.si.value])
    
    # generate unperturbed stream model
    potential_perturb = 2
    par_perturb = np.array([0*M.si.value, rs.si.value, 0, 0, 0])
    
    x1, x2, x3, v1, v2, v3 = interact.general_interact(par_perturb, xsub.si.value, vsub.si.value, Tenc.si.value, t_impact.si.value, dt.si.value, par_pot, potential, potential_perturb, model.x.si.value, model.y.si.value, model.z.si.value, model.v_x.si.value, model.v_y.si.value, model.v_z.si.value)
    stream = {}
    stream['x'] = (np.array([x1, x2, x3])*u.m).to(u.pc)
    stream['v'] = (np.array([v1, v2, v3])*u.m/u.s).to(u.km/u.s)
    
    c = coord.Galactocentric(x=stream['x'][0], y=stream['x'][1], z=stream['x'][2], v_x=stream['v'][0], v_y=stream['v'][1], v_z=stream['v'][2], **gc_frame_dict)
    cg = c.transform_to(gc.GD1)
    outdict = {'cg': cg}
    w0 = gd.PhaseSpacePosition(cg.transform_to(gc_frame).cartesian)
    orbit0 = ham.integrate_orbit(w0, dt=0.1*u.Myr, n_steps=1)
    E0 = orbit0.energy()[0]
    pickle.dump(outdict, open('../data/fiducial_noperturb.{:03.0f}.{:03.0f}.pkl'.format(vnorm.to(u.km/u.s).value, theta.to(u.deg).value), 'wb'))

    wangle = 180*u.deg
    ind = (cg.phi1.wrap_at(wangle)>-80*u.deg) & (cg.phi1.wrap_at(wangle)<0*u.deg)
    pvr = np.polyfit(cg.phi1.wrap_at(wangle)[ind], cg.radial_velocity[ind], 4)
    polyvr = np.poly1d(pvr)
    
    # generate perturbed stream model
    potential_perturb = 2
    par_perturb = np.array([M.si.value, rs.si.value, 0, 0, 0])
    
    x1, x2, x3, v1, v2, v3 = interact.general_interact(par_perturb, xsub.si.value, vsub.si.value, Tenc.si.value, t_impact.si.value, dt.si.value, par_pot, potential, potential_perturb, model.x.si.value, model.y.si.value, model.z.si.value, model.v_x.si.value, model.v_y.si.value, model.v_z.si.value)
    stream = {}
    stream['x'] = (np.array([x1, x2, x3])*u.m).to(u.pc)
    stream['v'] = (np.array([v1, v2, v3])*u.m/u.s).to(u.km/u.s)
    
    c = coord.Galactocentric(x=stream['x'][0], y=stream['x'][1], z=stream['x'][2], v_x=stream['v'][0], v_y=stream['v'][1], v_z=stream['v'][2], **gc_frame_dict)
    cg = c.transform_to(gc.GD1)
    outdict = {'cg': cg}
    pickle.dump(outdict, open('../data/fiducial_perturb.{:03.0f}.{:03.0f}.pkl'.format(vnorm.to(u.km/u.s).value, theta.to(u.deg).value), 'wb'))
    
    w = gd.PhaseSpacePosition(cg.transform_to(gc_frame).cartesian)
    orbit = ham.integrate_orbit(w, dt=0.1*u.Myr, n_steps=1)
    E = orbit.energy()[0]
    dE = 1 - E/E0
    perturbed = np.abs(dE)>0.001
    
    dVr = cg.radial_velocity - polyvr(cg.phi1.wrap_at(wangle))*u.km/u.s
    
    plt.close()
    fig, ax = plt.subplots(2,1,figsize=(10,7), sharex=True)
    
    plt.sca(ax[0])
    plt.plot(cg.phi1.wrap_at(wangle), cg.phi2, 'k.', ms=1)
    plt.plot(cg.phi1.wrap_at(wangle)[perturbed], cg.phi2[perturbed], 'ro', ms=1)
    plt.xlim(-60,-20)
    plt.ylim(-5,5)
    
    plt.sca(ax[1])
    plt.plot(cg.phi1.wrap_at(wangle), dVr, 'k.', ms=1)
    plt.plot(cg.phi1.wrap_at(wangle)[perturbed], dVr[perturbed], 'ro', ms=1)
    plt.ylim(-10,10)
    
    plt.tight_layout()
    plt.savefig('../plots/fiducial_search/vnorm.{:03.0f}.theta.{:03.0f}.png'.format(vnorm.to(u.km/u.s).value, theta.to(u.deg).value))

def fiducial_search():
    """"""
    
    vnorm = 250*u.km/u.s
    thetas = np.arange(0,180,5)*u.deg
    
    for theta in thetas:
        print(theta)
        generate_streakline_fiducial(vnorm=vnorm, theta=theta)

def show_fiducial():
    """"""
    pkl = pickle.load(open('../data/fiducial_noperturb.pkl', 'rb'))
    cg0 = pkl['cg']
    pkl = pickle.load(open('../data/fiducial_perturb.pkl', 'rb'))
    cg = pkl['cg']
    
    w0 = gd.PhaseSpacePosition(cg0.transform_to(gc_frame).cartesian)
    w = gd.PhaseSpacePosition(cg.transform_to(gc_frame).cartesian)
    
    orbit0 = ham.integrate_orbit(w0, dt=0.1*u.Myr, n_steps=1)
    orbit = ham.integrate_orbit(w, dt=0.1*u.Myr, n_steps=1)
    
    E0 = orbit0.energy()[0]
    E = orbit.energy()[0]
    dE = 1 - E/E0
    perturbed = np.abs(dE)>0.001
    
    
    wangle = 180*u.deg
    ind = (cg0.phi1.wrap_at(wangle)>-80*u.deg) & (cg0.phi1.wrap_at(wangle)<0*u.deg)
    pvr = np.polyfit(cg0.phi1.wrap_at(wangle)[ind], cg0.radial_velocity[ind], 4)
    polyvr = np.poly1d(pvr)
    dVr = cg.radial_velocity - polyvr(cg.phi1.wrap_at(wangle))*u.km/u.s
    
    pd = np.polyfit(cg0.phi1.wrap_at(wangle)[ind], cg0.distance[ind], 4)
    polyd = np.poly1d(pd)
    dDist = cg.distance - polyd(cg.phi1.wrap_at(wangle))*u.pc
    
    plt.close()
    fig, ax = plt.subplots(3,1,figsize=(10,10), sharex=True)
    
    plt.sca(ax[0])
    plt.plot(cg.phi1.wrap_at(wangle), cg.phi2, 'k.', ms=1)
    plt.plot(cg.phi1.wrap_at(wangle)[perturbed], cg.phi2[perturbed], 'ro', ms=1)
    plt.xlim(-60,-20)
    plt.ylim(-10,10)
    
    plt.sca(ax[1])
    plt.plot(cg.phi1.wrap_at(wangle), dVr, 'k.', ms=1)
    plt.plot(cg.phi1.wrap_at(wangle)[perturbed], dVr[perturbed], 'ro', ms=1)
    plt.ylim(-10,10)
    
    plt.sca(ax[2])
    plt.plot(cg.phi1.wrap_at(wangle), dDist, 'k.', ms=1)
    plt.ylim(-200,200)
    #plt.plot(cg.phi1.wrap_at(wangle), dE, 'k.', ms=1)
    #plt.axhline(-0.001)
    #plt.axhline(0.001)
    
    plt.tight_layout()


# idealized model
wangle = 180*u.deg

def make_ball_stream():
    """"""
    
    np.random.seed(358)
    
    # GD-1 orbit
    pos = np.load('/home/ana/projects/gd1_spur/data/log_orbit.npy')
    phi1, phi2, d, pm1, pm2, vr = pos
    c = gc.GD1(phi1=phi1*u.deg, phi2=phi2*u.deg, distance=d*u.kpc, pm_phi1_cosphi2=pm1*u.mas/u.yr, pm_phi2=pm2*u.mas/u.yr, radial_velocity=vr*u.km/u.s)
    w0 = gd.PhaseSpacePosition(c.transform_to(gc_frame).cartesian)
    
    # best-fitting orbit
    dt = 0.5*u.Myr
    n_steps = 120
    wangle = 180*u.deg

    # integrate back in time
    fit_orbit = ham.integrate_orbit(w0, dt=dt, n_steps=n_steps)
    
    prog_phi0 = -40*u.deg
    cg = fit_orbit.to_coord_frame(gc.GD1, galactocentric_frame=gc_frame)
    prog_i = np.abs(cg.phi1.wrap_at(180*u.deg) - prog_phi0).argmin()
    w0 = fit_orbit[prog_i]
    
    dt = 0.5*u.Myr
    tback = 3*u.Gyr
    n_steps = np.int64((tback/dt).decompose())
    #t_offset = 0.5*u.Gyr
    #n_offset = np.int64((t_offset/dt).decompose())
    
    # integrate back in time
    orbit = ham.integrate_orbit(w0, dt=-dt, n_steps=n_steps)
    
    # initialize ball
    prog_w0 = orbit[-1]
    Nstar = 500
    sigma = 0.7*u.km/u.s
    x = prog_w0.xyz[:,np.newaxis] + np.zeros(Nstar*3).reshape(-1,Nstar)
    v = np.random.randn(Nstar*3).reshape(-1,Nstar)*sigma + prog_w0.v_xyz[:,np.newaxis]
    c = coord.Galactocentric(x[0], x[1], x[2], v[0], v[1], v[2])
    prog_w = gd.PhaseSpacePosition(c.transform_to(gc_frame).cartesian)
    
    # integrate forward
    stream_gc = ham.integrate_orbit(prog_w, dt=dt, n_steps=n_steps)
    stream = stream_gc[-1].to_coord_frame(gc.GD1, galactocentric_frame=gc_frame)
    
    # rv trend
    ind = (cg.phi1.wrap_at(wangle)>-80*u.deg) & (cg.phi1.wrap_at(wangle)<0*u.deg)
    pvr = np.polyfit(cg.phi1.wrap_at(wangle)[ind], cg.radial_velocity[ind], 4)
    polyvr = np.poly1d(pvr)
    dVr = stream.radial_velocity - polyvr(stream.phi1.wrap_at(wangle))*u.km/u.s
    
    # observations
    t = Table.read('../data/master_catalog.fits')
    ind = (t['priority']<=3) & (t['delta_Vrad']<-1) & (t['delta_Vrad']>-20) & (t['FeH']<-2) & ((t['field']==1) | (t['field']==3) | (t['field']==8))
    t = t[ind]
    
    plt.close()
    fig, ax = plt.subplots(2,1,figsize=(10,8), sharex=True)
    
    plt.sca(ax[0])
    plt.plot(stream.phi1.wrap_at(wangle), stream.phi2, 'k.')
    #plt.plot(cg.phi1.wrap_at(wangle), cg.phi2, 'r-')
    
    plt.ylim(-5,5)
    plt.ylabel('$\phi_2$ [deg]')
    
    plt.sca(ax[1])
    plt.plot(stream.phi1.wrap_at(wangle), dVr, 'k.')
    plt.axhline(0, color='r')
    #plt.plot(cg.phi1.wrap_at(wangle), cg.radial_velocity, 'r-')
    
    plt.xlim(-60,-20)
    plt.ylim(-5,5)
    plt.xlabel('$\phi_1$ [deg]')
    plt.ylabel('$\Delta V_r$ [km s$^{-1}$]')
    
    plt.tight_layout()


# evaluate models

def loop_stars(N=1000, t_impact=0.5*u.Gyr, bnorm=0.06*u.kpc, bx=0.06*u.kpc, vnorm=200*u.km/u.s, vx=200*u.km/u.s, M=1e7*u.Msun):
    """Identify loop stars"""
    
    ########################
    # Perturber at encounter
    
    pkl = Table.read('../data/gap_present.fits')
    xunit = pkl['x_gap'].unit
    vunit = pkl['v_gap'].unit
    c = coord.Galactocentric(x=pkl['x_gap'][0]*xunit, y=pkl['x_gap'][1]*xunit, z=pkl['x_gap'][2]*xunit, v_x=pkl['v_gap'][0]*vunit, v_y=pkl['v_gap'][1]*vunit, v_z=pkl['v_gap'][2]*vunit, **gc_frame_dict)
    w0 = gd.PhaseSpacePosition(c.transform_to(gc_frame).cartesian)
    
    # best-fitting orbit
    dt = 0.5*u.Myr
    n_steps = np.int(t_impact / dt)

    # integrate back in time
    fit_orbit = ham.integrate_orbit(w0, dt=-dt, n_steps=n_steps)
    x = fit_orbit.pos.get_xyz()
    v = fit_orbit.vel.get_d_xyz()
    
    # gap location at the time of impact
    xgap = x[:,-1]
    vgap = v[:,-1]
    
    i = np.array([1,0,0], dtype=float)
    j = np.array([0,1,0], dtype=float)
    k = np.array([0,0,1], dtype=float)
    
    # find positional plane
    bi = np.cross(j, vgap)
    bi = bi/np.linalg.norm(bi)
    
    bj = np.cross(vgap, bi)
    bj = bj/np.linalg.norm(bj)
    
    # pick b
    by = np.sqrt(bnorm**2 - bx**2)
    b = bx*bi + by*bj
    xsub = xgap + b
    
    # find velocity plane
    vi = np.cross(vgap, b)
    vi = vi/np.linalg.norm(vi)
    
    vj = np.cross(b, vi)
    vj = vj/np.linalg.norm(vj)
    
    # pick v
    vy = np.sqrt(vnorm**2 - vx**2)
    vsub = vx*vi + vy*vj
    
    # load one orbital point
    pos = np.load('../data/log_orbit.npy')
    phi1, phi2, d, pm1, pm2, vr = pos

    c = gc.GD1(phi1=phi1*u.deg, phi2=phi2*u.deg, distance=d*u.kpc, pm_phi1_cosphi2=pm1*u.mas/u.yr, pm_phi2=pm2*u.mas/u.yr, radial_velocity=vr*u.km/u.s)
    w0 = gd.PhaseSpacePosition(c.transform_to(gc_frame).cartesian)
    
    # best-fitting orbit
    dt = 0.5*u.Myr
    n_steps = np.int(t_impact / dt)

    # integrate back in time
    fit_orbit = ham.integrate_orbit(w0, dt=-dt, n_steps=n_steps)
    x = fit_orbit.pos.get_xyz()
    v = fit_orbit.vel.get_d_xyz()
    xend = x[:,-1]
    vend = v[:,-1]
    
    # fine-sampled orbit at the time of impact
    c_impact = coord.Galactocentric(x=xend[0], y=xend[1], z=xend[2], v_x=vend[0], v_y=vend[1], v_z=vend[2], **gc_frame_dict)
    w0_impact = gd.PhaseSpacePosition(c_impact.transform_to(gc_frame).cartesian)
    
    # best-fitting orbit
    t = 56*u.Myr
    n_steps = N
    dt = t/n_steps

    stream = ham.integrate_orbit(w0_impact, dt=dt, n_steps=n_steps)
    xs = stream.pos.get_xyz()
    vs = stream.vel.get_d_xyz()
    
    Ep = 0.5*(225*u.km/u.s)**2*np.log(np.sum(xs.value**2, axis=0))
    Ek = 0.5*np.sum(vs**2, axis=0)
    Etot = Ep + Ek
    
    Ep_true = stream.potential_energy()
    Etot_true = stream.energy()
    Ek_true = stream.kinetic_energy()
    
    #################
    # Encounter setup
    
    # impact parameters
    Tenc = 0.01*u.Gyr
    T = 0.5*u.Gyr
    dt = 0.05*u.Myr
    rs = 0*u.pc
    
    # potential parameters
    potential = 3
    Vh = 225*u.km/u.s
    q = 1*u.Unit(1)
    rhalo = 0*u.pc
    par_pot = np.array([Vh.to(u.m/u.s).value, q.value, rhalo.si.value])
    #par_pot = np.array([Vh.si.value, q.value, rhalo.si.value])
    
    # generate unperturbed model
    # calculate model
    potential_perturb = 1
    par_perturb = np.array([0*M.si.value, 0., 0., 0.])
    
    x1, x2, x3, v1, v2, v3 = interact.general_interact(par_perturb, xsub.si.value, vsub.to(u.m/u.s).value, Tenc.to(u.s).value, t_impact.to(u.s).value, dt.to(u.s).value, par_pot, potential, potential_perturb, xs[0].si.value, xs[1].si.value, xs[2].si.value, vs[0].to(u.m/u.s).value, vs[1].to(u.m/u.s).value, vs[2].to(u.m/u.s).value)
    c0 = coord.Galactocentric(x=x1*u.m, y=x2*u.m, z=x3*u.m, v_x=v1*u.m/u.s, v_y=v2*u.m/u.s, v_z=v3*u.m/u.s, **gc_frame_dict)
    cg0 = c0.transform_to(gc.GD1)
    
    # generate stream model
    potential_perturb = 1
    par_perturb = np.array([M.si.value, 0., 0., 0.])

    x1, x2, x3, v1, v2, v3 = interact.general_interact(par_perturb, xsub.si.value, vsub.to(u.m/u.s).value, Tenc.to(u.s).value, t_impact.to(u.s).value, dt.to(u.s).value, par_pot, potential, potential_perturb, xs[0].si.value, xs[1].si.value, xs[2].si.value, vs[0].to(u.m/u.s).value, vs[1].to(u.m/u.s).value, vs[2].to(u.m/u.s).value)
    #x1, x2, x3, v1, v2, v3 = interact.general_interact(par_perturb, xsub.si.value, vsub.si.value, Tenc.si.value, t_impact.si.value, dt.si.value, par_pot, potential, potential_perturb, xs[0].si.value, xs[1].si.value, xs[2].si.value, vs[0].si.value, vs[1].si.value, vs[2].si.value)
    stream = {}
    stream['x'] = (np.array([x1, x2, x3])*u.m).to(u.kpc)
    stream['v'] = (np.array([v1, v2, v3])*u.m/u.s).to(u.km/u.s)
    
    c = coord.Galactocentric(x=stream['x'][0], y=stream['x'][1], z=stream['x'][2], v_x=stream['v'][0], v_y=stream['v'][1], v_z=stream['v'][2], **gc_frame_dict)
    cg = c.transform_to(gc.GD1)
    
    Ep_stream = 0.5*(225*u.km/u.s)**2*np.log(stream['x'][0].value**2 + stream['x'][1].value**2 + stream['x'][2].value**2)
    Ek_stream = 0.5*(stream['v'][0]**2 + stream['v'][1]**2 + stream['v'][2]**2)
    Etot_stream = Ep_stream + Ek_stream
    
    rE = np.abs(1 - Etot_stream/Etot)
    dE = Etot - Etot_stream
    Ntrue = np.size(rE)
    N2 = int(Ntrue/2)
    
    m1 = np.median(rE[:N2])
    m2 = np.median(rE[N2:])
    
    offset = 0.001
    top1 = np.percentile(dE[:N2], 3)*dE.unit
    top2 = np.percentile(dE[N2:], 92)*dE.unit
    ind_loop1 = np.where(dE[:N2]<top1)[0][0]
    ind_loop2 = np.where(dE[N2:]>top2)[0][-1]
    
    #ind_loop1 = np.where(rE[:N2]>m1+offset)[0][0]
    #ind_loop2 = np.where(rE[N2:]>m2+offset)[0][-1]
    
    print(ind_loop1, ind_loop2)
    
    loop_mask = np.zeros(Ntrue, dtype=bool)
    loop_mask[ind_loop1:ind_loop2+N2] = True
    phi1_mask = (cg.phi1.wrap_at(180*u.deg)>-50.*u.deg) & (cg.phi1.wrap_at(180*u.deg)<-30*u.deg)
    loop_mask = loop_mask & phi1_mask
    print(np.sum(loop_mask))
    
    # chi spur
    sp = np.load('../data/spur_track.npz')
    f = scipy.interpolate.interp1d(sp['x'], sp['y'], kind='quadratic')
    
    Nloop = np.sum(loop_mask)
    chi_spur = np.sum((cg.phi2[loop_mask].value - f(cg.phi1.wrap_at(180*u.deg).value[loop_mask]))**2/0.5**2)/Nloop
    print(chi_spur)
    
    # chi vr
    ind = (cg.phi1.wrap_at(180*u.deg)>-60*u.deg) & (cg.phi1.wrap_at(180*u.deg)<-20*u.deg)
    pvr = np.polyfit(cg.phi1[~loop_mask & ind].wrap_at(180*u.deg), cg.radial_velocity[~loop_mask & ind], 4)
    polyvr = np.poly1d(pvr)
    dVr = cg.radial_velocity - polyvr(cg.phi1.wrap_at(180*u.deg))*u.km/u.s
    
    phi1_list = np.array([-33.7, -30])*u.deg
    delta_phi1 = 1*u.deg
    mu_vr = np.array([0,0])*u.km/u.s
    sigma_vr = np.array([1,1])*u.km/u.s
    
    chi_vr = 0
    for e, phi in enumerate(phi1_list):
        ind_phi = np.abs(cg.phi1.wrap_at(180*u.deg) - phi) < delta_phi1
        ind_phi0 = np.abs(cg0.phi1.wrap_at(180*u.deg) - phi) < delta_phi1
        print(np.median(cg.radial_velocity[ind_phi & loop_mask]), np.median(cg0.radial_velocity[ind_phi0]))
        mu_vr[e] = np.median(cg0.radial_velocity[ind_phi0])
        chi_vr += (np.median(cg.radial_velocity[ind_phi & loop_mask]) - mu_vr[e])**2*sigma_vr[e]**-2
    
    outdict = {'delta_phi1': delta_phi1, 'phi1_list': phi1_list, 'mu_vr': mu_vr, 'sigma_vr': sigma_vr}
    pickle.dump(outdict, open('../data/vr_unperturbed.pkl', 'wb'))
    print(chi_vr)
    
    plt.close()
    fig, ax = plt.subplots(3,1,figsize=(10,7), sharex=True)
    
    plt.sca(ax[0])
    plt.plot(cg.phi1.wrap_at(180*u.deg), dE, 'o')
    plt.plot(cg.phi1.wrap_at(180*u.deg)[loop_mask], dE[loop_mask], 'o')
    
    plt.sca(ax[1])
    plt.plot(cg.phi1.wrap_at(180*u.deg), cg.phi2, 'o')
    plt.plot(cg.phi1.wrap_at(180*u.deg)[loop_mask], cg.phi2[loop_mask], 'o')
    plt.plot(sp['x'], sp['y'], 'r-', alpha=0.3)
    
    plt.xlim(-60,-20)
    plt.ylim(-5,5)
    
    plt.sca(ax[2])
    plt.plot(cg.phi1.wrap_at(180*u.deg), dVr, 'o')
    plt.plot(cg.phi1.wrap_at(180*u.deg)[loop_mask], dVr[loop_mask], 'o')
    plt.ylim(-5,5)
    
    plt.tight_layout()

def tophat(x, base_level, hat_level, hat_mid, hat_width):
    ret=[]
    for xx in x:
        if hat_mid-hat_width/2. < xx < hat_mid+hat_width/2.:
            ret.append(hat_level)
        else:
            ret.append(base_level)
    return np.array(ret)

def lnprob(x, params_units, xend, vend, dt_coarse, dt_fine, Tenc, Tstream, Nstream, par_pot, potential, potential_perturb, poly, wangle, delta_phi2, Nb, bins, bc, base_mask, hat_mask, Nside_min, f_gap, gap_position, gap_width, N2, percentile1, percentile2, phi1_min, phi1_max, phi2_err, spx, spy, quad_phi1, quad_phi2, Nquad, phi1_list, delta_phi1, mu_vr, sigma_vr, chigap_max, chispur_max):
    """Calculate pseudo-likelihood of a stream==orbit model, evaluating against the gap location & width, spur location & extent, and radial velocity offsets"""
    
    if (x[0]<0) | (x[0]>14) | (np.sqrt(x[3]**2 + x[4]**2)>500):
        return -np.inf
    
    x[5] = 10**x[5]
    params = [x_*u_ for x_, u_ in zip(x, params_units)]
    if potential_perturb==1:
        t_impact, bx, by, vx, vy, M, Tgap = params
        par_perturb = np.array([M.si.value, 0., 0., 0.])
    else:
        t_impact, bx, by, vx, vy, M, rs, Tgap = params
        par_perturb = np.array([M.si.value, rs.si.value, 0., 0., 0.])
        if x[6]<0:
            return -np.inf
    
    if (Tgap<0*u.Myr) | (Tgap>Tstream):
        return -np.inf
    
    # calculate model
    x1, x2, x3, v1, v2, v3, dE = interact.abinit_interaction(xend, vend, dt_coarse.si.value, dt_fine.si.value, t_impact.si.value, Tenc.si.value, Tstream.si.value, Tgap.si.value, Nstream, par_pot, potential, par_perturb, potential_perturb, bx.si.value, by.si.value, vx.si.value, vy.si.value)
    
    c = coord.Galactocentric(x=x1*u.m, y=x2*u.m, z=x3*u.m, v_x=v1*u.m/u.s, v_y=v2*u.m/u.s, v_z=v3*u.m/u.s, **gc_frame_dict)
    cg = c.transform_to(gc.GD1)
    
    # spur chi^2
    top1 = np.percentile(dE[:N2], percentile1)
    top2 = np.percentile(dE[N2:], percentile2)
    ind_loop1 = np.where(dE[:N2]<top1)[0][0]
    ind_loop2 = np.where(dE[N2:]>top2)[0][-1]
    
    f = scipy.interpolate.interp1d(spx, spy, kind='quadratic')
    
    aloop_mask = np.zeros(Nstream, dtype=bool)
    aloop_mask[ind_loop1:ind_loop2+N2] = True
    phi1_mask = (cg.phi1.wrap_at(wangle)>phi1_min) & (cg.phi1.wrap_at(wangle)<phi1_max)
    loop_mask = aloop_mask & phi1_mask
    Nloop = np.sum(loop_mask)
    
    loop_quadrant = (cg.phi1.wrap_at(wangle)[loop_mask]>quad_phi1) & (cg.phi2[loop_mask]>quad_phi2)
    if np.sum(loop_quadrant)<Nquad:
        return -np.inf
    
    chi_spur = np.sum((cg.phi2[loop_mask].value - f(cg.phi1.wrap_at(wangle).value[loop_mask]))**2/phi2_err**2)/Nloop
    
    # vr chi^2
    chi_vr = 0
    for e, phi in enumerate(phi1_list):
        ind_phi = np.abs(cg.phi1.wrap_at(180*u.deg) - phi) < delta_phi1
        #print(np.median(cg.radial_velocity[ind_phi & loop_mask]), np.median(cg.radial_velocity[ind_phi & ~loop_mask]))
        chi_vr += (np.median(cg.radial_velocity[ind_phi & aloop_mask]) - mu_vr[e])**2*sigma_vr[e]**-2
    
    # gap chi^2
    phi2_mask = np.abs(cg.phi2.value - poly(cg.phi1.wrap_at(wangle).value))<delta_phi2
    h_model, be = np.histogram(cg.phi1[phi2_mask].wrap_at(wangle).value, bins=bins)
    yerr = np.sqrt(h_model+1)
    
    model_base = np.median(h_model[base_mask])
    model_hat = np.median(h_model[hat_mask])
    if (model_base<Nside_min) | (model_hat>model_base*f_gap):
        return -np.inf
    
    ytop_model = tophat(bc, model_base, model_hat,  gap_position, gap_width)
    chi_gap = np.sum((h_model - ytop_model)**2/yerr**2)/Nb
    
    if np.isfinite(chi_gap) & np.isfinite(chi_spur) & np.isfinite(chi_vr):
        return -(chi_gap + chi_spur + chi_vr)
    else:
        return -np.inf

def lnprob_verbose(x, params_units, xend, vend, dt_coarse, dt_fine, Tenc, Tstream, Nstream, par_pot, potential, potential_perturb, poly, wangle, delta_phi2, Nb, bins, bc, base_mask, hat_mask, Nside_min, f_gap, gap_position, gap_width, N2, percentile1, percentile2, phi1_min, phi1_max, phi2_err, spx, spy, quad_phi1, quad_phi2, Nquad, phi1_list, delta_phi1, mu_vr, sigma_vr, chigap_max, chispur_max, colored=True, plot_comp=True, chi_label=True):
    """Calculate pseudo-likelihood of a stream==orbit model, evaluating against the gap location & width, spur location & extent, and radial velocity offsets"""
    
    if (x[0]<0) | (x[0]>14) | (np.sqrt(x[3]**2 + x[4]**2)>500):
        return -np.inf
    
    x[5] = 10**x[5]
    params = [x_*u_ for x_, u_ in zip(x, params_units)]
    if potential_perturb==1:
        t_impact, bx, by, vx, vy, M, Tgap = params
        par_perturb = np.array([M.si.value, 0., 0., 0.])
    else:
        t_impact, bx, by, vx, vy, M, rs, Tgap = params
        par_perturb = np.array([M.si.value, rs.si.value, 0., 0., 0.])
        if x[6]<0:
            return -np.inf
    
    if (Tgap<0*u.Myr) | (Tgap>Tstream):
        return -np.inf
    
    # calculate model
    x1, x2, x3, v1, v2, v3, dE = interact.abinit_interaction(xend, vend, dt_coarse.si.value, dt_fine.si.value, t_impact.si.value, Tenc.si.value, Tstream.si.value, Tgap.si.value, Nstream, par_pot, potential, par_perturb, potential_perturb, bx.si.value, by.si.value, vx.si.value, vy.si.value)
    
    c = coord.Galactocentric(x=x1*u.m, y=x2*u.m, z=x3*u.m, v_x=v1*u.m/u.s, v_y=v2*u.m/u.s, v_z=v3*u.m/u.s, **gc_frame_dict)
    cg = c.transform_to(gc.GD1)
    
    # spur chi^2
    top1 = np.percentile(dE[:N2], percentile1)
    top2 = np.percentile(dE[N2:], percentile2)
    ind_loop1 = np.where(dE[:N2]<top1)[0][0]
    ind_loop2 = np.where(dE[N2:]>top2)[0][-1]
    
    f = scipy.interpolate.interp1d(spx, spy, kind='quadratic')
    
    aloop_mask = np.zeros(Nstream, dtype=bool)
    aloop_mask[ind_loop1:ind_loop2+N2] = True
    phi1_mask = (cg.phi1.wrap_at(wangle)>phi1_min) & (cg.phi1.wrap_at(wangle)<phi1_max)
    loop_mask = aloop_mask & phi1_mask
    Nloop = np.sum(loop_mask)
    
    loop_quadrant = (cg.phi1.wrap_at(wangle)[loop_mask]>quad_phi1) & (cg.phi2[loop_mask]>quad_phi2)
    if np.sum(loop_quadrant)<Nquad:
        return -np.inf
    
    chi_spur = np.sum((cg.phi2[loop_mask].value - f(cg.phi1.wrap_at(wangle).value[loop_mask]))**2/phi2_err**2)/Nloop
    
    # vr chi^2
    chi_vr = 0
    for e, phi in enumerate(phi1_list):
        ind_phi = np.abs(cg.phi1.wrap_at(180*u.deg) - phi) < delta_phi1
        #print(np.median(cg.radial_velocity[ind_phi & loop_mask]), np.median(cg.radial_velocity[ind_phi & ~loop_mask]))
        chi_vr += (np.median(cg.radial_velocity[ind_phi & aloop_mask]) - mu_vr[e])**2*sigma_vr[e]**-2
    
    print(chi_vr)
    # gap chi^2
    phi2_mask = np.abs(cg.phi2.value - poly(cg.phi1.wrap_at(wangle).value))<delta_phi2
    h_model, be = np.histogram(cg.phi1[phi2_mask].wrap_at(wangle).value, bins=bins)
    yerr = np.sqrt(h_model+1)
    
    model_base = np.median(h_model[base_mask])
    model_hat = np.median(h_model[hat_mask])
    if (model_base<Nside_min) | (model_hat>model_base*f_gap):
        return -np.inf
    
    ytop_model = tophat(bc, model_base, model_hat,  gap_position, gap_width)
    chi_gap = np.sum((h_model - ytop_model)**2/yerr**2)/Nb
    
    plt.close()
    fig, ax = plt.subplots(3,2,figsize=(13,9))
    
    plt.sca(ax[0][0])
    plt.plot(bc, h_model, 'o')
    if plot_comp:
        plt.plot(bc, ytop_model, 'k-')

    if chi_label:
        plt.text(0.95, 0.15, '$\chi^2_{{gap}}$ = {:.2f}'.format(chi_gap), ha='right', transform=plt.gca().transAxes, fontsize='small')
    plt.ylabel('N')
    plt.xlim(-60,-20)
    
    plt.sca(ax[1][0])
    plt.plot(cg.phi1.wrap_at(wangle).value, dE, 'o')
    if colored:
        plt.plot(cg.phi1.wrap_at(wangle).value[aloop_mask], dE[aloop_mask], 'o')
    plt.ylabel('$\Delta$ E')
    
    plt.sca(ax[2][0])
    plt.plot(cg.phi1.wrap_at(wangle).value, cg.phi2.value, 'o')
    if colored:
        plt.plot(cg.phi1.wrap_at(wangle).value[loop_mask], cg.phi2.value[loop_mask], 'o')
    if plot_comp:
        isort = np.argsort(cg.phi1.wrap_at(wangle).value[loop_mask])
        plt.plot(cg.phi1.wrap_at(wangle).value[loop_mask][isort], f(cg.phi1.wrap_at(wangle).value[loop_mask])[isort], 'k-')
    
    if chi_label:
        plt.text(0.95, 0.15, '$\chi^2_{{spur}}$ = {:.2f}'.format(chi_spur), ha='right', transform=plt.gca().transAxes, fontsize='small')
    plt.xlabel('$\phi_1$ [deg]')
    plt.ylabel('$\phi_2$ [deg]')
    plt.xlim(-60,-20)
    plt.ylim(-5,5)
    
    plt.sca(ax[0][1])
    plt.plot(c.x.to(u.kpc), c.y.to(u.kpc), 'o')
    if colored:
        plt.plot(c.x.to(u.kpc)[loop_mask], c.y.to(u.kpc)[loop_mask], 'o') #, color='orange')
    
    plt.xlabel('x [kpc]')
    plt.ylabel('y [kpc]')
    
    plt.sca(ax[1][1])
    cr = np.sqrt(c.x**2 + c.y**2)
    plt.plot(cr.to(u.kpc), c.z.to(u.kpc), 'o')
    if colored:
        plt.plot(cr.to(u.kpc)[loop_mask], c.z.to(u.kpc)[loop_mask], 'o') #, color='orange')
    
    plt.xlabel('R [kpc]')
    plt.ylabel('z [kpc]')
    
    plt.sca(ax[2][1])
    isort = np.argsort(cg.phi1.wrap_at(wangle).value[~aloop_mask])
    vr0 = np.interp(cg.phi1.wrap_at(wangle).value, cg.phi1.wrap_at(wangle).value[~aloop_mask][isort], cg.radial_velocity.to(u.km/u.s)[~aloop_mask][isort])*u.km/u.s
    dvr = vr0 - cg.radial_velocity.to(u.km/u.s)
    plt.plot(cg.phi1.wrap_at(wangle).value, dvr, 'o')
    if colored:
        plt.plot(cg.phi1.wrap_at(wangle).value[loop_mask], dvr[loop_mask], 'o')
    
    if chi_label:
        print(chi_vr)
        plt.text(0.95, 0.15, '$\chi^2_{{V_r}}$ = {:.2f}'.format(chi_vr), ha='right', transform=plt.gca().transAxes, fontsize='small')
    plt.xlabel('$\phi_1$ [deg]')
    plt.ylabel('$\Delta$ $V_r$ [km s$^{-1}$]')
    plt.ylim(-5,5)
    plt.xlim(-60,-20)
    
    plt.tight_layout()
    
    return fig, ax, chi_gap, chi_spur, chi_vr, np.sum(loop_quadrant), -(chi_gap + chi_spur + chi_vr)

def sort_on_runtime(p):
    """Improve runtime by starting longest jobs first (sorts on first parameter -- in our case, the encounter time)"""
    
    p = np.atleast_2d(p)
    idx = np.argsort(p[:, 0])[::-1]
    
    return p[idx], idx

def run(cont=False, steps=100, nwalkers=100, nth=8, label='', potential_perturb=1, test=False):
    """"""
    
    pkl = Table.read('../data/gap_present.fits')
    xunit = pkl['x_gap'].unit
    vunit = pkl['v_gap'].unit
    c = coord.Galactocentric(x=pkl['x_gap'][0]*xunit, y=pkl['x_gap'][1]*xunit, z=pkl['x_gap'][2]*xunit, v_x=pkl['v_gap'][0]*vunit, v_y=pkl['v_gap'][1]*vunit, v_z=pkl['v_gap'][2]*vunit, **gc_frame_dict)
    w0 = gd.PhaseSpacePosition(c.transform_to(gc_frame).cartesian)
    xgap = np.array([w0.pos.x.si.value, w0.pos.y.si.value, w0.pos.z.si.value])
    vgap = np.array([w0.vel.d_x.si.value, w0.vel.d_y.si.value, w0.vel.d_z.si.value])
    
    # load orbital end point
    pos = np.load('../data/log_orbit.npy')
    phi1, phi2, d, pm1, pm2, vr = pos

    c_end = gc.GD1(phi1=phi1*u.deg, phi2=phi2*u.deg, distance=d*u.kpc, pm_phi1_cosphi2=pm1*u.mas/u.yr, pm_phi2=pm2*u.mas/u.yr, radial_velocity=vr*u.km/u.s)
    w0_end = gd.PhaseSpacePosition(c_end.transform_to(gc_frame).cartesian)
    xend = np.array([w0_end.pos.x.si.value, w0_end.pos.y.si.value, w0_end.pos.z.si.value])
    vend = np.array([w0_end.vel.d_x.si.value, w0_end.vel.d_y.si.value, w0_end.vel.d_z.si.value])
    
    dt_coarse = 0.5*u.Myr
    Tstream = 56*u.Myr
    Tgap = 29.176*u.Myr
    Nstream = 2000
    N2 = int(Nstream*0.5)
    dt_stream = Tstream/Nstream
    dt_fine = 0.05*u.Myr
    wangle = 180*u.deg
    Tenc = 0.01*u.Gyr
    
    # gap comparison
    bins = np.linspace(-60,-20,30)
    bc = 0.5 * (bins[1:] + bins[:-1])
    Nb = np.size(bc)
    Nside_min = 5
    f_gap = 0.5
    delta_phi2 = 0.5
    
    gap = np.load('../data/gap_properties.npz')
    phi1_edges = gap['phi1_edges']
    gap_position = gap['position']
    gap_width = gap['width']
    gap_yerr = gap['yerr']
    base_mask = ((bc>phi1_edges[0]) & (bc<phi1_edges[1])) | ((bc>phi1_edges[2]) & (bc<phi1_edges[3]))
    hat_mask = (bc>phi1_edges[4]) & (bc<phi1_edges[5])
    
    p = np.load('../data/polytrack.npy')
    poly = np.poly1d(p)
    x_ = np.linspace(-100,0,100)
    y_ = poly(x_)
    
    # spur comparison
    sp = np.load('../data/spur_track.npz')
    spx = sp['x']
    spy = sp['y']
    phi2_err = 0.2
    phi1_min = -50*u.deg
    phi1_max = -30*u.deg
    percentile1 = 3
    percentile2 = 92
    quad_phi1 = -32*u.deg
    quad_phi2 = 0.8*u.deg
    Nquad = 1
    
    # vr comparison
    pkl = pickle.load(open('../data/vr_unperturbed.pkl', 'rb'))
    phi1_list = pkl['phi1_list']
    delta_phi1 = pkl['delta_phi1']
    delta_phi1 = 1.5*u.deg
    mu_vr = pkl['mu_vr']
    sigma_vr = pkl['sigma_vr']
    #phi1_list = np.array([-33.7, -30])*u.deg
    #delta_phi1 = 1.5*u.deg
    #mu_vr = np.array([0,0])*u.km/u.s
    #sigma_vr = np.array([1,1])*u.km/u.s
    
    potential = 3
    Vh = 225*u.km/u.s
    q = 1*u.Unit(1)
    rhalo = 0*u.pc
    par_pot = np.array([Vh.si.value, q.value, rhalo.si.value])
    
    chigap_max = 0.6567184385873621
    chispur_max = 1.0213837095314207
    
    chigap_max = 0.8
    chispur_max = 1.2
    
    # parameters to sample
    t_impact = 0.5*u.Gyr
    bx = 40*u.pc
    by = 1*u.pc
    vx = 225*u.km/u.s
    vy = 1*u.km/u.s
    M = 7e6*u.Msun
    rs = 0.5*u.pc
    
    t_impact = 0.49*u.Gyr
    M = 2.2e7*u.Msun
    rs = 0.55*u.pc
    bx=21*u.pc
    by=15*u.pc
    vx=330*u.km/u.s
    vy=-370*u.km/u.s

    if potential_perturb==1:
        params_list = [t_impact, bx, by, vx, vy, M, Tgap]
    elif potential_perturb==2:
        params_list = [t_impact, bx, by, vx, vy, M, rs, Tgap]
    params_units = [p_.unit for p_ in params_list]
    params = [p_.value for p_ in params_list]
    params[5] = np.log10(params[5])
    
    model_args = [params_units, xend, vend, dt_coarse, dt_fine, Tenc, Tstream, Nstream, par_pot, potential, potential_perturb]
    gap_args = [poly, wangle, delta_phi2, Nb, bins, bc, base_mask, hat_mask, Nside_min, f_gap, gap_position, gap_width]
    spur_args = [N2, percentile1, percentile2, phi1_min, phi1_max, phi2_err, spx, spy, quad_phi1, quad_phi2, Nquad]
    vr_args = [phi1_list, delta_phi1, mu_vr, sigma_vr]
    lnp_args = [chigap_max, chispur_max]
    lnprob_args = model_args + gap_args + spur_args + vr_args + lnp_args
    
    ndim = len(params)
    if cont==False:
        seed = 614398
        np.random.seed(seed)
        p0 = [np.random.randn(ndim) for i in range(nwalkers)]
        p0 = (np.random.randn(ndim * nwalkers).reshape((nwalkers, ndim))*1e-4 + 1.)*np.array(params)[np.newaxis,:]
        
        seed = 3465
        prng = np.random.RandomState(seed)
        genstate = np.random.get_state()
    else:
        rgstate = pickle.load(open('../data/state{}.pkl'.format(label), 'rb'))
        genstate = rgstate['state']
        
        smp = np.load('../data/samples{}.npz'.format(label))
        flatchain = smp['chain']
        chain = np.transpose(flatchain.reshape(nwalkers, -1, ndim), (1,0,2))
        nstep = np.shape(chain)[0]
        flatchain = chain.reshape(nwalkers*nstep, ndim)
        
        positions = np.arange(-nwalkers, 0, dtype=np.int64)
        p0 = flatchain[positions]
    
    if test:
        N = np.size(p0[:,0])
        lnp = np.zeros(N)
        
        for i in range(N):
            args = copy.deepcopy(lnprob_args[:])
            lnp[i] = lnprob(p0[i], *args)
        
        print(lnp)
        print(N, np.sum(np.isfinite(lnp)))
        
        return
    
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, threads=nth, args=lnprob_args, runtime_sortingfn=sort_on_runtime)
    
    t1 = time.time()
    pos, prob, state = sampler.run_mcmc(p0, steps, rstate0=genstate)
    t2 = time.time()
    
    if cont==False:
        np.savez('../data/samples{}'.format(label), lnp=sampler.flatlnprobability, chain=sampler.flatchain, nwalkers=nwalkers)
    else:
        np.savez('../data/samples{}_temp'.format(label), lnp=sampler.flatlnprobability, chain=sampler.flatchain, nwalkers=nwalkers)
        np.savez('../data/samples{}'.format(label), lnp=np.concatenate([smp['lnp'], sampler.flatlnprobability]), chain=np.concatenate([smp['chain'], sampler.flatchain]), nwalkers=nwalkers)
    
    rgstate = {'state': state}
    pickle.dump(rgstate, open('../data/state{}.pkl'.format(label), 'wb'))
    
    print('Chain: {:5.2f} s'.format(t2 - t1))
    print('Average acceptance fraction: {}'.format(np.average(sampler.acceptance_fraction[0])))
    
    sampler.pool.terminate()


# chain diagnostics

def plot_corner(label='', full=False):
    """"""
    sampler = np.load('../data/samples{}.npz'.format(label))
    chain = sampler['chain']
    Npar = np.shape(chain)[1]
    print(np.sum(np.isfinite(sampler['lnp'])), np.size(sampler['lnp']))
    
    params = ['T', 'bx', 'by', 'vx', 'vy', 'logM', 'rs', 'Tgap']
    if full==False:
        params = ['T [Gyr]', 'B [pc]', 'V [km s$^{-1}$]', 'log M/M$_\odot$']
        abr = chain[:,:-3]
        abr[:,1] = np.sqrt(chain[:,1]**2 + chain[:,2]**2)
        abr[:,2] = np.sqrt(chain[:,3]**2 + chain[:,4]**2)
        abr[:,0] = chain[:,0]
        abr[:,3] = chain[:,5]
        if Npar>7:
            abr[:,3] = chain[:,6]
            abr[:,4] = chain[:,5]
            params = ['T [Gyr]', 'B [pc]', 'V [km s$^{-1}$]', '$r_s$ [pc]', 'log M/M$_\odot$']
            #lims = [[0.,2], [0.1,100], [10,1000], [0.001,1000], [5,9]]
        chain = abr
    
    plt.close()
    corner.corner(chain, bins=50, labels=params, plot_datapoints=True)
    
    plt.savefig('../plots/corner{}{:d}.png'.format(label, full))

def plot_chains(label=''):
    """"""
    sampler = np.load('../data/samples{}.npz'.format(label))
    chain = sampler['chain']
    lnp = sampler['lnp']
    nwalkers = sampler['nwalkers']
    ntot, Npar = np.shape(chain)
    nstep = int(ntot/nwalkers)
    steps = np.arange(nstep)
    
    Npanel = Npar + 1
    nrow = np.int(np.ceil(np.sqrt(Npanel)))
    ncol = np.int(np.ceil(Npanel/nrow))
    da = 2.5
    params = ['T [Gyr]', '$B_x$ [pc]', '$B_y$ [pc]', '$V_x$ [km s$^{-1}$]', '$V_y$ [km s$^{-1}$]', 'log M/M$_\odot$', '$r_s$ [pc]', '$T_{gap}$ [Myr]']
    
    plt.close()
    fig, ax = plt.subplots(nrow, ncol, figsize=(1.5*ncol*da, nrow*da), sharex=True)
    
    for i in range(Npar):
        plt.sca(ax[int(i/nrow)][i%nrow])
        plt.plot(steps, chain[:,i].reshape(nstep,-1), '-')
        plt.ylabel(params[i])
    
    plt.sca(ax[nrow-1][ncol-1])
    plt.plot(steps, lnp.reshape(nstep,-1), '-')
    plt.ylabel('ln P')
    
    for i in range(ncol):
        plt.sca(ax[nrow-1][i])
        plt.xlabel('Step')
        
    plt.tight_layout()
    plt.savefig('../plots/chain{}.png'.format(label))

def get_unique(label=''):
    """Save unique models in a separate file"""
    
    sampler = np.load('../data/samples{}.npz'.format(label))
    models, ind = np.unique(sampler['chain'], axis=0, return_index=True)
    ifinite = np.isfinite(sampler['lnp'][ind])
    
    np.savez('../data/unique_samples{}'.format(label), chain=models[ifinite], lnp=sampler['lnp'][ind][ifinite])

def check_model(fiducial=False, label='', rand=True, Nc=10, fast=True):
    """"""
    chain = np.load('../data/unique_samples{}.npz'.format(label))['chain']
    vnorm = np.sqrt(chain[:,3]**2 + chain[:,4]**2)
    if fast:
        ind = vnorm>490
    else:
        ind = vnorm<350
    chain = chain[ind]
    Nsample = np.shape(chain)[0]
    #print(Nsample)
    if rand:
        np.random.seed(59)
        ind = np.random.randint(Nsample, size=Nc)
    else:
        ind = np.load('../data/hull_points{}.npy'.format(label))
    Nc = np.size(ind)
    
    for k in range(Nc):
        x = chain[ind[k]]
        
        pkl = Table.read('../data/gap_present.fits')
        xunit = pkl['x_gap'].unit
        vunit = pkl['v_gap'].unit
        c = coord.Galactocentric(x=pkl['x_gap'][0]*xunit, y=pkl['x_gap'][1]*xunit, z=pkl['x_gap'][2]*xunit, v_x=pkl['v_gap'][0]*vunit, v_y=pkl['v_gap'][1]*vunit, v_z=pkl['v_gap'][2]*vunit, **gc_frame_dict)
        w0 = gd.PhaseSpacePosition(c.transform_to(gc_frame).cartesian)
        xgap = np.array([w0.pos.x.si.value, w0.pos.y.si.value, w0.pos.z.si.value])
        vgap = np.array([w0.vel.d_x.si.value, w0.vel.d_y.si.value, w0.vel.d_z.si.value])
        
        # load orbital end point
        pos = np.load('../data/log_orbit.npy')
        phi1, phi2, d, pm1, pm2, vr = pos

        c_end = gc.GD1(phi1=phi1*u.deg, phi2=phi2*u.deg, distance=d*u.kpc, pm_phi1_cosphi2=pm1*u.mas/u.yr, pm_phi2=pm2*u.mas/u.yr, radial_velocity=vr*u.km/u.s)
        w0_end = gd.PhaseSpacePosition(c_end.transform_to(gc_frame).cartesian)
        xend = np.array([w0_end.pos.x.si.value, w0_end.pos.y.si.value, w0_end.pos.z.si.value])
        vend = np.array([w0_end.vel.d_x.si.value, w0_end.vel.d_y.si.value, w0_end.vel.d_z.si.value])
        
        dt_coarse = 0.5*u.Myr
        Tstream = 56*u.Myr
        Tgap = 29.176*u.Myr
        Nstream = 2000
        N2 = int(Nstream*0.5)
        dt_stream = Tstream/Nstream
        dt_fine = 0.05*u.Myr
        wangle = 180*u.deg
        Tenc = 0.01*u.Gyr
        
        # gap comparison
        bins = np.linspace(-60,-20,30)
        bc = 0.5 * (bins[1:] + bins[:-1])
        Nb = np.size(bc)
        Nside_min = 5
        f_gap = 0.5
        delta_phi2 = 0.5
        
        gap = np.load('../data/gap_properties.npz')
        phi1_edges = gap['phi1_edges']
        gap_position = gap['position']
        gap_width = gap['width']
        gap_yerr = gap['yerr']
        base_mask = ((bc>phi1_edges[0]) & (bc<phi1_edges[1])) | ((bc>phi1_edges[2]) & (bc<phi1_edges[3]))
        hat_mask = (bc>phi1_edges[4]) & (bc<phi1_edges[5])
        
        p = np.load('../data/polytrack.npy')
        poly = np.poly1d(p)
        x_ = np.linspace(-100,0,100)
        y_ = poly(x_)
        
        # spur comparison
        sp = np.load('../data/spur_track.npz')
        spx = sp['x']
        spy = sp['y']
        phi2_err = 0.2
        phi1_min = -50*u.deg
        phi1_max = -30*u.deg
        percentile1 = 3
        percentile2 = 92
        quad_phi1 = -32*u.deg
        quad_phi2 = 0.8*u.deg
        Nquad = 1
        
        # vr comparison
        pkl = pickle.load(open('../data/vr_unperturbed.pkl', 'rb'))
        phi1_list = pkl['phi1_list']
        delta_phi1 = pkl['delta_phi1']
        mu_vr = pkl['mu_vr']
        sigma_vr = pkl['sigma_vr']
        #phi1_list = np.array([-33.7, -30])*u.deg
        #delta_phi1 = 1*u.deg
        #mu_vr = np.array([0,0])*u.km/u.s
        #sigma_vr = np.array([1,1])*u.km/u.s
        
        potential = 3
        Vh = 225*u.km/u.s
        q = 1*u.Unit(1)
        rhalo = 0*u.pc
        par_pot = np.array([Vh.si.value, q.value, rhalo.si.value])
        
        chigap_max = 0.6567184385873621
        chispur_max = 1.0213837095314207
        
        chigap_max = 0.8
        chispur_max = 1.2
        
        # parameters to sample
        potential_perturb = 2
        t_impact = 0.5*u.Gyr
        M = 1e7*u.Msun
        rs = 0.5*u.pc
        bx=60*u.pc
        by=1*u.pc
        vx=200*u.km/u.s
        vy=1*u.km/u.s

        if potential_perturb==1:
            params_list = [t_impact, bx, by, vx, vy, M, Tgap]
        elif potential_perturb==2:
            params_list = [t_impact, bx, by, vx, vy, M, rs, Tgap]
        params_units = [p_.unit for p_ in params_list]
        params = [p_.value for p_ in params_list]
        params[5] = np.log10(params[5])
        
        model_args = [params_units, xend, vend, dt_coarse, dt_fine, Tenc, Tstream, Nstream, par_pot, potential, potential_perturb]
        gap_args = [poly, wangle, delta_phi2, Nb, bins, bc, base_mask, hat_mask, Nside_min, f_gap, gap_position, gap_width]
        spur_args = [N2, percentile1, percentile2, phi1_min, phi1_max, phi2_err, spx, spy, quad_phi1, quad_phi2, Nquad]
        vr_args = [phi1_list, delta_phi1, mu_vr, sigma_vr]
        lnp_args = [chigap_max, chispur_max]
        lnprob_args = model_args + gap_args + spur_args + vr_args + lnp_args
        
        fig, ax, chi_gap, chi_spur, chi_vr, N, lnp = lnprob_verbose(x, *lnprob_args)
        print(lnp)
        
        plt.suptitle('  '.join(['{:.2g} {}'.format(x_, u_) for x_, u_ in zip(x,params_units)]), fontsize='medium')
        plt.tight_layout(rect=[0,0,1,0.96])
        
        plt.savefig('../plots/model_diag/likelihood_f{:d}_r{:d}_{}.png'.format(fast, rand, k))




