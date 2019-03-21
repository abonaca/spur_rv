import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from astropy.table import Table
import astropy.units as u
import astropy.coordinates as coord
from astropy.io import ascii

#from pyia import GaiaData
import gala.coordinates as gc
import gala.dynamics as gd
import gala.potential as gp
from gala.units import galactic

import pickle
import corner

gc_frame_dict = {'galcen_distance':8*u.kpc, 'z_sun':0*u.pc}
gc_frame = coord.Galactocentric(**gc_frame_dict)
ham = gp.Hamiltonian(gp.LogarithmicPotential(v_c=225*u.km/u.s, r_h=0*u.kpc, q1=1, q2=1, q3=1, units=galactic))
wangle = 180*u.deg

def delta_energy():
    """"""
    
    # orbit fitting setup
    n_steps = 1
    dt = 0.1*u.Myr

    pkl_p = pickle.load(open('/home/ana/projects/gd1_spur/data/fiducial_perturb_python3.pkl', 'rb'))
    c_p = pkl_p['cg']
    w0_p = gd.PhaseSpacePosition(c_p.transform_to(gc_frame).cartesian)
    orbit_p = ham.integrate_orbit(w0_p, dt=dt, n_steps=n_steps)
    e_p = orbit_p.energy()[0]
    
    pkl_np = pickle.load(open('/home/ana/projects/gd1_spur/data/fiducial_noperturb_python3.pkl', 'rb'))
    c_np = pkl_np['cg']
    w0_np = gd.PhaseSpacePosition(c_np.transform_to(gc_frame).cartesian)
    orbit_np = ham.integrate_orbit(w0_np, dt=dt, n_steps=n_steps)
    e_np = orbit_np.energy()[0]
    
    de = e_p - e_np
    ind = (c_np.phi1.wrap_at(wangle)>-50*u.deg) & (c_np.phi1.wrap_at(wangle)<-27*u.deg)
    
    plt.close()
    plt.figure(figsize=(10,5))
    plt.plot(c_np.phi1.wrap_at(wangle), de, 'k.')
    plt.plot(c_np.phi1.wrap_at(wangle)[ind], de[ind], 'r.')
    
    plt.tight_layout()

def configuration_space(perturbed=True):
    """"""
    
    pkl_p = pickle.load(open('/home/ana/projects/gd1_spur/data/fiducial_perturb_python3.pkl', 'rb'))
    c_p = pkl_p['cg']
    pkl_np = pickle.load(open('/home/ana/projects/gd1_spur/data/fiducial_noperturb_python3.pkl', 'rb'))
    c_np = pkl_np['cg']
    
    ind = (c_np.phi1.wrap_at(wangle)>-50*u.deg) & (c_np.phi1.wrap_at(wangle)<-27*u.deg)
    
    if perturbed:
        label = 'p'
        c = c_p[ind]
    else:
        label = 'np'
        c = c_np[ind]
    
    data = np.array([c.phi1.wrap_at(wangle), c.phi2, c.distance.to(u.kpc), c.radial_velocity, c.pm_phi1_cosphi2, c.pm_phi2]).T
    params = ['$\phi_1$ [deg]', '$\phi_2$ [deg]', 'd [kpc]', '$V_r$ [km s$^{-1}$]', '$\mu_{\phi_1}$ [mas yr$^{-1}$]', '$\mu_{\phi_2}$ [mas yr$^{-1}$]']
    ranges = [[-50, -22], [-2, 2], [7.4, 8.4], [-120, 70], [-13.8, -12], [-3.5, -2]]
    
    plt.close()
    fig, ax = plt.subplots(6,6,figsize=(13,13))
    corner.corner(data, bins=30, labels=params, plot_datapoints=True, plot_density=False, plot_contours=False, fig=fig, max_n_ticks=4, range=ranges)
    
    plt.tight_layout(h_pad=0, w_pad=0)
    plt.savefig('../plots/configuration_space_{}.png'.format(label))

def configuration_4d(observed=True, perturbed=True):
    """"""
    
    if observed:
        t = Table.read('/home/ana/data/gd1-better-selection.fits')
        label = 'obs'
        ind = (t['phi1']>-50) & (t['phi1']<-27)
        t = t[ind]
        data = np.array([t['phi1'], t['phi2'], t['pm_phi1_cosphi2'], t['pm_phi2']]).T
    else:
        pkl_p = pickle.load(open('/home/ana/projects/gd1_spur/data/fiducial_perturb_python3.pkl', 'rb'))
        c_p = pkl_p['cg']
        pkl_np = pickle.load(open('/home/ana/projects/gd1_spur/data/fiducial_noperturb_python3.pkl', 'rb'))
        c_np = pkl_np['cg']
        
        ind = (c_np.phi1.wrap_at(wangle)>-50*u.deg) & (c_np.phi1.wrap_at(wangle)<-27*u.deg)
        
        if perturbed:
            label = 'p'
            c = c_p[ind]
        else:
            label = 'np'
            c = c_np[ind]
    
        data = np.array([c.phi1.wrap_at(wangle), c.phi2, c.pm_phi1_cosphi2, c.pm_phi2]).T
        
    params = ['$\phi_1$ [deg]', '$\phi_2$ [deg]', '$\mu_{\phi_1}$ [mas yr$^{-1}$]', '$\mu_{\phi_2}$ [mas yr$^{-1}$]']
    #ranges = [[-50, -22], [-1, 2], [-13.8, -12], [-3.5, -2]]
    ranges = [[-50, -22], [-1, 2], [-14, -12], [-5, -1.5]]
    
    plt.close()
    fig, ax = plt.subplots(4,4,figsize=(10,10))
    corner.corner(data, bins=30, labels=params, plot_datapoints=True, plot_density=False, plot_contours=False, fig=fig, max_n_ticks=4, range=ranges, data_kwargs={'alpha': 0.8})
    
    plt.tight_layout(h_pad=0, w_pad=0)
    plt.savefig('../plots/configuration_4d_{}.png'.format(label))
