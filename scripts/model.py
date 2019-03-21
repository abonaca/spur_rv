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
#import gala.dynamics as gd
#import gala.potential as gp
#from gala.units import galactic

#import scipy.optimize
#import scipy.spatial
#import scipy.interpolate
#import time
#import emcee
#import corner

#from colossus.cosmology import cosmology
#from colossus.halo import concentration

#import interact
#import myutils

import pickle

def check_vr():
    pkl = pickle.load(open('/home/ana/projects/gd1_spur/data/fiducial.pkl', 'rb'))
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
    pkl = pickle.load(open('/home/ana/projects/gd1_spur/data/fiducial.pkl', 'rb'))
    cg = pkl['cg']
    ceq = cg.transform_to(coord.ICRS)
    cgal = cg.transform_to(coord.Galactocentric)
    
    plt.close()
    fig, ax = plt.subplots(2,2,figsize=(8,8))
    
    plt.sca(ax[0][0])
    plt.plot(cgal.x, cgal.y, 'k.', ms=1)
    
    plt.sca(ax[0][1])
    plt.plot(cgal.z, cgal.y, 'k.', ms=1)
    
    plt.sca(ax[1][0])
    plt.plot(cgal.x, cgal.z, 'k.', ms=1)
    
    plt.tight_layout()
