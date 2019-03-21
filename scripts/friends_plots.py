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

plt.style.use('si_lgray_ucondensed')

def phi1_vr():
    """"""
    tall = Table.read('/home/ana/data/gd1-better-selection.fits')
    t = Table.read('../data/master_catalog.fits')
    ind = (t['priority']<4) & (t['delta_Vrad']>-20) & (t['delta_Vrad']<-1)
    t = t[ind]
    
    spur = (t['field']==2) | (t['field']==4) | (t['field']==5)
    stream = ~spur
    
    cspur = mpl.cm.Blues_r(0.15)
    cstream = mpl.cm.Blues_r(0.4)
    colors = [cstream, cspur]
    
    plt.close()
    fig, ax = plt.subplots(2,1,figsize=(12,6), sharex=True)
    
    for e, ind in enumerate([stream, spur]):
        plt.sca(ax[0])
        plt.plot(t['phi1'][ind], t['phi2'][ind], 'o', color=colors[e], ms=10)
        
        plt.sca(ax[1])
        plt.plot(t['phi1'][ind], t['Vrad'][ind], 'o', color=colors[e], ms=10)
        plt.errorbar(t['phi1'][ind], t['Vrad'][ind], yerr=(t['lerr_Vrad'][ind], t['uerr_Vrad'][ind]), fmt='none', color=colors[e], zorder=0, lw=3)
    
    g = Table(fits.getdata('/home/ana/projects/GD1-DR2/output/gd1_members.fits'))

    plt.sca(ax[0])
    plt.scatter(g['phi1'], g['phi2'], s=g['pmem']*4, c=g['pmem'], cmap=mpl.cm.binary, vmin=0.5, vmax=1.1, zorder=0)
    plt.ylim(-3,3)
    plt.xlim(-45,-25)
    plt.ylabel('$\phi_2$ [deg]')
    
    plt.sca(ax[1])
    plt.ylim(-90,-30)
    plt.ylabel('Velocity [km s$^{-1}$]')
    plt.xlabel('$\phi_1$ [deg]')
    
    plt.tight_layout(h_pad=0)
    plt.savefig('../plots/friends/phi1_vr.png', dpi=200)

