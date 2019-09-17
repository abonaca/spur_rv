import numpy as np
from astropy.io import fits
from astropy.table import Table
import matplotlib.pyplot as plt

fn = '../data/acat_ta007897_ga1.charlieformat.fits'
dat = np.array(fits.getdata(fn))
acat_dtype_file = dat.dtype


def acat_format():
    """"""
    
    t = Table.read(fn)
    keys = t.colnames
    surveys = sorted(['PS', 'TMASS', 'WISE', 'GAIA', 'SDSS'])
    
    print(keys)
    
    for s in surveys:
        ks = [x for x in keys if s in x]
        print(ks)

def chelle_input_fields():
    """Check which acat fields are already in the gd1 hectochelle input catalog"""
    
    #tacat = Table.read(fn)
    aa = Table.read('../data/gd1_input_catalog.fits')
    #print(aa.colnames)
    
    st = np.zeros(len(aa), dtype=acat_dtype_file)
    st["RA"]  = aa["ra"]
    st["DEC"] = aa["dec"]
    #st["L"]   = aa["gaia_dr2_source.l"]
    #st["B"]   = aa["gaia_dr2_source.b"]
    
    # IDs
    #st["PS_ID"] = aa["ucal_fluxqz.obj_id"]
    #st["TMASS_ID"] = aa["tmass.twomass_id"]
    #st["WISE_ID"] = aa["allwise.source_id"]
    st["GAIA_ID"] = aa["source_id"]
    try:
        st["H3_ID"] = aa["H3_ID"]
    except(KeyError):
        st["H3_ID"] = -np.arange(len(aa))
    
    ## Fluxes from PS, SDSS, TMASS, WISE
    #for i, b in enumerate("GRIZY"):
        #band = "PS_{}".format(b)
        #flux, err = aa["ucal_fluxqz.median"][:, i], aa["ucal_fluxqz.err"][:, i]
        #st[band] = np.clip(-2.5 * np.log10(flux), 0.0, np.inf)
        #st[band + "_ERR"] = 1.086 * err / flux
    #for i, b in enumerate("UGRIZ"):
        #band = "SDSS_{}".format(b)
        #flux = aa["sdss_dr14_starsweep.psfflux"][:, i] * 1e-9,
        #err = 1. / np.sqrt(aa["sdss_dr14_starsweep.psfflux_ivar"][:, i]) * 1e-9
        #st[band] = np.clip(-2.5 * np.log10(flux), 0.0, np.inf)
        #st[band + "_ERR"] = 1.086 * err / flux
    #for i, b in enumerate("JHK"):
        #band = "TMASS_{}".format(b)
        #st[band] = aa["tmass.{}_m".format(b.lower())]
        #st[band + "_ERR"] = aa["tmass.{}_msigcom".format(b.lower())]
    #for i, b in enumerate(["W1", "W2"]):
        #band = "WISE_{}".format(b)
        #st[band] = aa["allwise.{}mpro".format(b.lower())]
        #st[band + "_ERR"] = aa["allwise.{}sigmpro".format(b.lower())]
    
    eq = ["PARALLAX", "PMRA", "PMDEC"]#, "RA", "DEC"]
    eq += ["PHOT_{}_MEAN_FLUX".format(b) for b in ["G", "BP", "RP"]]
    qs = eq + [q + "_ERROR" for q in eq]
    #qs += ["ASTROMETRIC_EXCESS_NOISE", "ASTROMETRIC_EXCESS_NOISE_SIG",
           #'VISIBILITY_PERIODS_USED', "PHOT_BP_RP_EXCESS_FACTOR"] #+ ["PARALLAX_OVER_ERROR"]

    for i, q in enumerate(qs):
        st["GAIA_" + q] = aa[q.lower()]

def lsd_query():
    """Columns to query from lsd"""
    
    gd1 = {'name': 'gd1chelle',
           'cols': ['source_id', 'ra', 'dec', 'parallax', 'pmra', 'pmdec', 'phot_g_mean_flux','phot_bp_mean_flux', 'phot_rp_mean_flux', 'parallax_error', 'pmra_error', 'pmdec_error', 'phot_g_mean_flux_error','phot_bp_mean_flux_error', 'phot_rp_mean_flux_error', 'priority', 'name', 'type'],
           'match': 0.5}
    
    gaia = {'name': 'gaia_dr2_source',
            'cols': ['l', 'b', 'astrometric_excess_noise', 'astrometric_excess_noise_sig', 'visibility_periods_used', 'phot_bp_rp_excess_factor'],
            'match_radius': 0.5}
    
    ps = {'name': 'ucal_fluxqz',
          'cols': ['obj_id', 'median', 'err'],
          'match_radius': 0.5}
    
    tmass = {'name': 'tmass',
             'cols': ['twomass_id', 'j_m', 'j_msigcom', 'h_m', 'h_msigcom', 'k_m', 'k_msigcom'],
             'match_radius': 0.5}
    
    wise = {'name': 'allwise',
             'cols': ['source_id', 'w1mpro', 'w1sigmpro', 'w2mpro', 'w2sigmpro'],
             'match_radius': 0.5}
    
    sdss = {'name': 'sdss_dr14_starsweep',
            'cols': ['psfflux', 'psfflux_ivar'],
            'match_radius': 0.5}
    

def make_gd1_acat():
    """"""
    
    aa = Table.read('../data/gd1_acat.fits')
    print(aa.colnames)
    
    st = np.zeros(len(aa), dtype=acat_dtype_file)
    st["RA"]  = aa["ra"]
    st["DEC"] = aa["dec"]
    st["L"]   = aa["gaia_dr2_source.l"]
    st["B"]   = aa["gaia_dr2_source.b"]
    
    # IDs
    st["PS_ID"] = aa["ucal_fluxqz.obj_id"]
    st["TMASS_ID"] = aa["tmass.twomass_id"]
    st["WISE_ID"] = aa["allwise.source_id"]
    st["GAIA_ID"] = aa["source_id"]
    try:
        st["H3_ID"] = aa["H3_ID"]
    except(KeyError):
        st["H3_ID"] = -np.arange(len(aa))
    
    # Fluxes from PS, SDSS, TMASS, WISE
    for i, b in enumerate("GRIZY"):
        band = "PS_{}".format(b)
        flux, err = aa["ucal_fluxqz.median"][:, i], aa["ucal_fluxqz.err"][:, i]
        ind = flux==0
        flux[ind] = np.nan
        err[ind] = np.nan
        st[band] = np.clip(-2.5 * np.log10(flux), 0.0, np.inf)
        st[band + "_ERR"] = 1.086 * err / flux
    for i, b in enumerate("UGRIZ"):
        band = "SDSS_{}".format(b)
        flux = aa["sdss_dr14_starsweep.psfflux"][:, i] * 1e-9
        err = 1. / np.sqrt(aa["sdss_dr14_starsweep.psfflux_ivar"][:, i]) * 1e-9
        ind = flux==0
        flux[ind] = np.nan
        err[ind] = np.nan
        st[band] = np.clip(-2.5 * np.log10(flux), 0.0, np.inf)
        st[band + "_ERR"] = 1.086 * err / flux
    for i, b in enumerate("JHK"):
        band = "TMASS_{}".format(b)
        st[band] = aa["tmass.{}_m".format(b.lower())]
        st[band + "_ERR"] = aa["tmass.{}_msigcom".format(b.lower())]
        ind = st[band]==0
        st[band][ind] = np.nan
        st[band + "_ERR"][ind] = np.nan
    for i, b in enumerate(["W1", "W2"]):
        band = "WISE_{}".format(b)
        st[band] = aa["allwise.{}mpro".format(b.lower())]
        st[band + "_ERR"] = aa["allwise.{}sigmpro".format(b.lower())]
        ind = st[band]==0
        st[band][ind] = np.nan
        st[band + "_ERR"][ind] = np.nan
    
    eq = ["PARALLAX", "PMRA", "PMDEC"]#, "RA", "DEC"]
    eq += ["PHOT_{}_MEAN_FLUX".format(b) for b in ["G", "BP", "RP"]]
    qs = eq + [q + "_ERROR" for q in eq]
    for i, q in enumerate(qs):
        st["GAIA_" + q] = aa[q.lower()]
    
    qsa = ["ASTROMETRIC_EXCESS_NOISE", "ASTROMETRIC_EXCESS_NOISE_SIG",
           'VISIBILITY_PERIODS_USED', "PHOT_BP_RP_EXCESS_FACTOR"]
    
    for i, q in enumerate(qsa):
        st["GAIA_" + q] = aa['gaia_dr2_source.' + q.lower()]
    
    st['GUIDE'] = aa['priority']
    
    #tout = Table(st)
    #tout.pprint()
    #print(tout.colnames)
    #print(tout['PS_G'], np.sum(np.isfinite(tout['PS_G'])))
    
    out = '../data/acat_gd1_cluster.fits'
    fits.writeto(out, st, overwrite=True)

def check_acat():
    """Plot CMD of top priority stars to make sure magnitudes are calculated correctly"""
    
    t = Table.read('../data/acat_gd1_cluster.fits')
    t = t[t['GUIDE']<4]
    
    plt.close()
    plt.figure()
    
    plt.plot(t['PS_G'] - t['PS_R'], t['PS_G'], 'k.')
    
    plt.tight_layout()
