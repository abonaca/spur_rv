import numpy as np
from astropy.io import fits


fn = '../../test_data/acat_ta007897_ga1.charlieformat.fits'
dat = np.array(fits.getdata(fn))
acat_dtype_file = dat.dtype



def convert_lsd_to_acat(aa):
    """aa: FITS binary table, IDL format
    """

    st = np.zeros(len(aa), dtype=acat_dtype_file)
    st["RA"]  = aa["ucal_fluxqz.ra"]
    st["DEC"] = aa["ucal_fluxqz.dec"]
    st["L"]   = aa["gaia_dr2_source.l"]
    st["B"]   = aa["gaia_dr2_source.b"]
    #st["EBV"] = aa["EBV"]

    # IDs
    st["PS_ID"] = aa["ucal_fluxqz.obj_id"]
    st["TMASS_ID"] = aa["tmass.twomass_id"]
    st["WISE_ID"] = aa["allwise.source_id"]
    st["GAIA_ID"] = aa["gaia_dr2_source.source_id"]
    try:
        st["H3_ID"] = aa["H3_ID"]
    except(KeyError):
        st["H3_ID"] = -np.arange(len(aa))
    
    # Fluxes from PS, SDSS, TMASS, WISE
    for i, b in enumerate("GRIZY"):
        band = "PS_{}".format(b)
        flux, err = aa["ucal_fluxqz.median"][:, i], aa["ucal_fluxqz.err"][:, i]
        st[band] = np.clip(-2.5 * np.log10(flux), 0.0, np.inf)
        st[band + "_ERR"] = 1.086 * err / flux
    for i, b in enumerate("UGRIZ"):
        band = "SDSS_{}".format(b)
        flux = aa["sdss_dr14_starsweep.psfflux"][:, i] * 1e-9,
        err = 1. / np.sqrt(aa["sdss_dr14_starsweep.psfflux_ivar"][:, i]) * 1e-9
        st[band] = np.clip(-2.5 * np.log10(flux), 0.0, np.inf)
        st[band + "_ERR"] = 1.086 * err / flux
    for i, b in enumerate("JHK"):
        band = "TMASS_{}".format(b)
        st[band] = aa["tmass.{}_m".format(b.lower())]
        st[band + "_ERR"] = aa["tmass.{}_msigcom".format(b.lower())]
    for i, b in enumerate(["W1", "W2"]):
        band = "WISE_{}".format(b)
        st[band] = aa["allwise.{}mpro".format(b.lower())]
        st[band + "_ERR"] = aa["allwise.{}sigmpro".format(b.lower())]


    eq = ["PARALLAX", "PMRA", "PMDEC"]#, "RA", "DEC"]
    eq += ["PHOT_{}_MEAN_FLUX".format(b) for b in ["G", "BP", "RP"]]
    qs = eq + [q + "_ERROR" for q in eq]
    qs += ["ASTROMETRIC_EXCESS_NOISE", "ASTROMETRIC_EXCESS_NOISE_SIG",
           'VISIBILITY_PERIODS_USED', "PHOT_BP_RP_EXCESS_FACTOR"] #+ ["PARALLAX_OVER_ERROR"]
    
    for i, q in enumerate(qs):
        st["GAIA_" + q] = aa["gaia_dr2_source." + q.lower()]
        

    acat = st
    return acat


if __name__ == "__main__":

    clusters = "m3", "m13", "m107"
    for c in clusters:
        fn = "{}.fits.gz".format(c)
        out = "acat_{}_cluster.fits".format(c)
        aa = fits.getdata(fn)
        acat = convert_lsd_to_acat(aa)
        fits.writeto(out, acat, overwrite=True)
