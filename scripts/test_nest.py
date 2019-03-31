import numpy as np
import dynesty
from dynesty import utils as dyfunc
from dynesty import plotting as dyplot
from dynesty import utils as dyfunc

import pickle
import matplotlib.pyplot as plt
import corner

def test():
    # Define the dimensionality of our problem.
    ndim = 3

    # Define our 3-D correlated multivariate normal likelihood.
    C = np.identity(ndim)  # set covariance to identity matrix
    C[C==0] = 0.95  # set off-diagonal terms
    Cinv = np.linalg.inv(C)  # define the inverse (i.e. the precision matrix)
    lnorm = -0.5 * (np.log(2 * np.pi) * ndim + np.log(np.linalg.det(C)))  # ln(normalization)

    print('static')
    # "Static" nested sampling.
    sampler = dynesty.NestedSampler(loglike, ptform, ndim, nlive=500, logl_args=[Cinv, lnorm])
    sampler.run_nested()
    sresults = sampler.results
    pickle.dump(sresults, open('../data/test_static.pkl','wb'))

    print('dynamic')
    # "Dynamic" nested sampling.
    dsampler = dynesty.DynamicNestedSampler(loglike, ptform, ndim, nlive=500, logl_args=[Cinv, lnorm])
    dsampler.run_nested()
    dresults = dsampler.results
    pickle.dump(dresults, open('../data/test_dynamic.pkl','wb'))


def plot():
    # Combine results from "Static" and "Dynamic" runs.
    
    sresults = pickle.load(open('../data/test_static.pkl', 'rb'))
    dresults = pickle.load(open('../data/test_dynamic.pkl', 'rb'))
    results = dyfunc.merge_runs([sresults, dresults])

    ## Plot a summary of the run.
    #rfig, raxes = dyplot.runplot(results)

    ## Plot traces and 1-D marginalized posteriors.
    #tfig, taxes = dyplot.traceplot(results)

    ## Plot the 2-D marginalized posteriors.
    #plt.close()
    #cfig, caxes = dyplot.cornerplot(results)


    # Extract sampling results.
    samples = results.samples  # samples
    weights = np.exp(results.logwt - results.logz[-1])  # normalized weights

    ## Compute 5%-95% quantiles.
    #quantiles = dyfunc.quantile(samples, [0.05, 0.95], weights=weights)

    ## Compute weighted mean and covariance.
    #mean, cov = dyfunc.mean_and_cov(samples, weights)

    # Resample weighted samples.
    samples_equal = dyfunc.resample_equal(samples, weights)
    
    plt.close()
    corner.corner(samples_equal, bins=70, plot_datapoints=False, smooth=1, show_titles=True)

    ## Generate a new set of results with statistical+sampling uncertainties.
    #results_sim = dyfunc.simulate_run(results)

def loglike(x, Cinv, lnorm):
    """The log-likelihood function."""

    return -0.5 * np.dot(x, np.dot(Cinv, x)) + lnorm

# Define our uniform prior.
def ptform(u):
    """Transforms samples `u` drawn from the unit cube to samples to those
    from our uniform prior within [-10., 10.) for each variable."""

    return 10. * (2. * u - 1.)
