
import numpy as np
import ot
import warnings
import xarray as xr

#####################################################################

def barycenterL2(models ,weights):  
    r""" Return the Wasserstein barycenter with Gaussian mapping

    Parameters
    ----------
    models : list of forecast data. Forecasts must be xarray data arrays with the members being saved in a dimension naemd "number". 
    weights : list of model weights

    """

    # Check data
    if len(models) == len(weights):
        n_models = len(models)
    else:
        raise ValueError("models and weigths should have the same lenght")


    for km in range(1,n_models):
        if models[km].shape[:-1] != models[0].shape[:-1]:
            raise ValueError('models must have matching '
                            'shapes except along `axis=-1`')
        
    # Model's weights must sum to one
    weights = np.array(weights)
    weights /= np.sum(weights)
    

    # Compute L2 barycenter support (concatenation)
    L2_bary = xr.concat(models,dim="number")

    # Compute L2 barycenter weights
    measures_weights = []
    for km in range(0,len(models)):
        n_ens = models[km].shape[-1]
        measures_weights.append(np.ones(models[km].shape)/n_ens*weights[km])
    L2_w = np.concatenate(measures_weights,axis=-1) 

    return L2_bary, L2_w






def barycenterW2gauss(models, weights):
    r""" Return the Wasserstein barycenter with Gaussian mapping

    Parameters
    ----------
    models : list of forecast data array (either numpy or xarray) in the shape (starting-dates, coordinates, lead-times, members)
    weights : list of model weights

    """
    # Check data
    if len(models) == len(weights):
        n_models = len(models)
    else:
        raise ValueError("models and weigths should have the same lenght")


    for km in range(1,n_models):
        if models[km].shape[:-1] != models[0].shape[:-1]:
            raise ValueError('models must have matching '
                            'shapes except along `axis=-1`')
        
       
    # Model's weights must sum to one
    weights = np.array(weights)
    weights /= np.sum(weights)

    # Dimensions
    nsim = models[0].shape[0]   # number of starting dates
    ngrid = models[0].shape[1]  # number of grid points
    nt = models[0].shape[2]     # number of time steps
    nens_tot = 0                # number of members 
    for km in range(n_models):
        nens_tot += models[km].shape[-1]

    # Compute Gaussian parameters for the input distributions (Step 1)
    models_mean = []        # list of the ensemble means
    models_cov = []         # list of the ensemble covariances
    models_temp = []
    for km in range(n_models):
            model = np.array(models[km])        # transform into array for faster computation
            models_temp.append(model)
            nens = model[km].shape[-1]
            mean = model.mean(axis=-1,keepdims=True)
            models_mean.append(mean.squeeze())
            cov = np.matmul((model-mean),(model-mean).transpose((0,1,3,2))) / nens
            models_cov.append(cov)

    models = models_temp

    # Compute the GaussW2-barycenter support (for each starting date and each grid points)
    W2_bary = np.zeros((nsim,ngrid,nt,nens_tot))    # Intialization of the W2-barycenter support
    for ks in range(nsim):      # Loop on starting dates
        for kg in range(ngrid):     # Loop on coordinates  
            # Select Gaussian parameters (pre-computed above)
            dist_mean = []
            dist_cov = []
            models_sel = []
            for km in range(n_models):
                    models_sel.append(models[km][ks,kg,:,:])
                    dist_mean.append(models_mean[km][ks,kg,:])
                    dist_cov.append(models_cov[km][ks,kg,:,:])

            if np.isnan(np.concatenate(models_sel,axis=-1)).any():
                # Return NaN if input contained any NaN
                W2_bary[ks,kg,:,:] = np.nan

            else: 
                # Compute the barycenter of the Gaussian distributions (Step2)
                mb, Cb = ot.gaussian.bures_wasserstein_barycenter(np.stack(dist_mean), np.stack(dist_cov), weights)

                # Map the input distributions into the Gaussian barycenter
                w2_bary_sel = []
                for km in range(n_models):      # Loop on the inputs (distributions/models)
                    # Compute the mappings between the (Gaussian) input distribution and Gaussian barycenter (Step 3)
                    A, b = ot.gaussian.bures_wasserstein_mapping(dist_mean[km], mb, dist_cov[km], Cb)
                    # Apply mapping to original discrete distribution (Step 4, part 1)
                    x = models_sel[km]
                    w2_bary_sel.append(np.dot(A,x) + np.expand_dims(b,axis=-1))
                # Pool the adjusted discrete distribution together (Step 4, part 2)    
                W2_bary[ks,kg,:,:] = np.concatenate(w2_bary_sel, axis=-1)


    # Compute the GaussW2-barycenter weights
    W2_w = []
    for km in range(n_models):      # Loop on the inputs (distributions/models)
        nens = models[km].shape[-1]
        W2_w.append(weights[km]/nens * np.ones(nens))
    W2_w = np.concatenate(W2_w)

    return W2_bary, W2_w




def barycenterW2(models ,weights, nbar,seed=None):
    r""" Return the Wasserstein barycenter

    Parameters
    ----------
    models : list of model's data in the shape (starting-dates, coordinates, lead-times, members)
    weights : list of model weights
    nbar : number of members/points of the Wasserstein barycenter ensemble. 
    """

    # Check data
    if len(models) == len(weights):
        n_models = len(models)
    else:
        raise ValueError("models and weigths should have the same lenght")


    for km in range(1,n_models):
        if models[km].shape[:-1] != models[0].shape[:-1]:
            raise ValueError('models must have matching '
                            'shapes except along `axis=-1`')
        
       

    # Model's weights must sum to one
    weights = np.array(weights)
    weights /= np.sum(weights)

    # Dimension
    nsim = models[0].shape[0]   # number of starting dates
    ngrid = models[0].shape[1]  # number of grid points
    nt = models[0].shape[2]     # number of time steps


    # Intialization of the W2-barycenter
    c = np.ones((nbar,),dtype=models[0].dtype) / nbar       # member's weights
    W2_bary = np.zeros((nsim,ngrid,nt,nbar))                # support

    models_concat = np.concatenate(models,axis=-1)
    n_ens_tot = models_concat.shape[-1]
    if nbar > n_ens_tot:
        warnings.warn("Warning: required number of members is higher than the sum of the input models' members")

    for ks in range(nsim):      # Loop on starting dates
        # Create first guess from input distributions
        # Random initialization of the support locations of the barycenter
        if seed is not None:
            np.random.seed(seed)    # fix the seed for repeatability
        rand_ind = np.random.randint(0, n_ens_tot, size=(nbar,))
        random_ens = np.take(models_concat[ks,:,:,:],rand_ind,axis=-1)
        
        for kg in range(ngrid):            # Loop on coordinates  (the barycenter is computed for each grid points separately)
            if np.isnan(random_ens[kg,:,:]).any(): 
                W2_bary[ks,kg,:,:] = np.nan
            else:
                # First guess 
                X_init = random_ens[kg,:,:].T

                # Format distributions
                measures_locations = []
                measures_weights = []
                for km in range(n_models):
                    measures_locations.append(np.array(models[km][ks,kg,:,:]).T)
                    n_ens = models[km].shape[-1]
                    measures_weights.append(np.ones(n_ens)/n_ens)

                # Compute barycenter support
                X = ot.lp.free_support_barycenter(measures_locations, measures_weights, X_init, c, weights=weights)
                W2_bary[ks,kg,:,:] = X.T

    return W2_bary


