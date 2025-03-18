import numpy as np
import xarray as xr
import properscoring as ps



#########################################################################################################
# CRPS 

def compute_crps_model(X,ref,weights=None,save_file=None):

    if weights is None:
        nens = X.shape[-1]
        weights = np.ones(X.shape) / nens
    
    elif len(weights.shape) == 1:
        if weights.shape[0] == X.shape[-1]:
            weights = np.broadcast_to(weights, X.shape)
        else:
            raise ValueError('weights should either have the same shape as X'
                            'or be a vector of the same dimension as `axis=-1` of X')
    elif weights.shape != X.shape:
        raise ValueError('weights should either have the same shape as X'
                        'or be a vector of the same dimension as `axis=-1` of X')

    crps_temp = np.zeros(ref.shape)
    for k in range(crps_temp.shape[0]):     # loop over first dimension to avoid MemoryError
        crps_temp[k,:,:] = ps.crps_ensemble(ref.values[k,:,:],X.values[k,:,:,:],weights=weights[k,:,:,:])


    crps = xr.DataArray(np.array(crps_temp), coords=[X.date, X.coord, X.time], dims=["date","coord","time"])
    crps.name = "crps"
    
    if save_file is not None:
        crps.unstack().to_netcdf(save_file,mode="w")

    return crps.unstack()



def compute_CRPS_skillscore(crps_model,crps_clim,score):
    weights_lat = np.cos(np.deg2rad(crps_model[0].latitude))
    weights_lat /= weights_lat.mean()

    crps_model = crps_model.isel(time=slice(2,None))
    crps_clim = crps_clim.isel(time=slice(2,None))

    if score=="CRPSmean":
        crps_mean = crps_model.weighted(weights_lat).mean(("date","latitude","longitude"))
        return crps_mean
    elif score=="CRPSS":
        crps_model_mean = crps_model.weighted(weights_lat).mean(("date","latitude","longitude"))
        crps_clim_mean = crps_clim.weighted(weights_lat).mean(("date","latitude","longitude"))
        crpss_mean = 1 - crps_model_mean / crps_clim_mean
        return crpss_mean
    elif score=="CRPSp":
        nb_fc = (~np.isnan(crps_clim)).sum(("latitude","longitude"))
        score_temp = ((1-crps_model/crps_clim)>=0).sum(("latitude","longitude")) / nb_fc * 100
        crpsp = score_temp.mean("date")
        return crpsp
    elif score=="CRPSf":
        nb_fc = (~np.isnan(crps_clim)).sum(("latitude","longitude"))
        thres_fail = 2*crps_clim
        score_temp = (crps_model > thres_fail).sum(("latitude","longitude")) / nb_fc * 100
        crpsf = score_temp.mean("date")
        return crpsf
    else:
        print("Unknown score")
        exit()


#########################################################################################################
# SSR

def compute_ssr_model(X,ref,weights=None):

    if weights is None:         # if no weight is given, we assume all members have equal weights
        nens = X.shape[-1]
        weights = np.ones(nens) / nens
    
    elif weights.shape[0] != X.shape[-1]:
        raise ValueError('weights should be a vector of length equal to the number of members (i.e. `axis=-1` of X)')
    
    weights = xr.DataArray(data=weights, dims=["number"], coords=dict(number=(["number"],X.number.values)))
    fc = X.weighted(weights.fillna(0))
    fc_mean = fc.mean("number")
    fc_var = fc.var("number")
    fc_mse = (fc_mean-ref)**2
    ssr = np.sqrt( fc_var.mean(("date")) /fc_mse.mean(("date")) )
    
    return ssr.unstack()