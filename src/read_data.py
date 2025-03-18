import numpy as np
import pandas as pd
import xarray as xr

##########################################################

def weekly_average(data,nw): 
    r""" Return the weekly averaged data.

    Parameters
    ----------
    data : 
    nw : number of weeks to return

    """
    data_week = []
    for k in range(nw):
        dw = data.sel(time=slice(k*7,k*7+6)).mean("time")
        data_week.append(dw)
    new_data = xr.concat(data_week,pd.Index(np.array(range(1,nw+1)),name="time"))
    return new_data


def load_data(list_model,var,folder_root,nw):
    r""" Read forecasts, compute weekly average, and return them as a list.
    
    Parameters
    ----------
    list_model : list of models to read
    var : variable to select
    folder_root : folder where the forecasts data is saved
    nw : number of weeks to return

    """
    # Read forecasts data
    fc_cal_list = []
    for k in range(len(list_model)):
        model = list_model[k]
        folder_data = "{}/{}/".format(folder_root,var)
        fc_cal = xr.open_mfdataset("{}/{}_cal*.nc".format(folder_data,model.lower()))["fc"]
        fc_cal =  weekly_average(fc_cal,nw)        
        fc_cal = fc_cal.stack(coord=("latitude","longitude"))
        fc_cal = fc_cal.transpose("date","coord","time","number") 
        fc_cal_list.append(fc_cal)
    
    coordinates = {"date":fc_cal.date, "time": fc_cal.time, "coords": fc_cal.coord}

    return fc_cal_list, coordinates


def read_ref(folder_data,var,nw):
    r""" Read reference data and return weekly average.
    
    Parameters
    ----------
    folder_data : folder where the reference data is saved
    var : variable to select
    nw : number of weeks to return

    """
    ref = xr.open_mfdataset("{}/{}/ref_merra*.nc".format(folder_data,var))["ref"]
    ref =  weekly_average(ref,nw)
    ref = ref.stack(coord=("latitude","longitude"))
    ref = ref.transpose("date","coord","time") 
    return ref


def read_clim(folder_data,var,nw):
    r""" Read climatology data data and return weekly average.
    
    Parameters
    ----------
    folder_data : folder where the reference data is saved
    var : variable to select
    nw : number of weeks to return

    """

    file_data = "{}/clim_{}.nc".format(folder_data,var)
    clim = xr.open_dataset(file_data)["fc"]
    clim =  weekly_average(clim,nw)
    clim = clim.stack(coord=("latitude","longitude"))
    clim = clim.transpose("date","coord","time","number")
    return clim




def get_model_info(att,list_name):
    # Information for plots
    
    models = ["ECCC", "ECMWF", "KMA", "NCEP"]
    colors = ["C4","C0","C5","C1"]
    labels = ["ECCC", "ECMWF", "KMA", "NCEP"]
    
    models_info = pd.DataFrame(data={'color':colors,'label':labels},index=models)
    
    return models_info.loc[list_name][att].values













############################







########################################################
########################################################
##########################################################
# Without putting it in torch
def read_fc_cal_xr(list_model,list_name,land_only=True):
    # Read forecast data listed in list_model and calibrated with list_name and return a list
    # 
    if land_only:
        # Create sea-mask
        mask_file = "./land_mask_era-land.nc"
        mask_xr = xr.open_dataset(mask_file)
        land_mask = mask_xr["crps_weekly_mean"]
        mask_xr.close()

    fc_cal_list = []
    for k in range(len(list_name)):
        name = list_name[k]
        model = list_model[k]
        print(model,name)
        folder_data = "/homedata/clecoz/DATA_S2S/multi/{}_fc_t2m/cal_s2s_{}/".format(model.upper(),name)
        print(folder_data)
        if land_only:
            fc_cal = xr.open_mfdataset("{}/{}_cal*.nc".format(folder_data,model.lower()))["fc"].where(~land_mask)
        else:
            fc_cal = xr.open_mfdataset("{}/{}_cal*.nc".format(folder_data,model.lower()))["fc"]
        fc_cal = fc_cal.sel(time=np.arange(1,29))
        #fc_cal = fc_cal.stack(coord=("latitude","longitude")).dropna(dim="coord")
        #fc_cal = fc_cal.transpose("date","coord","time","number")
        fc_cal_list.append(fc_cal)
        print(fc_cal.shape)
    return fc_cal_list


def read_ref_xr(land_only=True):
    # Read reference data
    folder_data = "/homedata/clecoz/DATA_S2S/multi/ECMWF_fc_t2m/cal_s2s_merra/"
    if land_only:
        # Create sea-mask
        mask_file = "./land_mask_era-land.nc"
        mask_xr = xr.open_dataset(mask_file)
        land_mask = mask_xr["crps_weekly_mean"]
        mask_xr.close()
        ref = xr.open_dataset("{}/ref_merra.nc".format(folder_data))["ref"].where(~land_mask)
    else:
        ref = xr.open_dataset("{}/ref_merra.nc".format(folder_data))["ref"]
    ref = ref.sel(time=np.arange(1,29))
    #ref = ref.stack(coord=("latitude","longitude")).dropna(dim="coord")
    #ref = ref.transpose("date","coord","time")
    return ref


def read_clim_xr(land_only=True):
    # Read climatology data
    file_data = "/homedata/clecoz/DATA_S2S/multi/climatology_30y_merra.nc"
    if land_only:
        # Create sea-mask
        mask_file = "./land_mask_era-land.nc"
        mask_xr = xr.open_dataset(mask_file)
        land_mask = mask_xr["crps_weekly_mean"]
        mask_xr.close()
        clim = xr.open_dataset(file_data)["clim"].where(~land_mask)
    else:
        clim = xr.open_dataset(file_data)["clim"]
    clim = clim.sel(time=np.arange(1,29))
    #clim = clim.stack(coord=("latitude","longitude")).dropna(dim="coord")
    #clim = clim.transpose("date","coord","time","number")
    return clim




