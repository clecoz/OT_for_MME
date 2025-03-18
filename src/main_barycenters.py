import numpy as np
import xarray as xr
import matplotlib.pyplot as pl

from read_data import *
from barycenter_functions import *


#########################################################################################################

# Settings
list_models = ["ECMWF","NCEP","ECCC","KMA"]                 # Choose models to combine
weights = np.ones(len(list_models)) / len(list_models)      # Choose model weights
var = "t2m"     # Choose variable (e.g. "t2m", "wind", "z500") #
nw = 4          # Number of weeks to select


# Folders
folder_fc_data = "./data/fc_cal/".format(var)       # folder containing forecast data
folder_results = "./results/output/"    # folder where to save the results
folder_plots = "./results/plots/"     # folder where to save the plots   



#########################################################################################################
# Load data

# Read forecast data
X, coordinates = load_data(list_models,var,folder_fc_data,nw)   
n_model = len(X)            # number of models
nsim = X[0].shape[0]        # number of simulations/starting dates
ngrid = X[0].shape[1]       # number of grid points
#nt = X[0].shape[2]          # should be nw

# Get coordinates
coord = coordinates["coords"]
date = coordinates["date"]
ltime = coordinates["time"]



#########################################################################################################
# Compute barycenters

X_L2, w_L2 = barycenterL2(X ,weights)      # Compute L2-barycenter

X_W2, w_W2 = barycenterW2gauss(X ,weights)      # Compute GaussW2-barycenter

# Convert GaussW2-barycenter to a xarray.DataArray and save in netcdf 
w2_bary = xr.Dataset(
    data_vars=dict(
        fc=(["date", "coord", "time", "number"], X_W2),
        weight=(["number"], w_W2),
    ),
    coords=dict(
        date=coordinates["date"],
        time=coordinates["time"],
        coord=coordinates["coords"],
        number=np.arange(0,X_W2.shape[-1]),
    ),
    attrs=dict(description="W2gauss",variable=var,models=list_models)
)
w2_bary.unstack().to_netcdf("{}/W2gaus_{}.nc".format(folder_results,var),mode="w")  



#########################################################################################################
# Visualisation (optional)

# Select a case
pos_lat = 49.0
pos_lon = 2.0
date_sel = "2023-02-16"

# Ingredient for plots
colors = list(get_model_info("color",list_models))
labels = list_models.copy()
markers = ["o"]*n_model


fig, axes = pl.subplots(nrows=1,ncols=3, figsize=(15,5),sharex=True, sharey=True,layout='tight')
# Plot SME
handles = []
for km in range(n_model):
    ek = axes[0].plot(X[km].time,X[km].sel(latitude=pos_lat,longitude=pos_lon,date=date_sel)-273.15,color=colors[km],lw=80/X[km].shape[-1],marker=markers[km], alpha=0.6,label=labels[km])
    handles.append(ek[0])
axes[0].legend(handles=handles,labels=labels)

for i in range(X_L2.shape[-1]):
    el2 = axes[1].plot(X_L2.time,X_L2.sel(latitude=pos_lat,longitude=pos_lon,date=date_sel).isel(number=i)-273.15,c="C3",lw=80*w_L2[0,0,0,i], marker='^', alpha=0.6, label=r"$L_2$ bary")
axes[1].legend(handles=[el2[0]],labels=[r"$L_2$ bary"])

for i in range(X_W2.shape[-1]):
    ew2 = axes[2].plot(w2_bary["fc"].time,w2_bary["fc"].sel(latitude=pos_lat,longitude=pos_lon,date=date_sel).isel(number=i)-273.15,c="C2",lw=80*w2_bary["weight"].isel(number=i), marker='v', alpha=0.6, label=r"$GaussW_2 bary")
axes[2].legend(handles=[ew2[0]],labels=[r"$GaussW_2$ bary"])

axes[0].set_ylabel(r"2m temperature (in ${}^\circ$C)")
axes[0].set_xticks([1,2,3,4])
axes[0].set_xlabel("Week")
axes[1].set_xlabel("Week")
axes[2].set_xlabel("Week")
pl.suptitle("Forecasted 2m temperature in Paris (lat={}, lon={}), initialized the {}".format(pos_lat,pos_lon,date_sel))
pl.savefig("{}/visualization.png".format(folder_plots))







