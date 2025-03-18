import numpy as np
import xarray as xr
import matplotlib.pyplot as pl
import cartopy.crs as ccrs
import cartopy.feature as cfeature

from read_data import *
from barycenter_functions import *
from evaluate_functions import *



#########################################################################################################

# Settings
list_models = ["ECMWF","NCEP","ECCC","KMA"]                 # Choose models to combine
weights = np.ones(len(list_models)) / len(list_models)      # Choose model weights
var = "t2m"     # Choose variable (e.g. "t2m", "wind", "z500") #
nw = 4          # Number of weeks to select


# Folders
folder_fc_data = "./data/fc_cal/"       # folder containing forecast data
folder_clim = "./data/climatology/"       # folder containing forecast data
folder_ref = "./data/reference/"       # folder containing forecast data
folder_results = "./results/output/"    # folder where to save the results
folder_plots = "./results/plots/"    # folder where to save the plots

GaussW2_file = "{}/W2gaus_{}.nc".format(folder_results,var)     # File where the GaussW2 barycenter is saved (computed in main_barycenters.py)



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

# Read climatology
clim = read_clim(folder_clim,var,nw).sel(date=X[0].date)   # only select dates present in forecasts

# Read reference
ref = read_ref(folder_ref,var,nw).sel(date=X[0].date)      # only select dates present in forecasts

# Read GaussW2 barycenter
bary = xr.open_dataset(GaussW2_file)
X_W2 = bary["fc"].stack(coord=("latitude","longitude")).transpose("date","coord","time","number")
w_W2 = bary["weight"]

# Compute L2 barycenter 
X_L2, w_L2 = barycenterL2(X ,weights) 



#########################################################################################################
#########################################################################################################
# CRPS related skill scores

# Compute CRPS for each date, grid point and lead time
crps_SME = []
for km in range(n_model):
    crps_SME.append(compute_crps_model(X[km], ref, save_file="{}/crps_{}.nc".format(folder_results,list_models[km])))

crps_L2 = compute_crps_model(X_L2, ref , w_L2, save_file="{}/crps_L2.nc".format(folder_results))
crps_W2 = compute_crps_model(X_W2, ref, w_W2, save_file="{}/crps_W2gauss.nc".format(folder_results))

crps_clim = compute_crps_model(clim, ref, save_file="{}/crps_clim.nc".format(folder_results))



# Compute skill scores and save them in table
labels = list_models+["L2 bary","GaussW2 bary"]
crps_models = crps_SME.copy()
crps_models.extend((crps_L2,crps_W2))
scores = ["CRPSS","CRPSp","CRPSf"]
SCORES = xr.DataArray(dims=["model","score","time"],
                    coords=dict(
                    model=(["model"],list_models+["L2 bary","GaussW2 bary"]),
                    score=(["score"],scores),
                    time=(["time"],np.array([3,4])),
                    ))

for km in range(len(SCORES.model)):
    for score in scores:
        SCORES.sel(model=labels[km],score=score)[:] = compute_CRPS_skillscore(crps_models[km],crps_clim, score)


# Plot skill scores
colors = list(get_model_info("color",list_models))
colors.extend(("C3","C2"))
nscore = len(scores)
n_label = len(labels)
fig, axes = pl.subplots(nrows=nscore,ncols=1, figsize=(8,10),sharex=True, sharey="row",layout="tight")
for i in range(nscore):
    score = scores[i]

    ind = np.arange(nw-2)
    width = 1/(n_label+2)
    legends = []
    for km in range(n_label):
        score_model = SCORES.sel(model=labels[km],score=score)
        lm = axes[i].bar(ind+km*width, score_model, width=width, color=colors[km],label=labels[km],edgecolor="k")
        legends.append(lm)

    if score=="CRPSS":
        axes[i].axhline(y=0,c="k",ls="--")
        axes[i].set_ylabel(r"CRPSS (-) $\rightarrow$")
    if score=="CRPSp":
        axes[i].axhline(y=50,c="k",ls="--")
        axes[i].set_ylabel(r"CRPSp (%) $\rightarrow$")
        axes[i].set_ylim((45,None))
    if score=="CRPSf":
        axes[i].set_ylabel(r"CRPSf (%) $\leftarrow$")

    axes[0].set_title(var)

pl.xticks(ind+(n_label-1)/2*width, ["week 3", "week 4"])
fig.legend(legends,labels,loc='lower center',ncol=n_label)
pl.tight_layout()
pl.subplots_adjust(bottom=0.08)

pl.savefig("{}/summary_score.png".format(folder_plots))
pl.close()



#########################################################################################################
#########################################################################################################
# Spread-Skill-Ratio (SSR)

# Compute SSR and save them in table (scores for weeks 3 and 4 are averaged)
SSR_SME = []
for km in range(n_model):
    SSR_SME.append(compute_ssr_model(X[km].isel(time=slice(2,None)), ref))
SSR_L2 = compute_ssr_model(X_L2.isel(time=slice(2,None)), ref , w_L2[0,0,0,:])
SSR_W2 = compute_ssr_model(X_W2.isel(time=slice(2,None)), ref, w_W2)
SSR_models = SSR_SME + [SSR_L2, SSR_W2]

labels = list_models+["L2 bary","GaussW2 bary"]


# Plot SSR maps
proj = ccrs.PlateCarree()
x = SSR_L2.longitude
y = SSR_W2.latitude
X, Y = np.meshgrid(x,y)
fig, axes = pl.subplots(nrows=n_label,ncols=1, figsize=(5,15), subplot_kw={'projection': proj, "aspect": 1})    #,sharey=True,sharex=True
for i in range(n_label):
    score_model = SSR_models[i].mean('time')      # average weeks 3 and 4
    axes[i].coastlines(resolution='50m',color='grey')
    axes[i].add_feature(cfeature.BORDERS,color='grey')
    axes[i].set_yticks(np.arange(-90,90,10), crs=proj)
    axes[i].set_ylabel(labels[i])
    bm = axes[i].pcolormesh(x,y,score_model,cmap="PuOr_r",vmin=0.6,vmax=1.4)

axes[0].set_title(var)
axes[-1].set_xticks(np.arange(-10,50,10), crs=proj)
cbar = fig.colorbar(bm, ax=axes,location="right",aspect=50,pad=0.05)
cbar.set_label("SSR (-)",fontsize=14)
pl.savefig("{}/spatial_SSR.png".format(folder_plots))
pl.close()

