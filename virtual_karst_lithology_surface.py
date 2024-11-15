
"""
Use the Lithology component to track two layers: a limestone layer, and a basement.
Water is partitioned between surface runoff (different infiltration excess fractions 
on limestone and basement) and recharge to karst aquifer (only on limestone).
Karst aquifer routes water on the contact surface, which may have different flow
directions than the topography. Discharge is the sum of the infiltration excess
and aquifer discharge past the contact (on the basement).
"""

#%%
import os
import numpy as np
import pandas as pd
import xarray as xr
from tqdm import tqdm
import matplotlib.pyplot as plt

from landlab import RasterModelGrid
from landlab.components import (
    FastscapeEroder, 
    FlowAccumulator,
    LakeMapperBarnes,
    LinearDiffuser,
    LithoLayers,
)

from virtual_karst_funcs import *

save_directory = '/Users/dlitwin/Documents/Research/Karst-landscape-evolution/landlab_virtual_karst/figures'
filename = "virtual_karst_surface_3"

#%% parameters

U = 1e-4 # uplift (m/yr)
E_limestone = 0.0 #5e-5 # limestone surface chemical denudation rate (m/yr)
E_weathered_basement = 0.0 # weathered basement surface chemical denudation rate (m/yr)
K_sp = 1e-5 # streampower incision (yr^...)
m_sp = 0.5 # exponent on discharge
n_sp = 1.0 # exponent on slope
D_ld = 1e-3 # diffusivity (m2/yr)

b_limestone = 50 # limestone unit thickness (m)
b_basement = 1000 # basement thickness (m)
bed_dip = 0.000 #0.002 # dip of bed (positive = toward bottom boundary)

r_tot = 1.0 #/ (3600 * 24 * 365) # total runoff m/yr
ie_frac = 0.1 # fraction of r_tot that becomes overland flow on limestone. ie_frac on basement=1. 
# ibar = 1e-3 / 3600 # mean storm intensity (m/s) equiv. 1 mm/hr 

T = 2e6 # total geomorphic time
dt = 500 # geomorphic timestep (yr)
N = int(T//dt) # number of geomorphic timesteps

Nx = 150
Ny = 100
dx = 50
noise_scale = 0.01

save_freq = 25 # steps between saving output
Ns = N//save_freq


#%% set up grid and lithology

mg = RasterModelGrid((Ny,Nx), xy_spacing=dx)
mg.set_closed_boundaries_at_grid_edges(right_is_closed=True,
                                       left_is_closed=True,
                                       top_is_closed=False,
                                       bottom_is_closed=False)
bottom_nodes = mg.nodes_at_bottom_edge
top_nodes = mg.nodes_at_top_edge
z = mg.add_zeros("node", "topographic__elevation")
np.random.seed(10010)
z += noise_scale*np.random.rand(len(z))

# two layers, both with bottoms below ground. Top layer is limestone, bottom is basement.
layer_elevations = [b_limestone,b_basement]
layer_ids = [0,1]
attrs = {
         "ie_frac": {0: ie_frac, 1: 1}, # fraction of r_tot that becomes infiltration excess runoff
         "chemical_denudation_rate": {0: E_limestone, 1: E_weathered_basement}
         }

lith = LithoLayers(
    mg, layer_elevations, layer_ids, function=lambda x, y: - bed_dip * y + noise_scale*np.random.rand(len(y)), attrs=attrs
)
dz_ad = np.zeros(mg.size("node"))
dz_ad[mg.core_nodes] = U * dt
# dz_ad[top_nodes] += 0.5 * U * dt # effectively decrease the baselevel fall rate of the upper boundary


# the karst or basement surface is the topographic elevation minus the depth to the top of the basement
zk = mg.add_zeros("node", "karst__elevation")
zk[:] = z-lith.z_top[0,:]


# instantiate runoff fields
# infiltration excess based on the fraction of r (p-e) that goes to ie. 
q_ie = mg.add_zeros("node", "infiltration_excess")
q_ie[mg.core_nodes] = mg.at_node['ie_frac'][mg.core_nodes] * r_tot
r = mg.add_zeros("node", "recharge_rate")
r[mg.core_nodes] = (1 - mg.at_node['ie_frac'][mg.core_nodes]) * r_tot

# instantiate discharge fields
Q1 = mg.add_zeros("node", "ie_discharge")
Q2 = mg.add_zeros("node", "karst_discharge")
Q_tot = mg.add_zeros("node", "total_discharge")
rock_ID = mg.at_node['rock_type__id']


#%% instantiate components 


# flow management for the topographic surface. runoff_rate="infiltration_excess",
# but we will only set this with water__unit_flux_in before running to avoid overwriting.
fa1 = FlowAccumulator(
    mg,
    surface="topographic__elevation",
    flow_director='FlowDirectorD8',
)
lmb1 = LakeMapperBarnes(
    mg,
    method="D8",
    fill_flat=False,
    surface="topographic__elevation",
    fill_surface="topographic__elevation",
    redirect_flow_steepest_descent=False,
    reaccumulate_flow=False,
    track_lakes=False,
    ignore_overfill=True,
)


# flow management for the basement surface, assuming all recharge accumulates at the base of
# the limestone, which is the top of the basement. Recharge only occurs on limestone. On the 
# basement it is set to zero. runoff_rate="recharge_rate", 
# but we will only set this with water__unit_flux_in before running to avoid overwriting.
fa2 = FlowAccumulator(
    mg,
    surface="karst__elevation",
    flow_director='FlowDirectorD8',
)
lmb2 = LakeMapperBarnes(
    mg,
    method="D8",
    fill_flat=False,
    surface="karst__elevation",
    fill_surface="karst__elevation",
    redirect_flow_steepest_descent=False,
    reaccumulate_flow=False,
    track_lakes=False,
    ignore_overfill=True,
)

fs2 = FastscapeEroder(mg, K_sp=K_sp, m_sp=m_sp, n_sp=n_sp, discharge_field='total_discharge')
ld2 = LinearDiffuser(mg, linear_diffusivity=D_ld)

#%% xarray to save output

output_fields = [
    "topographic__elevation",
    "karst__elevation",
    "rock_type__id",
    "total_discharge",
    "ie_discharge",
    "karst_discharge",
]
save_vals = ['limestone_exposed__area',
            'mean_limestone__thickness',
            'min_boundary_to_limestone__distance',
            'mean_boundary_to_limestone__distance',
            'max_boundary_to_limestone__distance',
            'mean_limestone__elevation',
            'max_limestone__elevation',
            'mean__elevation',
            'max__elevation',
            'erosion__rate',
            ]
df_out = pd.DataFrame(np.zeros((N,len(save_vals))), columns=save_vals)
params = {'U':U, 'K':K_sp, 'D_ld':D_ld, 'm_sp':m_sp,'n_sp':n_sp, 
        'E_limestone': E_limestone, 'E_weathered_basement':E_weathered_basement,
        'b_limestone':b_limestone, 'b_basement':b_basement, 'bed_dip':bed_dip,
        'r_tot':r_tot, 'ie_frac':ie_frac, 
        'T':T, 'N':N, 'dt':dt, 'save_freq':save_freq}

ds = xr.Dataset(
    data_vars={
        "topographic__elevation": (
            ("time", "y", "x"),  # tuple of dimensions
            np.empty((Ns, mg.shape[0], mg.shape[1])),
            {"units": "meters", "long_name": "Topographic Elevation"},
        ),
        "karst__elevation": (
            ("time", "y", "x"),
            np.empty((Ns, mg.shape[0], mg.shape[1])),
            {"units": "meters", "long_name": "Base of karst and top of basement"},
        ),
        "rock_type__id": (
            ("time", "y", "x"),
            np.empty((Ns, mg.shape[0], mg.shape[1])),
            {"units": "-", "long_name": "Rock Type ID Code"},
        ),
        "total_discharge": (
            ("time", "y", "x"),
            np.empty((Ns, mg.shape[0], mg.shape[1])),
            {"units": "m3/s", "long_name": "Discharge on topographic surface"},
        ),
        "ie_discharge": (
            ("time", "y", "x"),
            np.empty((Ns, mg.shape[0], mg.shape[1])),
            {"units": "m3/yr", "long_name": "Discharge from infiltration excess"},
        ),
        "karst_discharge": (
            ("time", "y", "x"),
            np.empty((Ns, mg.shape[0], mg.shape[1])),
            {"units": "m3/s", "long_name": "Discharge from karst"},
        ),
    },
    coords={
        "x": (
            ("x"),  # tuple of dimensions
            mg.x_of_node.reshape(mg.shape)[0, :],  # 1-d array of coordinate data
            {"units": "meters"},
        ),  # dictionary with data attributes
        "y": (("y"), mg.y_of_node.reshape(mg.shape)[:, 1], {"units": "meters"}),
        "time": (
            ("time"),
            dt * np.arange(Ns) / 1e3,
            {"units": "thousands of years since model start", "standard_name": "time"},
        ),
    },
)
ds = ds.assign_attrs(params)

#%% run forward


for i in tqdm(range(N)):

    # infiltration excess based on the fraction of r (p-e) that goes to ie. 
    q_ie[mg.core_nodes] = mg.at_node['ie_frac'][mg.core_nodes] * r_tot
    r[mg.core_nodes] = (1 - mg.at_node['ie_frac'][mg.core_nodes]) * r_tot

    # get flow directions and discharge on the topographic surface
    # first set discharge field to infiltration excess rate
    mg.at_node['water__unit_flux_in'][:] = q_ie
    lmb1.run_one_step()
    fa1.run_one_step()
    Q1[:] = mg.at_node['surface_water__discharge'].copy()

    # get flow directions and discharge on the basement surface. Some of this is in the karst.
    # first set discharge equal to recharge field
    mg.at_node['water__unit_flux_in'][:] = r
    lmb2.run_one_step()
    fa2.run_one_step()
    Q2[:] = mg.at_node['surface_water__discharge'].copy()

    # add conditionally to get total surface water discharge
    Q2_masked = np.zeros_like(Q2)
    Q2_masked[rock_ID==1] = Q2[rock_ID==1]
    Q_tot[:] = Q1 + Q2_masked

    # update topography
    fs2.run_one_step(dt)
    ld2.run_one_step(dt)
    z[mg.core_nodes] -= mg.at_node['chemical_denudation_rate'][mg.core_nodes] * dt
    z += dz_ad

    # update lithologic model, accounting erosion and advection (uplift)
    lith.rock_id = 0
    lith.dz_advection = dz_ad
    lith.run_one_step()

    # update the definition of the karst surface
    zk[:] = z-lith.z_top[0,:]

    ######## Save output

    if i%save_freq==0:
        # print(f"Finished iteration {i}")

        for of in output_fields:
            ds[of][i//save_freq, :, :] = mg["node"][of].reshape(mg.shape)


ds.to_netcdf(os.path.join(save_directory,f"{filename}.nc"))


# %% plot topographic change

# topography
plt.figure()
mg.imshow("topographic__elevation", colorbar_label='Elevation [m]')
plt.savefig(os.path.join(save_directory,f"{filename}_elevation.png"))

plt.figure()
mg.imshow("rock_type__id", cmap="viridis", colorbar_label='Rock ID')
plt.savefig(os.path.join(save_directory,f"{filename}_rockid.png"))

plt.figure()
mg.imshow('total_discharge', cmap="plasma", colorbar_label='Discharge')
plt.savefig(os.path.join(save_directory,f"{filename}_totalQ.png"))


# %%
