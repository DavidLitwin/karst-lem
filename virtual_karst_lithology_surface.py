
"""
Slightly more complex test of virtual karst in landlab. Now we will use 
the Lithology component to track two layers: a limestone layer, and a basement.
Use lithology difference to determine where there is loss in flow accumulation,
and use the lithologic contact to determine where there will be springs.
"""

#%%
import os
import copy
import pickle
import numpy as np
import pandas as pd
import xarray

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import LightSource

from landlab import RasterModelGrid
from landlab.components import (
    FastscapeEroder, 
    LossyFlowAccumulator, 
    FlowAccumulator,
    FlowDirectorD8,
    LakeMapperBarnes,
    LinearDiffuser,
    LithoLayers,
)

from virtual_karst_funcs import *

save_directory = '/Users/dlitwin/Documents/Research/Karst landscape evolution/landlab_virtual_karst/figures'

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

r_tot = 1.0 / (3600 * 24 * 365) # total runoff m/s
ie_frac = 0.3 # fraction of r_tot that becomes overland flow on limestone. ie_frac on basement=1. 
# ibar = 1e-3 / 3600 # mean storm intensity (m/s) equiv. 1 mm/hr 

T = 5e6 # total geomorphic time
dt = 500 # geomorphic timestep (yr)
N = int(T//dt) # number of geomorphic timesteps

Nx = 150
Ny = 100
dx = 50

noise_scale = 0.01


save_freq = 100 # steps between saving output
Ns = N//save_freq
output_fields = [
        "topographic__elevation",
        "aquifer_base__elevation",
        "water_table__elevation",
        "surface_water__discharge",
        "local_runoff",
        "rock_type__id",
        "exfiltration_rate",
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
            ]
df_out = pd.DataFrame(np.zeros((N,len(save_vals))), columns=save_vals)
df_params = pd.DataFrame({'U':U, 'K':K_sp, 'D_ld':D_ld, 'm_sp':m_sp,'n_sp':n_sp, 
                          'E_limestone': E_limestone, 'E_weathered_basement':E_weathered_basement,
                          'b_limestone':b_limestone, 'b_basement':b_basement, 'bed_dip':bed_dip,
                          'r_tot':r_tot, 'ie_frac':ie_frac, 
                          'T':T, 'N':N, 'dt':dt, 'save_freq':save_freq}, index=[0])
df_params.to_csv(os.path.join(save_directory, id, f"params_{id}.csv"))


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
    mg, layer_elevations, layer_ids, function=lambda x, y: - bed_dip * y, attrs=attrs
)
dz_ad = np.zeros(mg.size("node"))
dz_ad[mg.core_nodes] = U * dt
# dz_ad[top_nodes] += 0.5 * U * dt # effectively decrease the baselevel fall rate of the upper boundary


# the karst or basement surface is the topographic elevation minus the depth to the top of the basement
zk = mg.add_zeros("node", "karst__elevation")
zk[:] = z-lith.z_top[0,:]


# infiltration excess based on the fraction of r (p-e) that goes to ie. 
q_ie = mg.add_zeros("node", "infiltration_excess")
q_ie[mg.core_nodes] = mg.at_node['ie_frac'][mg.core_nodes] * r_tot
r = mg.add_zeros("node", "recharge_rate")
r[mg.core_nodes] = (1 - mg.at_node['ie_frac'][mg.core_nodes]) * r_tot


#%% instantiate components 


# flow management for the topographic surface, routing only infiltration excess
fa1 = FlowAccumulator(
    mg,
    surface="topographic__elevation",
    flow_director='FlowDirectorD8',
    runoff_rate="infiltration_excess",
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
# basement it is set to zero
fa2 = FlowAccumulator(
    mg,
    surface="karst__elevation",
    flow_director='FlowDirectorD8',
    runoff_rate="recharge_rate",
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

# get flow directions and discharge on the topographic surface
lmb1.run_one_step()
fa1.run_one_step()
Q1 = mg.at_node['surface_water__discharge'].copy()

# get flow directions and discharge on the basement surface. Some of this is in the karst.
lmb2.run_one_step()
fa2.run_one_step()
Q2 = mg.at_node['surface_water__discharge'].copy()

# add conditionally to get total surface water discharge
Q_tot = mg.add_zeros("node", "total_discharge")
rock_ID = mg.at_node['rock_type__id']
Q2_masked = np.zeros_like(Q2)
Q2_masked[rock_ID==1] = Q2[rock_ID==1]
Q_tot[:] = Q1 + Q2_masked


fs2 = FastscapeEroder(mg, K_sp=K_sp, m_sp=m_sp, n_sp=n_sp, discharge_field='total_discharge')
ld2 = LinearDiffuser(mg, linear_diffusivity=D_ld)


#%% run forward


for i in range(N):

    # get flow directions and discharge on the topographic surface
    lmb1.run_one_step()
    fa1.run_one_step()
    Q1 = mg.at_node['surface_water__discharge'].copy()

    # get flow directions and discharge on the basement surface. Some of this is in the karst.
    lmb2.run_one_step()
    fa2.run_one_step()
    Q2 = mg.at_node['surface_water__discharge'].copy()

    # add conditionally to get total surface water discharge
    Q2_masked = np.zeros_like(Q2)
    Q2_masked[rock_ID==1] = Q2[rock_ID==1]
    Q_tot[:] = Q1 + Q2_masked

    # update topography
    fs2.run_one_step(dt)
    ld2.run_one_step(dt)
    z[mg.core_nodes] -= mg.at_node['chemical_denudation_rate'] * dt
    z += dz_ad

    lith.rock_id = 0
    lith.dz_advection = dz_ad
    lith.run_one_step()


    ######## Save output

    if i%save_freq==0:
        print(f"Finished iteration {i}")


# %% plot topogrpahic change

# topography
plt.figure()
mg.imshow("topographic__elevation", colorbar_label='Elevation [m]')
plt.savefig(os.path.join(save_directory,"litholayers_topog_springs_0.5.png"))

plt.figure()
mg.imshow("rock_type__id", cmap="viridis", colorbar_label='Rock ID')
plt.savefig(os.path.join(save_directory,"litholayers_rocks_springs_0.5.png"))

plt.figure()
mg.imshow('surface_water__discharge', cmap="plasma", colorbar_label='Discharge')
plt.savefig(os.path.join(save_directory,"litholayers_discharge_springs_0.5.png"))

