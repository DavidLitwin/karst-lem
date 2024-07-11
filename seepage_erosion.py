
#%%
import os
import numpy as np
import pandas as pd
from tqdm import tqdm

import matplotlib.pyplot as plt

from landlab import RasterModelGrid
from landlab.components import (
    FastscapeEroder, 
    FlowAccumulator,
    FlowDirectorD8,
    LakeMapperBarnes,
    LinearDiffuser,
    GroundwaterDupuitPercolator,
)
from landlab.io.netcdf import to_netcdf

fig_directory = '/Users/dlitwin/Documents/Research/Karst landscape evolution/landlab_virtual_karst/figures'
save_directory = '/Users/dlitwin/Documents/Research Data/Local/karst_lem'
id = "seepage_test_1"

#%% parameters

# U = 1e-4 # uplift m/yr
K = 1e-2 # streampower incision (yr^...)
D = 1e-3 # diffusivity (m2/yr)

b = 20
n = 0.01
ksat = 1000 / (365 * 24 * 3600)
r = 1e-8 # total runoff m/s

N = 500 # number of geomorphic timesteps
dt = 500 # geomorphic timestep (yr)
dt_gw = 1 * 24 * 3600 # groundwater timestep (s)
save_freq = 100 # steps between saving output
output_fields = [
        "at_node:topographic__elevation",
        "at_node:aquifer_base__elevation",
        "at_node:water_table__elevation",
        "at_node:surface_water_discharge",
        "at_node:local_runoff",
        ]
save_vals = [
               'mean_relief',
               'max_relief', 
               'median_aquifer_thickness', 
               'discharge_lower', 
               'discharge_upper',
               'area_lower', 
               'area_upper',
               'first_wtdelta', 
               'wt_iterations'
               ]

#%% set up grid and lithology

Nx = 100
Ny = 50
dx = 10
mg = RasterModelGrid((Ny,Nx), xy_spacing=dx)
mg.set_closed_boundaries_at_grid_edges(right_is_closed=True,
                                       left_is_closed=True,
                                       top_is_closed=True,
                                       bottom_is_closed=False)

np.random.seed(10010)
z = mg.add_zeros("node", "topographic__elevation")
z += 0.1*np.random.rand(len(z))
z[mg.core_nodes] += b #force thickness to zero at boundary
# z[np.where(mg.y_of_node==dx)[0][::10]] = 0.1
z[np.where(mg.x_of_node==500)[0][0:Ny//2]] = np.linspace(0,b/2,Ny//2)
zb = mg.add_zeros('node', 'aquifer_base__elevation')
zwt = mg.add_zeros('node', 'water_table__elevation')
zwt[:] = 0.9*b

# exfiltration
q_ex = mg.add_zeros("node", "exfiltration")


#%%

gdp = GroundwaterDupuitPercolator(
    mg, 
    hydraulic_conductivity=ksat,
    porosity=n,
    recharge_rate=r,
)
fd = FlowDirectorD8(mg)
fa = FlowAccumulator(
    mg,
    surface="topographic__elevation",
    flow_director=fd,
    runoff_rate="average_surface_water__specific_discharge",
)
lmb = LakeMapperBarnes(
    mg,
    method="Steepest",
    fill_flat=False,
    surface="topographic__elevation",
    fill_surface="topographic__elevation",
    redirect_flow_steepest_descent=False,
    reaccumulate_flow=False,
    track_lakes=False,
    ignore_overfill=True,
)

fs = FastscapeEroder(mg, K_sp=K, discharge_field='surface_water__discharge')
ld = LinearDiffuser(mg, linear_diffusivity=D)

lmb.run_one_step()

#%% run forward

h = mg.at_node['aquifer__thickness']
Ns = N//save_freq

df_out = pd.DataFrame(np.zeros((N,len(save_vals))), columns=save_vals)

for i in tqdm(range(N)):
    
    # iterate for steady state water table
    wt_delta = 1
    wt_iter = 0
    while wt_delta > 1e-4:

        zwt0 = zwt.copy()
        gdp.run_with_adaptive_time_step_solver(dt_gw)
        wt_delta = np.mean(np.abs(zwt0 - zwt))/(dt_gw/3600)

        if wt_iter == 0:
            df_out['first_wtdelta'].loc[i] = wt_delta
        wt_iter += 1
    df_out['wt_iterations'].loc[i] = wt_iter
    

    # local runoff is sum of saturation and infiltration excess. Convert units to m/yr. 
    q_ex[mg.core_nodes] = np.maximum(mg.at_node['average_surface_water__specific_discharge'][mg.core_nodes] -r ,0)

    # update areas
    fa.run_one_step()

    # update topography
    fs.run_one_step(dt)
    # ld.run_one_step(dt)

    # remove depressions
    lmb.run_one_step()

    # something to handle the aquifer itself - see regolith models in DupuitLEM
    # this should cover it, but again check boundary conditions
    zwt[zwt>z] = z[zwt>z]
    h[mg.core_nodes] = (zwt - zb)[mg.core_nodes]

    # metrics of change
    df_out['mean_relief'].loc[i] = np.mean(z[mg.core_nodes])
    df_out['max_relief'].loc[i] = np.max(z[mg.core_nodes])
    df_out['median_aquifer_thickness'].loc[i] = np.median(h[mg.core_nodes])

    # save output
    if i%save_freq==0:
        print(f"Finished iteration {i}")
        print(f"Mean aquifer thickness: {np.mean(h[mg.core_nodes])}")

        # save the specified grid fields
        filename = os.path.join(save_directory, id, f"{id}_grid_{i}.nc")
        to_netcdf(mg, filename, include=output_fields, format="NETCDF4")

#%% save out

df_out['time'] = np.arange(0,N*dt,dt)
df_out.set_index('time', inplace=True)
df_out.to_csv(os.path.join(save_directory, id, f"{id}_output.csv"))

# %% plot topogrpahic change

# topography
plt.figure()
mg.imshow("topographic__elevation", colorbar_label='Elevation [m]')
plt.savefig(os.path.join(save_directory, id, "elevation.png"))

Q_an = mg.at_node['surface_water__discharge']/mg.at_node['drainage_area']
Q_an[np.isnan(Q_an)] = 0
plt.figure()
mg.imshow(Q_an, cmap="Blues", colorbar_label='Discharge')
plt.savefig(os.path.join(save_directory, id, "lith_gdp_discharge.png"))

# %%

plt.figure()
inds = np.where(mg.x_of_node==500)[0][0:Ny//2]
inds1 = np.where(mg.x_of_node==500)[0]
plt.plot(mg.y_of_node[inds1], z[inds1], label='final')
plt.plot(mg.y_of_node[inds], np.linspace(0,b/2,Ny//2), label='initial')
plt.legend()
# %%
