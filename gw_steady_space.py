"""
No uplift, fluvial erosion with space and nonlinear hillslope diffusion

"""
#%%
import os
import numpy as np
import pandas as pd
from tqdm import tqdm

import matplotlib.pyplot as plt
from landlab import RasterModelGrid
from landlab.components import (
    SpaceLargeScaleEroder,
    GroundwaterDupuitPercolator,
    TaylorNonLinearDiffuser,
    PriorityFloodFlowRouter,
)
from landlab.io.netcdf import to_netcdf

fig_directory = '/Users/dlitwin/Documents/Research/Karst landscape evolution/landlab_virtual_karst/figures'
save_directory = '/Users/dlitwin/Documents/Research Data/Local/karst_lem'
id = "seepage_test_2"

#%% parameters

Nx = 50 # grid Nx
Ny = 50 # grid Ny

v0 = 10 # grid spacing
D = 0.1 #0.2 # # hillslope diffusivity (m2/yr)
Sc = 0.1 # critical slope
Kr = 1e-4 #5e-4 5e-5# bedrock erodibility
Ks = 50 * Kr #4*Kr # sediment erodibility
Ff = 0.0 # fraction fine material (that does not contribute to sediment transport)
m = 0.5 # streampower m
n = 1 # streampower n
H_star = 1 # bed roughness scale
Vs = 0.0 # 5.0 # settling velocity
r = 1.0 # runoff rate

b = 50
n = 0.01
ksat = 20 / (3600*24)
r = 0.0 #0.1 / (3600*24) # total runoff m/s -- try no runoff

N = 5000 # number of geomorphic timesteps
dt = 100 # geomorphic timestep (yr)

dt_gw = 1 * 3600 # groundwater timestep (s)
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


mg = RasterModelGrid((Ny,Nx), xy_spacing=v0)
mg.set_status_at_node_on_edges(right=mg.BC_NODE_IS_FIXED_VALUE,
                                top=mg.BC_NODE_IS_FIXED_VALUE,
                                left=mg.BC_NODE_IS_FIXED_VALUE,
                                bottom=mg.BC_NODE_IS_FIXED_VALUE,
)
channel_x = (v0 * Nx) // 2
channel_nodes = np.where(mg.x_of_node==channel_x)[0][0:5]
mg.status_at_node[channel_nodes] = mg.BC_NODE_IS_FIXED_VALUE


np.random.seed(10010)
z = mg.add_zeros("node", "topographic__elevation")
zba = mg.add_zeros('node', 'aquifer_base__elevation')
zwt = mg.add_zeros('node', 'water_table__elevation')
zb = mg.add_zeros("bedrock__elevation", at="node")
hs = mg.add_zeros("node", "soil__depth")

z[:] = b + 0.1*np.random.rand(len(z))
z[np.where(mg.y_of_node==0)[0][:]] = 0.0
z[channel_nodes] = 0.0
zba[:] = -1.0
zwt[:] = b - 45 #0.5
zwt[zwt>z] = z[zwt>z]

zb[:] = zba
hs[:] = z - zb

plt.figure()
mg.imshow('topographic__elevation')

plt.figure()
mg.imshow(mg.status_at_node)

#%%

plt.figure()
inds = np.where(mg.x_of_node==channel_x)[0][0:Ny]
plt.fill_between(mg.y_of_node[inds], z[inds], y2=0, color='0.5')
plt.fill_between(mg.y_of_node[inds], zwt[inds], y2=0, color='dodgerblue', alpha=0.5)
plt.xlabel('y coordinate')
plt.ylabel('elevation')



#%%

gdp = GroundwaterDupuitPercolator(
    mg, 
    hydraulic_conductivity=ksat,
    porosity=n,
    recharge_rate=r,
)

# Instantiate flow router
fr = PriorityFloodFlowRouter(mg, flow_metric="D8", runoff_rate='average_surface_water__specific_discharge')

# Instantiate SPACE model with chosen parameters
sp = SpaceLargeScaleEroder(
    mg,
    K_sed=Ks,
    K_br=Kr,
    F_f=Ff,
    phi=0.0,
    H_star=H_star,
    v_s=Vs,
    m_sp=m,
    n_sp=n,
    sp_crit_sed=0,
    sp_crit_br=0,
)

nld = TaylorNonLinearDiffuser(mg, linear_diffusivity=D, slope_crit=Sc, dynamic_dt=True)

#%% Single step test

wt_delta = 1
wt_iter = 0
while wt_delta > 1e-5:

    zwt0 = zwt.copy()
    gdp.run_with_adaptive_time_step_solver(dt_gw)
    wt_delta = np.mean(np.abs(zwt0 - zwt))/(dt_gw/3600)

    wt_iter += 1

#%% run forward

h = mg.at_node['aquifer__thickness']
Ns = N//save_freq

# outlet adjacent nodes
nodes = np.arange(len(z)).reshape(mg.shape)
adjacent_nodes = nodes[1,:]

elapsed_time = 0.0  # years
count = 0
sed_flux = np.zeros(N)
relief = np.zeros(N)


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

    # diffusion
    nld.run_one_step(dt)

    # Run the flow router
    fr.run_one_step()

    # Run SPACE for one time step
    sp.run_one_step(dt=dt)

    # Save sediment flux and relief to array
    sed_flux[count] = np.sum(mg.at_node["sediment__flux"][adjacent_nodes])

    # something to handle the aquifer itself - see regolith models in DupuitLEM
    # this should cover it, but again check boundary conditions
    zwt[zwt>z] = z[zwt>z]
    h[mg.core_nodes] = (zwt - zb)[mg.core_nodes]

    # metrics of change
    df_out['mean_relief'].loc[i] = np.mean(z[mg.core_nodes])
    df_out['max_relief'].loc[i] = np.max(z[mg.core_nodes])
    df_out['median_aquifer_thickness'].loc[i] = np.median(h[mg.core_nodes])

    # save output
    # if i%save_freq==0:
    #     print(f"Finished iteration {i}")
    #     print(f"Mean aquifer thickness: {np.mean(h[mg.core_nodes])}")

    #     # save the specified grid fields
    #     filename = os.path.join(save_directory, id, f"{id}_grid_{i}.nc")
    #     to_netcdf(mg, filename, include=output_fields, format="NETCDF4")

#%% save out

df_out['time'] = np.arange(0,N*dt,dt)
df_out.set_index('time', inplace=True)
df_out.to_csv(os.path.join(save_directory, id, f"{id}_output.csv"))

# %% plot topogrpahic change

# topography
plt.figure()
mg.imshow("topographic__elevation", colorbar_label='Elevation [m]')
# plt.savefig(os.path.join(save_directory, id, "elevation.png"))

Q_an = mg.at_node['surface_water__discharge']/mg.at_node['drainage_area']
Q_an[np.isnan(Q_an)] = 0

plt.figure()
mg.imshow(Q_an, cmap="Blues", colorbar_label='Discharge')
# plt.savefig(os.path.join(save_directory, id, "lith_gdp_discharge.png"))
#%%

plt.figure()
mg.imshow(zwt, cmap="Blues", colorbar_label='DTW')
# plt.savefig(os.path.join(save_directory, id, "lith_gdp_discharge.png"))

# %%


plt.figure()
inds = np.where(mg.x_of_node==channel_x)[0][0:Ny]
plt.fill_between(mg.y_of_node[inds], z[inds], y2=0, color='0.5')
plt.fill_between(mg.y_of_node[inds], zwt[inds], y2=0, color='dodgerblue', alpha=0.5)
plt.xlabel('y coordinate')
plt.ylabel('elevation')

# %%
