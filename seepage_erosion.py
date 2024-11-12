
#%%
import os
import numpy as np
import pandas as pd
from tqdm import tqdm

import matplotlib.pyplot as plt
from matplotlib import colors

from landlab import RasterModelGrid
from landlab.components import (
    FastscapeEroder, 
    FlowAccumulator,
    FlowDirectorD8,
    FlowDirectorSteepest,
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
ksat = 20 / (3600*24)
r = 0.1 / (3600*24) # total runoff m/s

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


Nx = 50
Ny = 50
dx = 10
channel_x = (dx * Nx) // 2
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
z[np.where(mg.x_of_node==channel_x)[0][0:Ny//2]] = np.linspace(0,b/2,Ny//2)
zb = mg.add_zeros('node', 'aquifer_base__elevation')
zwt = mg.add_zeros('node', 'water_table__elevation')
zwt[:] = 0.8*z

# exfiltration
qex = mg.add_zeros("node", "exfiltration")

plt.figure()
mg.imshow('topographic__elevation')

#%%

plt.figure()
inds = np.where(mg.x_of_node==channel_x)[0][0:Ny]
plt.fill_between(mg.y_of_node[inds], z[inds], y2=0, color='0.5')
plt.fill_between(mg.y_of_node[inds], zwt[inds], y2=0, color='dodgerblue')
plt.xlabel('y coordinate')
plt.ylabel('elevation')



#%%

gdp = GroundwaterDupuitPercolator(
    mg, 
    hydraulic_conductivity=ksat,
    porosity=n,
    recharge_rate=r,
)
fd = FlowDirectorSteepest(mg)
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

#%% Single step test

wt_delta = 1
wt_iter = 0
while wt_delta > 1e-5:

    zwt0 = zwt.copy()
    gdp.run_with_adaptive_time_step_solver(dt_gw)
    wt_delta = np.mean(np.abs(zwt0 - zwt))/(dt_gw/3600)

    wt_iter += 1

fd.run_one_step()

#%%

# initialize updated qs field
qs = mg.at_node['surface_water__specific_discharge']
qs[np.isnan(qs)] = 0
qs_update = qs.copy()
frl = mg.at_node['flow__link_to_receiver_node']
frn = mg.at_node['flow__receiver_node']

# exfiltration at recevier is the amount that qs exceeds recharge.
qex[mg.core_nodes] = np.maximum(qs[mg.core_nodes] - r ,0)
qex_receiver = qex[frn]

##  allocation should scale with flux into receiver (optional -- need to think about how to incorporate this)
# sign_link_into_receiver = (mg.node_at_link_head[frl] == frn)*2 - 1
# flux = mg.at_link['groundwater__specific_discharge'][frl] * sign_link_into_receiver
# faces = mg.face_at_link[frl]
# face_widths = mg.length_of_face[faces]
# flux_norm = (flux * face_widths / mg.cell_area_at_node[frn])/qex_receiver
# flux_norm[np.isnan(flux_norm)] = 0
# flux_norm[np.isinf(flux_norm)] = 0

## allocation should scale with flux into node (optional)
# gw = - mg.at_link["groundwater__specific_discharge"][mg.links_at_node] * mg.link_dirs_at_node
# gw[gw<0] = 0
# gw_rel = gw / (gw.sum(axis=1).reshape((len(gw),1)) + np.spacing(1))
# gw_rel[gw_rel<0] = 0

# gw_rel_frl = np.zeros_like(qex_receiver)
# for n, rl in enumerate(frl):
#     if rl != -1:
#         lan = mg.links_at_node[n,:]
#         ind = np.where(lan==rl)[0]
#         gw_rel_frl[n] = gw_rel[n,ind]
#         if ind.size == 0:
#             print('empty')

# allocation from receiver to upstream is proportional to slope
gr = np.abs(mg.calc_grad_at_link(z)) # comfortable taking abs because slope should always be toward receiver
sinegr = np.sin(np.arctan(gr)) # use sine -- zero when slope is zero between source and receiver, maximum in the limit pure vertical
sinegr_receiver = sinegr[frl]

# add up to 0.25 of exfiltration from receiver with exfiltration to upstream node. Remove from receiver.
# 0.25 because this ensures that we never exceed 1 for raster grid with 4 possible receivers. 
# More complex way would be to set a value based on the flux contribution, calculated above.
for n, rn in enumerate(frn):
    value = 0.25 * sinegr_receiver[n] * qex_receiver[n] # * gw_rel_frl[n]
    qs_update[n] += value
    qs_update[rn] -= value


## does not vectorize right...
# values = 0.25 * sinegr_receiver * qex_receiver
# qs_update += values
# qs_update[frn] -= values

assert np.sum(qs_update) == np.sum(qs)

#%%


# rg = RasterModelGrid((5, 5), xy_spacing=10.0)
# z = rg.add_zeros("topographic__elevation", at="node")
# z[:] = 50.0
# z[10:14] = 25
# rg.imshow(z)
# hg = rg.calc_grad_at_link(z)
# gr = - rg.calc_grad_at_link(z)[rg.links_at_node] * rg.link_dirs_at_node
# np.sum(gr>0, axis=1)

# nalh = rg.node_at_link_head
# nalt = rg.node_at_link_tail

#%%
q=mg.at_node['surface_water__specific_discharge']
q[mg.open_boundary_nodes] = 0
mg.imshow('surface_water__specific_discharge', cmap='plasma')

fig, ax = plt.subplots()
inds = np.where(mg.x_of_node==channel_x)[0][0:Ny]
ax.fill_between(mg.y_of_node[inds], z[inds], y2=0, color='0.5')
ax.fill_between(mg.y_of_node[inds], zwt[inds], y2=0, color='dodgerblue')
ax.scatter(mg.y_of_node[inds], zwt[inds], c='k', s=5)
ax.set_xlabel('y coordinate')
ax.set_ylabel('elevation')
ax.set_ylim((0,2*max(z)))

ax1 = plt.twinx()
ax1.plot(mg.y_of_node[inds], q[inds])
ax1.plot(mg.y_of_node[inds], qs_update[inds])
ax1.set_ylim(2*max(q),0)


plt.figure()
qs_update[mg.open_boundary_nodes] = 0
mg.imshow(qs_update, cmap='plasma')


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
    qex[mg.core_nodes] = np.maximum(mg.at_node['average_surface_water__specific_discharge'][mg.core_nodes] -r ,0)

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
