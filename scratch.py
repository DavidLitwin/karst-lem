"""
tests to figure out next steps with karst lem
"""

#%%
import os
import copy
import pickle
import numpy as np
import pandas as pd
import xarray as xr

import holoviews as hv

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


#%% multiple flow accumulators

mg = RasterModelGrid((100,150), xy_spacing=50)
mg.set_closed_boundaries_at_grid_edges(right_is_closed=True,
                                       left_is_closed=True,
                                       top_is_closed=False,
                                       bottom_is_closed=False)

z = mg.add_zeros("node", "topographic__elevation")
zk = mg.add_zeros("node", "karst__elevation")
np.random.seed(10010)
z += 5 + 0.1*np.random.rand(len(z))
zk += 0.1*np.random.rand(len(zk))

# flow accumulators for topographic and karst surfaces
fa1 = FlowAccumulator(mg, surface='topographic__elevation', flow_director='FlowDirectorD8', runoff_rate=1.0)
fa2 = FlowAccumulator(mg, surface='karst__elevation', flow_director='FlowDirectorD8', runoff_rate=1.0)

#%% run and test

# run them in sequence, and then go back and run the first. Fields are overwritten, so will need to save in between, and re-run flow directors each time.
fa1.run_one_step()
q1 = mg.at_node['surface_water__discharge'].copy()
fa2.run_one_step()
q2 = mg.at_node['surface_water__discharge'].copy()

fa1.run_one_step()
q1a = mg.at_node['surface_water__discharge'].copy()

# test that running fa2 in between does not affect q1
np.allclose(q1, q1a)

# test that discharge is different on the two surfaces (different random noise added)
np.allclose(q1,q2)


#####################################
#%% second test of flow routing


mg = RasterModelGrid((100,150), xy_spacing=1)
mg.set_closed_boundaries_at_grid_edges(right_is_closed=True,
                                       left_is_closed=True,
                                       top_is_closed=False,
                                       bottom_is_closed=False)

z = mg.add_zeros("node", "topographic__elevation")
zk = mg.add_zeros("node", "karst__elevation")
np.random.seed(10010)
z += 5 + 0.1*np.random.rand(len(z))
zk += 0.1*np.random.rand(len(zk))

# np.random.seed(10010)
# z += 0.1*np.random.rand(len(zk))
# zk[:] = z.copy()


# infiltration excess based on the fraction of r (p-e) that goes to ie. 
q_ie = mg.add_zeros("node", "infiltration_excess")
q_ie[mg.core_nodes] = 0.1
r = mg.add_zeros("node", "recharge_rate")
r[mg.core_nodes] =  0.1

# flow management for the topographic surface, routing only infiltration excess
fa1 = FlowAccumulator(
    mg,
    surface="topographic__elevation",
    flow_director='FlowDirectorD8',
    # runoff_rate="infiltration_excess",
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
    # runoff_rate="recharge_rate",
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
Q1 = mg.add_zeros("node", "ie_discharge")
mg.at_node['water__unit_flux_in'][:] = mg.at_node['infiltration_excess']
print(max(mg.at_node['water__unit_flux_in']))
lmb1.run_one_step()
fa1.run_one_step()
Q1[:] = mg.at_node['surface_water__discharge'].copy()
print(np.sum(Q1[mg.open_boundary_nodes]))

# get flow directions and discharge on the basement surface. Some of this is in the karst.
Q2 = mg.add_zeros("node", "karst_discharge")
mg.at_node['water__unit_flux_in'][:] = mg.at_node['recharge_rate']
print(max(mg.at_node['water__unit_flux_in']))
lmb2.run_one_step()
fa2.run_one_step()
Q2[:] = mg.at_node['surface_water__discharge'].copy()
print(np.sum(Q2[mg.open_boundary_nodes]))


np.allclose(Q1,Q2)


# %% ################################# 

# Test lithology to make sure we can get flow director on the right surface

mg = RasterModelGrid((100,150), xy_spacing=1)
mg.set_closed_boundaries_at_grid_edges(right_is_closed=True,
                                       left_is_closed=True,
                                       top_is_closed=False,
                                       bottom_is_closed=False)
bottom_nodes = mg.nodes_at_bottom_edge
top_nodes = mg.nodes_at_top_edge
z = mg.add_zeros("node", "topographic__elevation")
np.random.seed(10010)
z += 0.1*np.random.rand(len(z))

# two layers, depths below surface. Top layer is limestone, bottom is basement.
layer_elevations = [2,20]
layer_ids = [0,1]
attrs = {
         "ie_frac": {0: 0.5, 1: 1}, # fraction of r_tot that becomes infiltration excess runoff
         }

# initiate litholayers.
bed_dip = -0.001
# z_surf = generate_conditioned_surface(mg, lambda x, y: -bed_dip * y, noise_coeff=0.1, random_seed=100111)
# lith = LithoLayers(
#     mg, layer_elevations, layer_ids, function=lambda x, y: 0*x + 0*y + z_surf, attrs=attrs
# )
lith = LithoLayers(
    mg, layer_elevations, layer_ids, function=lambda x, y: -bed_dip * y + 0.1*np.random.rand(len(z)), attrs=attrs
)
dz_ad = np.zeros(mg.size("node"))
dz_ad[mg.core_nodes] = 5
lith.dz_advection = dz_ad #advection = uplift in timestep
lith.rock_id = 0

zk = mg.add_zeros("node", "karst__elevation")
zk[:] = z-lith.z_top[0,:]


#%%

# mimic erosion that increases in the positive y direction. Add a random component to make it different than the subsurface.
z[mg.core_nodes] -= 0.035 * mg.y_of_node[mg.core_nodes]
z += dz_ad # add uplift to topography
lith.run_one_step() # update lithology model
zk[:] = z-lith.z_top[0,:] # update karst elevation

plt.figure()
mg.imshow(z)
plt.figure()
mg.imshow('rock_type__id')

# %%

# cross section
# z_bottom and z_top refer to elevation below surface topography. To get the actual surface elevation, need elevation - lithology surface
x = 50
nodes = np.where(mg.x_of_node==x)[0][1:-1]
plt.figure()
plt.fill_between(mg.y_of_node[nodes], y1=z[nodes]-lith.z_bottom[0,nodes], y2=z[nodes]-lith.z_top[0,nodes])
plt.fill_between(mg.y_of_node[nodes], y1=z[nodes]-lith.z_bottom[1,nodes], y2=z[nodes]-lith.z_top[1,nodes])
# plt.plot(mg.y_of_node[nodes], z[nodes]-lith.z_top[0,nodes], 'k--')  
plt.plot(mg.y_of_node[nodes], zk[nodes], 'k--')  

# %%
# flow accumulation on these surfaces 


r_tot = 1

# infiltration excess based on the fraction of r (p-e) that goes to ie. 
q_ie = mg.add_zeros("node", "infiltration_excess")
q_ie[mg.core_nodes] = mg.at_node['ie_frac'][mg.core_nodes] * r_tot
r = mg.add_zeros("node", "recharge_rate")
r[mg.core_nodes] = (1 - mg.at_node['ie_frac'][mg.core_nodes]) * r_tot


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

#%%
lmb1.run_one_step()
fa1.run_one_step()

Q1 = mg.at_node['surface_water__discharge'].copy()

lmb2.run_one_step()
fa2.run_one_step()

Q2 = mg.at_node['surface_water__discharge'].copy()


plt.figure()
mg.imshow(Q1, cmap='viridis')
plt.title('overland Discharge')

plt.figure()
mg.imshow(Q2, cmap='viridis')
mg.imshow('rock_type__id', cmap='Greys', alpha=0.3, allow_colorbar=False)
plt.title('Karst Discharge')

rock_ID = mg.at_node['rock_type__id']
Q2_masked = np.zeros_like(Q2)
Q2_masked[rock_ID==1] = Q2[rock_ID==1]
Q_tot = Q1 + Q2_masked

plt.figure()
mg.imshow(Q_tot, cmap='viridis')
# mg.imshow('rock_type__id', cmap='Greys', alpha=0.3, allow_colorbar=False)
plt.title('Total Discharge')

########################
#%% test erode one surface and then deposit limestone


mg = RasterModelGrid((100,150), xy_spacing=1)
mg.set_closed_boundaries_at_grid_edges(right_is_closed=True,
                                       left_is_closed=True,
                                       top_is_closed=False,
                                       bottom_is_closed=False)
bottom_nodes = mg.nodes_at_bottom_edge
top_nodes = mg.nodes_at_top_edge
z = mg.add_zeros("node", "topographic__elevation")
np.random.seed(10010)
y0 = np.mean(mg.y_of_node)
hmax = 1e-3
z += - 5 + 0.1*np.random.rand(len(z)) + (-hmax/y0) * mg.y_of_node**2 + 2*hmax * mg.y_of_node

plt.figure()
mg.imshow(z)


# two layers, depths below surface. Top layer is limestone, bottom is basement.
layer_elevations = [20]
layer_ids = [1]
attrs = {
         "ie_frac": {0: 0.5, 1: 1}, # fraction of r_tot that becomes infiltration excess runoff
         }

# initiate litholayers.
bed_dip = 0.0
lith = LithoLayers(
    mg, layer_elevations, layer_ids, function=lambda x, y: -bed_dip * y, attrs=attrs, layer_type='MaterialLayers'
)

# flow management for the topographic surface, routing only infiltration excess
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
    ignore_overfill=True
)
sp = FastscapeEroder(mg)

# fill depressions on the basement surface
lmb1.run_one_step()
fa1.run_one_step()
sp.run_one_step(500)

# update lithology so it knows the new surface
lith.dz_advection = 0.0
lith.rock_id = 1
lith.run_one_step()

z0 = z.copy()

#%% 

plt.figure()
mg.imshow(z)

plt.figure()
mg.imshow('drainage_area')

#%%

# deposit the limestone
z[:] = 0.1*np.random.rand(len(z))
lith.dz_advection = 0.0
lith.rock_id = 0
lith.run_one_step()

zk = mg.add_zeros("node", "karst__elevation")
zk[:] = z-lith.z_bottom[1,:]

plt.figure()
mg.imshow(z)

np.all(zk==z0)

#%%

# mimic erosion that increases in the positive y direction. Add a random component to make it different than the subsurface.
z[mg.core_nodes] -= 0.1 * mg.y_of_node[mg.core_nodes] + 0.1*np.random.rand(mg.number_of_core_nodes)
z += dz_ad # add uplift to topography
lith.run_one_step() # update lithology model
zk[:] = z-lith.z_top[0,:] # update karst elevation

plt.figure()
mg.imshow(z)
plt.figure()
mg.imshow('rock_type__id')

# %%

# cross section
# z_bottom and z_top refer to elevation below surface topography. To get the actual surface elevation, need elevation - lithology surface
x = 50
nodes = np.where(mg.x_of_node==x)[0][1:-1]
plt.figure()
plt.fill_between(mg.y_of_node[nodes], y1=z[nodes]-lith.z_bottom[0,nodes], y2=z[nodes]-lith.z_top[0,nodes])
plt.fill_between(mg.y_of_node[nodes], y1=z[nodes]-lith.z_bottom[1,nodes], y2=z[nodes]-lith.z_top[1,nodes])
# plt.plot(mg.y_of_node[nodes], z[nodes]-lith.z_top[0,nodes], 'k--')  
plt.plot(mg.y_of_node[nodes], zk[nodes], 'k--')  

# %%
# flow accumulation on these surfaces 


r_tot = 1

# infiltration excess based on the fraction of r (p-e) that goes to ie. 
q_ie = mg.add_zeros("node", "infiltration_excess")
q_ie[mg.core_nodes] = mg.at_node['ie_frac'][mg.core_nodes] * r_tot
r = mg.add_zeros("node", "recharge_rate")
r[mg.core_nodes] = (1 - mg.at_node['ie_frac'][mg.core_nodes]) * r_tot

# flow management for the basement surface, assuming all recharge accumulates at the base of
# the limestone, which is the top of the basement. Recharge only occurs on limestone. On the 
# basement it is set to zero
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

#%%

mg.at_node['water__unit_flux_in'][:] = q_ie
lmb1.run_one_step()
fa1.run_one_step()
Q1 = mg.at_node['surface_water__discharge'].copy()

mg.at_node['water__unit_flux_in'][:] = r
# lmb2.run_one_step()
fa2.run_one_step()
Q2 = mg.at_node['surface_water__discharge'].copy()


plt.figure()
mg.imshow(Q1, cmap='viridis')
plt.title('overland Discharge')

plt.figure()
mg.imshow(Q2, cmap='viridis')
mg.imshow('rock_type__id', cmap='Greys', alpha=0.3, allow_colorbar=False)
plt.title('Karst Discharge')

rock_ID = mg.at_node['rock_type__id']
Q2_masked = np.zeros_like(Q2)
Q2_masked[rock_ID==1] = Q2[rock_ID==1]
Q_tot = Q1 + Q2_masked

plt.figure()
mg.imshow(Q_tot, cmap='viridis')
# mg.imshow('rock_type__id', cmap='Greys', alpha=0.3, allow_colorbar=False)
plt.title('Total Discharge')



################################
# %% 
# test xarray saving datasets


mg = RasterModelGrid((10,15), xy_spacing=1)
mg.set_closed_boundaries_at_grid_edges(right_is_closed=True,
                                       left_is_closed=True,
                                       top_is_closed=False,
                                       bottom_is_closed=False)
nts = 5
z = mg.add_zeros("node", "topographic__elevation")
rock_id = mg.add_zeros("node", "rock_type__id")
dt = 1000

ds = xr.Dataset(
    data_vars={
        "topographic__elevation": (
            ("time", "y", "x"),  # tuple of dimensions
            np.empty((nts, mg.shape[0], mg.shape[1])),  # n-d array of data
            {
                "units": "meters",  # dictionary with data attributes
                "long_name": "Topographic Elevation",
            },
        ),
        "rock_type__id": (
            ("time", "y", "x"),
            np.empty((nts, mg.shape[0], mg.shape[1])),
            {"units": "-", "long_name": "Rock Type ID Code"},
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
            dt * np.arange(nts) / 1e3,
            {"units": "thousands of years since model start", "standard_name": "time"},
        ),
    },
)

grid_out_fields = ["topographic__elevation", "rock_type__id"]
for i in range(nts):
    z[:] += 1
    rock_id[:] = i%2
    sum_z = np.sum(z)

    for of in grid_out_fields:
        ds[of][i, :, :] = mg["node"][of].reshape(mg.shape)
    ds['sum_z'][i] = sum_z

# %%
ds.assign_attrs({'dt':1000, 'dx':1})
# %%

# test a method of getting upper and lower boundaries

a = 10
idx = np.ones((10,5))
idx[6:,:] = 0
idx[0:2,::2] = 0

# idx[0:5,:] = 0

# idx[5:,:] = 0
# idx[-1,::2] = 1
print(idx)
print('')

idx = np.pad(idx, ((1,),(0,)), 'constant', constant_values=0)

print(idx)
print('')
idx_diff = np.diff(idx, axis=0)
upper_pos = (np.argwhere(idx_diff.T==1)[:,1]) * a
lower_pos = (idx.shape[0] -2 - np.argwhere(idx_diff.T==-1)[:,1]) * a

print(upper_pos)
print(lower_pos)




def find_contact(mg, lith, rock_id=0):
    """
    Find distance from upper and lower boundaries to the contact between the 
    lithologies. Distance will be zero if rock_id is still at the boundary.
    """

    idx = ((lith.dz[rock_id,:] > 1e-2)*1).reshape(mg.shape)

    idx = np.pad(idx, ((1,),(0,)), 'constant', constant_values=0)
    idx_diff = np.diff(idx, axis=0)
    upper_pos = (np.argwhere(idx_diff.T==1)[:,1]) * a
    lower_pos = (idx.shape[0] -2 - np.argwhere(idx_diff.T==-1)[:,1]) * a



# %%
