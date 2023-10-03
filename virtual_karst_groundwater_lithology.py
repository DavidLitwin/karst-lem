"""
More complex test of virtual karst in landlab. Use the Lithology component to 
track two layers: a limestone layer, and a basement. Now use the thicknesses and
permeabilities of those lithologies to parametrize a GroundwaterDupuitPercolator
model.
"""

import os
import copy
import pickle
import numpy as np

import matplotlib.pyplot as plt
from matplotlib import cm

from landlab import RasterModelGrid
from landlab.components import (
    FastscapeEroder, 
    FlowAccumulator,
    LossyFlowAccumulator, 
    FlowDirectorD8,
    LakeMapperBarnes,
    LinearDiffuser,
    LithoLayers,
    GroundwaterDupuitPercolator,
)
from landlab.grid.mappers import map_max_of_node_links_to_node

#%% parameters

U = 1e-4
K = 1e-5
D = 1e-3


#%% set up grid and lithology

mg = RasterModelGrid((200,300), xy_spacing=25)
mg.set_closed_boundaries_at_grid_edges(right_is_closed=True,
                                       left_is_closed=True,
                                       top_is_closed=False,
                                       bottom_is_closed=False)

bottom_nodes = mg.nodes_at_bottom_edge
top_nodes = mg.nodes_at_top_edge
z = mg.add_zeros("node", "topographic__elevation")
np.random.seed(10010)
z += 0.1*np.random.rand(len(z))

# two layers, both with bottoms below ground. Top layer is limestone, bottom is basement.
# weathered_thickness is a way to add some somewhat realistic weathered zone in the
# basement that will host the aquifer there. In the limestone, the aquifer is just the whole unit.
layer_elevations = [100,1000]
layer_ids = [0,1]
attrs = {"Ksat": {0: 1e-3, 1: 1e-4}, "weathered_thickness": {0: 0.0, 1: 0.5}}

lith = LithoLayers(
    mg, layer_elevations, layer_ids, function=lambda x, y: - 0.002 * y, attrs=attrs
)

rock_id = 1
zb = mg.add_zeros('node', 'aquifer_base__elevation')
zb[:] = z - lith.z_bottom[rock_id,:]
zb[mg.open_boundary_nodes] = z[mg.open_boundary_nodes] - 0.10
zb[basement_not_contact] = zb[basement_not_contact] - 0.10
zwt = mg2.add_zeros('node', 'water_table__elevation')
zwt[mg2.core_nodes] = zb[mg2.core_nodes] + 0.01


#%%


gdp = GroundwaterDupuitPercolator(
    mg, 
    hydraulic_conductivity="Ksat",
    recharge_rate=3e-8 # ~ 1 m/yr
)
fd = FlowDirectorD8(mg)
fa = FlowAccumulator(
    mg,
    surface="topographic__elevation",
    flow_director=fd2,
    runoff_rate="surface_water__specific_discharge",
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

lmb2.run_one_step()

Q = mg.at_node['surface_water__discharge']



#%% run forward

N = 4000
R = np.zeros(N)
dt = 500

a0 = mg.cell_area_at_node[mg.core_nodes][0]
area = mg.at_node['drainage_area']
q_loss = mg.at_node["surface_water__discharge_loss"]
lower_bndry = np.arange(0, mg.number_of_node_columns)

dz_ad = np.zeros(mg.size("node"))
dz_ad[mg.core_nodes] = U * dt
dz_ad[top_nodes] += 0.5 * U * dt # effectively decrease the baselevel fall rate of the upper boundary

save_freq = 50
Ns = N//save_freq
z_all = np.zeros((len(z),Ns))
rock_id_all = np.zeros((len(z),Ns))

for i in range(N):

    lmb2.run_one_step()

    # update areas
    fa2.run_one_step()

    # update topography
    fs2.run_one_step(dt)
    ld2.run_one_step(dt)
    z += dz_ad

    lith.rock_id = 0
    lith.dz_advection = dz_ad
    lith.run_one_step()

    # calculate spring discharge that will be used next time
    total_loss = np.sum(q_loss)
    contact = find_contact(mg, lith)
    lower_drainage = get_divide_mask(mg, lower_bndry)
    lower_contact = np.logical_and(contact, lower_drainage)
    Q_spring = area_weighted_springs(mg, lower_contact, total_loss)
    q[:] = 1 + Q_spring/a0

    # metrics of change
    R[i] = np.mean(z[mg.core_nodes])

    if i%save_freq==0:
        print(f"Finished iteration {i}")

        z_all[:,i//save_freq] = z
        rock_id_all[:,i//save_freq] = mg.at_node['rock_type__id']