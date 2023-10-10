"""
More complex test of virtual karst in landlab. Use the Lithology component to 
track two layers: a limestone layer, and a basement. Now use the thicknesses and
permeabilities of those lithologies to parametrize a GroundwaterDupuitPercolator
model.

Experiment with some partial partitioning - may need to allow some amount of 
surface runoff in order to get incision to begin with. Part of the challenge here
is that karst will change conductivity with age, which we are not capturing yet.
"""

#%%

import os
import copy
import pickle
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.colors import LightSource

from landlab import RasterModelGrid
from landlab.components import (
    FastscapeEroder, 
    FlowAccumulator,
    FlowDirectorD8,
    LakeMapperBarnes,
    LinearDiffuser,
    LithoLayers,
    GroundwaterDupuitPercolator,
)
from landlab.grid.mappers import map_value_at_max_node_to_link
from landlab.io.netcdf import to_netcdf

fig_directory = '/Users/dlitwin/Documents/Research/Karst landscape evolution/landlab_virtual_karst/figures'
save_directory = '/Users/dlitwin/Documents/Research Data/Local/karst_lem'
id = "test_1"

#%% parameters

U = 1e-4 # uplift m/yr
K = 1e-5 # streampower incision (yr^...)
D = 1e-3 # diffusivity (m2/yr)

b_limestone = 20 # limestone unit thickness (m)
b_basement = 1000 # basement thickness (m)
bed_dip = 0.002 # dip of bed (positive = toward bottom boundary)
ksat_limestone = 1e-5 # ksat limestone (m/s)
ksat_basement = 1e-6 # ksat basement (m/s)
n_limestone = 0.1 # drainable porosity 
n_weathered_basement = 0.1 # drainable porosity 
b_weathered_basement = 0.5 # thickness of regolith that can host aquifer in basement (m)

ie_rate = 1e-8 # infiltration excess average rate (m/s)
recharge_rate = 2e-8 # recharge rate (m/s)

N = 4000 # number of geomorphic timesteps
dt = 500 # geomorphic timestep (yr)
dt_gw = 10 * 24 * 3600 # groundwater timestep (s)
save_freq = 100 # steps between saving output
output_fields = [
        "at_node:topographic__elevation",
        "at_node:aquifer_base__elevation",
        "at_node:water_table__elevation",
        "at_node:surface_water_discharge",
        "at_node:local_runoff",
        "at_node:rock_type__id"
        ]

#%% set up grid and lithology

mg = RasterModelGrid((100,150), xy_spacing=50)
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
layer_elevations = [b_limestone,b_basement]
layer_ids = [0,1]
attrs = {"Ksat_node": {0: ksat_limestone, 1: ksat_basement}, 
         "weathered_thickness": {0: 0.0, 1: b_weathered_basement},
         "porosity": {0: n_limestone, 1: n_weathered_basement}
         }

lith = LithoLayers(
    mg, layer_elevations, layer_ids, function=lambda x, y: - bed_dip * y, attrs=attrs
)

# we just start with aquifer base at the base of the limestone, because limestone
# covers the whole surface
rock_id = 1
zb = mg.add_zeros('node', 'aquifer_base__elevation')
zb[:] = z - lith.z_bottom[rock_id,:]
zwt = mg.add_zeros('node', 'water_table__elevation')
zwt[mg.core_nodes] = z[mg.core_nodes] - 0.1

# Lithology model tracks Ksat at nodes, but gdp needs ksat at links
ks = mg.add_zeros("link", "Ksat")
ks[:] = map_value_at_max_node_to_link(mg, "water_table__elevation", "Ksat_node")

# add a local runoff field - both infiltration excess and saturation excess
q_local = mg.add_zeros("node", "local_runoff")

q_ie = mg.add_zeros("node", "infiltration_excess")
q_ie[mg.core_nodes] = ie_rate

#%%


gdp = GroundwaterDupuitPercolator(
    mg, 
    hydraulic_conductivity='Ksat',
    porosity="porosity",
    recharge_rate=recharge_rate
)
fd = FlowDirectorD8(mg)
fa = FlowAccumulator(
    mg,
    surface="topographic__elevation",
    flow_director=fd,
    runoff_rate="local_runoff",
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

dz_ad = np.zeros(mg.size("node"))
dz_ad[mg.core_nodes] = U * dt
# dz_ad[top_nodes] += 0.5 * U * dt # effectively decrease the baselevel fall rate of the upper boundary

Ns = N//save_freq
R = np.zeros(N)
first_wtdelta = np.zeros(N)
wt_iterations = np.zeros(N)

for i in range(N):

    # remove depressions
    lmb.run_one_step()

    # iterate for steady state water table
    wt_delta = 1
    wt_iter = 0
    while wt_delta > 1e-4:

        zwt0 = zwt.copy()
        gdp.run_with_adaptive_time_step_solver(dt_gw)
        wt_delta = np.max(np.abs(zwt0 - zwt))/(dt_gw/3600)

        if wt_iter == 0:
            first_wtdelta[i] = wt_delta
        wt_iter += 1
    wt_iterations[i] = wt_iter

    # local runoff is sum of saturation and infiltration excess. Convert units to m/yr. 
    q_local[mg.core_nodes] = (mg.at_node['average_surface_water__specific_discharge'][mg.core_nodes] + mg.at_node['infiltration_excess'][mg.core_nodes]) * 3600 * 24 * 365

    # update areas
    fa.run_one_step()

    # update topography
    fs.run_one_step(dt)
    ld.run_one_step(dt)
    z += dz_ad

    # update lithologic model
    lith.rock_id = 0
    lith.dz_advection = dz_ad
    lith.run_one_step()

    # map new ksat to link
    ks[:] = map_value_at_max_node_to_link(mg, "water_table__elevation", "Ksat_node")

    # update lower aquifer boundary condition
    zb[mg.core_nodes] = ((z - lith.z_bottom[rock_id,:]) - mg.at_node["weathered_thickness"])[mg.core_nodes]
    ## something to handle boundary nodes (because lith thickness there is not updated)

    # something to handle the aquifer itself - see regolith models in DupuitLEM
    # this should cover it, but again check boundary conditions
    zwt[mg.core_nodes] = (zb + gdp._thickness)[mg.core_nodes]

    # metrics of change
    R[i] = np.mean(z[mg.core_nodes])

    # save output
    if i%save_freq==0:
        print(f"Finished iteration {i}")

        # save the specified grid fields
        filename = os.path.join(save_directory, f"{id}_grid_{i}.nc")
        to_netcdf(mg, filename, include=output_fields, format="NETCDF4")


# %% plot topogrpahic change

# topography
plt.figure()
mg.imshow("topographic__elevation", colorbar_label='Elevation [m]')
plt.savefig(os.path.join(fig_directory,"lith_gdp_topog.png"))

plt.figure()
mg.imshow("rock_type__id", cmap="viridis", colorbar_label='Rock ID')
plt.savefig(os.path.join(fig_directory,"lith_gdp_rocks.png"))

Q_an = mg.at_node['surface_water__discharge']/mg.at_node['drainage_area']
plt.figure()
mg.imshow('surface_water__discharge', cmap="plasma", colorbar_label='Discharge')
plt.savefig(os.path.join(fig_directory,"lith_gdp_discharge.png"))



#%%

plt.figure()
# mg.imshow('aquifer__thickness', cmap="plasma", colorbar_label='thickness')
mg.imshow('water_table__elevation', cmap="plasma", colorbar_label='WT Elevation')
#%%
# hillshade

elev = mg.at_node['topographic__elevation'].copy()
elev[mg.boundary_nodes] = np.nan
y = np.arange(mg.shape[0] + 1) * mg.dx - mg.dx * 0.5
x = np.arange(mg.shape[1] + 1) * mg.dy - mg.dy * 0.5

elev_plot = elev.reshape(mg.shape)
elev_profile = np.nanmean(elev_plot, axis=1)

f, (ax0, ax1) = plt.subplots(1, 2, width_ratios=[4, 1], figsize=(10,5))
ls = LightSource(azdeg=135, altdeg=45)
ax0.imshow(
        ls.hillshade(elev_plot, 
            vert_exag=1, 
            dx=mg.dx, 
            dy=mg.dy), 
        origin="lower", 
        extent=(x[0], x[-1], y[0], y[-1]), 
        cmap='gray',
        )
for i in range(mg.shape[1]):
    ax1.plot(elev_plot[:,i], y[0:-1], alpha=0.1, color='k')
ax1.plot(elev_profile, y[0:-1], linewidth=2, color='r')

ax0.set_xlabel('X [m]')
ax0.set_ylabel('Y [m]')
ax1.set_xlabel('Z [m]')
f.tight_layout()
plt.savefig(os.path.join(fig_directory,"lith_gdp_hillshade.png"))