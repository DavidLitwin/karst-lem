"""
Use the Lithology component to track two layers: a limestone layer, and a basement. 
Precipitation partitioned between infiltration excess and recharge. Fraction
depends on whether on limestone or basement. Recharge to GroundwaterDupuitPercolator, 
with different hydraulic properties.

"""

#%%

import os
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt

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
# from virtual_karst_funcs import *

save_directory = '/Users/dlitwin/Documents/Research Data/Local/karst_lem'
filename = "virtual_karst_gw_1"
# os.mkdir(os.path.join(save_directory,filename))

#%% parameters

U = 1e-4 # uplift (m/yr)
E_limestone = 0.0 #5e-5 # limestone surface chemical denudation rate (m/yr)
E_weathered_basement = 0.0 # weathered basement surface chemical denudation rate (m/yr)
K_sp_limestone = 1e-5 # streampower incision (yr^...)
K_sp_basement = 1e-5 # streampower incision (yr^...)
m_sp = 0.5 # exponent on discharge
n_sp = 1.0 # exponent on slope
D_ld = 1e-3 # diffusivity (m2/yr)

b_limestone = 25 # limestone unit thickness (m)
b_basement = 1000 # basement thickness (m)
bed_dip = 0.0 #0.002 # dip of bed (positive = toward bottom boundary)

ksat_limestone = 1.5e-5 # ksat limestone (m/s) ksat=1.5e-5 similar to sauter (1992) Fissured limestone
ksat_basement = 1e-6 # ksat basement (m/s) # 1e-7: fine grained sedimentary (Gleeson)
n_limestone = 0.002 # drainable porosity 
n_weathered_basement = 0.005 # drainable porosity 
b_weathered_basement = 0.5 # thickness of regolith that can host aquifer in basement (m)

r_tot = 1 / (3600 * 24 * 365) # total runoff m/s
# ibar = 1e-3 / 3600 # mean storm intensity (m/s) equiv. 1 mm/hr 
ie_frac = 0.1 # fraction of r_tot that becomes overland flow on limestone. ie_frac on basement=1. 

wt_delta_tol = 1e-6 # acceptable rate of water table change before moving on (m/s)
dt_gw = 1 * 24 * 3600 # groundwater timestep (s)

T = 1e6 # total geomorphic time (yr)
dt = 500 # geomorphic timestep (yr)
N = int(T//dt) # number of geomorphic timesteps

Nx = 150
Ny = 100
dx = 50
noise_scale = 0.1

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
         "Ksat_node": {0: ksat_limestone, 1: ksat_basement}, 
         "weathered_thickness": {0: b_weathered_basement, 1: b_weathered_basement}, # was 0.0 for limestone but this can lead to discontinuities.
         "porosity": {0: n_limestone, 1: n_weathered_basement},
         "chemical_denudation_rate": {0: E_limestone, 1: E_weathered_basement},
         "erodibility":{0: K_sp_limestone, 1: K_sp_basement}
         }

lith = LithoLayers(
    mg, layer_elevations, layer_ids, function=lambda x, y: - bed_dip * y + noise_scale*np.random.rand(len(y)), 
    attrs=attrs, layer_type='MaterialLayers',
)
dz_ad = np.zeros(mg.size("node"))
dz_ad[mg.core_nodes] = U * dt
# dz_ad[top_nodes] += 0.5 * U * dt # effectively decrease the baselevel fall rate of the upper boundary

# the karst surface is the topographic elevation minus the depth to the top of the basement
# as this evolves it continues to the top of the basement
zk = mg.add_zeros("node", "karst__elevation")
zk[:] = z-lith.z_top[0,:]

# start with aquifer base at the base of the limestone, plus the small weathered thickness underlying 
# limestone (used for continuity)
zb = mg.add_zeros('node', 'aquifer_base__elevation')
zb[:] = (z - lith.z_bottom[1,:]) - mg.at_node["weathered_thickness"]
zwt = mg.add_zeros('node', 'water_table__elevation')
zwt[mg.core_nodes] = z[mg.core_nodes] - 0.2

# Lithology model tracks Ksat at nodes, but gdp needs ksat at links
ks = mg.add_zeros("link", "Ksat")
ks[:] = map_value_at_max_node_to_link(mg, "water_table__elevation", "Ksat_node")

# infiltration excess and recharge fields
# model rainfall rates as exponentially distributed with mean ibar
# then the fraction of rainfall rates that falls at higher intensity
# than the hydraulic conductivity (if isotropic, steady state infiltration, etc.)
# becomes infiltration excess.
q_ie = mg.add_zeros("node", "infiltration_excess")
r_m = mg.add_zeros("node", "recharge_matrix")

# add a local runoff field - both infiltration excess and saturation excess
q_local = mg.add_zeros("node", "local_runoff")

# add a field for exfiltration -- derived from local gdp runoff
qe_local = mg.add_zeros("node", "exfiltration_rate")

h = mg.at_node['aquifer__thickness']

#%%


gdp = GroundwaterDupuitPercolator(
    mg, 
    hydraulic_conductivity='Ksat',
    porosity="porosity",
    recharge_rate="recharge_matrix"
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

fs = FastscapeEroder(mg, K_sp='erodibility', discharge_field='surface_water__discharge')
ld = LinearDiffuser(mg, linear_diffusivity=D_ld)

lmb.run_one_step()

#%% xarray to save output

output_fields = [
    "topographic__elevation",
    "karst__elevation",
    "aquifer_base__elevation",
    "water_table__elevation",
    "rock_type__id",
    "average_surface_water__specific_discharge",
    "surface_water__discharge",
]
save_vals = ['limestone_exposed__area',
            'mean_limestone__thickness',
            'mean_limestone__elevation',
            'max_limestone__elevation',
            'mean__elevation',
            'max__elevation',
            'denudation__rate',
            'median_aquifer_thickness',
            'discharge_lower',
            'discharge_upper',
            'area_lower',
            'area_upper',
            'limestone_upper',
            'limestone_lower',
            'wt_iterations',
            'mean_ie',
            'mean_r_matrix',
            ]
df_out = pd.DataFrame(np.zeros((N,len(save_vals))), columns=save_vals)
params = {'U':U, 'K_limestone':K_sp_limestone, 'K_basement':K_sp_basement, 'D_ld':D_ld, 'm_sp':m_sp,
            'E_limestone': E_limestone, 'E_weathered_basement':E_weathered_basement,
            'n_sp':n_sp,
            'b_limestone':b_limestone, 'b_basement':b_basement, 'bed_dip':bed_dip,
            'ksat_limestone':ksat_limestone, 'ksat_basement':ksat_basement, 
            'n_limestone':n_limestone,'n_weathered_basement':n_weathered_basement,
            'b_weathered_basement':b_weathered_basement, 
            'r_tot':r_tot, 'ie_frac':ie_frac, #'ibar':ibar, 
            'wt_delta_tol':wt_delta_tol, 'T':T, 'N':N, 'dt':dt, 'dt_gw':dt_gw, 'save_freq':save_freq}

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
        "aquifer_base__elevation": (
            ("time", "y", "x"),
            np.empty((Ns, mg.shape[0], mg.shape[1])),
            {"units": "meters", "long_name": "Base of aquifer for Dupuit model"},
        ),
        "water_table__elevation": (
            ("time", "y", "x"),
            np.empty((Ns, mg.shape[0], mg.shape[1])),
            {"units": "meters", "long_name": "Elevation of water table"},
        ),
        "rock_type__id": (
            ("time", "y", "x"),
            np.empty((Ns, mg.shape[0], mg.shape[1])),
            {"units": "-", "long_name": "Rock Type ID Code"},
        ),
        "average_surface_water__specific_discharge": (
            ("time", "y", "x"),
            np.empty((Ns, mg.shape[0], mg.shape[1])),
            {"units": "m/s", "long_name": "Local saturation excess overland flow"},
        ),
        "surface_water__discharge": (
            ("time", "y", "x"),
            np.empty((Ns, mg.shape[0], mg.shape[1])),
            {"units": "m3/s", "long_name": "Discharge on topographic surface"},
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

df_out = pd.DataFrame(np.zeros((N,len(save_vals))), columns=save_vals)
bnds = mg.fixed_value_boundary_nodes
bnds_lower = bnds[0:mg.shape[1]]
bnds_upper = bnds[mg.shape[1]:]

for i in range(N):
    
    # iterate for steady state water table
    wt_delta = 1
    wt_iter = 0
    while wt_delta > wt_delta_tol:

        zwt0 = zwt.copy()
        gdp.run_with_adaptive_time_step_solver(dt_gw)
        wt_delta = np.max(np.abs(zwt0 - zwt))/dt_gw

        if wt_iter == 0:
            df_out['first_wtdelta'].loc[i] = wt_delta
        wt_iter += 1
    df_out['wt_iterations'].loc[i] = wt_iter

    # local runoff is sum of saturation and infiltration excess. Convert units to m/yr. 
    q_local[mg.core_nodes] = (mg.at_node['average_surface_water__specific_discharge'][mg.core_nodes] + mg.at_node['infiltration_excess'][mg.core_nodes]) * 3600 * 24 * 365

    # update areas
    fa.run_one_step()

    # update topography
    fs.run_one_step(dt)
    ld.run_one_step(dt)
    z += dz_ad

    # update lithologic model
    lith.rock_id = mg.at_node['rock_type__id'] # deposited material is the same as what was there before.
    lith.dz_advection = dz_ad
    lith.run_one_step()

    # remove depressions
    lmb.run_one_step()

    # map new ksat to link
    ks[:] = map_value_at_max_node_to_link(mg, "water_table__elevation", "Ksat_node")

    # update lower aquifer boundary condition
    zb[mg.core_nodes] = ((z - lith.z_bottom[1,:]) - mg.at_node["weathered_thickness"])[mg.core_nodes]
    ## something to handle boundary nodes (because lith thickness there is not updated)

    # something to handle the aquifer itself - see regolith models in DupuitLEM
    # this should cover it, but again check boundary conditions
    h[mg.core_nodes] = (zwt - zb)[mg.core_nodes]

    # save output
    if i%save_freq==0:
        for of in output_fields:
            ds[of][i//save_freq, :, :] = mg["node"][of].reshape(mg.shape)

ds.to_netcdf(os.path.join(save_directory, filename, f"{filename}.nc"))

# %% plot topogrpahic change

# topography
plt.figure()
mg.imshow("topographic__elevation", colorbar_label='Elevation [m]')
plt.savefig(os.path.join(save_directory, filename, f"{filename}_elevation.png"))

plt.figure()
mg.imshow("rock_type__id", cmap="viridis", colorbar_label='Rock ID')
plt.savefig(os.path.join(save_directory, filename, f"{filename}_rockid.png"))

plt.figure()
mg.imshow('surface_water__discharge', cmap="plasma", colorbar_label='Discharge')
plt.savefig(os.path.join(save_directory, filename, f"{filename}_totalQ.png"))
