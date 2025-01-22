
"""
Use the Lithology component to track two layers: a limestone layer, and a basement.
Water is partitioned between surface runoff (different infiltration excess fractions 
on limestone and basement) and recharge. Recharge is partitioned between matrix and conduit
pathways. Conduit recharge routed on the lithologic contact surface, matrix recharge
goes to the unconfined aquifer. Discharge is the sum of the infiltration excess (Horton), 
saturation excess (Dunne), routed on topography, and the conduit discharge routed on the
contact surface or topography where contact is eroded away. Conduit recharge happens only
on the limestone.
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
    GroundwaterDupuitPercolator,
)
from landlab.grid.mappers import map_value_at_max_node_to_link

from virtual_karst_funcs import *

save_directory = '/Users/dlitwin/Documents/Research Data/Local/karst_lem'
filename = "virtual_karst_conduit_gw_1"
os.mkdir(os.path.join(save_directory,filename))

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
conduit_frac = 0.5 # initial fraction of recharge going to conduits
n_weathered_basement = 0.005 # drainable porosity 
b_weathered_basement = 0.5 # thickness of regolith that can host aquifer in basement (m)

r_tot = 1 / (3600 * 24 * 365) # total runoff m/s
ibar = 1e-3 / 3600 # mean storm intensity (m/s) equiv. 1 mm/hr 
# ie_frac = 0.0 # fraction of r_tot that becomes overland flow on limestone. ie_frac on basement=1. 

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
         "conduit_frac": {0: conduit_frac, 1: 0.0},
         "chemical_denudation_rate": {0: E_limestone, 1: E_weathered_basement},
         "erodibility":{0: K_sp_limestone, 1: K_sp_basement}
         }

lith = LithoLayers(
    mg, layer_elevations, layer_ids, function=lambda x, y: - bed_dip * y + noise_scale*np.random.rand(len(y)), 
    layer_type='MaterialLayers', attrs=attrs,
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
r_c = mg.add_zeros("node", "recharge_conduit")
r_m = mg.add_zeros("node", "recharge_matrix")

# add a local runoff field - both infiltration excess and saturation excess
q_local = mg.add_zeros("node", "local_runoff")

# add a field for exfiltration -- derived from local gdp runoff
qe_local = mg.add_zeros("node", "exfiltration_rate")

# instantiate discharge fields
Q1 = mg.add_zeros("node", "horton_dunne_discharge")
Q2 = mg.add_zeros("node", "karst_discharge")
Q_tot = mg.add_zeros("node", "total_discharge")
rock_ID = mg.at_node['rock_type__id']


#%% instantiate components 

# groundwater model
gdp = GroundwaterDupuitPercolator(
    mg, 
    hydraulic_conductivity='Ksat',
    porosity="porosity",
    recharge_rate="recharge_matrix",
    courant_coefficient=0.9,
    vn_coefficient=0.9,
)
h = mg.at_node['aquifer__thickness']

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

# erosion components
fs2 = FastscapeEroder(mg,  K_sp='erodibility', m_sp=m_sp, n_sp=n_sp, discharge_field='total_discharge')
ld2 = LinearDiffuser(mg, linear_diffusivity=D_ld)

#%% xarray to save output

output_fields = [
    "topographic__elevation",
    "karst__elevation",
    "aquifer_base__elevation",
    "water_table__elevation",
    "rock_type__id",
    "total_discharge",
    "horton_dunne_discharge",
    "karst_discharge",
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
            'mean_r_conduit',
            'mean_r_matrix',
            ]
df_out = pd.DataFrame(np.zeros((N,len(save_vals))), columns=save_vals)
params = {'U':U, 'K_limestone':K_sp_limestone, 'K_basement':K_sp_basement, 'D_ld':D_ld, 'm_sp':m_sp,
            'E_limestone': E_limestone, 'E_weathered_basement':E_weathered_basement,
            'n_sp':n_sp, 'conduit_frac':conduit_frac,
            'b_limestone':b_limestone, 'b_basement':b_basement, 'bed_dip':bed_dip,
            'ksat_limestone':ksat_limestone, 'ksat_basement':ksat_basement, 
            'n_limestone':n_limestone,'n_weathered_basement':n_weathered_basement,
            'b_weathered_basement':b_weathered_basement, 
            'r_tot':r_tot, 'ibar':ibar, 
            'wt_delta_tol':wt_delta_tol, 'T':T, 'N':N, 'dt':dt, 'dt_gw':dt_gw, 'save_freq':save_freq}
df_params = pd.DataFrame(params, index=[0])
df_params.to_csv(os.path.join(save_directory, filename, f"params_{filename}.csv"))


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
        "total_discharge": (
            ("time", "y", "x"),
            np.empty((Ns, mg.shape[0], mg.shape[1])),
            {"units": "m3/s", "long_name": "Discharge on topographic surface"},
        ),
        "horton_dunne_discharge": (
            ("time", "y", "x"),
            np.empty((Ns, mg.shape[0], mg.shape[1])),
            {"units": "m3/yr", "long_name": "Discharge from combined Hortonian and Dunnian overland flow"},
        ),
        "karst_discharge": (
            ("time", "y", "x"),
            np.empty((Ns, mg.shape[0], mg.shape[1])),
            {"units": "m3/s", "long_name": "Discharge from discrete karst system"},
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
            dt * save_freq * np.arange(Ns) / 1e3,
            {"units": "thousands of years since model start", "standard_name": "time"},
        ),
    },
)
ds = ds.assign_attrs(params)

#%% run forward


for i in tqdm(range(N)):
    z0 = z.copy()

    # map new ksat to link
    ks[:] = map_value_at_max_node_to_link(mg, "water_table__elevation", "Ksat_node")

    # partition infiltration excess, conduit recharge, matrix recharge 
    ie_frac = np.exp(-mg.at_node["Ksat_node"]/ibar) 
    q_ie[mg.core_nodes] = r_tot * ie_frac[mg.core_nodes]
    r_c[mg.core_nodes] = r_tot * (1 - ie_frac[mg.core_nodes]) * mg.at_node["conduit_frac"][mg.core_nodes]
    r_m[mg.core_nodes] = r_tot * (1 - ie_frac[mg.core_nodes]) * (1 - mg.at_node["conduit_frac"][mg.core_nodes])

    if (r_m > 0.0).any():
        # iterate for steady state water table
        wt_delta = 1
        wt_iter = 0
        while wt_delta > wt_delta_tol:
            zwt0 = zwt.copy()
            gdp.run_with_adaptive_time_step_solver(dt_gw)
            wt_delta = np.max(np.abs(zwt0 - zwt))/dt_gw

            # if wt_iter == 0:
            #     df_out['first_wtdelta'].loc[i] = wt_delta
            wt_iter += 1
        df_out.loc[i, 'wt_iterations'] = wt_iter
    # else:
    #     h[:] = np.zeros_like(h)
    #     zwt[:] = zb

    # local runoff is sum of saturation and infiltration excess.
    # use surface_water__specific_discharge rather than average because we are going for steady-state (end of timestep)
    q_local[mg.core_nodes] = (mg.at_node['surface_water__specific_discharge'][mg.core_nodes] + 
                              mg.at_node['infiltration_excess'][mg.core_nodes])
    qe_local[mg.core_nodes] = np.maximum(0, mg.at_node['average_surface_water__specific_discharge'][mg.core_nodes]-r_m[mg.core_nodes])

    if (q_local > 0.0).any():
        # first set discharge field to Horton+Dunne runoff rate. Convert units to m/yr. 
        # get flow directions and discharge on the topographic surface
        mg.at_node['water__unit_flux_in'][:] = q_local * 3600 * 24 * 365
        lmb1.run_one_step()
        fa1.run_one_step()
        Q1[:] = mg.at_node['surface_water__discharge']
    else:
        Q1[:] = np.zeros_like(Q1)

    if (r_c > 0.0).any():
        # first set discharge equal to conduit recharge rate. Convert units to m/yr. 
        # get flow directions and discharge on the basement surface. Some of this is in the karst.
        mg.at_node['water__unit_flux_in'][:] = r_c * 3600 * 24 * 365
        lmb2.run_one_step()
        fa2.run_one_step()
        Q2[:] = mg.at_node['surface_water__discharge']
    else:
        Q2[:] = np.zeros_like(Q2)

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
    lith.rock_id = mg.at_node['rock_type__id'] # deposited material is the same as what was there before.
    lith.dz_advection = dz_ad
    lith.run_one_step()

    # update the definition of the karst surface
    zk[:] = z-lith.z_top[0,:]

    # update lower aquifer boundary condition
    zb[mg.core_nodes] = (zk - mg.at_node["weathered_thickness"])[mg.core_nodes]

    # something to handle the aquifer itself - see regolith models in DupuitLEM
    # this should cover it, but again check boundary conditions
    h[h>z-zb] = (z-zb)[h>z-zb]
    zwt[:] = zb+h

    ######## Save output

    # save change metrics
    df_out.loc[i,'limestone_exposed__area'] = np.sum(mg.at_node['rock_type__id'][mg.core_nodes]==0)/len(mg.core_nodes)
    df_out.loc[i,'mean_limestone__thickness'] = np.mean(lith.z_bottom[1,:][mg.at_node['rock_type__id']==0])
    df_out.loc[i,'mean_limestone__elevation'] = np.mean(z[mg.at_node['rock_type__id']==0])
    df_out.loc[i,'mean__elevation'] = np.mean(z[mg.core_nodes])
    df_out.loc[i,'max__elevation'] = np.max(z[mg.core_nodes])
    df_out.loc[i,'denudation__rate'] = -np.mean((z - z0 - U*dt)[mg.core_nodes])
    df_out.loc[i,'median_aquifer_thickness'] = np.median(h[mg.core_nodes])
    # at, ab = get_lower_upper_area(mg, bottom_nodes, top_nodes)
    # df_out.loc[i,'area_lower'] = ab
    # df_out.loc[i,'area_upper'] = at
    # Qt, Qb = get_lower_upper_water_flux(mg, bottom_nodes, top_nodes)
    # df_out.loc[i,'discharge_lower'] = Qb
    # df_out.loc[i,'discharge_upper'] = Qt
    df_out.loc[i,'mean_ie'] = np.mean(q_ie[mg.core_nodes])
    df_out.loc[i,'mean_r_conduit'] = np.mean(r_c[mg.core_nodes])
    df_out.loc[i,'mean_r_matrix'] = np.mean(r_m[mg.core_nodes])

    if i%save_freq==0:
        for of in output_fields:
            ds[of][i//save_freq, :, :] = mg["node"][of].reshape(mg.shape)


ds.to_netcdf(os.path.join(save_directory, filename, f"{filename}.nc"))

df_out['time'] = np.arange(0, N*dt, dt)
df_out.set_index('time', inplace=True)
df_out.to_csv(os.path.join(save_directory, filename, f"output_{filename}.csv"))


# %% plot topographic change

# topography
plt.figure()
mg.imshow("topographic__elevation", colorbar_label='Elevation [m]')
plt.savefig(os.path.join(save_directory, filename, f"{filename}_elevation.png"))

plt.figure()
mg.imshow("rock_type__id", cmap="viridis", colorbar_label='Rock ID')
plt.savefig(os.path.join(save_directory, filename, f"{filename}_rockid.png"))

plt.figure()
mg.imshow('total_discharge', cmap="plasma", colorbar_label='Discharge')
plt.savefig(os.path.join(save_directory, filename, f"{filename}_totalQ.png"))


# %%
