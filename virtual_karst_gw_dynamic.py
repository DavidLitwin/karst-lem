"""
More complex test of virtual karst in landlab. Use the Lithology component to 
track three layers: a limestone layer, weathered basement, and unweathered basement. 
Thicknesses and permeabilities of those lithologies to parametrize a GroundwaterDupuitPercolator
model. Limestone porosity and permeability change with age, which changes partitioning between recharge
and infiltration excess, and flow in the subsurface.

"""

#%%

from tqdm import tqdm
import os
import numpy as np
import pandas as pd

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
from landlab.io.netcdf import write_raster_netcdf
from virtual_karst_funcs import *

save_directory = '/Users/dlitwin/Documents/Research Data/Local/karst_lem'
print('Enter base filename:')
id = input()
# id = "flat_dynamic_ksat_9"

#%% parameters

U = 1e-4 # uplift (m/yr)
E_limestone = 5e-5 # limestone surface chemical denudation rate (m/yr)
E_weathered_basement = 0.0 # weathered basement surface chemical denudation rate (m/yr)
K_sp = 1e-5 # streampower incision (yr^...)
m_sp = 0.5 # exponent on discharge
n_sp = 1.0 # exponent on slope
D_ld = 1e-3 # diffusivity (m2/yr)

D0 = 2e-5 # initial eq diameter (m)
Df = 1e-4 # final eq diameter (m)
n0 = 0.002 # initial porosity (-)
nf = 0.01 # final porosity (-)
t0 = 1e5 #1.25e6 # midpoint of logistic (yr)
kt = 1/1e4 # sharpness of logistic (1/yr)

b_limestone = 50 # limestone unit thickness (m)
b_basement = 1000 # basement thickness (m)
bed_dip = 0.000 #0.002 # dip of bed (positive = toward bottom boundary)
ksat_limestone = calc_ksat(n0, D0) # ksat limestone (m/s)
ksat_basement = 1e-6 # ksat basement (m/s)
n_limestone = n0 # drainable porosity 
n_weathered_basement = 0.1 # drainable porosity 
b_weathered_basement = 0.5 # thickness of regolith that can host aquifer in basement (m)

r_tot = 1 / (3600 * 24 * 365) # total runoff m/s
ibar = 1e-3 / 3600 # mean storm intensity (m/s) equiv. 1 mm/hr 

wt_delta_tol = 1e-6 # acceptable rate of water table change before moving on (m/s)

T = 5e6 # total geomorphic time
dt = 500 # geomorphic timestep (yr)
N = int(T//dt) # number of geomorphic timesteps

dt_gw = 1 * 24 * 3600 # groundwater timestep (s)
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
save_vals = ['limestone_exposed',
               'mean_relief',
               'max_relief',
               'median_aquifer_thickness',
               'discharge_lower',
               'discharge_upper',
               'area_lower',
               'area_upper',
               'first_wtdelta',
               'wt_iterations',
               'ksat_limestone',
               'n_limestone',
               'mean_ie',
               'mean_r',
               ]
df_out = pd.DataFrame(np.zeros((N,len(save_vals))), columns=save_vals)
df_params = pd.DataFrame({'U':U, 'K':K_sp, 'D_ld':D_ld, 'm_sp':m_sp,
                          'E_limestone': E_limestone, 'E_weathered_basement':E_weathered_basement,
                          'n_sp':n_sp, 'D0':D0, 'Df':Df, 'n0':n0, 'nf':nf, 't0':t0, 'kt':kt, 
                          'b_limestone':b_limestone, 'b_basement':b_basement, 'bed_dip':bed_dip,
                          'ksat_limestone':ksat_limestone, 'ksat_basement':ksat_basement, 
                          'n_limestone':n_limestone,'n_weathered_basement':n_weathered_basement,
                          'b_weathered_basement':b_weathered_basement, 
                          'r_tot':r_tot, 'ibar':ibar, 
                          'wt_delta_tol':wt_delta_tol, 'T':T, 'N':N, 'dt':dt, 'dt_gw':dt_gw, 'save_freq':save_freq}, index=[0])
df_params.to_csv(os.path.join(save_directory, id, f"params_{id}.csv"))

#%% set up grid and lithology

mg = RasterModelGrid((100,150), xy_spacing=50)
mg.set_closed_boundaries_at_grid_edges(right_is_closed=True,
                                       left_is_closed=True,
                                       top_is_closed=False,
                                       bottom_is_closed=False)
bnds_lower = mg.nodes_at_bottom_edge
bnds_upper = mg.nodes_at_top_edge
z = mg.add_zeros("node", "topographic__elevation")
np.random.seed(10010)
z += 0.1*np.random.rand(len(z))
atot = np.sum(mg.cell_area_at_node[mg.core_nodes])

# two layers, both with bottoms below ground. Top layer is limestone, bottom is basement.
# weathered_thickness is a way to add some somewhat realistic weathered zone in the
# basement that will host the aquifer there. In the limestone, the aquifer is just the whole unit.
layer_elevations = [b_limestone,b_basement]
layer_ids = [0,1]
attrs = {"Ksat_node": {0: ksat_limestone, 1: ksat_basement}, 
         "weathered_thickness": {0: b_weathered_basement, 1: b_weathered_basement}, # was 0.0 for limestone but this can lead to discontinuities.
         "porosity": {0: n_limestone, 1: n_weathered_basement},
         "chemical_denudation_rate": {0: E_limestone, 1: E_weathered_basement}
         }

lith = LithoLayers(
    mg, layer_elevations, layer_ids, function=lambda x, y: - bed_dip * y, attrs=attrs
)
dz_ad = np.zeros(mg.size("node"))
dz_ad[mg.core_nodes] = U * dt
# dz_ad[top_nodes] += 0.5 * U * dt # effectively decrease the baselevel fall rate of the upper boundary

# we just start with aquifer base at the base of the limestone, plus the small weathered thickness underlying limestone (used for continuity)
rock_id = 1
zb = mg.add_zeros('node', 'aquifer_base__elevation')
zb[:] = (z - lith.z_bottom[rock_id,:]) - mg.at_node["weathered_thickness"]
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
ie_frac = np.exp(-mg.at_node["Ksat_node"]/ibar) 
ie_rate = r_tot * ie_frac # infiltration excess average rate (m/s)
r_rate = r_tot * (1 - ie_frac) # recharge rate (m/s)
q_ie = mg.add_zeros("node", "infiltration_excess")
q_ie[mg.core_nodes] = ie_rate[mg.core_nodes]
r = mg.add_zeros("node", "recharge_rate")
r[mg.core_nodes] = r_rate[mg.core_nodes]

# add a local runoff field - both infiltration excess and saturation excess
q_local = mg.add_zeros("node", "local_runoff")

# add a field for exfiltration -- derived from local gdp runoff
qe_local = mg.add_zeros("node", "exfiltration_rate")

#%% initialize components

gdp = GroundwaterDupuitPercolator(
    mg, 
    hydraulic_conductivity='Ksat',
    porosity="porosity",
    recharge_rate="recharge_rate",
    courant_coefficient=0.9,
    vn_coefficient=0.9,
)
h = mg.at_node['aquifer__thickness']

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

fs = FastscapeEroder(mg, K_sp=K_sp, discharge_field='surface_water__discharge', m_sp=m_sp, n_sp=n_sp)
ld = LinearDiffuser(mg, linear_diffusivity=D_ld)

lmb.run_one_step()


#%% run forward

for i in tqdm(range(N)):

    # iterate for steady state water table
    wt_delta = 1
    wt_iter = 0
    while wt_delta > wt_delta_tol:

        # if wt_iter == 0:
        #     print('first round:', np.sum(zwt>z))
        # else:
        #     print(np.sum(zwt>z))

        # h[zwt>z] = (z-zb)[zwt>z]
        # zwt[zwt>z] = z[zwt>z]

        zwt0 = zwt.copy()
        gdp.run_with_adaptive_time_step_solver(dt_gw)
        wt_delta = np.max(np.abs(zwt0 - zwt))/dt_gw

        if wt_iter == 0:
            df_out['first_wtdelta'].loc[i] = wt_delta
        wt_iter += 1
    df_out['wt_iterations'].loc[i] = wt_iter

    # local runoff is sum of saturation and infiltration excess. Convert units to m/yr. 
    q_local[mg.core_nodes] = (mg.at_node['average_surface_water__specific_discharge'][mg.core_nodes] + 
                              mg.at_node['infiltration_excess'][mg.core_nodes]) * 3600 * 24 * 365
    qe_local[mg.core_nodes] = np.maximum(0, mg.at_node['average_surface_water__specific_discharge'][mg.core_nodes]-r[mg.core_nodes])


    # update areas
    fa.run_one_step()

    # update topography
    fs.run_one_step(dt)
    ld.run_one_step(dt)
    z[mg.core_nodes] -= mg.at_node["chemical_denudation_rate"][mg.core_nodes] * dt
    z += dz_ad

    # update lithologic model
    lith.rock_id = 0
    lith.dz_advection = dz_ad
    lith.run_one_step()

    # remove depressions
    # if np.sum(mg.at_node['drainage_area'][mg.open_boundary_nodes]) < 0.9 * atot:
    #     lmb.run_one_step()

    # update limestone rock properties
    D = calc_pore_diam_logistic(i*dt, t0, kt, D0, Df)
    n = calc_porosity_logistic(i*dt, t0, kt, D0, Df, n0, nf)
    ksat = calc_ksat(n,D)
    lith.update_rock_properties("porosity", 0, n)
    lith.update_rock_properties("Ksat_node", 0, ksat)

    # map new ksat to link
    ks[:] = map_value_at_max_node_to_link(mg, "water_table__elevation", "Ksat_node")

    # update infiltration excess and recharge
    ie_frac = np.exp(-mg.at_node["Ksat_node"]/ibar) 
    q_ie[mg.core_nodes] = r_tot * ie_frac[mg.core_nodes]
    r[mg.core_nodes] = r_tot * (1 - ie_frac[mg.core_nodes])

    # update lower aquifer boundary condition
    zb[mg.core_nodes] = ((z - lith.z_bottom[rock_id,:]) - mg.at_node["weathered_thickness"])[mg.core_nodes]
    ## something to handle boundary nodes (because lith thickness there is not updated)

    # something to handle the aquifer itself - see regolith models in DupuitLEM
    # this should cover it, but again check boundary conditions
    h[h>z-zb] = (z-zb)[h>z-zb]
    zwt[:] = zb+h

    ###################### save output #################

    # save change metrics
    df_out['limestone_exposed'].loc[i] = np.sum(mg.at_node['rock_type__id'][mg.core_nodes]==0)/len(mg.core_nodes)
    df_out['mean_relief'].loc[i] = np.mean(z[mg.core_nodes])
    df_out['max_relief'].loc[i] = np.max(z[mg.core_nodes])
    df_out['median_aquifer_thickness'].loc[i] = np.median(h[mg.core_nodes])
    at, ab = get_lower_upper_area(mg, bnds_lower, bnds_upper)
    df_out['area_lower'].loc[i] = ab
    df_out['area_upper'].loc[i] = at
    Qt, Qb = get_lower_upper_water_flux(mg, bnds_lower, bnds_upper)
    df_out['discharge_lower'].loc[i] = Qb
    df_out['discharge_upper'].loc[i] = Qt
    df_out['ksat_limestone'].loc[i] = ksat
    df_out['n_limestone'].loc[i] = n
    df_out['mean_ie'].loc[i] = np.mean(q_ie[mg.core_nodes])
    df_out['mean_r'].loc[i] = np.mean(r[mg.core_nodes])

    # save grid
    if i%save_freq==0:
        # print(f"Finished iteration {i}")

        # save the specified grid fields
        # filename = os.path.join(save_directory, id, f"grid_{id}_{i}.nc")
        filename1 = os.path.join(save_directory, id, f"grid_{id}.nc")
        # write_raster_netcdf(filename, mg, names=output_fields, time=i*dt)
        write_raster_netcdf(filename1, mg, names=output_fields, time=i*dt, append=True)

#%% save out

df_out['time'] = np.arange(0,N*dt,dt)
df_out.set_index('time', inplace=True)
df_out.to_csv(os.path.join(save_directory, id, f"output_{id}.csv"))

# %% plot topographic change

# topography
plt.figure()
mg.imshow("topographic__elevation", colorbar_label='Elevation [m]')
plt.savefig(os.path.join(save_directory, id, "elevation.png"))

plt.figure()
mg.imshow("rock_type__id", cmap="viridis", colorbar_label='Rock ID')
plt.savefig(os.path.join(save_directory, id, "rock_id.png"))

Q_an = mg.at_node['surface_water__discharge']/mg.at_node['drainage_area']
plt.figure()
mg.imshow('surface_water__discharge', cmap="Blues", colorbar_label='Discharge')
plt.savefig(os.path.join(save_directory, id, "lith_gdp_discharge.png"))


# %%

fig, ax = plt.subplots()
ax.plot(df_out['discharge_upper'], 'r', label='Qupper')
ax.plot(df_out['discharge_lower'], 'r--', label='Qlower')
ax.set_ylabel("Discharge (m3/yr)", color='r')
ax1 = ax.twinx()
ax1.plot(df_out['area_upper'], 'b', label='Aupper')
ax1.plot(df_out['area_lower'], 'b--', label='Alower')
ax1.set_ylabel('Area (m2)', color='b')
ax.legend(loc='upper right')
ax1.legend(loc='lower right')
plt.savefig(os.path.join(save_directory, id, "area_discharge.png"))

# %%

plt.figure()
plt.plot(df_out['mean_relief'], color='r', label='Mean relief')
plt.plot(df_out['max_relief'], color='g', label='Max relief')
plt.xlabel('Time (yr)')
plt.ylabel('Relief (m)')
plt.legend()
plt.savefig(os.path.join(save_directory, id, "relief.png"))

#%%

plt.figure()
plt.plot(df_out['limestone_exposed'])
plt.ylim((0.01,1.1))
plt.ylabel('Limestone exposed (-)')
plt.xlabel('Time (yr)')
plt.savefig(os.path.join(save_directory, id, "exposed_limestone.png"))
