"""
Make some plots for the virtual karst models
"""

#%%
import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from landlab.io.netcdf import write_raster_netcdf, from_netcdf
from virtual_karst_funcs import *

fig_directory = '/Users/dlitwin/Documents/Research/Karst landscape evolution/landlab_virtual_karst/figures'
save_directory = '/Users/dlitwin/Documents/Research Data/Local/karst_lem'
id = "flat_dynamic_ksat_4"

df_out = pd.read_csv(os.path.join(save_directory,id,f'output_{id}.csv'))
df_params = pd.read_csv(os.path.join(save_directory,id,f'params_{id}.csv')).loc[0]

#%% write raster netcdfs for paraview (for earlier cases using to_netcdf)

grid_files = glob.glob(os.path.join(save_directory,id,'*.nc'))
files = sorted(grid_files, key=lambda x:int(x.split('_')[-1][:-3]))
iteration = int(files[-1].split('_')[-1][:-3])

for i in range(len(files)):
    grid = from_netcdf(files[i])
    write_raster_netcdf(os.path.join(save_directory,id,'paraview',f'{id}_grid_{i}.nc'),
                        grid,
                        time=df_out['time'].loc[i])
    

# %% import params and output

N = df_params['N']
dt = df_params['dt']
t0 = df_params['t0']
kt = df_params['kt']
D0 = df_params['D0']
Df = df_params['Df']
n0 = df_params['n0']
nf = df_params['nf']
t = np.arange(0, N*dt, dt)
D = calc_pore_diam_logistic(t, t0, kt, D0, Df)
n = calc_porosity_logistic(t, t0, kt, D0, Df, n0, nf)
ksat = calc_ksat(n,D)


#%% ksat and n change in time

fig, ax = plt.subplots()
ax.plot(t, n, color='seagreen')
ax.set_ylim((0.0,0.012))
ax.set_ylabel('Porosity (-)', color='seagreen')
ax.set_xlabel('Time (yr)')

ax1 = ax.twinx()
ax1.plot(t, ksat, color='dodgerblue')
ax1.set_ylabel(r'$k_{sat}$ (m/s)', color='dodgerblue')
ax1.set_ylim((0.0,4e-5))
plt.savefig(os.path.join(save_directory, id, "limestone_props.png"))


#%% ksat change and limestone exposure


fig, ax = plt.subplots()
ax.plot(t, df_out['limestone_exposed'], color='darkgoldenrod')
ax.set_ylim((0.0,1.01))
ax.set_ylabel('Limestone exposed (-)', color='darkgoldenrod')
ax.set_xlabel('Time (yr)')

ax1 = ax.twinx()
ax1.plot(t, ksat, color='dodgerblue')
ax1.set_ylabel(r'$k_{sat}$ (m/s)', color='dodgerblue')
ax1.set_ylim((0.0,3e-5))
plt.savefig(os.path.join(save_directory, id, "limestone_ksat.png"))


# %% ksat and mean aquifer thickness

fig, ax = plt.subplots()
ax.plot(t, df_out['median_aquifer_thickness'], color='b')
# ax.set_ylim((0.01,1.01))
ax.set_ylabel('Median Aquifer Thickness (m)', color='b')
ax.set_xlabel('Time (yr)')

ax1 = ax.twinx()
ax1.plot(t, ksat, color='dodgerblue')
ax1.set_ylabel(r'$k_{sat}$ (m/s)', color='dodgerblue')
ax1.set_ylim((0.0,3e-5))
plt.savefig(os.path.join(save_directory, id, "med_aquifer_thickness.png"))


# %%

fig, ax = plt.subplots()
ax.plot(t, df_out['mean_r']/df_params['r_tot'], color='b')
ax.set_ylim((0.0,1.01))
ax.set_ylabel('Fraction Recharge (-)', color='b')
ax.set_xlabel('Time (yr)')

ax1 = ax.twinx()
ax1.plot(t, ksat, color='dodgerblue')
ax1.set_ylabel(r'$k_{sat}$ (m/s)', color='dodgerblue')
ax1.set_ylim((0.0,3e-5))
plt.savefig(os.path.join(save_directory, id, "recharge.png"))
# %%
