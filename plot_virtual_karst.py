"""
Make some plots for the virtual karst models
"""

#%%
import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from landlab.io.netcdf import write_raster_netcdf, from_netcdf, read_netcdf
from virtual_karst_funcs import *

fig_directory = '/Users/dlitwin/Documents/Research/Karst landscape evolution/landlab_virtual_karst/figures'
save_directory = '/Users/dlitwin/Documents/Research Data/Local/karst_lem'
id = "flat_dynamic_ksat_2"

df_out = pd.read_csv(os.path.join(save_directory,id,f'output_{id}.csv'))
df_params = pd.read_csv(os.path.join(save_directory,id,f'params_{id}.csv')).loc[0]

#%% write raster netcdfs for paraview (for earlier cases using to_netcdf)

grid_files = glob.glob(os.path.join(save_directory,id,'*.nc'))
files = sorted(grid_files, key=lambda x:int(x.split('_')[-1][:-3]))
iterations = [int(x.split('_')[-1][:-3]) for x in files]
iteration = int(files[-1].split('_')[-1][:-3])

#%%
# for i in range(len(files)):
#     grid = from_netcdf(files[i])
#     write_raster_netcdf(os.path.join(save_directory,id,'paraview',f'{id}_grid_{i}.nc'),
#                         grid,
#                         time=df_out['time'].loc[i])
    

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
# plt.savefig(os.path.join(save_directory, id, "limestone_props.png"))


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
# plt.savefig(os.path.join(save_directory, id, "limestone_ksat.png"))


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
# plt.savefig(os.path.join(save_directory, id, "med_aquifer_thickness.png"))


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
# plt.savefig(os.path.join(save_directory, id, "recharge.png"))

# %% Big figure (to animate?)


elev_max = 250
# N = 4000

for i, N in enumerate(iterations):

    fig = plt.figure(figsize=(8,5))
    t1 = t[N]
    file = files[np.where(np.equal(iterations,N))[0][0]]
    mg = from_netcdf(file)

    ax1 = fig.add_subplot(2,2,2)
    ax1.plot(t, n, color='seagreen')
    ax1.axvline(x=t1, linestyle='--', color='r')
    ax1.set_ylim((0.0,0.012))
    ax1.set_ylabel('Porosity (-)', color='seagreen')
    ax1.set_xlabel('Time (yr)')

    ax1t = ax1.twinx()
    ax1t.plot(t, ksat, color='dodgerblue')
    ax1t.set_ylabel(r'$k_{sat}$ (m/s)', color='dodgerblue')
    ax1t.set_ylim((0.0,4e-5))

    ax2 = fig.add_subplot(2,2,4)
    ax2.plot(t, df_out['limestone_exposed'], color='darkgoldenrod')
    ax2.axvline(x=t1, linestyle='--', color='r')
    ax2.set_ylim((0.0,1.01))
    ax2.set_ylabel('Limestone exposed (-)', color='darkgoldenrod')
    ax2.set_xlabel('Time (yr)')

    ax2t = ax2.twinx()
    ax2t.plot(t, df_out['mean_relief'], color='brown')
    # ax2t.set_ylim((0.0,1.01))
    ax2t.set_ylabel('Mean Relief (m)', color='brown')
    ax2t.set_xlabel('Time (yr)')

    ax3 = fig.add_subplot(2,2,1)
    rockid = mg.at_node['rock_type__id'].copy()
    local_runoff = mg.at_node['local_runoff'].copy()
    elev = mg.at_node['topographic__elevation'].copy()
    runoff_ma = np.ma.masked_array(data=local_runoff, mask=local_runoff<0.5)
    y = np.arange(mg.shape[0] + 1) * mg.dx - mg.dx * 0.5
    x = np.arange(mg.shape[1] + 1) * mg.dy - mg.dy * 0.5

    ax3.imshow(
            rockid.reshape(mg.shape),
            origin="lower", 
            extent=(x[0], x[-1], y[0], y[-1]), 
            cmap='gray',
            alpha=0.3,
            )
    ax3.imshow(
            runoff_ma.reshape(mg.shape),
            origin="lower", 
            extent=(x[0], x[-1], y[0], y[-1]), 
            cmap='plasma',
            alpha=0.9,
            vmin=0,
            vmax=5
            )

    ax4 = fig.add_subplot(2,2,3, projection='3d')
    surf = ax4.plot_surface(mg.x_of_node.reshape(mg.shape), 
                            mg.y_of_node.reshape(mg.shape),
                            elev.reshape(mg.shape), 
                            rstride=1, 
                            cstride=1, 
                            cmap='gist_earth',
                            vmin=0, vmax=elev_max,
                            linewidth=0,
                            antialiased=False)
    ax4.axis('off')
    # ax4.set_aspect('equal')
    # ax4.set_xlim(0,7500)
    # ax4.set_ylim(0,5000)
    ax4.set_box_aspect((np.ptp(mg.x_of_node), np.ptp(mg.y_of_node), 2*np.max(elev)))
    ax4.set_xlim(2500,5000)
    ax4.set_ylim(1500,3000)

    fig.tight_layout()
    plt.savefig(os.path.join(save_directory,id,'%s.%04d.png'%(id,i)), dpi=300)
    plt.close()

# %%

fig = plt.figure()
ax4 = fig.add_subplot(1,1,1, projection='3d')
surf = ax4.plot_surface(mg.x_of_node.reshape(mg.shape), 
                        mg.y_of_node.reshape(mg.shape),
                        elev.reshape(mg.shape), 
                        rstride=1, cstride=1, 
                        cmap='gist_earth', 
                        alpha=1.0, linewidth=0,
                        antialiased=False)
# ax4.set_aspect('equal')
ax4.set_box_aspect((np.ptp(mg.x_of_node), np.ptp(mg.y_of_node), 2*elev_max))
ax4.axis('off')
ax4.set_xlim(0,7500)
ax4.set_ylim(0,5000)
# ax4.set_zlim(0, 1000)
fig.tight_layout()
# %%

# just elevation animation


elev_max = 250
# N = 4000

for i, N in enumerate(iterations):

    fig = plt.figure(figsize=(8,5))
    t1 = t[N]
    file = files[np.where(np.equal(iterations,N))[0][0]]
    mg = from_netcdf(file)
    elev = mg.at_node['topographic__elevation'].copy()

    ax4 = fig.add_subplot(1,1,1, projection='3d')
    surf = ax4.plot_surface(mg.x_of_node.reshape(mg.shape), 
                            mg.y_of_node.reshape(mg.shape),
                            elev.reshape(mg.shape), 
                            rstride=1, 
                            cstride=1, 
                            cmap='gist_earth',
                            vmin=0, vmax=elev_max,
                            linewidth=0,
                            antialiased=False)
    ax4.axis('off')
    # ax4.set_aspect('equal')
    ax4.set_xlim(0,7500)
    ax4.set_ylim(0,5000)
    ax4.set_box_aspect((np.ptp(mg.x_of_node), np.ptp(mg.y_of_node), 2*np.max(elev)))
    # ax4.set_xlim(2500,5000)
    # ax4.set_ylim(1500,3000)

    fig.tight_layout()
    plt.savefig(os.path.join(save_directory,id,'%s_elev.%04d.png'%(id,i)), dpi=300)
    plt.close()
# %%
