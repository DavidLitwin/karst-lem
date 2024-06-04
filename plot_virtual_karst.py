"""
Make some plots for the virtual karst models
"""

#%%
import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource
from netCDF4 import Dataset


from landlab.io.netcdf import write_raster_netcdf, from_netcdf, read_netcdf
from virtual_karst_funcs import *

fig_directory = '/Users/dlitwin/Documents/Research/Karst landscape evolution/landlab_virtual_karst/figures'
save_directory = '/Users/dlitwin/Documents/Research Data/Local/karst_lem'
id = "flat_dynamic_ksat_8"

df_out = pd.read_csv(os.path.join(save_directory,id,f'output_{id}.csv'))
df_params = pd.read_csv(os.path.join(save_directory,id,f'params_{id}.csv')).loc[0]

#%%

file = os.path.join(save_directory,id,f'grid_{id}.nc')
fp = Dataset(file, "r", format="NETCDF4")
elev = fp.variables['topographic__elevation'][-1,:,:].data
rid = fp.variables['rock_type__id'][-1,:,:].data

plt.figure()
im = plt.imshow(elev, cmap='gist_earth')
plt.colorbar(im, label='Elevation')

plt.figure()
im = plt.imshow(rid, cmap='Blues')
# plt.colorbar(im, label='Elevation')

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
ax.set_ylim((0.0,1.02*nf))
ax.set_ylabel('Porosity (-)', color='seagreen')
ax.set_xlabel('Time (yr)')

ax1 = ax.twinx()
ax1.plot(t, ksat, color='dodgerblue')
ax1.set_ylabel(r'$k_{sat}$ (m/s)', color='dodgerblue')
ax1.set_ylim((0.0,1.02*np.max(ksat)))
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
ax1.set_ylim((0.0,1.02*np.max(ksat)))
plt.savefig(os.path.join(save_directory, id, "limestone_ksat.png"))


# %% ksat and mean aquifer thickness

fig, ax = plt.subplots()
ax.plot(t, df_out['median_aquifer_thickness'], color='b')
ax.set_ylim((0.01,1.02*np.max(df_out['median_aquifer_thickness'])))
ax.set_ylabel('Median Aquifer Thickness (m)', color='b')
ax.set_xlabel('Time (yr)')

ax1 = ax.twinx()
ax1.plot(t, ksat, color='dodgerblue')
ax1.set_ylabel(r'$k_{sat}$ (m/s)', color='dodgerblue')
ax1.set_ylim((0.0,1.02*np.max(ksat)))
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
ax1.set_ylim((0.0,1.02*np.max(ksat)))
plt.savefig(os.path.join(save_directory, id, "recharge.png"))

# %% Big figure (to animate?)

time = fp.variables['t'][:].data
elev_max = np.max(fp.variables['topographic__elevation'][:].data)
runoff_max = np.max(fp.variables['local_runoff'][:].data)
x = fp.variables['x'][:].data + 0.5 * np.diff(fp.variables['x'][:].data)[0]
y = fp.variables['y'][:].data + 0.5 * np.diff(fp.variables['y'][:].data)[0]
X, Y = np.meshgrid(x,y)

for i, t1 in enumerate(time):

    fig = plt.figure(figsize=(9,5))

    elev = fp.variables['topographic__elevation'][i,:,:].data
    rockid = fp.variables['rock_type__id'][i,:,:].data
    local_runoff = fp.variables['local_runoff'][i,:,:].data
    runoff_limestone = np.ma.masked_array(data=local_runoff, mask=rockid==1)
    runoff_basement = np.ma.masked_array(data=local_runoff, mask=rockid==0)

    ax1 = fig.add_subplot(2,2,2)
    ax1.plot(t, n, color='seagreen')
    ax1.axvline(x=t1, linestyle='--', color='r')
    ax1.set_ylim((0.0,1.02*nf))
    ax1.set_ylabel('Porosity (-)', color='seagreen')
    ax1.set_xlabel('Time (yr)')

    ax1t = ax1.twinx()
    ax1t.plot(t, ksat, color='dodgerblue')
    ax1t.set_ylabel(r'$k_{sat}$ (m/s)', color='dodgerblue')
    ax1t.set_ylim((0.0,1.02*np.max(ksat)))

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

    im1 = ax3.imshow(
            runoff_basement,
            origin="lower", 
            extent=(x[0], x[-1], y[0], y[-1]), 
            cmap='Blues',
            alpha=1.0,
            vmin=-1.0,
            vmax=5,
            )
    im2 = ax3.imshow(
            runoff_limestone,
            origin="lower", 
            extent=(x[0], x[-1], y[0], y[-1]), 
            cmap='Greens',
            alpha=1.0,
            vmin=-1.0,
            vmax=5,
            )
    cbb = plt.colorbar(im1,label=r'$q_s$ Basement', shrink=0.5)
    cbb = plt.colorbar(im2,label=r'$q_s$ Limestone', shrink=0.5)

    ax4 = fig.add_subplot(2,2,3, projection='3d')
    surf = ax4.plot_surface(X, 
                            Y,
                            elev, 
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
    ax4.set_box_aspect((np.ptp(x), np.ptp(y), 2*np.max(elev)))
    ax4.set_xlim(2500,5000)
    ax4.set_ylim(1500,3000)

    fig.tight_layout()
    plt.savefig(os.path.join(save_directory,id,'%s.%04d.png'%(id,i)), dpi=300)
    plt.close()


# %%  images for elevation animation: from appended NETCDF4

file = os.path.join(save_directory,id,f'grid_{id}.nc')
fp = Dataset(file, "r", format="NETCDF4")
elev = fp.variables['topographic__elevation'][:].data
x = fp.variables['x'][:].data
y = fp.variables['y'][:].data
t = fp.variables['t'][:].data
X, Y = np.meshgrid(x,y)
elev_max = np.max(elev)

for i, t in enumerate(t):

    fig = plt.figure(figsize=(8,5))

    ax4 = fig.add_subplot(1,1,1, projection='3d')
    surf = ax4.plot_surface(X, Y, elev[i,:,:], 
                            rstride=1, 
                            cstride=1, 
                            cmap='gist_earth',
                            vmin=0, vmax=elev_max,
                            linewidth=0,
                            antialiased=False)
    ax4.axis('off')
    ax4.set_xlim(0,7500)
    ax4.set_ylim(0,5000)
    ax4.set_box_aspect((np.ptp(x), np.ptp(y), 2*np.max(elev[i,:,:])))
    # ax4.set_xlim(2500,5000)
    # ax4.set_ylim(1500,3000)

    fig.tight_layout()
    plt.savefig(os.path.join(save_directory,id,'%s_elev.%04d.png'%(id,i)), dpi=300)
    plt.close()

# %%

# hillshade

# elev = mg.at_node['topographic__elevation'].copy()
# elev[mg.boundary_nodes] = np.nan
# y = np.arange(mg.shape[0] + 1) * mg.dx - mg.dx * 0.5
# x = np.arange(mg.shape[1] + 1) * mg.dy - mg.dy * 0.5

# elev_plot = elev.reshape(mg.shape)
# elev_profile = np.nanmean(elev_plot, axis=1)

# f, (ax0, ax1) = plt.subplots(1, 2, width_ratios=[4, 1], figsize=(10,5))
# ls = LightSource(azdeg=135, altdeg=45)
# ax0.imshow(
#         ls.hillshade(elev_plot, 
#             vert_exag=1, 
#             dx=mg.dx, 
#             dy=mg.dy), 
#         origin="lower", 
#         extent=(x[0], x[-1], y[0], y[-1]), 
#         cmap='gray',
#         )
# for i in range(mg.shape[1]):
#     ax1.plot(elev_plot[:,i], y[0:-1], alpha=0.1, color='k')
# ax1.plot(elev_profile, y[0:-1], linewidth=2, color='r')

# ax0.set_xlabel('X [m]')
# ax0.set_ylabel('Y [m]')
# ax1.set_xlabel('Z [m]')
# f.tight_layout()
# plt.savefig(os.path.join(save_directory, id, "hillshade.png"))

#%%

a = np.array([[0,1]])
plt.figure(figsize=(1, 3))
img = plt.imshow(a, cmap="plasma")
plt.gca().set_visible(False)
cax = plt.axes([0.1, 0.1, 0.4, 0.8])
plt.colorbar(orientation="vertical", cax=cax, ticks=[0.0, 0.25, 0.5, 0.75, 1.0])
# plt.tight_layout()
plt.savefig(os.path.join(save_directory,id,"colorbar.pdf"))

# plt.axes([])
# %% cross section through domain

file = os.path.join(save_directory,id,f'grid_{id}.nc')
fp = Dataset(file, "r", format="NETCDF4")
z = fp.variables['topographic__elevation'][:].data
zb = fp.variables['aquifer_base__elevation'][:].data
zwb = zb+0.5


wt = fp.variables['water_table__elevation'][:].data
x = fp.variables['x'][:].data
y = fp.variables['y'][:].data
t = fp.variables['t'][:].data
X, Y = np.meshgrid(x,y)
nt = 10
nx = 75

#%%
fig, axs = plt.subplots(figsize=(4,1.5))

axs.plot(y,z[nt, :, nx], color='k', linewidth=0.25)
axs.fill_between(y,z[nt, :, nx],zb[nt, :, nx],facecolor='#EBEBEB' ) # permeable rock
axs.fill_between(y,wt[nt, :, nx],zb[nt, :, nx],facecolor='#2481E9', alpha=1.0) # aquifer
axs.fill_between(y,zb[nt, :, nx],np.zeros_like(zb[nt, :, nx]),facecolor='#808080') # impermeable
axs.set_xlim((min(y),max(y)))
axs.set_ylim((0,np.nanmax(z[:, :, nx])*1.05))
axs.spines['right'].set_visible(False)
axs.spines['top'].set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join(save_directory,id,f"XS_{nt}.pdf"), transparent=True, dpi=300)


# %%
