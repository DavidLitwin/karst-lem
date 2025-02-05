"""
Make some plots for the virtual karst models
"""

#%%

import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource, TwoSlopeNorm, ListedColormap, CenteredNorm, LogNorm
from netCDF4 import Dataset
from scipy import ndimage
import joypy

from virtual_karst_funcs import locate_drainage_divide, calc_slope_d4, calc_chi


fig_directory = '/Users/dlitwin/Documents/Research/Karst landscape evolution/landlab_virtual_karst/figures'
save_directory = '/Users/dlitwin/Documents/Research Data/Local/karst_lem'
id = "virtual_karst_null_9"
# id = 'virtual_karst_conduit_8'
# id = 'virtual_karst_gw_8'
model = id[14:-2]
dx = 50

#%% Load data

# file = os.path.join(save_directory,id,f'grid_{id}.nc')
file = os.path.join(save_directory,id,f'{id}.nc')
fp = Dataset(file, "r", format="NETCDF4")
elev = fp.variables['topographic__elevation'][:,:,:].data
rid = fp.variables['rock_type__id'][:,:,:].data
denud = fp.variables['denudation__rate'][:,:,:].data
zk = fp.variables['karst__elevation'][:,:,:].data
zk[rid==1] = np.nan

if model=='gw':
    # qs = fp.variables['average_surface_water__specific_discharge'][:,:,:].data
    Q = fp.variables['surface_water__discharge'][:,:,:].data # gw
elif model=='conduit':
    Q = fp.variables['total_discharge'][:,:,:].data # conduit
elif model=='null':
    Q = fp.variables['ie_discharge'][:,:,:].data # null
else:
    Q = np.empty(elev.shape)

t = fp['time'][:].data
max_elev = np.max(elev[:,1:-1,1:-1], axis=(1,2))
frac_limestone = 1 - np.mean(rid[:,1:-1,1:-1], axis=(1,2))

# times = [500, 1000, 1500]
# times = [1000, 1750, 2500]
# times = [2000, 3000, 4000]
times = [1000, 2500, 4000]
id_ts_1 = [np.argmin(np.absolute(t-i)) for i in times]

fracs = [0.9, 0.5, 0.1]
id_ts = [np.argmin(np.absolute(frac_limestone-i)) for i in fracs]

#%% Time-decay of fraction limestone exposed and elevation

fig, ax = plt.subplots(figsize=(5,3.5))
ax.plot(t, max_elev, label='Max elevation', color='r')
# ax.scatter(t[id_ts], max_elev[id_ts], color='r', s=10)
ax.plot(t, fp.U*t*1e3-fp.b_limestone, label='Contact elevation', color='r', linestyle='--') #*fp.save_freq
ax.set_ylabel('Elevation (m)', color='r')
ax1 = ax.twinx()
ax1.plot(t, frac_limestone, label='Limestone', color='b')
# ax1.scatter(t[id_ts], frac_limestone[id_ts], color='b', s=10)
ax1.set_ylabel('Fraction exposed (-)', color='b')
fig.legend(ncols=2, loc=(0.2,0.86), frameon=False)
ax.set_xlabel('Time (kyr)')
plt.savefig(os.path.join(save_directory, id, f"elev_frac_time.png"), dpi=300)


#%% slopes on the contact

median_contact_slope = np.zeros(len(t))
contact_slope_25 = np.zeros(len(t))
contact_slope_75 = np.zeros(len(t))

for i in range(len(t)):
    zi = elev[i,:,:]
    slope = calc_slope_d4(zi, dx=dx)
    edges_eros = (zi - ndimage.binary_erosion(zi)).astype(bool)
    slopes = slope[edges_eros]
    # edges_dil = zi - ndimage.binary_dilation(zi)
    # slopes = slope[np.astype(edges_dil, bool)]
    median_contact_slope[i] = np.median(slopes)
    contact_slope_25[i] = np.percentile(slope, 25)
    contact_slope_75[i] = np.percentile(slope, 75)

#%% slope at the lithologic contact through time

fig, ax = plt.subplots(figsize=(5,3))
ax.plot(t, median_contact_slope, 'r-', label='Median')
ax.fill_between(t, contact_slope_25, contact_slope_75, color='0.5', alpha=0.5, label='25-75th percentile')
ax1 = ax.twinx()
ax1.plot(t, frac_limestone, label='Limestone', color='b')
# ax1.scatter(t[id_ts], frac_limestone[id_ts], color='b', s=10)
ax1.set_ylabel('Fraction exposed (-)', color='b')

ax.set_xlabel('Time (yr)')
ax.set_ylabel('Slope at lithologic contact')
ax.legend()
plt.savefig(os.path.join(save_directory, id, f"contact_slope_time.png"), dpi=300)

#%% Elevation CDFs through time

zi = elev[-1,:,:]
zi_sort = np.sort(zi.flatten())
cdf = np.linspace(0,1,len(zi_sort))

n = 10
dn = len(t)//n
cmap1 = plt.cm.get_cmap('Blues_r', n+2)
plt.figure()
cdf = np.linspace(0,1,len(zi_sort))
for i in range(n-1):
    zi = elev[i*dn,:,:]
    zi_sort = np.sort(zi.flatten()) / (t[i*dn]*1e3 * fp.U)
    plt.plot(zi_sort, cdf, color=cmap1(i+1))
plt.ylabel('CDF')
plt.xlabel('Elevation (m)')
plt.title(f'model:{model}, ie_frac = {fp.ie_frac:.2f}')
plt.savefig(os.path.join(save_directory, id, f"elev_cdfs.png"), dpi=300)

#%%


# fig, axs = plt.subplots(ncols=3, nrows=1, figsize=(6,2), layout='constrained')
# for i, id_t in enumerate(id_ts_1):
#     zi = elev[id_t,:,:]
#     si = calc_slope_d4(elev, 50.0)

# id_t = id_ts[2]
# zi = elev[-1,:,:]
# si = calc_slope_d4(zi, 50.0)
# zi_sort = np.sort(zi.flatten())
# cdf = np.linspace(0,1,len(zi_sort))

# plt.figure()
# plt.scatter(si, zi, s=3, alpha=0.2)


#%% Sequence of kernel density plots of elevation

for i in [50, 100, 150]:
    data = elev[i::10,1:-1,1:-1]
    t1 = t[i::10]
    x = np.ones(data.shape) * t1.reshape(t1.shape[0],1,1)

    df = pd.DataFrame(data=np.array([x.flatten(), data.flatten()]).T, columns=['timestep',"elev"])

    labels=[y if y%100==0 else None for y in list(df.timestep.unique())]
    fig, axes = joypy.joyplot(df, by="timestep", labels=labels, column="elev", range_style='own', 
                            grid="y", linewidth=1, legend=False, figsize=(6,5),
                            title=f'model:{model}, ie_frac = {fp.ie_frac:.2f}',
                            colormap=plt.cm.autumn_r, alpha=0.7)
    for i, ax in enumerate(axes[:-1]):
        # ax.axvline(np.mean(data,axis=(1,2))[i],0)
        ax.scatter(np.mean(data,axis=(1,2))[i], 0, color='k')
        # ax.scatter(np.max(data,axis=(1,2))[i], 0)
        ax.scatter(t1[i]*1e3 * fp.U, 0, color='b')
        # ax.set_xlim(-20,400)
    # axes[len(axes)//2].set_ylabel('Time (kyr)')
    plt.xlabel('Elevation (m)')
    plt.savefig(os.path.join(save_directory, id, f"elev_kernel_density_{i}.png"), dpi=300)

# %% Elevation, lithology, and erosion rate at three fractions of limestone exposed

cmap1 = ListedColormap(['floralwhite', 'sienna'])
cmap2 = plt.get_cmap('RdBu_r').copy()
cmap2.set_over('m')
fig, axs = plt.subplots(ncols=3, nrows=3, figsize=(6,4), layout='constrained')
for i, id_t in enumerate(id_ts):
    im1 =  axs[0,i].imshow(elev[id_t,:,:], cmap='gist_earth', vmax=np.max(elev), vmin=0)
    im2 = axs[1,i].imshow(rid[id_t,:,:], cmap=cmap1)
    im3 = axs[2,i].imshow(denud[id_t,:,:]*1e3, cmap=cmap2, norm=TwoSlopeNorm(vcenter=fp.U*1e3, vmax=1, vmin=0.0)) #vmax=0.5*np.max(denud)*1e3

    axs[0,i].axis('off')
    axs[1,i].axis('off')
    axs[2,i].axis('off')
    axs[0,i].set_title(f'Limestone {fracs[i]}')

cb1 = fig.colorbar(im1, ax=axs[0,:], label='Elevation (m)', shrink=0.8)
cb2 = fig.colorbar(im2, ax=axs[1,:], shrink=0.8) #label='Lithology'
cb3 = fig.colorbar(im3, ax=axs[2,:], label='Erosion (mm/yr)', shrink=0.8, extend='max')

cb2.set_ticks([0.25,0.75], labels=['Carbonate','Basement'])
# cb3.set_ticks([0.0, 0.1,0.5*np.max(denud)*1e3])
cb3.set_ticks([0.0, 0.1,1])
# fig.tight_layout()
plt.savefig(os.path.join(save_directory, id, f"elev_frac_space.png"), dpi=300)


# %% Elevation, lithology, and erosion rate at three timesteps

cmap1 = ListedColormap(['floralwhite', 'sienna'])
fig, axs = plt.subplots(ncols=3, nrows=3, figsize=(6,4), layout='constrained')
for i, id_t in enumerate(id_ts_1):
    im1 =  axs[0,i].imshow(elev[id_t,:,:], cmap='gist_earth', vmax=np.max(elev), vmin=0)
    im2 = axs[1,i].imshow(rid[id_t,:,:], cmap=cmap1)
    im3 = axs[2,i].imshow(denud[id_t,:,:]*1e3, cmap=cmap2, norm=TwoSlopeNorm(vcenter=fp.U*1e3, vmax=1, vmin=0.0)) #vmax=0.5*np.max(denud)*1e3

    axs[0,i].axis('off')
    axs[1,i].axis('off')
    axs[2,i].axis('off')
    axs[0,i].set_title(f'{times[i]} kyr')

cb1 = fig.colorbar(im1, ax=axs[0,:], label='Elevation (m)', shrink=0.8)
cb2 = fig.colorbar(im2, ax=axs[1,:], shrink=0.8) #label='Lithology'
cb3 = fig.colorbar(im3, ax=axs[2,:], label='Erosion (mm/yr)', shrink=0.8, extend='max')

cb2.set_ticks([0.25,0.75], labels=['Carbonate','Basement'])
# cb3.set_ticks([0.0, 0.1,0.5*np.max(denud)*1e3])
cb3.set_ticks([0.0, 0.1,1])
# fig.tight_layout()
plt.savefig(os.path.join(save_directory, id, f"elev_frac_space_times.png"), dpi=300)

#%% Elevation profiles at three timesteps

y_pos = fp["y"][1:-1].data
fig, axs = plt.subplots(ncols=3, nrows=1, figsize=(6,1.5))
for i, id_t in enumerate(id_ts_1):
    axs[i].fill_between(y_pos, np.max(elev[id_t,1:-1,1:-1], axis=1), y2=np.min(elev[id_t,1:-1,1:-1], axis=1), color='0.7', alpha=0.4)
    axs[i].plot(y_pos, np.min(elev[id_t,1:-1,1:-1], axis=1), color='k')
    axs[i].plot(y_pos, np.mean(elev[id_t,1:-1,1:-1], axis=1))
    axs[i].plot(y_pos, np.nanmax(zk[id_t,1:-1,1:-1], axis=1), color='r', linestyle='--')
    axs[i].set_ylim((0,np.max(elev)))
    axs[i].axis('off')
    axs[i].set_title(f'{times[i]} kyr')
    
fig.tight_layout()
plt.savefig(os.path.join(save_directory, id, f"profile_frac_space.png"), dpi=300)


#%% Overlay limestone exposed and discharge at 3 timesteps

cmap1 = ListedColormap(['floralwhite', 'sienna'])
fig, axs = plt.subplots(ncols=3, nrows=1, figsize=(6,2), layout='constrained')
for i, id_t in enumerate(id_ts_1):
    im1 =  axs[i].imshow(Q[id_t,:,:], cmap='Blues', vmax=0.5*np.max(Q), vmin=0)
    im2 = axs[i].imshow(rid[id_t,:,:], cmap=cmap1, alpha=0.3)

    # axs[i].axis('off')
    # axs[i].set_title(f'Limestone {fracs[i]}')
    axs[i].set_title(f'{times[i]} kyr')

# cb1 = fig.colorbar(im1, ax=axs[:], label='Saturation Excess (m/s)', shrink=0.8)
cb1 = fig.colorbar(im1, ax=axs[:], label='Discharge (m$^3$/yr)', shrink=0.8)
plt.savefig(os.path.join(save_directory, id, f"discharge_space.png"), dpi=300)

#%% Slope maps at three timesteps

fig, axs = plt.subplots(ncols=3, nrows=1, figsize=(6,2), layout='constrained')
for i, id_t in enumerate(id_ts_1):
    slope = calc_slope_d4(elev[id_t,:,:], dx=dx)
    im1 =  axs[i].imshow(slope, cmap='cubehelix_r', vmax=2.0, vmin=0)

    # axs[i].axis('off')
    # axs[i].set_title(f'Limestone {fracs[i]}')
    axs[i].set_title(f'{times[i]} kyr')

cb1 = fig.colorbar(im1, ax=axs[:], label='Slope (m/m)', shrink=0.8)
plt.savefig(os.path.join(save_directory, id, f"slope.png"), dpi=300)

#%% Chi maps at three timesteps

fig, axs = plt.subplots(ncols=3, nrows=1, figsize=(6,2), layout='constrained')
for i, id_t in enumerate(id_ts_1):
    chi = calc_chi(elev[id_t,:,:], dx=dx, reference_concavity=0.5, min_drainage_area=0.0)
    im1 =  axs[i].imshow(chi.reshape(elev.shape[1:]), cmap='cubehelix', vmax=10.0, vmin=0)

    # axs[i].axis('off')
    # axs[i].set_title(f'Limestone {fracs[i]}')
    axs[i].set_title(f'{times[i]} kyr')

cb1 = fig.colorbar(im1, ax=axs[:], label='Chi (m)', shrink=0.8)
plt.savefig(os.path.join(save_directory, id, f"chi.png"), dpi=300)

#%%

core_nodes = np.arange(elev.shape[1]*elev.shape[2]).reshape(elev.shape[1:])[1:-1,1:-1].flatten()
fig, axs = plt.subplots(ncols=3, nrows=1, figsize=(6,2), layout='constrained')
for i, id_t in enumerate(id_ts_1):
    chi = calc_chi(elev[id_t,:,:], dx=dx, reference_concavity=0.5, min_drainage_area=0.0)
    x = chi[core_nodes]
    y = elev[id_t,1:-1,1:-1].flatten()
    c = Q[id_t,1:-1,1:-1].flatten()
    inds = np.argsort(c)
    im1 =  axs[i].scatter(x[inds], y[inds], c=c[inds], 
                          cmap='plasma', norm=LogNorm(vmin=10, vmax=np.max(Q[id_ts_1,:,:])),
                          s=4, alpha=0.5)

    # axs[i].axis('off')
    # axs[i].set_title(f'Limestone {fracs[i]}')
    axs[i].set_ylim(0,1.1*np.max(elev[id_ts_1,:,:]))
    axs[i].set_title(f'{times[i]} kyr')
    axs[i].set_xlabel(r'$chi$ (m)')
axs[0].set_ylabel('Elevation (m)')
cb1 = fig.colorbar(im1, ax=axs[:], label='Q (m3/s)', shrink=0.8)
plt.savefig(os.path.join(save_directory, id, f"chi_elevation.png"), dpi=300)

# %% Map of scarp edges at one timestep

im = rid[id_ts[1],:,:]

edges_dil = im - ndimage.binary_dilation(im)
edge_frac_dil = np.abs(np.sum(edges_dil[1:-1,1:-1]))/len(edges_dil[1:-1,1:-1].flatten())

edges_eros = im - ndimage.binary_erosion(im)
edge_frac_eros = np.abs(np.sum(edges_eros[1:-1,1:-1]))/len(edges_eros[1:-1,1:-1].flatten())
print(edge_frac_dil,edge_frac_eros)

plt.figure()
plt.imshow(np.astype(edges_dil, bool), cmap='pink')
plt.title(f'Limestone {fracs[1]}, fraction edges: {edge_frac_dil:0.3f}')

plt.figure()
plt.imshow(im, cmap='pink')
plt.title(f'Limestone {fracs[1]}, fraction edges: {edge_frac_dil:0.3f}')

# plt.colorbar()
# plt.savefig(os.path.join(save_directory, id, f"edges.png"), dpi=300)

#%% drainage divide at one timestep

im = elev[id_ts_1[1],:,:]
# plt.imshow(im)

divide_image, mask = locate_drainage_divide(im, 50.0)
divide_image[0,:]
plt.imshow(divide_image, interpolation=None)


# profiles along the way -- how much does additional water affect the channel form?
# maximum steady-state relief of the basement (based on chi and steepness)
# maximum steady-state relief across both (based on chi-p and steepness, assuming water is all lost and not returned)?

# seepage and seepage diffusive erosion

# %% cross-run comparisons

runs_all = {}
runs = ['virtual_karst_null_1', 'virtual_karst_null_2', 'virtual_karst_null_3', 'virtual_karst_null_4']
# runs = ['virtual_karst_conduit_1', 'virtual_karst_conduit_2', 'virtual_karst_conduit_3', 'virtual_karst_conduit_4']
# runs = ['virtual_karst_gw_1', 'virtual_karst_gw_2', 'virtual_karst_gw_3', 'virtual_karst_gw_4']
for run in runs:
    file = os.path.join(save_directory,run,f'{run}.nc')
    runs_all[run] = Dataset(file, "r", format="NETCDF4")

cmap1 = plt.cm.get_cmap('Blues_r', len(runs)+2)
fracs = [0.9, 0.5, 0.1]
fig, ax = plt.subplots(figsize=(5,3.5))
for j, run in enumerate(runs):

    var = runs_all[run].ie_frac
    t = runs_all[run]['time'][:].data
    rid = runs_all[run].variables['rock_type__id'][:,:,:].data
    frac_limestone = 1 - np.mean(rid[:,1:-1,1:-1], axis=(1,2))
    id_ts = [np.argmin(np.absolute(frac_limestone-i)) for i in fracs]
    ax.plot(t, frac_limestone, label=f'{var:0.2f}', color=cmap1(j+1))
    # ax.scatter(t[id_ts], frac_limestone[id_ts], color='b', s=10)

ax.set_ylabel('Fraction exposed (-)')
ax.set_xlabel('Time (kyr)')
ax.set_title(f'Model: {run[14:-2]}, Vary: ie_frac')
ax.legend(frameon=False)
plt.savefig(os.path.join(save_directory, runs[0], f"exposed_frac_time_ie_frac.png"), dpi=300)

#%%

runs_all = {}
runs = ['virtual_karst_null_6', 'virtual_karst_gw_6', 'virtual_karst_conduit_6']
colors = ['k', 'b', 'r']
for run in runs:
    file = os.path.join(save_directory,run,f'{run}.nc')
    runs_all[run] = Dataset(file, "r", format="NETCDF4")

# cmap1 = plt.cm.get_cmap('Blues_r', len(runs)+2)
fracs = [0.9, 0.5, 0.1]
fig, ax = plt.subplots(figsize=(5,3.5))
for j, run in enumerate(runs):

    var = run[14:-2]
    print(runs_all[run].ie_frac)
    t = runs_all[run]['time'][:].data
    rid = runs_all[run].variables['rock_type__id'][:,:,:].data
    frac_limestone = 1 - np.mean(rid[:,1:-1,1:-1], axis=(1,2))
    id_ts = [np.argmin(np.absolute(frac_limestone-i)) for i in fracs]
    ax.plot(t, frac_limestone, label=f'{var}', color=colors[j]) # color=cmap1(j+1)
    # ax.scatter(t[id_ts], frac_limestone[id_ts], color='b', s=10)

ax.set_ylabel('Fraction exposed (-)')
ax.set_xlabel('Time (kyr)')
ax.set_title(f'Vary model, ie_frac = {runs_all[run].ie_frac:.2f}')
ax.legend(frameon=False)
plt.savefig(os.path.join(save_directory, runs[0], f"exposed_frac_time_model.png"), dpi=300)


# %%
