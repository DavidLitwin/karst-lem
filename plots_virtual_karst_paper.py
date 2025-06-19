"""
Make some plots for the virtual karst models -- Paper Figure 3
"""

#%%

import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource, TwoSlopeNorm, ListedColormap, CenteredNorm, LogNorm
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.patches as patches
from netCDF4 import Dataset
from scipy import ndimage

from virtual_karst_funcs import *


fig_directory = '/Users/dlitwin/Documents/Papers/geol_2025_karst/figures'
save_directory = '/Users/dlitwin/Documents/Research Data/Local/karst_lem'
id_gw = 'virtual_karst_gw_8' # water loss to continuum gw model
id_null = 'virtual_karst_null_8' #same as id_gw, but infiltration lost, does not go to gw
id_null_K = 'virtual_karst_null_9' # no infiltration contrast, just erodibility contrast
id_null_0 = 'virtual_karst_null_10' # no infiltration contrast, no erodibility contrast
ids = {'id_gw': id_gw, 'id_null': id_null, 'id_null_0': id_null_0} #'id_null_K': id_null_K,
labels = {'id_gw': 'Infiltration+Seepage', 'id_null':'Infiltration', 'id_null_K':'Erodibility', 'id_null_0':'Null'}
colors = {'id_gw': 'b', 'id_null':'dodgerblue', 'id_null_K':'darkorange', 'id_null_0':'gray'}
dx = 50

time = 2500
id_t = np.argmin(np.absolute(t-time))

#%% Time of limestone persistence plot

# fig1, ax1 = plt.subplots(figsize=(5,3.5))
# fig2, ax2 = plt.subplots(figsize=(5,3.5))

fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(8,3))
for name, nid in ids.items():

    model = nid.split('_')[-2]
    file = os.path.join(save_directory,nid,f'{nid}.nc')
    fp = Dataset(file, "r", format="NETCDF4")
    elev = fp.variables['topographic__elevation'][:,:,:].data
    rid = fp.variables['rock_type__id'][:,:,:].data

    t = fp['time'][:].data
    mean_elev = np.mean(elev[:,1:-1,1:-1], axis=(1,2))
    frac_limestone = 1 - np.mean(rid[:,1:-1,1:-1], axis=(1,2))

    ax1.plot(t, frac_limestone, label=f'{labels[name]}', color=colors[name])
    ax2.plot(t, mean_elev, label=f'{labels[name]}', color=colors[name])
    
    if name=='id_gw':
        ax1.scatter(time, frac_limestone[id_t], s=50, color=colors[name])
        ax2.scatter(time, mean_elev[id_t], s=50, color=colors[name])

ax1.set_ylabel('Fraction Limestone Exposed (-)')
ax1.set_xlabel('Time (kyr)')

ax2.set_ylim(-10, 300)
ax2.set_ylabel('Mean Elevation (m)')
ax2.set_xlabel('Time (kyr)')
ax2.legend(frameon=False, loc='upper left')
fig.tight_layout()

metadata = {
    'Title': 'Compare Limestone Fraction and Mean Elevation Time Evolution',
    'Creator': 'David Litwin',
    'Subject': f'Infiltration+Seepage:{id_gw}, Infiltration:{id_null}, Null:{id_null_0}'
}
fig.savefig(os.path.join(fig_directory, 'time_evolution.pdf'), metadata=metadata, dpi=300, transparent=True)


 #%% Snapshot plot


name = 'id_gw'
nid = ids[name]
model = nid.split('_')[-2]
file = os.path.join(save_directory,nid,f'{nid}.nc')
fp = Dataset(file, "r", format="NETCDF4")
elev = fp.variables['topographic__elevation'][:,:,:].data
rid = fp.variables['rock_type__id'][:,:,:].data
qs = fp.variables['average_surface_water__specific_discharge'][:,:,:].data

fig, ax = plt.subplots(figsize=(6,2.5))
im1 =  ax.imshow(elev[id_t,:,:], cmap='gist_earth', vmax=np.max(elev), vmin=0, interpolation=None, origin='lower')
cbar = fig.colorbar(im1, label=r'Elevation (m)')
ax.set_xticks(np.arange(0,175,25))
ax.set_yticks(np.arange(0,125,25))
ax.set_title(f'Infiltration+Seepage Model, {time} kyr')
metadata = {
    'Title': f'Infiltration+Seepage Model, {time} kyr',
    'Creator': 'David Litwin',
    'Subject': f'Infiltration+Seepage:{id_gw}'
}
fig.savefig(os.path.join(fig_directory, 'elev_snapshot.pdf'), metadata=metadata, dpi=300, transparent=True)


ie_runoff = rid[id_t,:,:]
ie_runoff[ie_runoff<1] = 1e-2

pe_runoff = (ie_runoff + 3600*24*365*qs[id_t,:,:]) * elev[id_t,:,:] * 1000 * 9.81 # fudging the infiltration excess
fig, ax = plt.subplots(figsize=(6,2.5))
im1 =  ax.imshow(pe_runoff, cmap='magma_r', norm=LogNorm(vmax=1e8, vmin=1e2), interpolation=None, origin='lower')
cbar = fig.colorbar(im1, label=r'Potential Energy (J/yr/m$^2$)')
ax.set_xticks(np.arange(0,175,25))
ax.set_yticks(np.arange(0,125,25))
ax.set_title('Potential Energy from Local Runoff')
metadata = {
    'Title': 'Potential Energy from Local Runoff',
    'Creator': 'David Litwin',
    'Subject': f'Infiltration+Seepage:{id_gw}'
}
fig.savefig(os.path.join(fig_directory, 'PE_runoff.pdf'), metadata=metadata, dpi=300, transparent=True)


pe_uniform = 1 * elev[id_t,:,:] * 1000 * 9.81
fig, ax = plt.subplots(figsize=(6,2.5))
im1 =  ax.imshow(pe_uniform, cmap='magma_r', norm=LogNorm(vmax=1e8, vmin=1e2), interpolation=None, origin='lower')
cbar = fig.colorbar(im1, label=r'Potential Energy (J/yr/m$^2$)')
ax.set_xticks(np.arange(0,175,25))
ax.set_yticks(np.arange(0,125,25))
ax.set_title('Potential Energy if Uniform Runoff')
metadata = {
    'Title': 'Potential Energy if Uniform Runoff',
    'Creator': 'David Litwin',
    'Subject': f'Infiltration+Seepage:{id_gw}'
}
fig.savefig(os.path.join(fig_directory, 'PE_uniform.pdf'), metadata=metadata, dpi=300, transparent=True)


# fig, ax = plt.subplots(figsize=(6,2.5))
# im1 =  ax.imshow(pe_runoff/pe_uniform, cmap='bwr_r', norm=LogNorm(vmax=1e2, vmin=1e-2), interpolation=None)
# cbar = fig.colorbar(im1, label=r'PE Runoff / PE Uniform')
# ax.set_title('Runoff PE vs. Uniform PE')


#%%

fig, ax = plt.subplots(figsize=(6,2.5))
im1 = ax.imshow(pe_runoff/pe_uniform, cmap='bwr_r',
                norm=LogNorm(vmax=1e2, vmin=1e-2), interpolation=None, origin='lower')
cbar = fig.colorbar(im1, label=r'PE Runoff / PE Uniform')
ax.set_title('Runoff PE vs. Uniform PE')

ax.set_xticks(np.arange(0,175,25))
ax.set_yticks(np.arange(0,125,25))

# Coordinates of the zoom box in image (data) coordinates
x0, x1 = 70, 85  # columns (horizontal)
y0, y1 = 50, 65  # rows (vertical)
# x0, x1 = 55, 70  # columns (horizontal)
# y0, y1 = 30, 45  # rows (vertical)

# Inset axes in the lower right as a fraction of the main axes
axins = inset_axes(ax, width="35%", height="50%",
                   loc='lower right', borderpad=0)

# Plot zoomed-in region in the inset
im2 = axins.imshow(pe_runoff/pe_uniform, cmap='bwr_r',
                   norm=LogNorm(vmax=1e2, vmin=1e-2), interpolation=None, origin='lower')

# Set the limits to the zoomed region
axins.set_xlim(x0, x1)
axins.set_ylim(y0, y1)  # note: imshow's y-axis is top-to-bottom

# Optionally, remove ticks from inset
axins.set_xticks([])
axins.set_yticks([])

# Draw a rectangle on the main plot to indicate the zoomed area
rect = patches.Rectangle((x0, y0), x1-x0, y1-y0, linewidth=1,
                         edgecolor='black', facecolor='none', linestyle='--')
ax.add_patch(rect)

metadata = {
    'Title': 'Potential Energy Ratio',
    'Creator': 'David Litwin',
    'Subject': f'Infiltration+Seepage:{id_gw}'
}
fig.savefig(os.path.join(fig_directory, 'PE_ratio.pdf'), metadata=metadata, dpi=300, transparent=True)


#%%

ie_runoff = rid.copy()
ie_runoff[ie_runoff<1] = 1e-2
pe_runoff = np.mean((ie_runoff + 3600*24*365*qs) * elev * 1000 * 9.81, axis=(1,2))
pe_uniform = np.mean(1 * elev * 1000 * 9.81, axis=(1,2))

fig, ax = plt.subplots(figsize=(4,2.75))
ax.plot(t, pe_runoff, label='PE Runoff', color='b')
ax.plot(t, pe_uniform, label='PE Uniform', color='k', linestyle='--')
ax.fill_between(t, pe_uniform, pe_runoff, color='b', alpha=0.3)
ax.legend(frameon=False)
ax.set_ylabel(r'Mean PE (J/yr/m$^2$)')
ax.set_xlabel('Time (kyr)')
fig.tight_layout()
metadata = {
    'Title': 'Potential Energy Time Evolution',
    'Creator': 'David Litwin',
    'Subject': f'Infiltration+Seepage:{id_gw}'
}
fig.savefig(os.path.join(fig_directory, 'PE_time_evolution.pdf'), metadata=metadata, dpi=300, transparent=True)


#%%

ie_runoff = rid.copy()
ie_runoff[ie_runoff<1] = 1e-2
mean_runoff = np.mean((ie_runoff + 3600*24*365*qs), axis=(1,2))
mean_elevation = np.mean(elev, axis=(1,2))

fig, ax = plt.subplots()
ax.plot(t, mean_runoff, label='Runoff')
ax1 = ax.twinx()
ax1.plot(t, mean_elevation)
ax.set_ylim(0.5,1.0)



# %%
