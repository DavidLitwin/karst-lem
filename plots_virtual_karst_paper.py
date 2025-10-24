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

from landlab.components import FlowAccumulator

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

#%% Fraction Limestone

fig1, ax1 = plt.subplots(figsize=(4, 2.5))
for name, nid in ids.items():
    model = nid.split('_')[-2]
    file = os.path.join(save_directory, nid, f'{nid}.nc')
    fp = Dataset(file, "r", format="NETCDF4")
    elev = fp.variables['topographic__elevation'][:,:,:].data
    rid = fp.variables['rock_type__id'][:,:,:].data

    t = fp['time'][:].data
    id_t = np.argmin(np.absolute(t - time))
    mean_elev = np.mean(elev[:, 1:-1, 1:-1], axis=(1, 2))
    frac_limestone = 1 - np.mean(rid[:, 1:-1, 1:-1], axis=(1, 2))

    ax1.plot(t, frac_limestone, label=f'{labels[name]}', color=colors[name])
    if name == 'id_gw':
        ax1.scatter(time, frac_limestone[id_t], s=50, color=colors[name])

ax1.set_ylabel('Limestone Exposed (-)')
ax1.set_xlabel('Time (kyr)')
# ax1.legend(frameon=False, loc='upper left')
fig1.tight_layout()

metadata1 = {
    'Title': 'Compare Limestone Fraction Time Evolution',
    'Creator': 'David Litwin',
    'Subject': f'Infiltration+Seepage:{id_gw}, Infiltration:{id_null}, Null:{id_null_0}'
}
fig1.savefig(os.path.join(fig_directory, 'limestone_fraction_time_evolution.pdf'),
             metadata=metadata1, dpi=300, transparent=True)

#%% Mean Elevation

fig2, ax2 = plt.subplots(figsize=(4, 2.5))
for name, nid in ids.items():
    model = nid.split('_')[-2]
    file = os.path.join(save_directory, nid, f'{nid}.nc')
    fp = Dataset(file, "r", format="NETCDF4")
    elev = fp.variables['topographic__elevation'][:,:,:].data
    rid = fp.variables['rock_type__id'][:,:,:].data

    t = fp['time'][:].data
    id_t = np.argmin(np.absolute(t - time))
    mean_elev = np.mean(elev[:, 1:-1, 1:-1], axis=(1, 2))
    frac_limestone = 1 - np.mean(rid[:, 1:-1, 1:-1], axis=(1, 2))

    ax2.plot(t, mean_elev, label=f'{labels[name]}', color=colors[name])
    if name == 'id_gw':
        ax2.scatter(time, mean_elev[id_t], s=50, color=colors[name])

ax2.set_ylim(-10, 300)
ax2.set_ylabel('Mean Elevation (m)')
ax2.set_xlabel('Time (kyr)')
ax2.legend(frameon=False, loc='upper left')
fig2.tight_layout()

metadata2 = {
    'Title': 'Compare Mean Elevation Time Evolution',
    'Creator': 'David Litwin',
    'Subject': f'Infiltration+Seepage:{id_gw}, Infiltration:{id_null}, Null:{id_null_0}'
}
fig2.savefig(os.path.join(fig_directory, 'mean_elevation_time_evolution.pdf'),
             metadata=metadata2, dpi=300, transparent=True)


 #%% Snapshot plots - infiltration + seepage

name = 'id_gw'
nid = ids[name]
model = nid.split('_')[-2]
file = os.path.join(save_directory,nid,f'{nid}.nc')
fp = Dataset(file, "r", format="NETCDF4")
elev = fp.variables['topographic__elevation'][:,:,:].data

# runoff and energy
qs = fp.variables['average_surface_water__specific_discharge'][:,:,:].data
ie_runoff = rid[id_t,:,:]
ie_runoff[ie_runoff<1] = 1e-2
runoff = (ie_runoff + 3600*24*365*qs[id_t,:,:])  # recalculated total runoff
pe_runoff = runoff * elev[id_t,:,:] * 1000 * 9.81 # recalculated total runoff * elevation


# integrate pe_runoff
x = fp.variables['x'][:]
dx = x[1]-x[0]
mg = RasterModelGrid(elev[id_t,:,:].shape, xy_spacing=dx)
mg.set_closed_boundaries_at_grid_edges(right_is_closed=True,
                                       left_is_closed=True,
                                       top_is_closed=False,
                                       bottom_is_closed=False)
z = mg.add_zeros("topographic__elevation", at='node')
z[:] = elev[id_t,:,:].flatten()
pe = mg.add_zeros("local_potential__energy", at='node')
pe[:] = pe_runoff.flatten()
r = mg.add_zeros("local__runoff", at='node')
r[:] = runoff.flatten()

fa = FlowAccumulator(
    mg,
    surface="topographic__elevation",
    flow_director='D8',
    runoff_rate="local_potential__energy",
    # runoff_rate="local__runoff",
)
fa.run_one_step()

# surface lithology
rid = fp.variables['rock_type__id'][:,:,:].data
grid = np.absolute(rid[id_t,:,:]-1)
grid[0, :] = 0
grid[-1, :] = 0
grid[:, 0] = 0
grid[:, -1] = 0

#%% topo with hashed limestone

fig, ax = plt.subplots(figsize=(6,2.5))
im1 =  ax.imshow(elev[id_t,:,:], cmap='gist_earth', vmax=np.max(elev), vmin=0, interpolation=None, origin='lower')

for (i, j), val in np.ndenumerate(grid):
    if val:
        rect = patches.Rectangle((j-0.5, i-0.5), 1, 1, hatch='XXX', 
                                fill=False, edgecolor='white', linewidth=0)
        ax.add_patch(rect)


cbar = fig.colorbar(im1, label=r'Elevation (m)')
# cbar.ax.set_yticks([0, 100, ])
ax.set_xticks(np.arange(0,175,25))
ax.set_yticks(np.arange(0,125,25))
ax.set_title(f'Infiltration+Seepage Model, {time} kyr')
metadata = {
    'Title': f'Infiltration+Seepage Model, {time} kyr',
    'Creator': 'David Litwin',
    'Subject': f'Infiltration+Seepage:{id_gw}'
}
fig.savefig(os.path.join(fig_directory, 'elev_snapshot.pdf'), metadata=metadata, dpi=300, transparent=True)


#%% local potential energy runoff

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


#%% local relative PE -- inset zoom

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

# metadata = {
#     'Title': 'Potential Energy Ratio',
#     'Creator': 'David Litwin',
#     'Subject': f'Infiltration+Seepage:{id_gw}'
# }
# fig.savefig(os.path.join(fig_directory, 'PE_ratio.pdf'), metadata=metadata, dpi=300, transparent=True)

#%% Local Relative PE -- inset transect

fig, ax = plt.subplots(figsize=(6,2.5))
im1 = ax.imshow(pe_runoff/pe_uniform, cmap='bwr_r',
                norm=LogNorm(vmax=1e2, vmin=1e-2), interpolation=None, origin='lower')
cbar = fig.colorbar(im1, label=r'PE Runoff / PE Precip.')
ax.set_title('Potential Energy Fraction')

# --- Transect parameters ---
transect_x = 78  # x-index for the transect (column in the array)
segment_frac = 0.25  # Fraction of full image height to show as segment
center_y = 55       # Center y-index for the segment
full_h = pe_runoff.shape[0]
segment_h = int(full_h * segment_frac)
y_start = max(0, center_y - segment_h // 2)
y_end = min(full_h, center_y + segment_h // 2)
transect_vals = (pe_runoff / pe_uniform)[y_start:y_end, transect_x]
transect_vals_1 = elev[id_t, y_start:y_end, transect_x]
ys = np.arange(y_start, y_end)

# --- Inset for transect ---
axins = inset_axes(ax, width="25%", height="60%", loc='center right', borderpad=0.5)

axins.plot(transect_vals, ys, color='black', linewidth=1, zorder=99)
axins.scatter(transect_vals, ys, 
              c=transect_vals, cmap='bwr_r', 
              norm=LogNorm(vmax=1e2, vmin=1e-2), 
              s=32, edgecolors='black', zorder=100)
axins.axvline(1, color='k', linestyle='--', linewidth=1)

axins.set_xscale('log')
axins.set_xlim(1e-3, 1e3)
axins.set_ylim(ys[0], ys[-1])
axins.set_yticks([])
axins.set_xticks([1])

axins_1 = axins.twiny()
axins_1.plot(transect_vals_1, ys, color='gray', linewidth=2)
axins_1.set_xlim(0,300)
axins_1.tick_params(axis='x', colors='black')


ax.plot([transect_x, transect_x], [y_start, y_end-1], color='k', linestyle='--', linewidth=1.75, alpha=0.8)

# plt.tight_layout()
plt.show()

metadata = {
    'Title': 'Potential Energy Ratio',
    'Creator': 'David Litwin',
    'Subject': f'Infiltration+Seepage:{id_gw}'
}
fig.savefig(os.path.join(fig_directory, 'PE_ratio.pdf'), metadata=metadata, dpi=300, transparent=False)

#%% integrated PE runoff

pe = mg.at_node['surface_water__discharge'] # it's called surface_water__discharge because that's what landlab calls the output of flow accumulation
pe[pe==0.0] = np.min(pe[mg.core_nodes])


fig, ax = plt.subplots(figsize=(6,2.5))
im1 =  ax.imshow(pe.reshape(mg.shape), cmap='magma_r', norm=LogNorm(vmax=1e13, vmin=1e7), interpolation=None, origin='lower')
cbar = fig.colorbar(im1, label=r'Potential Energy (J/yr)')
ax.set_xticks(np.arange(0,175,25))
ax.set_yticks(np.arange(0,125,25))
ax.set_title('Integrated Potential Energy')
# metadata = {
#     'Title': 'Potential Energy from Local Runoff',
#     'Creator': 'David Litwin',
#     'Subject': f'Infiltration+Seepage:{id_gw}'
# }
# fig.savefig(os.path.join(fig_directory, 'PE_runoff.pdf'), metadata=metadata, dpi=300, transparent=True)



#%% PE timeseries -- mean

ie_runoff = rid.copy()
ie_runoff[ie_runoff<1] = 1e-2
pe_runoff = np.mean((ie_runoff + 3600*24*365*qs) * elev * 1000 * 9.81, axis=(1,2))
pe_uniform = np.mean(1 * elev * 1000 * 9.81, axis=(1,2))

fig, ax = plt.subplots(figsize=(4,2.6))
ax.plot(t, pe_uniform, label='PE Precip.', color='k', linestyle='-')
ax.plot(t, pe_runoff, label='PE Runoff', color='b')
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
fig.tight_layout()
fig.savefig(os.path.join(fig_directory, 'PE_time_evolution.pdf'), metadata=metadata, dpi=300, transparent=True)


#%%

# ie_runoff = rid.copy()
# ie_runoff[ie_runoff<1] = 1e-2
# mean_runoff = np.mean((ie_runoff + 3600*24*365*qs), axis=(1,2))
# mean_elevation = np.mean(elev, axis=(1,2))

# fig, ax = plt.subplots()
# ax.plot(t, mean_runoff, label='Runoff')
# ax1 = ax.twinx()
# ax1.plot(t, mean_elevation)
# ax.set_ylim(0.5,1.0)



# %%
