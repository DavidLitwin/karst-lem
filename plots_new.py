"""
Make some plots for the virtual karst models
"""

#%%

import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource, TwoSlopeNorm, ListedColormap, CenteredNorm
from netCDF4 import Dataset
from scipy import ndimage

fig_directory = '/Users/dlitwin/Documents/Research/Karst landscape evolution/landlab_virtual_karst/figures'
save_directory = '/Users/dlitwin/Documents/Research Data/Local/karst_lem'
# id = "virtual_karst_null_7"
# id = 'virtual_karst_conduit_4'
id = 'virtual_karst_gw_3'

#%%

# file = os.path.join(save_directory,id,f'grid_{id}.nc')
file = os.path.join(save_directory,id,f'{id}.nc')
fp = Dataset(file, "r", format="NETCDF4")
elev = fp.variables['topographic__elevation'][:,:,:].data
rid = fp.variables['rock_type__id'][:,:,:].data
denud = fp.variables['denudation__rate'][:,:,:].data


#%%

t = fp['time'][:].data
max_elev = np.max(elev[:,1:-1,1:-1], axis=(1,2))
frac_limestone = 1 - np.mean(rid[:,1:-1,1:-1], axis=(1,2))

fracs = [0.9, 0.5, 0.1]
id_ts = [np.argmin(np.absolute(frac_limestone-i)) for i in fracs]

fig, ax = plt.subplots(figsize=(5,3.5))
ax.plot(t, max_elev, label='Max elevation', color='r')
ax.scatter(t[id_ts], max_elev[id_ts], color='r', s=10)
ax.plot(t, fp.U*t*1e3-fp.b_limestone, label='Contact elevation', color='r', linestyle='--') #*fp.save_freq
ax.set_ylabel('Elevation (m)', color='r')
ax1 = ax.twinx()
ax1.plot(t, frac_limestone, label='Limestone', color='b')
ax1.scatter(t[id_ts], frac_limestone[id_ts], color='b', s=10)
ax1.set_ylabel('Fraction exposed (-)', color='b')
fig.legend(ncols=2, loc=(0.2,0.86), frameon=False)
ax.set_xlabel('Time (kyr)')
plt.savefig(os.path.join(save_directory, id, f"elev_frac_time.png"), dpi=300)

# %%

cmap1 = ListedColormap(['floralwhite', 'sienna'])
fig, axs = plt.subplots(ncols=3, nrows=3, figsize=(6,4), layout='constrained')
for i, id_t in enumerate(id_ts):
    im1 =  axs[0,i].imshow(elev[id_t,:,:], cmap='gist_earth', vmax=np.max(elev), vmin=0)
    im2 = axs[1,i].imshow(rid[id_t,:,:], cmap=cmap1)
    im3 = axs[2,i].imshow(denud[id_t,:,:]*1e3, cmap='RdBu_r', norm=TwoSlopeNorm(vcenter=fp.U*1e3, vmax=0.5*np.max(denud)*1e3, vmin=0.0))

    axs[0,i].axis('off')
    axs[1,i].axis('off')
    axs[2,i].axis('off')
    axs[0,i].set_title(f'Limestone {fracs[i]}')

cb1 = fig.colorbar(im1, ax=axs[0,:], label='Elevation (m)', shrink=0.8)
cb2 = fig.colorbar(im2, ax=axs[1,:], shrink=0.8) #label='Lithology'
cb3 = fig.colorbar(im3, ax=axs[2,:], label='Erosion (mm/yr)', shrink=0.8)

cb2.set_ticks([0.25,0.75], labels=['Carbonate','Basement'])
cb3.set_ticks([0.0, 0.1,0.5*np.max(denud)*1e3])
# fig.tight_layout()
# plt.savefig(os.path.join(save_directory, id, f"elev_frac_space.png"), dpi=300)

# %%

im = rid[id_ts[1],:,:]
plt.figure()
plt.imshow(im, cmap='pink_r')

edges_dil = im - ndimage.binary_dilation(im)
edge_frac_dil = np.abs(np.sum(edges_dil[1:-1,1:-1]))/len(edges_dil[1:-1,1:-1].flatten())

edges_eros = im - ndimage.binary_erosion(im)
edge_frac_eros = np.abs(np.sum(edges_eros[1:-1,1:-1]))/len(edges_eros[1:-1,1:-1].flatten())

print(edge_frac_dil,edge_frac_eros)

plt.figure()
plt.imshow(edges_dil, cmap='pink')
plt.title(f'Limestone {fracs[1]}, fraction edges: {edge_frac_dil:0.3f}')
plt.savefig(os.path.join(save_directory, id, f"edges.png"), dpi=300)
# plt.colorbar()

# %%

# profiles along the way -- how much does additional water affect the channel form?
# maximum steady-state relief of the basement (based on chi and steepness)
# maximum steady-state relief across both (based on chi-p and steepness, assuming water is all lost and not returned)?

# seepage and seepage diffusive erosion
#%%

y_pos = fp["y"][1:-1].data
fig, axs = plt.subplots(ncols=3, nrows=2, figsize=(6,3))
for i, id_t in enumerate(id_ts):
    axs[0,i].fill_between(y_pos, np.max(elev[id_t,1:-1,1:-1], axis=1), y2=np.min(elev[id_t,1:-1,1:-1], axis=1), color='0.7', alpha=0.4)
    axs[0,i].plot(y_pos, np.min(elev[id_t,1:-1,1:-1], axis=1), color='k')
    axs[0,i].plot(y_pos, np.mean(elev[id_t,1:-1,1:-1], axis=1))
    axs[0,i].set_ylim((0,np.max(elev)))
    axs[1,i].imshow(rid[id_t,:,:], cmap='pink_r')
    axs[0,i].axis('off')
    axs[1,i].axis('off')
    axs[0,i].set_title(f'Limestone {fracs[i]}')
    
fig.tight_layout()
plt.savefig(os.path.join(save_directory, id, f"profile_frac_space.png"), dpi=300)
# %%
