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
from scipy import ndimage

fig_directory = '/Users/dlitwin/Documents/Research/Karst landscape evolution/landlab_virtual_karst/figures'
save_directory = '/Users/dlitwin/Documents/Research Data/Local/karst_lem'
id = "virtual_karst_null_2"

#%%

# file = os.path.join(save_directory,id,f'grid_{id}.nc')
file = os.path.join(save_directory,id,f'{id}.nc')
fp = Dataset(file, "r", format="NETCDF4")
elev = fp.variables['topographic__elevation'][:,:,:].data
rid = fp.variables['rock_type__id'][:,:,:].data


#%%

fracs = [0.9, 0.5, 0.1]
id_ts = [np.argmin(np.absolute(frac_limestone-i)) for i in fracs]

t = fp['time'][:].data
max_elev = np.max(elev[:,1:-1,1:-1], axis=(1,2))
frac_limestone = 1 - np.mean(rid[:,1:-1,1:-1], axis=(1,2))
fig, ax = plt.subplots(figsize=(5,3.5))
ax.plot(t, max_elev, label='Max elevation', color='r')
ax.scatter(t[id_ts], max_elev[id_ts], color='r', s=10)
ax.plot(t, fp.U*t*1e3*fp.save_freq-fp.b_limestone, label='Contact elevation', color='r', linestyle='--')
ax.set_ylabel('Elevation (m)', color='r')
ax1 = ax.twinx()
ax1.plot(t, frac_limestone, label='Limestone', color='b')
ax1.scatter(t[id_ts], frac_limestone[id_ts], color='b', s=10)
ax1.set_ylabel('Fraction exposed (-)', color='b')
fig.legend(ncols=2, loc=(0.2,0.86), frameon=False)
ax.set_xlabel('Time (kyr)')
plt.savefig(os.path.join(save_directory, id, f"elev_frac_time.png"), dpi=300)

# %%

fig, axs = plt.subplots(ncols=3, nrows=2, figsize=(6,3))
for i, id_t in enumerate(id_ts):
    axs[0,i].imshow(elev[id_t,:,:], cmap='gist_earth', vmax=np.max(elev), vmin=0)
    axs[1,i].imshow(rid[id_t,:,:], cmap='pink_r')
    axs[0,i].axis('off')
    axs[1,i].axis('off')
    axs[0,i].set_title(f'Limestone {fracs[i]}')
    
fig.tight_layout()
plt.savefig(os.path.join(save_directory, id, f"elev_frac_space.png"), dpi=300)

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
plt.imshow(edges_eros, cmap='pink_r')
# plt.colorbar()

# %%
