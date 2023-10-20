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

fig_directory = '/Users/dlitwin/Documents/Research/Karst landscape evolution/landlab_virtual_karst/figures'
save_directory = '/Users/dlitwin/Documents/Research Data/Local/karst_lem'
id = "flat_dynamic_ksat_2"

#%%

df_out = pd.read_csv(os.path.join(save_directory,id,f'{id}_output.csv'))
# compiling netcdfs into a format that paraview likes

grid_files = glob.glob(os.path.join(save_directory,id,'*.nc'))
files = sorted(grid_files, key=lambda x:int(x.split('_')[-1][:-3]))
iteration = int(files[-1].split('_')[-1][:-3])

for i in range(len(files)):
    grid = from_netcdf(files[i])
    write_raster_netcdf(os.path.join(save_directory,id,'paraview',f'{id}_grid_{i}.nc'),
                        grid,
                        time=df_out['time'].loc[i])
    


# %%
