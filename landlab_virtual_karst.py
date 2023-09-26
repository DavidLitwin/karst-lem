
"""
Simple test with landlab, moving mass around
"""

#%%

import os
import numpy as np

import matplotlib.pyplot as plt
from matplotlib import cm

from landlab import RasterModelGrid, imshow_grid
from landlab.components import (
    FastscapeEroder, 
    FlowAccumulator,
    LossyFlowAccumulator, 
    FlowDirectorD8,
    LakeMapperBarnes,
    LinearDiffuser,
)
from landlab.utils import get_watershed_mask


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx, array[idx]

def find_all_near(array, value, tol):
    
    array = np.asarray(array)
    select = np.abs(array-value) < tol
    inds = np.argwhere(select)
    
    return inds, array[inds]

def get_mask_and_large_channels(mg, bnds_lower, bnds_upper, area_thresh):
    """
    Create a mask that selects just the points that drain to the lower boundary.
    Make a dictionary of the large channels draining to the lower boundary and the 
    upper boundary.

    """
    area = mg.at_node['drainage_area']
    lower_mask = np.zeros(len(area))

    lower_channels = {}
    for i in bnds_lower:
        mask = get_watershed_mask(mg,i)
        lower_mask += mask
        
        channel = np.argwhere(np.logical_and(mask, area>area_thresh))

        if channel.size > 0:
            lower_channels[i] = channel
    
    upper_channels = {}
    for i in bnds_upper:
        mask = get_watershed_mask(mg,i)
        channel = np.argwhere(np.logical_and(mask, area>area_thresh))

        if channel.size > 0:
            upper_channels[i] = channel

    return lower_mask, lower_channels, upper_channels

save_directory = '/Users/dlitwin/Documents/Research/karst_lem/landlab_virtual_karst/figures'


#%%

U = 1e-4
K = 1e-5
D = 1e-3

# %% New better idea on a two-boundary grid

mg = RasterModelGrid((100,100), xy_spacing=50)
mg.set_closed_boundaries_at_grid_edges(right_is_closed=True,
                                       left_is_closed=True,
                                       top_is_closed=False,
                                       bottom_is_closed=False)

z = mg.add_zeros("node", "topographic__elevation")
np.random.seed(10010)
z += 0.1*np.random.rand(len(z))

q = mg.add_ones("node", "local_runoff")
# q[:] = mg.cell_area_at_node


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

fs = FastscapeEroder(mg, K_sp=K, discharge_field='surface_water__discharge')
ld = LinearDiffuser(mg, linear_diffusivity=D)

lmb.run_one_step()

#%%

plt.figure()
imshow_grid(mg,mg.status_at_node)
# %% Steady state topography draining normally top right and bottom left

dt = 500
N = 4000

R = np.zeros(N)
for i in range(N):

    
    fa.run_one_step()
    fs.run_one_step(dt)
    ld.run_one_step(dt)

    z[mg.core_nodes] += U * dt

    R[i] = np.mean(z[mg.core_nodes])

    if i%1000==0:
        print(f"Finished iteration {i}")

z0 = z.copy()

plt.figure()
imshow_grid(mg,"topographic__elevation")
# imshow_grid(mg,"drainage_area")

plt.figure()
plt.plot(R)

#%% Approach 1: pick specific points to move water around

area = mg.at_node['drainage_area']
node_ids = np.arange(len(area))

s = 4

# pick top
t_inds = mg.y_of_node == 4200
t_ids = node_ids[t_inds]
t_area_inds = np.argsort(area[t_inds])[-s:]
t_inds = t_ids[t_area_inds]
x_t = mg.x_of_node[t_inds]
y_t = mg.y_of_node[t_inds]


# pick bottom
b_inds = mg.y_of_node == 800
b_ids = node_ids[b_inds]
b_area_inds = np.argsort(area[b_inds])[-s:]
b_inds = b_ids[b_area_inds]
x_b = mg.x_of_node[b_inds]
y_b = mg.y_of_node[b_inds]

plt.figure()
imshow_grid(mg,"topographic__elevation")
plt.scatter(x_b,y_b)
plt.scatter(x_t,y_t)


# %% instantiate

mg1 = RasterModelGrid((100,100), xy_spacing=50)
mg1.set_closed_boundaries_at_grid_edges(right_is_closed=True,
                                       left_is_closed=True,
                                       top_is_closed=False,
                                       bottom_is_closed=False)


# new discharge from previous state
z1 = mg1.add_zeros("node", "topographic__elevation")
z1[:] = z0

# instantiate a local runoff field
q = mg1.add_ones("node", "local_runoff")
# q[:] = mg.cell_area_at_node

fd1 = FlowDirectorD8(mg1)
fa1 = FlowAccumulator(
    mg1,
    surface="topographic__elevation",
    flow_director=fd1,
    runoff_rate="local_runoff",
)
lmb1 = LakeMapperBarnes(
    mg1,
    method="Steepest",
    fill_flat=False,
    surface="topographic__elevation",
    fill_surface="topographic__elevation",
    redirect_flow_steepest_descent=False,
    reaccumulate_flow=False,
    track_lakes=False,
    ignore_overfill=True,
)

fs1 = FastscapeEroder(mg1, K_sp=K, discharge_field='surface_water__discharge')
ld1 = LinearDiffuser(mg1, linear_diffusivity=D)

Q = mg1.at_node['surface_water__discharge']
q0 = Q.copy()

#%% run forward

N = 10000
f = 0.8


b_areas = np.zeros((N,s))
t_areas = np.zeros((N,s))
a0 = mg.cell_area_at_node[mg.core_nodes][0]

R = np.zeros(N)
dt = 50
for i in range(N):

    # find new areas
    fa1.run_one_step()

    area = mg1.at_node['drainage_area']
    # update drainage area (run FA)
    # find drainage areas at points
    b_areas[i,:] = area[b_inds]/a0
    t_areas[i,:] = area[t_inds]/a0

    # add and subtract local discharge
    q[b_inds] = area[t_inds]/a0 * f
    q[t_inds] = - area[t_inds]/a0 * f * 0.1

    # update discharge
    fa1.run_one_step()
    fs1.run_one_step(dt)
    ld1.run_one_step(dt)


    z1[mg1.core_nodes] += U * dt

    R[i] = np.mean(z1[mg1.core_nodes])

    if i%1000==0:
        print(f"Finished iteration {i}")


#%% Plot results

clrs_t = cm.Reds(np.linspace(0.2, 1, s))
clrs_b = cm.Blues(np.linspace(0.2, 1, s))
plt.figure()
for k in range(s):
    plt.plot(b_areas[:,k]-b_areas[0,k], color=clrs_b[k], label=f'Bottom {k}')
    plt.plot(t_areas[:,k]-t_areas[0,k], color=clrs_t[k], label=f'Top {k}')
plt.legend()

# Relief change and topography
plt.figure()
plt.plot(R)

plt.figure()
imshow_grid(mg1,"topographic__elevation")
# imshow_grid(mg1,"surface_water__discharge")
plt.scatter(x_b,y_b)
plt.scatter(x_t,y_t)

# elevation difference
plt.figure()
imshow_grid(mg1,z1-z0, symmetric_cbar=True, cmap='RdBu')
# imshow_grid(mg1,"surface_water__discharge")
plt.scatter(x_b,y_b)
plt.scatter(x_t,y_t)

# runoff
plt.figure()
imshow_grid(mg1,"local_runoff", cmap='plasma')
# imshow_grid(mg1,"surface_water__discharge", cmap='plasma')
# plt.scatter(x_b,y_b)
# plt.scatter(x_t,y_t)

# average profiles
y = mg.y_of_node.reshape(mg.shape)
z1p = z1.reshape(mg.shape)
z0p = z0.reshape(mg.shape)
plt.figure()
plt.plot(y[:,0], np.mean(z1p, axis=1), label='karst')
plt.plot(y[:,0], np.mean(z0p, axis=1), label='equilibrium' )
plt.legend(frameon=False)

#runoff difference from initial steady state
q1 = mg1.at_node['surface_water__discharge']
plt.figure()
imshow_grid(mg1,q1-q0, symmetric_cbar=True, cmap='RdBu')

#%% Approach 2: use LossyFlowAccumulator and identify springs at a given elevation

# get basins drainging one direction or the other
f = mg.at_node['flow__upstream_node_order']
area = mg.at_node['drainage_area']
bnds = mg.fixed_value_boundary_nodes
bnds_lower = bnds[0:mg.shape[0]]
bnds_upper = bnds[mg.shape[0]:]
atot = mg.number_of_core_nodes * mg.dx**2

a_thresh = 0.005*atot
lower_mask, lower_channels, upper_channels = get_mask_and_large_channels(mg, bnds_lower, bnds_upper, area_thresh=a_thresh)
plt.figure()
imshow_grid(mg,lower_mask)

plt.figure()
imshow_grid(mg,area>a_thresh)

#%% make grid and find the places to lose and gain water

mg2 = RasterModelGrid((100,100), xy_spacing=50)
mg2.set_closed_boundaries_at_grid_edges(right_is_closed=True,
                                       left_is_closed=True,
                                       top_is_closed=False,
                                       bottom_is_closed=False)

# new elevation from previous state
z2 = mg2.add_zeros("node", "topographic__elevation")
z2[:] = z0

# instantiate a local runoff field
q = mg2.add_ones("node", "local_runoff")

# define the loss from the lower area
loss = mg2.add_zeros("node", "loss_field")
y = mg2.y_of_node
loss[np.logical_and(y>=1000,y<=2000)] = 0.3
def loss_fxn(Qw, nodeID, linkID, grid):
    return (1. - grid.at_node['loss_field'][nodeID]) * Qw

# find springs
z_springs = 19
spring_pts = []
for key, arr in upper_channels.items():
    idx, zs = find_nearest(z2[arr], z_springs)
    if np.abs(z_springs-zs) < 2:
        spring_pts.append(arr[idx][0])

# plot the loss window, springs, and elevation
plt.figure()
imshow_grid(mg2,z2)
imshow_grid(mg2, loss, alpha=0.2, cmap='viridis')
plt.scatter(mg2.x_of_node[spring_pts], mg2.y_of_node[spring_pts])

#%% instantiate components 

fd2 = FlowDirectorD8(mg2)
fa2 = LossyFlowAccumulator(
    mg2,
    surface="topographic__elevation",
    flow_director=fd2,
    runoff_rate="local_runoff",
    loss_function=loss_fxn,
)
lmb2 = LakeMapperBarnes(
    mg2,
    method="Steepest",
    fill_flat=False,
    surface="topographic__elevation",
    fill_surface="topographic__elevation",
    redirect_flow_steepest_descent=False,
    reaccumulate_flow=False,
    track_lakes=False,
    ignore_overfill=True,
)

fs2 = FastscapeEroder(mg2, K_sp=K, discharge_field='surface_water__discharge')
ld2 = LinearDiffuser(mg2, linear_diffusivity=D)

Q = mg2.at_node['surface_water__discharge']

#%% run forward

N = 2000

a0 = mg2.cell_area_at_node[mg.core_nodes][0]
area = mg2.at_node['drainage_area']
q_loss = mg2.at_node["surface_water__discharge_loss"]

R = np.zeros(N)
dt = 50
for i in range(N):

    # update areas
    fa2.run_one_step()

    # update topography
    fs2.run_one_step(dt)
    ld2.run_one_step(dt)
    z2[mg2.core_nodes] += U * dt

    # calculate spring discharge that will be used next time
    total_loss = np.sum(q_loss)
    q[spring_pts] = 1 + total_loss/(len(spring_pts)*a0)

    # metrics of change
    R[i] = np.mean(z2[mg2.core_nodes])

    if i%100==0:
        print(f"Finished iteration {i}")

# %% plot topogrpahic change

# topography
plt.figure()
imshow_grid(mg2,z2, colorbar_label='Elevation [m]')
#%%

# topographic change
plt.figure()
imshow_grid(mg2, z2-z0, symmetric_cbar=True, cmap='RdBu', colorbar_label='Elevation change [m]')
plt.scatter(mg2.x_of_node[spring_pts], mg2.y_of_node[spring_pts])
plt.axhline(1000)
plt.axhline(2000)

#%% Plot the channel change

arr_max = 0
for key, arr in upper_channels.items():
    if len(arr) > arr_max:
        arr_max = len(arr)
        upper_key = key
arr_max = 0
for key, arr in lower_channels.items():
    if len(arr) > arr_max:
        arr_max = len(arr)
        lower_key = key
lower_basin = lower_channels[lower_key]
upper_basin = upper_channels[upper_key]


fig, axs = plt.subplots(nrows=2)
axs[1].scatter(mg2.y_of_node[lower_basin], z0[lower_basin], s=4, label='Steady State')
axs[1].scatter(mg2.y_of_node[lower_basin], z2[lower_basin], s=4, label='Post')
axs[1].set_ylabel('Elevation [m]')
axs[1].set_xlabel('y-position [m]')
axs[1].set_title('Lower (losing) Basin')
axs[1].legend(frameon=False)
axs[0].scatter(mg2.y_of_node[upper_basin], z0[upper_basin], s=4, label='Steady State')
axs[0].scatter(mg2.y_of_node[upper_basin], z2[upper_basin], s=4, label='Post')
axs[0].set_ylabel('Elevation [m]')
axs[0].set_xlabel('y-position [m]')
axs[0].set_title('Upper (gaining) Basin')
plt.tight_layout()
# %%

# motion of drainage divide
lower_mask2, _, _ = get_mask_and_large_channels(mg2, bnds_lower, bnds_upper, area_thresh=a_thresh)

diff = np.logical_xor(lower_mask, lower_mask2)

plt.figure()
imshow_grid(mg2,z0)
imshow_grid(mg2,diff, cmap='Blues', alpha=0.6)
# %%

