
"""
Slightly more complex test of virtual karst in landlab. Now we will use 
the Lithology component to track two layers: a limestone layer, and a basement.
Use lithology difference to determine where there is loss in flow accumulation,
and use the lithologic contact to determine where there will be springs.
"""

#%%
import os
import copy
import pickle
import numpy as np

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import LightSource

from landlab import RasterModelGrid
from landlab.components import (
    FastscapeEroder, 
    LossyFlowAccumulator, 
    FlowDirectorD8,
    LakeMapperBarnes,
    LinearDiffuser,
    LithoLayers,
)

from virtual_karst_funcs import *

save_directory = '/Users/dlitwin/Documents/Research/Karst landscape evolution/landlab_virtual_karst/figures'

#%% parameters

U = 1e-4
K = 1e-5
D = 1e-3


#%% set up grid and lithology

mg = RasterModelGrid((100,150), xy_spacing=50)
mg.set_closed_boundaries_at_grid_edges(right_is_closed=True,
                                       left_is_closed=True,
                                       top_is_closed=False,
                                       bottom_is_closed=False)

bottom_nodes = mg.nodes_at_bottom_edge
top_nodes = mg.nodes_at_top_edge
z = mg.add_zeros("node", "topographic__elevation")
np.random.seed(10010)
z += 0.1*np.random.rand(len(z))

# two layers, both with bottoms below ground. Top layer is limestone, bottom is basement.
layer_elevations = [100,1000]
layer_ids = [0,1]
attrs = {"K_loss": {0: 0.5, 1: 0.0}}

lith = LithoLayers(
    mg, layer_elevations, layer_ids, function=lambda x, y: - 0.002 * y, attrs=attrs
)
# mg.imshow("rock_type__id", cmap="viridis")

# z bottom is the distance from the surface to the layer, so we want shorter distances
# at the top of the domain
mg.imshow(lith.z_bottom[1,:])

#%%
# define the loss based on the lithology dependent K_loss field
# this is everywhere at first, but decreases in extent once we erode 
# through to the basement
def loss_fxn(Qw, nodeID, linkID, grid):
    return (1. - grid.at_node['K_loss'][nodeID]) * Qw

# springs

# approach S1

def area_weighted_springs(mg, contact, Q_total, boundary_row=1, rock_id=1):
    """
    Find the spring discharge by weighting the total lost Q by the drainage area 
    of points along the contact between the lithologies. When the contact is not
    present (before we have eroded to the contact), the spring discharge is zero.

    Parameters
    ----------
    mg: ModelGrid
        Landlab ModelGrid object
    contact: array, lenth: nodes
        Boolean array of the lower contact at which spring discharge will occur.
    Q_total: float
        The total discharge lost calculated by summing the result of LossyFlowAccumulator
    boundary_row: int
        The row on which to check how much of the basement lithology is exposed.
        Default: 1 (the row above the boundary)
    rock_id: int
        The ID associated with the basement in the Litholayers model.
        Default: 1

    Returns
    --------
    Q_spring: array, length: nodes
    
    """

    Q_spring = mg.zeros("node")

    if np.sum(contact) == 0:
        return Q_spring
    else:
        area = mg.at_node['drainage_area']

        # make sure none of the nodes in contact is a flow 
        # reciever for another node in contact. Choose the 
        # upstream point if one is.
        fr = mg.at_node["flow__receiver_node"]
        nodes = np.arange(len(area))[contact]
        recievers = fr[contact]
        test = np.isin(nodes, recievers, invert=True)
        contact_1 = nodes[test]


        # normalize to the proportion of the domain length
        # where the basement has appeared. This effectively
        # allows some of the lost water to drain via subsurface
        # if the basement hasn't appeared.
        rid = mg.at_node['rock_type__id'].reshape(mg.shape)
        rid_bndry = rid[boundary_row,1:-1]
        Q_total_norm = Q_total * np.sum(rid_bndry==rock_id)/len(rid_bndry)

        # normalize point discharge based on the upslope area
        Ac = area[contact_1]
        Ac_norm = Ac / np.sum(Ac)
        Q_spring[contact_1] = Ac_norm * Q_total_norm       

    return Q_spring


#%% instantiate components 


q = mg.add_ones("node", "local_runoff")
fd2 = FlowDirectorD8(mg)
fa2 = LossyFlowAccumulator(
    mg,
    surface="topographic__elevation",
    flow_director=fd2,
    runoff_rate="local_runoff",
    loss_function=loss_fxn,
)
lmb2 = LakeMapperBarnes(
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

fs2 = FastscapeEroder(mg, K_sp=K, discharge_field='surface_water__discharge')
ld2 = LinearDiffuser(mg, linear_diffusivity=D)

lmb2.run_one_step()

Q = mg.at_node['surface_water__discharge']

#%% run forward

N = 4000
R = np.zeros(N)
dt = 500

a0 = mg.cell_area_at_node[mg.core_nodes][0]
area = mg.at_node['drainage_area']
q_loss = mg.at_node["surface_water__discharge_loss"]
lower_bndry = np.arange(0, mg.number_of_node_columns)

dz_ad = np.zeros(mg.size("node"))
dz_ad[mg.core_nodes] = U * dt
dz_ad[top_nodes] += 0.5 * U * dt # effectively decrease the baselevel fall rate of the upper boundary

save_freq = 50
Ns = N//save_freq
z_all = np.zeros((len(z),Ns))
rock_id_all = np.zeros((len(z),Ns))

for i in range(N):

    lmb2.run_one_step()

    # update areas
    fa2.run_one_step()

    # update topography
    fs2.run_one_step(dt)
    ld2.run_one_step(dt)
    z += dz_ad

    lith.rock_id = 0
    lith.dz_advection = dz_ad
    lith.run_one_step()

    # calculate spring discharge that will be used next time
    total_loss = np.sum(q_loss)
    contact = find_contact(mg, lith)
    lower_drainage = get_divide_mask(mg, lower_bndry)
    lower_contact = np.logical_and(contact, lower_drainage)
    Q_spring = area_weighted_springs(mg, lower_contact, total_loss)
    q[:] = 1 + Q_spring/a0

    # metrics of change
    R[i] = np.mean(z[mg.core_nodes])

    if i%save_freq==0:
        print(f"Finished iteration {i}")

        z_all[:,i//save_freq] = z
        rock_id_all[:,i//save_freq] = mg.at_node['rock_type__id']

# %% plot topogrpahic change

# topography
plt.figure()
mg.imshow("topographic__elevation", colorbar_label='Elevation [m]')
plt.savefig(os.path.join(save_directory,"litholayers_topog_springs_0.5.png"))

plt.figure()
mg.imshow("rock_type__id", cmap="viridis", colorbar_label='Rock ID')
plt.savefig(os.path.join(save_directory,"litholayers_rocks_springs_0.5.png"))

plt.figure()
mg.imshow('surface_water__discharge', cmap="plasma", colorbar_label='Discharge')
plt.savefig(os.path.join(save_directory,"litholayers_discharge_springs_0.5.png"))


#%%

Q_an = mg.at_node['surface_water__discharge']/mg.at_node['drainage_area']
plt.figure()
mg.imshow(contact, cmap="plasma", colorbar_label='Discharge')
# plt.savefig(os.path.join(save_directory,"litholayers_discharge_springs_0.5.png"))

#%%
plt.figure()
mg.imshow('local_runoff', cmap="plasma", colorbar_label='Discharge')
#%%
# hillshade

elev = mg.at_node['topographic__elevation'].copy()
elev[mg.boundary_nodes] = np.nan
y = np.arange(mg.shape[0] + 1) * mg.dx - mg.dx * 0.5
x = np.arange(mg.shape[1] + 1) * mg.dy - mg.dy * 0.5

elev_plot = elev.reshape(mg.shape)
elev_profile = np.nanmean(elev_plot, axis=1)

f, (ax0, ax1) = plt.subplots(1, 2, width_ratios=[4, 1], figsize=(10,5))
ls = LightSource(azdeg=135, altdeg=45)
ax0.imshow(
        ls.hillshade(elev_plot, 
            vert_exag=1, 
            dx=mg.dx, 
            dy=mg.dy), 
        origin="lower", 
        extent=(x[0], x[-1], y[0], y[-1]), 
        cmap='gray',
        )
for i in range(mg.shape[1]):
    ax1.plot(elev_plot[:,i], y[0:-1], alpha=0.1, color='k')
ax1.plot(elev_profile, y[0:-1], linewidth=2, color='r')

ax0.set_xlabel('X [m]')
ax0.set_ylabel('Y [m]')
ax1.set_xlabel('Z [m]')
f.tight_layout()
plt.savefig(os.path.join(save_directory,"litholayers_hillshade_springs_0.5_blevel.png"))

#%%

plt.figure()
zplot = z_all[1]
zplot[np.isnan(zplot)] = 0
mg.imshow(zplot)

#%%
plt.figure()
ridplot = rock_id_all[1]
ridplot[np.isnan(ridplot)] = 0
mg.imshow(ridplot)


#%% save all

for i in range(Ns):
    elev = z_all[:,i]

    elev[mg.boundary_nodes] = np.nan
    y = np.arange(mg.shape[0] + 1) * mg.dx - mg.dx * 0.5
    x = np.arange(mg.shape[1] + 1) * mg.dy - mg.dy * 0.5

    elev_plot = elev.reshape(mg.shape)
    elev_profile = np.nanmean(elev_plot, axis=1)

    f, (ax0, ax1) = plt.subplots(1, 2, width_ratios=[4, 1], figsize=(10,5))
    ls = LightSource(azdeg=135, altdeg=45)
    ax0.imshow(
            ls.hillshade(elev_plot, 
                vert_exag=1, 
                dx=mg.dx, 
                dy=mg.dy), 
            origin="lower", 
            extent=(x[0], x[-1], y[0], y[-1]), 
            cmap='gray',
            )
    for j in range(mg.shape[1]):
        ax1.plot(elev_plot[:,j], y[0:-1], alpha=0.1, color='k')
    ax1.plot(elev_profile, y[0:-1], linewidth=2, color='r')

    ax0.set_xlabel('X [m]')
    ax0.set_ylabel('Y [m]')
    ax1.set_xlabel('Z [m]')
    f.tight_layout()

    plt.savefig(os.path.join(save_directory,f"litholayers_hillshade_springs_0.5_%d.png"%i))
    plt.close()

# %% find the lithologic contact


plt.figure()
contact = find_contact(mg, lith)
mg.imshow(contact, cmap="plasma", colorbar_label='Rock contact')


#%% Lower boundary

lower_bndry = np.arange(0, mg.number_of_node_columns)
lower_drainage = get_divide_mask(mg, lower_bndry)
lower_contact = np.logical_and(contact, lower_drainage)

plt.figure()
mg.imshow(lower_contact, cmap="plasma", colorbar_label='Rock contact')


# %%

mg1 = copy.copy(mg)

N = 4000

elev_fun = lambda y, N, U, dt: 0.002 * y + N * U * dt - 100

surface = elev_fun(mg1.y_of_node, N, U, dt)

plt.figure()
mg1.imshow("rock_type__id")


elev_plot = z.copy()
elev_plot[np.isnan(elev_plot)] = 0.0
surface_plot = elev_plot > surface
plt.figure()
mg1.imshow(surface_plot)


# %%
plt.figure()
mg.imshow(lith.dz[1,:]>0)

# %%
plt.figure()
mg.imshow("rock_type__id", alpha=0.5)


# %% pickle grid and lithology model for gdp testing

pickle_dir = '/Users/dlitwin/Documents/Research/Karst landscape evolution/landlab_virtual_karst'
with open(os.path.join(pickle_dir,'lith.pkl'), 'wb') as file:
    pickle.dump(lith, file)

with open(os.path.join(pickle_dir,'grid.pkl'), 'wb') as file:
    pickle.dump(mg, file)
# %%
