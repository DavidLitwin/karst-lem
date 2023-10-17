

import numpy as np

from landlab.utils import get_watershed_mask
from landlab.grid.mappers import map_max_of_node_links_to_node

def find_nearest(array, value):
    """Return index and value nearest to a value in an array."""
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx, array[idx]

def find_all_near(array, value, tol):
    """Return indices and values that are within tolerance of a value in an array"""
    array = np.asarray(array)
    select = np.abs(array-value) < tol
    inds = np.argwhere(select)
    
    return inds, array[inds]


def get_divide_mask(mg, bnds_lower):
    """
    Create a mask that selects just the points that drain to the lower boundary.
    """
    area = mg.at_node['drainage_area']
    lower_mask = np.zeros(len(area))

    for i in bnds_lower:
        mask = get_watershed_mask(mg, i)
        lower_mask += mask
        
    return lower_mask


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
        mask = get_watershed_mask(mg, i)
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


def find_contact(mg, lith, rock_id=1):
    """
    Find the nodes at the contact between lithologies. Select the nodes on the 
    rock_id side of the contact. 
    """
    # calculate gradient of rock type
    grad = mg.calc_grad_at_link((lith.dz[rock_id,:] > 1e-2)*1)

    # set links adjacent to boundaries to zero
    stat_nal = mg.status_at_node[mg.nodes_at_link]
    ind_nal = np.sum(stat_nal, axis=1) > 0.0
    grad[ind_nal] = 0

    # map to nodes
    grad_node = map_max_of_node_links_to_node(mg, np.abs(grad))

    # take nodes on the rock_id side 
    rid = mg.at_node["rock_type__id"]
    grad_intersect = np.logical_and(grad_node>0, rid==rock_id)

    return grad_intersect
    
def get_lower_upper_water_flux(mg, bnds_lower, bnds_upper):
    """sum discharge on lower and upper boundaries"""

    Q = mg.at_node['surface_water__discharge']
    return np.sum(Q[bnds_lower]), np.sum(Q[bnds_upper])

def get_lower_upper_area(mg, bnds_lower, bnds_upper):
    """sum area on lower and upper boundaries"""

    area = mg.at_node['drainage_area']
    return np.sum(area[bnds_lower]), np.sum(area[bnds_upper])

