
import copy
import numpy as np

from landlab.utils import get_watershed_mask
from landlab.grid.mappers import map_max_of_node_links_to_node
from landlab.components import (
    FlowAccumulator,
    LakeMapperBarnes,
)

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


def generate_conditioned_surface(mg, dip_func=lambda x, y: 0*x + 0*y, noise_coeff=0.01, random_seed=1123):

    mg1 = copy.copy(mg)
    z_surf = mg1.add_zeros("node", "surface__elevation")
    np.random.seed(random_seed)
    z_surf += dip_func(mg1.x_of_node,mg1.y_of_node) + noise_coeff * np.random.rand(len(z_surf))
    z_surf0 = z_surf.copy()

    # flow management for the topographic surface, routing only infiltration excess
    fa1 = FlowAccumulator(
        mg1,
        surface="surface__elevation",
        flow_director='FlowDirectorD8',
    )
    lmb1 = LakeMapperBarnes(
        mg1,
        method="D8",
        fill_flat=False,
        surface="surface__elevation",
        fill_surface="surface__elevation",
        redirect_flow_steepest_descent=False,
        reaccumulate_flow=False,
        track_lakes=False,
        ignore_overfill=True,
    )
    lmb1.run_one_step()
    fa1.run_one_step()

    print(np.sum(np.abs(z_surf0-z_surf)))
    return z_surf



def calc_ksat(n, D, alpha=1.0, rho=1000, g=9.81, mu=0.0010518):
    """
    Calculate hydraulic conductivity (m/s) from the parallel capillary model (See Vacher and Mylroie 2002).
    Follows Kozeny-Karman equation.
    
    Parameters:
    -----------
    n: porosity (-)
    D: Equivalent pore diameter (m)
    alpha: toruosity (-). Default = 1.
    rho: density of water (kg/m3).
    g: gravitational acceleration (m/s2).
    mu: dynamic viscosity (N*s/m2)
    """    

    return (rho * g * n * D**2)/(32 * mu * alpha**2)


def calc_pore_diam_logistic(t, t0, k, D0, Df):
    """
    Calculate the equivalent capillary pore diameter assuming a logistic increase in
    diameter with time.

    Parameters:
    ----------
    t: time (T)
    t0: logistic midpoint (T)
    k: logistic growth rate/steepness (1/T)
    D0: initial diameter (L)
    Df: final diameter (L)

    """
    return (Df-D0) / (1 + np.exp(-k * (t - t0))) + D0

def calc_porosity_logistic(t, t0, k, D0, Df, n0, nf):
    """
    Calculate porosity assuming a linear increase in porosity with eqivalent pore diameter, and a 
    logistic increase in equivalent pore diameter with time.

    Parameters:
    ----------
    t: time (T)
    t0: logistic midpoint (T)
    k: logistic growth rate/steepness (1/T)
    D0: initial diameter (L)
    Df: final diameter (L)
    n0: initial porosity (-)
    nf: final porosity (-)
    """

    return (nf - n0)/(Df - D0) * ((Df - D0) / (1 + np.exp(-k * (t - t0)))) + n0

def calc_ceq(Tc, gam_ca, gam_hco3, Pco2):
    """
    inputs:
    K1, Kc, Kh: fast reaction equilibrium coefficients
    gam_ca: activity coefficient for calcium
    gam_hco3: activity coefficient for bicarbonate
    Pco2: carbon dioxide partial pressure
    
    returns:
    ceq: calcium equilibrium concentration
    """

    T = 273.16 + Tc
    K1 = np.exp(- 356.3094 - 0.06091964*T + 21834.37/T + 126.8339*np.log(T) - 1684915 / T)
    K2 = np.exp(- 107.8871 - 0.03252849*T + 5151.79/T + 38.92561*np.log(T) - 563713.9 / T)
    Kc = np.exp(-171.9065 - 0.077993*T + 2839.319/T + 71.595*np.log(T))
    Kh = np.exp(108.3865 + 0.01985076*T - 6919.53/T - 40.45154*np.log(T) + 669365 / T)

    ceq = ((K1*Kc*Kh*Pco2)/(4*K2*gam_ca*gam_hco3))**(1/3)
    return ceq

def calc_max_chem_denudation(Tc, Pco2, rho, P, AET):
    """
    inputs:
    K1, Kc, Kh: fast reaction equilibrium coefficients
    Pco2: carbon dioxide partial pressure
    Tc: Temperature (deg C)
    P: precipitation (m/yr)
    AET: actual evapotranspiration (m/yr)
    
    returns:
    E: chemical denudation (m/yr)
    """

    T = 273.16 + Tc
    K1 = np.exp(- 356.3094 - 0.06091964*T + 21834.37/T + 126.8339*np.log(T) - 1684915 / T)
    K2 = np.exp(- 107.8871 - 0.03252849*T + 5151.79/T + 38.92561*np.log(T) - 563713.9 / T)
    Kc = np.exp(-171.9065 - 0.077993*T + 2839.319/T + 71.595*np.log(T))
    Kh = np.exp(108.3865 + 0.01985076*T - 6919.53/T - 40.45154*np.log(T) + 669365 / T)

    E = 1/(10 * rho) * ((Kc*K1*Kh*Pco2)/(K2*4))**(1/3) * (P - AET)
    return E


def calc_T_lapse(z, Tc0=15, Lr=6.5):
    """
    inputs:
    z: topographic elevation (m)
    T0: temperature at sealevel (dec C) Default=15 (Standard Sealevel conditions)
    Lr: lapse rate (dec C/km)
    """

    # lapse rate based on international standard atmosphere
    Tc = Tc0 - Lr * z/1000 
    return Tc

def calc_P_gaussian(z, P0=0.5, A=2.0, Hp=2000, Dv=1000):
    """
    inputs:
    z: topographic elevation (m)
    P0: precipitation at sealevel (m/yr)
    A: maximum additional orographic precipitation (m/yr)
    Hp: elevation of maximum precipitation (m)
    Dv: width of gradient (m)
    """

    # Precipitation Gaussian (Zavala et al 2020, Basin Research)
    P = P0 + A * np.exp((z-Hp)**2/(2*Dv**2))
    
    return P

# def calc_AET_linear(z, P, AET0, ):
#     """
#     inputs:
#     z: topographic elevation (m)
#     P0: precipitation at sealevel (m/yr)
#     A: maximum additional orographic precipitation (m/yr)
#     Hp: elevation of maximum precipitation (m)
#     Dv: width of gradient (m)
#     """

#     # Precipitation Gaussian (Zavala et al 2020, Basin Research)
#     P = P0 + A * np.exp((z-Hp)**2/(2*Dv**2))
    
#     return P