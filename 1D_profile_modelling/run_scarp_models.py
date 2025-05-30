#%%

import numpy as np
import matplotlib.pyplot as plt

from landlab import RasterModelGrid
from scarp_profile_model import ScarpProfileModel
from itertools import product


#%%

dx = 25
mg = RasterModelGrid((3,500), xy_spacing=dx)
mg.set_closed_boundaries_at_grid_edges(right_is_closed=False,
                                       left_is_closed=False,
                                       top_is_closed=True,
                                       bottom_is_closed=True)

T = 1e6
dt = 500
K = 1e-3
m = 0.5
n = 1.5
hack_coef = 6.69
hack_exp = 1.67
Sc = 0.5

x = mg.x_of_node
y = mg.y_of_node
zp = 200
xp = 1e3
z_spring = 100
q_spring = 1e3
delta = 1e-4
z1 = np.zeros_like(x)
z1[x<xp] = (zp/xp) * x[x<xp]
z1[x>=xp] = zp - delta * x[x>=xp]
z = mg.add_ones("node", "topographic__elevation")
z[:] = z1


mdl = ScarpProfileModel(mg, K_sp=K, m_sp=m, n_sp=n, exp_hack=hack_exp, coef_hack=hack_coef, z_spring=z_spring, q_spring=q_spring)

mdl.run_model(T, dt)

# %%
N = int(T//dt)
ni = 200
plt.figure()
for j in range(int(N//ni)):
    plt.plot(mdl.x_pos, mdl.z_all[j*ni,:])

plt.figure()
plt.plot(mdl.time, mdl.x_divide, label='Model')
plt.plot(mdl.time, mdl.time*mdl.v_divide+mdl.x0_divide, label=f'regression: c={mdl.v_divide:.4e} m/yr')
plt.legend()



#%%
ni = 200
fig, ax = plt.subplots()
for j in range(int(N//ni)):
    z_plot = mdl.z_all[j*ni,:]
    cond = z_plot<=z_spring
    ax.plot(x[y==dx], z_plot)
    ax.plot(x[y==dx][cond], z_plot[cond], linewidth=3, color='b')


# %% Run many models across parameter space


def initialize_grid(Lx, dx, z_p, x_d, slope_p):

    Nx = int(Lx/dx)
    mg = RasterModelGrid((3,Nx), xy_spacing=dx)
    mg.set_closed_boundaries_at_grid_edges(right_is_closed=False,
                                        left_is_closed=False,
                                        top_is_closed=True,
                                        bottom_is_closed=True)
    z = mg.add_ones("node", "topographic__elevation")
    x = mg.x_of_node

    z1 = np.zeros_like(z)
    z1[x<x_d] = (z_p/x_d) * x[x<x_d]
    z1[x>=x_d] = z_p - slope_p * x[x>=x_d]
    z[:] = z1

    return mg



T = 1e6
dt = 500
Lx = 10000
dx = 50 #25
K = 1e-5
m = 0.5
n = 1.0
hack_coef = 6.69
hack_exp = 1.67
Sc = 1.0

z_p = 200
x_d = 1e3
z_spring = np.sort(200 - np.geomspace(5,195,10))
q_spring = 10**np.linspace(3,5,10)
Z_s, Q_s = np.meshgrid(z_spring, q_spring)
run_shape= Z_s.shape
Z_s = Z_s.flatten()
Q_s = Q_s.flatten()
delta = 1e-4

velocities = np.zeros_like(Z_s)
profiles = np.zeros((len(Z_s), int(Lx//dx)))

for i in range(len(Z_s)):

    mg = initialize_grid(Lx, dx, z_p, x_d, delta)

    mdl = ScarpProfileModel(mg, K_sp=K, m_sp=m, n_sp=n, 
                            exp_hack=hack_exp, coef_hack=hack_coef, 
                            z_spring=Z_s[i], q_spring=Q_s[i])

    mdl.run_model(T, dt)

    velocities[i] = mdl.v_divide
    profiles[i,:] = mdl.z_all[-1,:]

#%%
# null case:

mg = initialize_grid(Lx, dx, z_p, x_d, delta)
mdl = ScarpProfileModel(mg, K_sp=K, m_sp=m, n_sp=n, 
                        exp_hack=hack_exp, coef_hack=hack_coef, 
                        z_spring=0.0, q_spring=0.0)
mdl.run_model(T, dt)

vel_0 = mdl.v_divide
profile_0 = mdl.z_all[-1,:]


#%%

V_matrix = velocities.reshape(run_shape)
Q_matrix = Q_s.reshape(run_shape)
fig, ax = plt.subplots()
im = ax.imshow(V_matrix, cmap='viridis', aspect='equal', origin='lower')
# ax.set_xticks(list(z_spring))
ax.set_xticks(range(run_shape[0]))
ax.set_yticks(range(run_shape[1]))
ax.set_yticklabels([f'{x:.2e}' for x in q_spring])
ax.set_xticklabels([f'{x:.1f}' for x in z_spring])
ax.set_xlabel('Spring Elevation')
ax.set_ylabel('Spring Discharge')
fig.colorbar(im, label='Retreat Velocity (m/yr)')


# %%


# levels = np.arange(0,30,5)
V_matrix = velocities.reshape(run_shape)
Q_matrix = Q_s.reshape(run_shape)
fig, ax = plt.subplots()
im = ax.imshow((V_matrix-vel_0)/vel_0, cmap='viridis', aspect='equal', origin='lower')
# ax.contour((V_matrix/vel_0), levels, colors='k', origin='lower')
# ax.set_xticks(list(z_spring))
ax.set_xticks(range(0,run_shape[0],2))
ax.set_yticks(range(0,run_shape[1],2))
ax.set_yticklabels([f'{x:.1e}' for x in q_spring[::2]])
ax.set_xticklabels([f'{x:.1f}' for x in z_spring[::2]/z_p])
ax.set_xlabel('Spring Elevation / Plateau Elevation')
ax.set_ylabel('Spring Discharge')
fig.colorbar(im, label='Increase relative to no-spring velocity')
# %%

# levels = np.arange(0,30,5)
V_matrix = velocities.reshape(run_shape)
Q_matrix = Q_s.reshape(run_shape)
fig, ax = plt.subplots()
im = ax.imshow((V_matrix-vel_0)/vel_0, cmap='viridis', aspect='equal', origin='lower')
# ax.contour((V_matrix/vel_0), levels, colors='k', origin='lower')
# ax.set_xticks(list(z_spring))
ax.set_xticks(range(0,run_shape[0],2))
ax.set_yticks(range(0,run_shape[1],2))
ax.set_yticklabels([f'{x:.1f}' for x in q_spring[::2]/mdl.Q_out.max()])
ax.set_xticklabels([f'{x:.2f}' for x in z_spring[::2]/z_p])
ax.set_xlabel('Spring Elevation / Plateau Elevation')
ax.set_ylabel('Spring Discharge / Precipitation')
fig.colorbar(im, label='Increase relative to no-spring velocity')
# %%

Qs = Q_s.flatten()
Zs = Z_s.flatten()

# rel_vel = ((V_matrix-vel_0)/vel_0).flatten()
rel_vel = ((V_matrix)/vel_0).flatten()

power_delivered = (np.log(Qs) * (Zs/z_p)**2)
# power_delivered = ((Qs/mdl.Q_out.max())**(1/8) * (Zs/z_p)**2)
# power_delivered = (np.log(Qs) * np.log(Zs))

fig, ax = plt.subplots(figsize=(6,4))
sc = ax.scatter(power_delivered, rel_vel, c=Zs/z_p, s=0.001*Qs)
ax.set_xlabel(r'$(Z_s/Z_p)^2 \log(Q_s) $')
ax.set_ylabel(r'$V_d/V_{d,0}$')
ax.set_yscale('log')
fig.colorbar(sc, label=r'$Z_s/Z_p$')

# Find the smallest, median, and largest Qs values
qs_sorted = np.sort(Qs)
qs_legend = [qs_sorted[0], np.median(qs_sorted), qs_sorted[-1]]

# Convert to scatter "s" parameter (area in points^2)
msizes = [0.001 * q for q in qs_legend]

# Create dummy plot handles with desired sizes
# You can set the color as needed; here, 'C0' is the default matplotlib blue
handles = [
    plt.Line2D([], [], marker='o', linestyle='', color='C0', markersize=np.sqrt(s), 
               label=f'{q:.0f} m³/s')
    for s, q in zip(msizes, qs_legend)
]
# Add legend
plt.legend(handles=handles, title='Spring Discharge (m³/s)', fontsize=10, title_fontsize=11)

# %% Model runs with lost water (distributed, as a fraction of recharge)
