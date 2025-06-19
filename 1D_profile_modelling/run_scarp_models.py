#%%

import numpy as np
import matplotlib.pyplot as plt

from landlab import RasterModelGrid
from scarp_profile_model import ScarpProfileModel
from itertools import product


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


#%%

dx = 25
mg = RasterModelGrid((3,1000), xy_spacing=dx)
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


mdl = ScarpProfileModel(mg, K_sp=K, m_sp=m, n_sp=n, exp_hack=hack_exp, coef_hack=hack_coef, z_contact=z_spring, q_spring=q_spring)

mdl.run_model(T, dt)

# %%
N = int(T//dt)
ni = 200

num_lines = int(N // ni)
cmap = plt.get_cmap('viridis')

plt.figure()
for j in range(num_lines):
    color = cmap(j / (num_lines - 1)) if num_lines > 1 else cmap(0.5)
    plt.plot(mdl.x_pos, mdl.z_all[j*ni, :], color=color)
plt.xlabel('Distance (m)')
plt.ylabel('Height (m)')

plt.figure()
plt.plot(mdl.time, mdl.x_divide, label='Model')
plt.plot(mdl.time, mdl.time*mdl.v_divide+mdl.x0_divide, label=f'regression: c={mdl.v_divide:.4e} m/yr')
plt.legend()
plt.xlabel('Time (yr)')
plt.ylabel('Divide x position (m)')


#%%

plt.figure()
plt.plot(mdl.time, mdl.A_contact_all, label='Model')
plt.legend()
plt.xlabel('Time (yr)')
plt.ylabel('Drainage area at contact (m2)')

#%%


ni = 200
fig, ax = plt.subplots()
for j in range(num_lines):
    color = cmap(j / (num_lines - 1)) if num_lines > 1 else cmap(0.5)
    z_plot = mdl.z_all[j*ni,:]
    cond = z_plot<=z_spring
    ax.plot(x[y==dx], z_plot, color=color)
    ax.plot(x[y==dx][cond], z_plot[cond], linewidth=3, color='b')
plt.xlabel('Distance (m)')
plt.ylabel('Height (m)')

# %% Run many models across parameter space

T = 1e6
dt = 500
Lx = 5000
dx = 25 #50 #25
K = 1e-5
m = 0.5
n = 1.0
hack_coef = 6.69
hack_exp = 1.67
Sc = 1.0

z_p = 200
x_d = 1e3
z_contact = np.sort(z_p - np.geomspace(0.025*z_p,0.975*z_p,10))
q_spring = 10**np.linspace(3,5,10)
Z_s, Q_s = np.meshgrid(z_contact, q_spring)
run_shape= Z_s.shape
Z_s = Z_s.flatten()
Q_s = Q_s.flatten()
delta = 1e-4

velocities = np.zeros_like(Z_s)
profiles = np.zeros((len(Z_s), int(Lx//dx)))
a_contact = np.zeros_like(Z_s)

for i in range(len(Z_s)):

    mg = initialize_grid(Lx, dx, z_p, x_d, delta)

    mdl = ScarpProfileModel(mg, K_sp=K, m_sp=m, n_sp=n, 
                            exp_hack=hack_exp, coef_hack=hack_coef, 
                            z_contact=Z_s[i], q_spring=Q_s[i])

    mdl.run_model(T, dt)

    velocities[i] = mdl.v_divide
    profiles[i,:] = mdl.z_all[-1,:]
    a_contact[i] = np.mean(mdl.A_contact_all[mdl.time > 0.8 * T])

#%%
# null case:

mg = initialize_grid(Lx, dx, z_p, x_d, delta)
mdl = ScarpProfileModel(mg, K_sp=K, m_sp=m, n_sp=n, 
                        exp_hack=hack_exp, coef_hack=hack_coef, 
                        z_contact=0.0, q_spring=0.0)
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
ax.set_xticklabels([f'{x:.1f}' for x in z_contact])
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
ax.set_xticklabels([f'{x:.1f}' for x in z_contact[::2]/z_p])
ax.set_xlabel('Spring Elevation / Plateau Elevation')
ax.set_ylabel('Spring Discharge')
fig.colorbar(im, label='Increase relative to no-spring velocity')

# %%

Qs = Q_s.flatten()
Zs = Z_s.flatten()

# rel_vel = ((V_matrix-vel_0)/vel_0).flatten()
rel_vel = ((V_matrix)/vel_0).flatten()

power_delivered = (np.log(Qs) * (Zs/z_p)**2)
# power_delivered = (Qs**0.1 * (Zs/z_p)**2)
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

#%% Profile concavity

plt.figure()
for i in range(profiles.shape[0]):
    plt.plot(profiles[i,:], alpha=0.1)
plt.plot(profile_0, color='k')

# %% Model runs with lost water (distributed, as a fraction of recharge)



T = 1e7
dt = 5000
Lx = 10000
dx = 50 #25
K = 1e-5
m = 0.5
n = 1.0
hack_coef = 6.69
hack_exp = 1.67
Sc = 1.0
r_basement = 1.0

z_p = 200
x_d = 1e3
z_contact = np.sort(z_p - np.geomspace(0.025*z_p,0.975*z_p,10))
r_limestone = np.linspace(0.2,2,10)
Z_l, R_l = np.meshgrid(z_contact, r_limestone)
run_shape= Z_l.shape
Z_l = Z_l.flatten()
R_l = R_l.flatten()
delta = 1e-4

velocities = np.zeros_like(Z_l)
profiles = np.zeros((len(Z_l), int(Lx//dx)))

for i in range(len(Z_l)):

    mg = initialize_grid(Lx, dx, z_p, x_d, delta)

    mdl = ScarpProfileModel(mg, K_sp=K, m_sp=m, n_sp=n, 
                            exp_hack=hack_exp, coef_hack=hack_coef, 
                            z_contact=Z_l[i], q_spring=0,
                            r_limestone=R_l[0], r_basement=r_basement)

    mdl.run_model(T, dt)

    velocities[i] = mdl.v_divide
    profiles[i,:] = mdl.z_all[-1,:]

#%%
# null case:

mg = initialize_grid(Lx, dx, z_p, x_d, delta)
mdl = ScarpProfileModel(mg, K_sp=K, m_sp=m, n_sp=n, 
                        exp_hack=hack_exp, coef_hack=hack_coef, 
                        z_contact=0.0, q_spring=0.0,
                        r_limestone=1.0, r_basement=1.0)
mdl.run_model(T, dt)

vel_0 = mdl.v_divide
profile_0 = mdl.z_all[-1,:]

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
