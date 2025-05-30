import numpy as np

from landlab import RasterModelGrid
from landlab.components import (
    FastscapeEroder, 
    FlowAccumulator,

)

class ScarpProfileModel:

    def __init__(self, 
                 grid, 
                 K_sp=1e-5, 
                 m_sp=0.75, 
                 n_sp=1.5, 
                 exp_hack=1.67, 
                 coef_hack=6.69, 
                 Sc=None, 
                 z_spring=0.0, 
                 q_spring=0.0,
                 ):
    
        self.z = grid.at_node['topographic__elevation']
        self.x = grid.x_of_node
        self.y = grid.y_of_node
        self.dx = grid.spacing[0]
        self.core_row = self.y==self.dx

        self.exp_hack = exp_hack
        self.coef_hack = coef_hack
        self.Sc = Sc
        self.q_spring = q_spring
        self.z_spring = z_spring

        self.fa = FlowAccumulator(grid)
        self.fs = FastscapeEroder(grid, K_sp=K_sp, m_sp=m_sp, n_sp=n_sp, discharge_field='surface_water__discharge')

        self.area = grid.at_node['drainage_area']
        self.Q = grid.at_node['surface_water__discharge']

    def reduce_to_critical_slope(self):
        z_core = self.z[self.core_row]

        n = len(z_core)
        z_new = np.copy(z_core)
        
        for i in range(1, n):
            z_test = z_new[i-1] + self.Sc * self.dx
            z_new[i] = min(z_test, z_core[i])
        
        self.z[self.core_row] = z_new

    def run_step(self, dt):

        self.fa.run_one_step()
        x_dist = self.area/self.dx**2

        self.Q[:] = self.coef_hack * x_dist**self.exp_hack
        if self.q_spring > 0.0:
            self.Q[np.logical_and(self.core_row, self.z<=self.z_spring)] = self.Q[np.logical_and(self.core_row, self.z<=self.z_spring)] + self.q_spring

        self.fs.run_one_step(dt)
        if self.Sc is not None:
            self.reduce_to_critical_slope()

    def finalize_step(self):

        self.z_out = self.z[self.core_row]
        self.Q_out = self.Q[self.core_row]

    def run_model(self, T, dt):

        N = int(T//dt)
        self.x_pos = self.x[self.core_row]
        self.time = np.arange(0,T,dt)

        self.z_all = np.zeros((N,len(self.z[self.core_row])))
        self.Q_all = np.zeros((N,len(self.z[self.core_row])))

        for i in range(N):

            self.run_step(dt)
            self.finalize_step()

            self.z_all[i,:] = self.z_out
            self.Q_all[i,:] = self.Q_out
            

        self.x_divide = self.x_pos[np.argmax(self.z_all, axis=1)]
        if (self.x_divide == np.max(self.x_pos)).any():
            print('Scarp reached boundary. Velocity only calculated on points prior.')

        cond = self.x_divide<np.max(self.x_pos)
        self.v_divide, self.x0_divide = np.polyfit(self.time[cond], self.x_divide[cond], 1)