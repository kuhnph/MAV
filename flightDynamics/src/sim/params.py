import numpy as np
from sim.rotations import body_to_inertial


class Params:
    """Aircraft, environment, and simulation configuration."""

    def __init__(self):
        self._simulation()
        self._environment()
        self._wind()
        self._aircraft()
        self._longitudinal_aerodynamics()
        self._lateral_aerodynamics()
        self._derived_constants()
        self._initial_condition()
        self._gains()

    def _simulation(self):
        self.speed_scale=4.0
        self.sim_hz=200
        self.render_hz=60
        self.time=0.0
        self.time_step=1.0/self.sim_hz
        self.nonzero_floor=1e-3
        self.end_sim = 10e6

    def _environment(self):
        self.g=9.81
        self.rho=1.2683

    def _wind(self):
        # Configurable starting values for the gust filters.
        self.Va0=20.0  # Reference airspeed (m/s).
        self.Lu=200.0  # Longitudinal turbulence length scale (m).
        self.Lv=200.0  # Lateral turbulence length scale (m).
        self.Lw=50.0   # Vertical turbulence length scale (m).
        self.sigma_u=0.05  # Longitudinal gust standard deviation (m/s).
        self.sigma_v=0.05  # Lateral gust standard deviation (m/s).
        self.sigma_w=0.05  # Vertical gust standard deviation (m/s).

    def _aircraft(self):
        self.m=1.56
        self.j_x=0.1147
        self.j_y=0.0576
        self.j_z=0.1712
        self.j_xz=0.0015
        self.s_wing=0.2589
        self.b=1.4224
        self.c=0.3302
        self.s_prop=0.0314
        self.k_motor=20.0
        self.k_t_p=0.0
        self.k_omega=0.0
        self.c_prop=1.0
        self.e=0.9

    def _longitudinal_aerodynamics(self):
        self.c_l_0=0.09167
        self.c_d_0=0.01631
        self.c_m_0=-0.02338
        self.c_l_alpha=3.5016
        self.c_d_alpha=0.2108
        self.c_m_alpha=-0.5675
        self.c_l_q=2.8932
        self.c_d_q=0.0
        self.c_m_q=-1.3990
        self.c_l_delta_e=0.2724
        self.c_d_delta_e=0.3045
        self.c_m_delta_e=-0.3254
        self.c_d_p=0.0254
        self.stall_slope=50.0
        self.alpha0=0.4712
        self.epsilon=0.1592

    def _lateral_aerodynamics(self):
        self.c_y_0=0.0
        self.c_ell_0=0.0
        self.c_n_0=0.0
        self.c_y_beta=-0.07359
        self.c_ell_beta=-0.02854
        self.c_n_beta=0.00040
        self.c_y_p=0.0
        self.c_ell_p=-0.3209
        self.c_n_p=-0.01297
        self.c_y_r=0.0
        self.c_ell_r=0.03066
        self.c_n_r=-0.00434
        self.c_y_delta_a=0.0
        self.c_ell_delta_a=0.1682
        self.c_n_delta_a=-0.00328
        self.c_y_delta_r=-0.17
        self.c_ell_delta_r=0.105
        self.c_n_delta_r=-0.00328

    def _derived_constants(self):
        self.aspect_ratio=self.b**2/self.s_wing
        self.gamma=self.j_x*self.j_z-self.j_xz**2
        self.gamma_1=self.j_xz*(self.j_x-self.j_y+self.j_z)/self.gamma
        self.gamma_2=(self.j_z*(self.j_z-self.j_y)+self.j_xz**2)/self.gamma
        self.gamma_3=self.j_z/self.gamma
        self.gamma_4=self.j_xz/self.gamma
        self.gamma_5=(self.j_z-self.j_x)/self.j_y
        self.gamma_6=self.j_xz/self.j_y
        self.gamma_7=((self.j_x-self.j_y)*self.j_x+self.j_xz**2)/self.gamma
        self.gamma_8=self.j_x/self.gamma

    def _initial_condition(self):
        # State order: pn,pe,pd,u,v,w,phi,theta,psi,p,q,r
        self.state0=np.array([
            0.0,0.0,0.0,
            19.97213816555619,-3.4106051316484816e-08,1.055318480807477,
            1.7053025658242407e-10,0.152790440292938,0.0,
            -3.38214786100923e-10,3.745413743113032e-19,2.1963338460718988e-09,
        ],dtype=np.float64).reshape(12,1)
        # Control order: elevator,throttle,aileron,rudder
        self.u=np.array([
            -1.65097275e-01,
             9.25892619e-01,
            -8.74552214e-10,
             1.66177574e-09,
        ],dtype=np.float64).reshape(4,1)

    def _gains(self):
        kp = 8
        kq = 8
        kr = 8
        self.K_w = np.array([[kp, 0, 0], 
                             [0, kq, 0],
                             [0, 0, kr]])

        k_phi = 2
        k_theta = 2
        k_psi = 2
        self.K_eta = np.array([[k_phi, 0, 0], 
                             [0, k_theta, 0],
                             [0, 0, k_psi]])

    def body_to_inertial(self,phi,theta,psi):
        return body_to_inertial(phi,theta,psi,dtype=np.float64)
