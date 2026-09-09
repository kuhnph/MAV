import numpy as np
from numpy import cos as c
from numpy import sin as s
from numpy import tan as t

from sim.FaM import ForcesAndMoments
from sim.params import Params


class Controller:
    def __init__(self,parameters: Params):
        self.params=parameters
        self.forces_and_moments=ForcesAndMoments(parameters)
    
    def control_loop(self,u,state,eta_c,etaDot_c=0):
        f_w,G_w = self.forces_and_moments.calculate_G_AND_f(state,u)



        #Calculate omega_c (outer loop)
        phi, theta, psi = state.item(6),state.item(7),state.item(8)
        eta = np.array([[phi,theta,psi]]).T

        T = np.array([
            [1, s(phi)*t(theta), c(phi)*t(theta)],
            [0, c(phi)         , -s(phi)],
            [0, s(phi)/c(theta), c(phi)/c(theta)]
        ])

        nu_outer = etaDot_c + self.params.K_eta @ (eta_c - eta)
        omega_c = np.linalg.solve(T,nu_outer)


        #Calculate u (inner loop)
        omegaDot_c = 0
        p,q,r=state.item(9),state.item(10),state.item(11)
        omega = np.array([[p,q,r]]).T
        nu_inner = omegaDot_c+self.params.K_w@(omega_c-omega)
        u_controlled=np.linalg.solve(G_w,nu_inner-f_w)

        

        # The effectiveness matrix columns are aileron, elevator, and rudder.
        aileron_commanded=u_controlled.item(0)
        elevator_commanded=u_controlled.item(1)
        rudder_commanded=u_controlled.item(2)
        throttle_commanded=u.item(1)

        aileron_commanded, elevator_commanded, rudder_commanded = self.saturate(aileron_commanded,elevator_commanded,rudder_commanded)

        u_commanded=np.array([
            [elevator_commanded],
            [throttle_commanded],
            [aileron_commanded],
            [rudder_commanded],
        ],dtype=np.float64)

        return u_commanded

    def saturate(self, ac,ec,rc):
        sat_a = np.deg2rad(20)
        sat_e = np.deg2rad(20)
        sat_r = np.deg2rad(20)

        ac = np.clip(ac, -sat_a, sat_a)
        ec = np.clip(ec, -sat_e, sat_e)
        rc = np.clip(rc, -sat_r, sat_r)

        return ac,ec,rc
