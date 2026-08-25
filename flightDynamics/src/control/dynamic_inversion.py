import numpy as np

from sim.FaM import ForcesAndMoments
from sim.params import Params


class Controller:
    def __init__(self,parameters: Params):
        self.params=parameters
        self.forces_and_moments=ForcesAndMoments(parameters)
    
    def control_loop(self,u,state,omegaDot_c,omega_c):
        f_w,G_w = self.forces_and_moments.calculate_G_AND_f(state,u)

        p,q,r=state.item(9),state.item(10),state.item(11)
        omega = np.array([[p],[q],[r]])
        nu = omegaDot_c+self.params.K_w@(omega_c-omega)
        surface_commands=np.linalg.solve(G_w,nu-f_w)

        #DEBUG
        self.omegaDot_DI = f_w+G_w@surface_commands
        

        # The effectiveness matrix columns are aileron, elevator, and rudder.
        aileron_commanded=surface_commands.item(0)
        elevator_commanded=surface_commands.item(1)
        rudder_commanded=surface_commands.item(2)
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