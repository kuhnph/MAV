from __future__ import annotations

import numpy as np
from numpy import cos,sin
from sim.params import Params


class ForcesAndMoments:
    def __init__(self,parameters: Params | None=None):
        self.parameters=parameters or Params()

    def calculate(self,state,controls):
        params=self.parameters
        u,v,w=state.item(3),state.item(4),state.item(5)
        phi,theta=state.item(6),state.item(7)
        p,q,r=state.item(9),state.item(10),state.item(11)
        delta_e,delta_t,delta_a,delta_r=(controls.item(index) for index in range(4))
        airspeed=max(np.sqrt(u**2+v**2+w**2),params.nonzero_floor)
        alpha=np.arctan2(w,u)
        beta=np.arcsin(np.clip(v/airspeed,-1.0,1.0))

        numerator=1.0+np.exp(-params.stall_slope*(alpha-params.alpha0))+np.exp(
            params.stall_slope*(alpha+params.alpha0)
        )
        denominator=(1.0+np.exp(-params.stall_slope*(alpha-params.alpha0)))*(
            1.0+np.exp(params.stall_slope*(alpha+params.alpha0))
        )
        sigma=numerator/denominator
        c_l=(1.0-sigma)*(params.c_l_0+params.c_l_alpha*alpha)+sigma*(
            2.0*np.sign(alpha)*sin(alpha)**2*cos(alpha)
        )
        c_d=params.c_d_p+(params.c_l_0+params.c_l_alpha*alpha)**2/(
            np.pi*params.e*params.aspect_ratio
        )
        c_x=-c_d*cos(alpha)+c_l*sin(alpha)
        c_x_q=-params.c_d_q*cos(alpha)+params.c_l_q*sin(alpha)
        c_x_delta_e=-params.c_d_delta_e*cos(alpha)+params.c_l_delta_e*sin(alpha)
        c_z=-c_d*sin(alpha)-c_l*cos(alpha)
        c_z_q=-params.c_d_q*sin(alpha)-params.c_l_q*cos(alpha)
        c_z_delta_e=-params.c_d_delta_e*sin(alpha)-params.c_l_delta_e*cos(alpha)

        gravity=np.array([
            [-params.m*params.g*np.sin(theta)],
            [params.m*params.g*np.cos(theta)*np.sin(phi)],
            [params.m*params.g*np.cos(theta)*np.cos(phi)],
        ],dtype=np.float64)
        aerodynamic=0.5*params.rho*airspeed**2*params.s_wing*np.array([
            [c_x+c_x_q*(params.c/(2.0*airspeed))*q+c_x_delta_e*delta_e],
            [params.c_y_0+params.c_y_beta*beta+params.c_y_p*(params.b/(2.0*airspeed))*p
             +params.c_y_r*(params.b/(2.0*airspeed))*r+params.c_y_delta_a*delta_a
             +params.c_y_delta_r*delta_r],
            [c_z+c_z_q*(params.c/(2.0*airspeed))*q+c_z_delta_e*delta_e],
        ],dtype=np.float64)
        propulsion=0.5*params.rho*params.s_prop*params.c_prop*np.array([
            [(params.k_motor*delta_t)**2-airspeed**2],[0.0],[0.0],
        ],dtype=np.float64)
        forces=gravity+aerodynamic+propulsion
        moments=0.5*params.rho*airspeed**2*params.s_wing*np.array([
            [params.b*(params.c_ell_0+params.c_ell_beta*beta
             +params.c_ell_p*(params.b/(2.0*airspeed))*p
             +params.c_ell_r*(params.b/(2.0*airspeed))*r
             +params.c_ell_delta_a*delta_a+params.c_ell_delta_r*delta_r)
             -params.k_t_p*(params.k_omega*delta_t)**2],
            [params.c*(params.c_m_0+params.c_m_alpha*alpha
             +params.c_m_q*(params.c/(2.0*airspeed))*q+params.c_m_delta_e*delta_e)],
            [params.b*(params.c_n_0+params.c_n_beta*beta
             +params.c_n_p*(params.b/(2.0*airspeed))*p
             +params.c_n_r*(params.b/(2.0*airspeed))*r
             +params.c_n_delta_a*delta_a+params.c_n_delta_r*delta_r)],
        ],dtype=np.float64)
        return forces,moments
