from __future__ import annotations

import numpy as np
from numpy import cos,sin
from sim.params import Params
from sim.rotations import body_to_inertial


rng = np.random.default_rng()
class ForcesAndMoments:
    def __init__(self,parameters: Params | None=None):
        self.parameters=parameters or Params()
        self.xu = 0
        self.xv1 = 0
        self.xv2 = 0
        self.xw1 = 0
        self.xw2 = 0

        self.wind_body = np.array([[0,0,0]]).T
        self.alpha = 0
        self.beta = 0

    def calculate(self,state,controls):
        #Inputs and aero measurements
        params=self.parameters
        u,v,w=state.item(3),state.item(4),state.item(5)
        phi,theta=state.item(6),state.item(7)
        p,q,r=state.item(9),state.item(10),state.item(11)
        delta_e,delta_t,delta_a,delta_r=(controls.item(index) for index in range(4))

        u_wind,v_wind,w_wind = [i[0] for i in self.wind_body]


        u_r = u - u_wind
        v_r = v - v_wind
        w_r = w - w_wind

        airspeed=np.sqrt(u_r**2+v_r**2+w_r**2)
        alpha=np.arctan2(w_r,u_r)
        beta=np.arcsin(np.clip(v_r/airspeed,-1.0,1.0))
        self.alpha = alpha
        self.beta = beta

        #Coefficients
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


        #gravity force
        gravity=np.array([
            [-params.m*params.g*np.sin(theta)],
            [params.m*params.g*np.cos(theta)*np.sin(phi)],
            [params.m*params.g*np.cos(theta)*np.cos(phi)],
        ],dtype=np.float64)

        #aerodynamic force
        aerodynamic=0.5*params.rho*airspeed**2*params.s_wing*np.array([
            [c_x+c_x_q*(params.c/(2.0*airspeed))*q+c_x_delta_e*delta_e],
            [params.c_y_0+params.c_y_beta*beta+params.c_y_p*(params.b/(2.0*airspeed))*p
             +params.c_y_r*(params.b/(2.0*airspeed))*r+params.c_y_delta_a*delta_a
             +params.c_y_delta_r*delta_r],
            [c_z+c_z_q*(params.c/(2.0*airspeed))*q+c_z_delta_e*delta_e],
        ],dtype=np.float64)

        #propulsion Force
        propulsion=0.5*params.rho*params.s_prop*params.c_prop*np.array([
            [(params.k_motor*delta_t)**2-airspeed**2],[0.0],[0.0],
        ],dtype=np.float64)

        #Force Sum
        forces=gravity+aerodynamic+propulsion

        q_times_S = 0.5*params.rho*airspeed**2*params.s_wing

        L_0 = q_times_S*(params.b*(params.c_ell_0+params.c_ell_beta*beta
             +params.c_ell_p*(params.b/(2.0*airspeed))*p
             +params.c_ell_r*(params.b/(2.0*airspeed))*r)
             -params.k_t_p*(params.k_omega*delta_t)**2)

        L_u = q_times_S*params.b*(params.c_ell_delta_a*delta_a
             +params.c_ell_delta_r*delta_r)

        L = L_0+L_u
        

        M_0 = q_times_S*params.c*(params.c_m_0+params.c_m_alpha*alpha
             +params.c_m_q*(params.c/(2.0*airspeed))*q)

        M_u = q_times_S*params.c*params.c_m_delta_e*delta_e

        M = M_0+M_u

        


        N_0 = q_times_S*params.b*(params.c_n_0+params.c_n_beta*beta
             +params.c_n_p*(params.b/(2.0*airspeed))*p
             +params.c_n_r*(params.b/(2.0*airspeed))*r)

        N_u = q_times_S*params.b*(params.c_n_delta_a*delta_a
             +params.c_n_delta_r*delta_r)

        N = N_0+N_u



        moments=np.array([
            [L],
            [M],
            [N],],dtype=np.float64)

        self.M_0 = np.array([
            [L_0],
            [M_0],
            [N_0]
        ],dtype=np.float64)

        self.B_M = q_times_S*np.array([
            [params.b*params.c_ell_delta_a,     0, params.b*params.c_ell_delta_r],
            [0,       params.c*params.c_m_delta_e,                             0],
            [params.b*params.c_n_delta_a,       0,   params.b*params.c_n_delta_r]
        ],dtype=np.float64)
        return forces,moments


    def calculate_G_AND_f(self,state,controls):
        params=self.parameters

        self.calculate(state,controls)
                
        p,q,r=state.item(9),state.item(10),state.item(11)

        omega = np.array([[p],[q],[r]])

        J = np.array([[params.j_x, 0, -params.j_xz],
                      [0, params.j_y, 0],
                      [-params.j_xz, 0, params.j_z]])

        J_omega = J @ omega

        omega_cross_J_omega = np.cross(omega.flatten(),J_omega.flatten())
        omega_cross_J_omega = omega_cross_J_omega.reshape(3,1)

        f_w = np.linalg.solve(J,self.M_0-omega_cross_J_omega)

        G_w = np.linalg.solve(J,self.B_M)

        return f_w, G_w


    def wind(self,state):
        params=self.parameters

        phi   = state.item(6)
        theta = state.item(7)
        psi   = state.item(8)

        self.stead_wind_ned = np.array([[0,0,0]]).T

        R_bi = body_to_inertial(phi, theta, psi).T

        stead_wind_body = R_bi @ self.stead_wind_ned

        #Generate noise
        dt = params.time_step
        xi_u, xi_v, xi_w = rng.standard_normal(3)

        n_u = xi_u / np.sqrt(dt)
        n_v = xi_v / np.sqrt(dt)
        n_w = xi_w / np.sqrt(dt)

        #Calculate state derivatives
        a_u = params.Va0 / params.Lu
        a_v = params.Va0 / params.Lv
        a_w = params.Va0 / params.Lw

        #longitudinal derivative
        xu_dot = -a_u * self.xu + n_u

        #Lateral Derivatives
        xv1_dot = self.xv2
        xv2_dot = -(a_v**2)*self.xv1 - 2.0*a_v*self.xv2 + n_v

        #Vertical derivatives
        xw1_dot = self.xw2
        xw2_dot = -(a_w**2)*self.xw1 - 2.0*a_w*self.xw2 + n_w

        #Euler Integration
        self.xu  += dt * xu_dot

        self.xv1 += dt * xv1_dot
        self.xv2 += dt * xv2_dot

        self.xw1 += dt * xw1_dot
        self.xw2 += dt * xw2_dot

        #output
        Ku = params.sigma_u * np.sqrt(2*params.Va0/params.Lu)
        Kv = params.sigma_v * np.sqrt(3*params.Va0/params.Lv)
        Kw = params.sigma_w * np.sqrt(3*params.Va0/params.Lw)

        bv = params.Va0/(np.sqrt(3.0)*params.Lv)
        bw = params.Va0/(np.sqrt(3.0)*params.Lw)

        u_gust = Ku*self.xu
        v_gust = Kv*(bv*self.xv1 + self.xv2)
        w_gust = Kw*(bw*self.xw1 + self.xw2)

        gust_body = np.array([[u_gust,v_gust,w_gust]]).T

        #DEBUG no wind
        self.wind_body = gust_body + stead_wind_body
        self.wind_body = np.array([[0,0,0]]).T