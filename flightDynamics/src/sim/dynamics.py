import numpy as np
from sim.params import Params
from sim.FaM import ForcesAndMoments
from sim.rotations import body_to_inertial


class Dynamics:
    def __init__(self,parameters: Params | None=None):
        self.parameters=parameters or Params()
        self.forces_and_moments=ForcesAndMoments(self.parameters)
        self.state=self.parameters.state0.copy()
        self.time_step=float(self.parameters.time_step)
        self.time=float(self.parameters.time)

    def rk4(self,controls):
        derivative_1=self.derivative(self.state,controls)
        derivative_2=self.derivative(self.state+self.time_step/2.0*derivative_1,controls)
        derivative_3=self.derivative(self.state+self.time_step/2.0*derivative_2,controls)
        derivative_4=self.derivative(self.state+self.time_step*derivative_3,controls)
        self.state+=self.time_step/6.0*(
            derivative_1+2.0*derivative_2+2.0*derivative_3+derivative_4
        )

    def derivative(self,state,controls):
        params=self.parameters
        u,v,w=state.item(3),state.item(4),state.item(5)
        phi,theta,psi=state.item(6),state.item(7),state.item(8)
        p,q,r=state.item(9),state.item(10),state.item(11)
        forces,moments=self.forces_and_moments.calculate(state,controls)
        force_x,force_y,force_z=(forces.item(index) for index in range(3))
        ell,pitch_moment,yaw_moment=(moments.item(index) for index in range(3))
        position_dot=body_to_inertial(phi,theta,psi,dtype=np.float64)@np.array([[u],[v],[w]])
        velocity_dot=np.array([
            [r*v-q*w],[p*w-r*u],[q*u-p*v],
        ],dtype=np.float64)+(1.0/params.m)*np.array(
            [[force_x],[force_y],[force_z]],dtype=np.float64
        )
        euler_dot=np.array([
            [1.0,np.sin(phi)*np.tan(theta),np.cos(phi)*np.tan(theta)],
            [0.0,np.cos(phi),-np.sin(phi)],
            [0.0,np.sin(phi)/np.cos(theta),np.cos(phi)/np.cos(theta)],
        ],dtype=np.float64)@np.array([[p],[q],[r]],dtype=np.float64)
        angular_acceleration=np.array([
            [params.gamma_1*p*q-params.gamma_2*q*r+params.gamma_3*ell+params.gamma_4*yaw_moment],
            [params.gamma_5*p*r-params.gamma_6*(p**2-r**2)+pitch_moment/params.j_y],
            [params.gamma_7*p*q-params.gamma_1*q*r+params.gamma_4*ell+params.gamma_8*yaw_moment],
        ],dtype=np.float64)
        return np.concatenate([position_dot,velocity_dot,euler_dot,angular_acceleration],axis=0)

    def update(self,controls):
        self.rk4(controls)
        self.time+=self.time_step
