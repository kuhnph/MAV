from dataclasses import dataclass

import numpy as np
from sim.dynamics import Dynamics


@dataclass(frozen=True)
class TrimTarget:
    airspeed: float=20.0
    climb_angle_deg: float=0.0
    turn_radius: float=float("inf")


@dataclass(frozen=True)
class TrimSolution:
    state: np.ndarray
    controls: np.ndarray
    residual_norm: float
    iterations: int

    def summary(self):
        elevator,throttle,aileron,rudder=self.controls[:,0]
        return (
            f"Trim residual: {self.residual_norm:.3e} after {self.iterations} iterations\n"
            f"Controls: elevator={elevator:.5f}, throttle={throttle:.5f}, "
            f"aileron={aileron:.5f}, rudder={rudder:.5f}"
        )


def _condition(variables,target,parameters):
    alpha,beta,phi=variables[:3]
    controls=variables[3:].reshape(4,1)
    climb_angle=np.deg2rad(target.climb_angle_deg)
    theta=alpha+climb_angle
    yaw_rate=0.0 if np.isinf(target.turn_radius) else target.airspeed/target.turn_radius
    u=target.airspeed*np.cos(alpha)*np.cos(beta)
    v=target.airspeed*np.sin(beta)
    w=target.airspeed*np.sin(alpha)*np.cos(beta)
    p=-yaw_rate*np.sin(theta)
    q=yaw_rate*np.sin(phi)*np.cos(theta)
    r=yaw_rate*np.cos(phi)*np.cos(theta)
    state=np.array([
        parameters.state0.item(0),parameters.state0.item(1),parameters.state0.item(2),
        u,v,w,phi,theta,parameters.state0.item(8),p,q,r,
    ],dtype=np.float64).reshape(12,1)
    desired=np.zeros((12,1),dtype=np.float64)
    desired[0,0]=target.airspeed*np.cos(climb_angle)
    desired[2,0]=-target.airspeed*np.sin(climb_angle)
    desired[8,0]=yaw_rate
    derivative=Dynamics(parameters).derivative(state,controls)
    scale=np.array([20,20,20,10,10,10,1,1,1,10,10,10],dtype=np.float64).reshape(12,1)
    return state,controls,(derivative-desired)/scale


def solve_trim(parameters,target,max_iterations=80,tolerance=1e-7):
    if target.airspeed<=0:
        raise ValueError("Airspeed must be positive")
    if not np.isinf(target.turn_radius) and abs(target.turn_radius)<1:
        raise ValueError("Turn radius must be infinite or have magnitude >= 1 meter")
    variables=np.array([
        0.05,0.0,0.0,
        parameters.u.item(0),parameters.u.item(1),
        parameters.u.item(2),parameters.u.item(3),
    ],dtype=np.float64)
    lower=np.array([-0.7,-0.4,-1.4,-1.0,0.0,-1.0,-1.0])
    upper=np.array([0.7,0.4,1.4,1.0,1.0,1.0,1.0])
    damping=1e-3
    for iteration in range(1,max_iterations+1):
        state,controls,residual=_condition(variables,target,parameters)
        flat_residual=residual.ravel()
        residual_norm=np.linalg.norm(flat_residual)
        if residual_norm<tolerance:
            break
        jacobian=np.empty((flat_residual.size,variables.size))
        step=1e-5
        for column in range(variables.size):
            perturbed=variables.copy()
            perturbed[column]+=step
            jacobian[:,column]=(
                _condition(perturbed,target,parameters)[2].ravel()-flat_residual
            )/step
        update=np.linalg.solve(
            jacobian.T@jacobian+damping*np.eye(variables.size),
            -jacobian.T@flat_residual,
        )
        candidate=np.clip(variables+update,lower,upper)
        candidate_norm=np.linalg.norm(_condition(candidate,target,parameters)[2])
        if candidate_norm<residual_norm:
            variables=candidate
            damping=max(damping/2,1e-8)
        else:
            damping=min(damping*10,1e8)
    state,controls,residual=_condition(variables,target,parameters)
    return TrimSolution(state,controls,float(np.linalg.norm(residual)),iteration)
