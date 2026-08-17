from dataclasses import dataclass

import numpy as np
from sim.dynamics import Dynamics


@dataclass(frozen=True)
class TrimTarget:
    """Requested steady-flight condition used by the trim solver."""

    airspeed: float=20.0
    climb_angle_deg: float=0.0
    turn_radius: float=float("inf")


@dataclass(frozen=True)
class TrimSolution:
    """Aircraft state and controls that best satisfy a trim target."""

    state: np.ndarray
    controls: np.ndarray
    residual_norm: float
    iterations: int

    def summary(self):
        """Return a compact, human-readable report of solver convergence."""

        elevator,throttle,aileron,rudder=self.controls[:,0]
        return (
            f"Trim residual: {self.residual_norm:.3e} after {self.iterations} iterations\n"
            f"Controls: elevator={elevator:.5f}, throttle={throttle:.5f}, "
            f"aileron={aileron:.5f}, rudder={rudder:.5f}"
        )


def _condition(variables,target,parameters):
    """Build a candidate state and its scaled trim-equation residual."""

    # The seven optimization variables are angle of attack, sideslip, bank angle,
    # and the four controls (elevator, throttle, aileron, and rudder).
    alpha,beta,phi=variables[:3]
    controls=variables[3:].reshape(4,1)

    # Flight-path angle gamma relates pitch attitude and angle of attack by
    # theta = alpha + gamma. Positive gamma is a climb in the NED convention.
    climb_angle=np.deg2rad(target.climb_angle_deg)
    theta=alpha+climb_angle

    # A coordinated steady turn advances heading at V/R. Infinite radius
    # therefore represents straight flight.
    yaw_rate=0.0 if np.isinf(target.turn_radius) else target.airspeed/target.turn_radius

    # Resolve the requested airspeed into body-axis velocity components using
    # the candidate aerodynamic angles.
    u=target.airspeed*np.cos(alpha)*np.cos(beta)
    v=target.airspeed*np.sin(beta)
    w=target.airspeed*np.sin(alpha)*np.cos(beta)

    # Convert the desired Euler yaw rate into body angular rates while holding
    # roll and pitch constant (phi_dot = theta_dot = 0).
    p=-yaw_rate*np.sin(theta)
    q=yaw_rate*np.sin(phi)*np.cos(theta)
    r=yaw_rate*np.cos(phi)*np.cos(theta)

    # Position and heading do not affect the steady aerodynamic balance, so
    # retain their configured initial values in the candidate state.
    state=np.array([
        parameters.state0.item(0),parameters.state0.item(1),parameters.state0.item(2),
        u,v,w,phi,theta,parameters.state0.item(8),p,q,r,
    ],dtype=np.float64).reshape(12,1)

    # At trim, only inertial position and heading change. North/down velocity
    # follow the flight-path angle, and heading changes at the turn rate.
    desired=np.zeros((12,1),dtype=np.float64)
    desired[0,0]=target.airspeed*np.cos(climb_angle)
    desired[2,0]=-target.airspeed*np.sin(climb_angle)
    desired[8,0]=yaw_rate
    derivative=Dynamics(parameters).derivative(state,controls)

    # Nondimensionalize groups of unlike derivatives so large translational
    # values do not dominate attitude and angular-rate errors in least squares.
    scale=np.array([20,20,20,10,10,10,1,1,1,10,10,10],dtype=np.float64).reshape(12,1)
    return state,controls,(derivative-desired)/scale


def solve_trim(parameters,target,max_iterations=80,tolerance=1e-7):
    """Solve for a bounded steady-flight state and control vector.

    A damped Gauss-Newton iteration minimizes the scaled dynamics residual.
    Bounds keep aerodynamic angles and actuator commands in plausible ranges.
    """

    if target.airspeed<=0:
        raise ValueError("Airspeed must be positive")
    if not np.isinf(target.turn_radius) and abs(target.turn_radius)<1:
        raise ValueError("Turn radius must be infinite or have magnitude >= 1 meter")

    # Seed the aerodynamic angles near level flight and the controls at their
    # configured initial values. A good control seed improves convergence.
    variables=np.array([
        0.05,0.0,0.0,
        parameters.u.item(0),parameters.u.item(1),
        parameters.u.item(2),parameters.u.item(3),
    ],dtype=np.float64)

    lower=np.array([-0.7,-0.4,-1.4,-1.0,0.0,-1.0,-1.0])
    upper=np.array([0.7,0.4,1.4,1.0,1.0,1.0,1.0])

    # Levenberg-Marquardt damping regularizes J^T J and is adapted after each
    # accepted or rejected candidate step.
    damping=1e-3

    for iteration in range(1,max_iterations+1):
        state,controls,residual=_condition(variables,target,parameters)
        flat_residual=residual.ravel()
        residual_norm=np.linalg.norm(flat_residual)
        if residual_norm<tolerance:
            break

        # Approximate the 12-by-7 residual Jacobian with forward differences.
        jacobian=np.empty((flat_residual.size,variables.size))
        step=1e-5
        for column in range(variables.size):
            perturbed=variables.copy()
            perturbed[column]+=step
            jacobian[:,column]=(
                _condition(perturbed,target,parameters)[2].ravel()-flat_residual
            )/step

        # Solve (J^T J + lambda I) Delta = -J^T r for the damped
        # least-squares update, then enforce the variable bounds.
        update=np.linalg.solve(
            jacobian.T@jacobian+damping*np.eye(variables.size),
            -jacobian.T@flat_residual,
        )
        candidate=np.clip(variables+update,lower,upper)
        candidate_norm=np.linalg.norm(_condition(candidate,target,parameters)[2])

        # Accept improving steps and relax damping; otherwise retain the old
        # iterate and increase damping to make the next step more conservative.
        if candidate_norm<residual_norm:
            variables=candidate
            damping=max(damping/2,1e-8)
        else:
            damping=min(damping*10,1e8)

    # Re-evaluate after the final iteration so the returned residual matches
    # the returned state and controls exactly.
    state,controls,residual=_condition(variables,target,parameters)
    return TrimSolution(state,controls,float(np.linalg.norm(residual)),iteration)
