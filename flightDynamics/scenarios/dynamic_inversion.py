"""Run a dynamic inversion controller in the loop."""

import numpy as np

from scenarios.setup import Scenario
from sim.params import Params
from control.trim import TrimTarget,solve_trim


OMEGA_COMMAND=np.array([[0.001],[0.0],[0.0]],dtype=np.float64)
OMEGA_DOT_COMMAND=np.zeros((3,1),dtype=np.float64)

# Edit the desired flight condition here.
AIRSPEED=20.0
CLIMB_ANGLE_DEG=1.0
TURN_RADIUS=1e9


def build_scenario():
    parameters = Params()

    target=TrimTarget(
        airspeed=AIRSPEED,
        climb_angle_deg=CLIMB_ANGLE_DEG,
        turn_radius=TURN_RADIUS,
    )
    solution=solve_trim(parameters,target)
    parameters.state0=solution.state
    return Scenario(
        name="Dynamic Inversion",
        description="DI controller commanding angular rates",
        parameters=parameters,
        controller_enabled=True,
        omega_dot_command=OMEGA_DOT_COMMAND.copy(),
        omega_command=OMEGA_COMMAND.copy(),
    )
