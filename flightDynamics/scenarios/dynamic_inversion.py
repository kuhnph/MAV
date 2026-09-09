"""Run a dynamic inversion controller in the loop."""

import numpy as np

from scenarios.setup import Scenario
from sim.params import Params

# ELEVATOR=-0.1639
ELEVATOR = 3
THROTTLE=0.9996
AILERON=0.0
RUDDER=0.0
OMEGA_COMMAND=np.array([[0.001],[0.0],[0.0]],dtype=np.float64)
OMEGA_DOT_COMMAND=np.zeros((3,1),dtype=np.float64)

def build_scenario():
    parameters = Params()

    parameters.u = np.array([
        [ELEVATOR],
        [THROTTLE],
        [AILERON],
        [RUDDER],
    ], dtype=np.float64)

    return Scenario(
        name="Dynamic Inversion",
        description="DI controller commanding angular rates",
        parameters=parameters,
        controller_enabled=True,
        omega_dot_command=OMEGA_DOT_COMMAND.copy(),
        omega_command=OMEGA_COMMAND.copy(),
    )
