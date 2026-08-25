"""Run a dynamic inversion controller in the loop."""

import numpy as np

from control.dynamic_inversion import Controller
from scenarios.setup import Scenario
from sim.params import Params

ELEVATOR=-0.1639
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

    di_controller=Controller(parameters)

    return Scenario(
        name="Dynamic Inversion",
        description="DI controller commanding angular rates",
        parameters=parameters,
        control_update=di_controller.control_loop,
        omega_dot_command=OMEGA_DOT_COMMAND.copy(),
        omega_command=OMEGA_COMMAND.copy(),
    )
