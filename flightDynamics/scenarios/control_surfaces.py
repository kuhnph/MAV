"""Run with directly specified control inputs."""

import numpy as np
from scenarios.setup import Scenario
from sim.params import Params

# Edit these four values for this scenario.
ELEVATOR=-0.1639
THROTTLE=0.9996
AILERON=0.0
RUDDER=0.0


def build_scenario():
    parameters=Params()
    parameters.u=np.array([
        [ELEVATOR],
        [THROTTLE],
        [AILERON],
        [RUDDER],
    ],dtype=np.float64)
    return Scenario(
        name="control surfaces",
        description="direct elevator, throttle, aileron, and rudder inputs",
        parameters=parameters,
    )
