"""Run with directly specified control inputs."""

import numpy as np
from experiments.setup import Experiment
from sim.params import Params

# Edit these four values for this experiment.
ELEVATOR=-0.1639
THROTTLE=0.9996
AILERON=0.0
RUDDER=0.0


def build_experiment():
    parameters=Params()
    parameters.u=np.array([
        [ELEVATOR],
        [THROTTLE],
        [AILERON],
        [RUDDER],
    ],dtype=np.float64)
    return Experiment(
        name="control surfaces",
        description="direct elevator, throttle, aileron, and rudder inputs",
        parameters=parameters,
    )
