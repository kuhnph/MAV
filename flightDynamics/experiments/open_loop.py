"""Interactive open-loop flight starting from straight-and-level trim."""

from control.trim import TrimTarget,solve_trim
from experiments.setup import Experiment
from sim.params import Params

# Edit the starting airspeed if desired.
AIRSPEED=20.0


def build_experiment():
    parameters=Params()
    solution=solve_trim(parameters,TrimTarget(
        airspeed=AIRSPEED,
        climb_angle_deg=0.0,
        turn_radius=float("inf"),
    ))
    parameters.state0=solution.state
    parameters.u=solution.controls
    print(solution.summary())
    return Experiment(
        name="open loop",
        description=f"interactive controls from level flight at {AIRSPEED:g} m/s",
        parameters=parameters,
        accepts_key_input=True,
    )
