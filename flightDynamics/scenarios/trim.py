"""Calculate and run a trimmed flight condition."""

from control.trim import TrimTarget,solve_trim
from scenarios.setup import Scenario
from sim.params import Params

# Edit the desired flight condition here.
AIRSPEED=20.0
CLIMB_ANGLE_DEG=0.0
TURN_RADIUS=100


def build_scenario():
    parameters=Params()
    target=TrimTarget(
        airspeed=AIRSPEED,
        climb_angle_deg=CLIMB_ANGLE_DEG,
        turn_radius=TURN_RADIUS,
    )
    solution=solve_trim(parameters,target)
    parameters.state0=solution.state
    parameters.u=solution.controls
    print(solution.summary())
    return Scenario(
        name="trim",
        description=(
            f"{AIRSPEED:g} m/s, {CLIMB_ANGLE_DEG:g} deg climb, "
            f"{TURN_RADIUS:g} m turn radius"
        ),
        parameters=parameters,
    )
