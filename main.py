import sys
from pathlib import Path

project_dir=Path(__file__).parent
sys.path.append(str(project_dir/"flightDynamics/src"))
sys.path.append(str(project_dir/"flightDynamics"))

# Choose "open_loop", "trim", or "control_surfaces".
SCENARIO="open_loop"


def load_scenario():
    if SCENARIO=="trim":
        from scenarios.trim import build_scenario
    elif SCENARIO=="control_surfaces":
        from scenarios.control_surfaces import build_scenario
    elif SCENARIO=="open_loop":
        from scenarios.open_loop import build_scenario
    elif SCENARIO == "dynamic_inversion":
        from scenarios.dynamic_inversion import build_scenario
    else:
        raise ValueError(f"Unknown scenario: {SCENARIO}")
    return build_scenario()


def main():
    import pyglet
    from plotting.dataLogging import Logger
    from sim.dynamics import Dynamics
    from viewer.mesh import aircraft_model_mesh
    from viewer.renderer import Renderer
    from viewer.window import SimWindow

    scenario=load_scenario()
    parameters=scenario.parameters
    dynamics=Dynamics(parameters)
    logger=Logger()
    vertices,indices=aircraft_model_mesh(scale=5.0)

    def sim_step(dt):
        del dt
        for _ in range(max(1,int(round(parameters.speed_scale)))):
            if scenario.control_update is None:
                u=window.u
            else:
                print("hello world")
                u=scenario.control_update(dynamics.time, dynamics.state.copy(),scenario.desired)
            logger.log(dynamics.time,dynamics.state,u)
            dynamics.update(u)


    def get_pose():
        state=dynamics.state
        return tuple(float(state[index,0]) for index in (0,1,2,6,7,8))

    window=SimWindow(
        sim_step_func=sim_step,
        get_pose_func=get_pose,
        get_sim_time_func=lambda: dynamics.time,
        renderer_factory=lambda ctx: Renderer(ctx,vertices,indices),
        initial_controls=parameters.u,
        accepts_key_input=scenario.accepts_key_input,
        width=1000,
        height=800,
        render_hz=parameters.render_hz,
        sim_hz=parameters.sim_hz,
    )
    print(f"Running {scenario.name}: {scenario.description}")
    try:
        pyglet.app.run()
    finally:
        print("Exporting telemetry")
        logger.export()


if __name__=="__main__":
    main()
