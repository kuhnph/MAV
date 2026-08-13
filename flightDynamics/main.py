import sys
from pathlib import Path

project_dir=Path(__file__).parent
sys.path.append(str(project_dir/"src"))

# Choose "open_loop", "trim", or "control_surfaces".
EXPERIMENT="open_loop"


def load_experiment():
    if EXPERIMENT=="trim":
        from experiments.trim import build_experiment
    elif EXPERIMENT=="control_surfaces":
        from experiments.control_surfaces import build_experiment
    elif EXPERIMENT=="open_loop":
        from experiments.open_loop import build_experiment
    else:
        raise ValueError(f"Unknown experiment: {EXPERIMENT}")
    return build_experiment()


def main():
    import pyglet
    from plotting.dataLogging import Logger
    from sim.dynamics import Dynamics
    from viewer.mesh import aircraft_model_mesh
    from viewer.renderer import Renderer
    from viewer.window import SimWindow

    experiment=load_experiment()
    parameters=experiment.parameters
    dynamics=Dynamics(parameters)
    logger=Logger()
    vertices,indices=aircraft_model_mesh(scale=5.0)

    def sim_step(dt):
        del dt
        for _ in range(max(1,int(round(parameters.speed_scale)))):
            logger.log(dynamics.time,dynamics.state,window.u)
            dynamics.update(window.u)

    def get_pose():
        state=dynamics.state
        return tuple(float(state[index,0]) for index in (0,1,2,6,7,8))

    window=SimWindow(
        sim_step_func=sim_step,
        get_pose_func=get_pose,
        get_sim_time_func=lambda: dynamics.time,
        renderer_factory=lambda ctx: Renderer(ctx,vertices,indices),
        initial_controls=parameters.u,
        accepts_key_input=experiment.accepts_key_input,
        width=1000,
        height=800,
        render_hz=parameters.render_hz,
        sim_hz=parameters.sim_hz,
    )
    print(f"Running {experiment.name}: {experiment.description}")
    try:
        pyglet.app.run()
    finally:
        print("Exporting telemetry")
        logger.export()


if __name__=="__main__":
    main()
