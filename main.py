import sys
from pathlib import Path
import numpy as np

project_dir=Path(__file__).parent
sys.path.append(str(project_dir/"flightDynamics/src"))
sys.path.append(str(project_dir/"flightDynamics"))

# Choose "open_loop", "trim", or "control_surfaces".
SCENARIO="dynamic_inversion"

#SIGNAL
from sim.signalGenerator import SignalGenerator
SG = SignalGenerator()


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
    from control.dynamic_inversion import Controller
    from control.terminal_commands import TerminalCommands
    from sim.dynamics import Dynamics
    from viewer.mesh import aircraft_model_mesh
    from viewer.renderer import Renderer
    from viewer.window import SimWindow

    scenario=load_scenario()
    parameters=scenario.parameters
    dynamics=Dynamics(parameters)
    controller=Controller(parameters) if scenario.controller_enabled else None
    logger=Logger()
    vertices,indices=aircraft_model_mesh(scale=5.0)

    current_controls=parameters.u.copy()
    eta_command = np.array([[0.0], [7.09205215e-02], [0.0]])
    terminal_commands = TerminalCommands() if controller is not None else None
    
    def sim_step(dt):
        nonlocal current_controls
        del dt
        if terminal_commands is not None:
            terminal_commands.apply(eta_command)
        for _ in range(max(1,int(round(parameters.speed_scale)))):
            #non controller logic
            if controller is None:
                current_controls=window.u
                logger.log(dynamics.time,dynamics.state,current_controls,wind=dynamics.wind_body)

            #controller logic
            else:
                current_controls=controller.control_loop(
                    current_controls,
                    dynamics.state.copy(),
                    eta_command,
                )
                logger.log(dynamics.time,dynamics.state,current_controls,controller.commanded_states,dynamics.wind_body,dynamics.aero)
            dynamics.update(current_controls)



            


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
    if terminal_commands is not None:
        terminal_commands.start()

    LOGGING = True
    if LOGGING:
        try:
            pyglet.app.run()
        finally:
            print("Exporting telemetry")
            logger.export()
    else:
        pyglet.app.run()


if __name__=="__main__":
    main()
