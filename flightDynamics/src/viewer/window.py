import pyglet
from pyglet.window import key
import moderngl
import time
import numpy as np
from sim.rotations import model_matrix_from_pose
from sim.params import Params


class SimWindow(pyglet.window.Window):
    def __init__(self, sim_step_func, get_pose_func, get_sim_time_func, renderer_factory,
                 initial_controls=None,accepts_key_input=False,width=1000,height=800,
                 render_hz=60,sim_hz=None):
        self.parameters = Params()
        sim_hz = self.parameters.sim_hz if sim_hz is None else sim_hz
        super().__init__(width=width, height=height, caption="MAV Viewer (GPU)", resizable=True)

        controls=self.parameters.u if initial_controls is None else initial_controls
        self.del_e=controls[0,0]
        self.del_t=controls[1,0]
        self.del_a=controls[2,0]
        self.del_r=controls[3,0]
        self.u = np.array([[self.del_e,self.del_t,self.del_a,self.del_r]]).T
        self.accepts_key_input=accepts_key_input

        self.ctx = moderngl.create_context()
        self.renderer = renderer_factory(self.ctx)
        self.sim_step = sim_step_func
        self.get_pose = get_pose_func
        self.get_sim_time = get_sim_time_func
        self.sim_dt = 1.0 / sim_hz
        self.accum = 0.0
        self.last = time.perf_counter()
        self.time_label = pyglet.text.Label(
            "t = 0.00 s",
            font_name="Arial",
            font_size=14,
            x=10,
            y=self.height - 10,
            anchor_x="left",
            anchor_y="top",
            color=(255, 255, 255, 255),
        )
        pyglet.clock.schedule(self._tick)
        pyglet.clock.schedule_interval(self._render, 1.0 / render_hz)

    def _tick(self, _dt):
        now = time.perf_counter()
        frame_dt = now - self.last
        self.last = now

        self.accum += frame_dt
        if self.accum > 0.25:
            self.accum = 0.25

        while self.accum >= self.sim_dt:
            self.sim_step(self.sim_dt)
            self.accum -= self.sim_dt

            if self.get_sim_time() >= self.parameters.end_sim:
                self.close()
                return

    def _render(self, _dt):
        self.dispatch_event("on_draw")
        self.flip()

    def on_draw(self):
        self.clear()

        pn, pe, pd, phi, theta, psi = self.get_pose()
        model = model_matrix_from_pose(pn, pe, pd, phi, theta, psi)

        self.renderer.draw(self.width, self.height, model)
        self._draw_overlay()

    def _draw_overlay(self):
        sim_t = self.get_sim_time()
        self.time_label.text = f"t = {sim_t:8.2f} s"
        self.time_label.y = self.height - 10

        try:
            self.ctx.screen.use()
            self.ctx.disable(moderngl.DEPTH_TEST)
        except Exception:
            pass

        self.time_label.draw()

        try:
            self.ctx.enable(moderngl.DEPTH_TEST)
        except Exception:
            pass

    def on_key_press(self, symbol, modifiers):
        if symbol == key.ESCAPE:
            self.close()
            return pyglet.event.EVENT_HANDLED
        
        if not self.accepts_key_input:
            return
        if symbol == key.SPACE:
            self.del_t += .001
            print(f"Thrust: {self.del_t:.3}")
        if symbol == key.X:
            self.del_t -= .001
            print(f"Thrust: {self.del_t:.3}")

        if symbol == key.UP:
            self.del_e += .001
            print(f"Elevator Deflection: {self.del_e:.3}")
        if symbol == key.DOWN:
            self.del_e -= .001
            print(f"Elevator Deflection: {self.del_e:.3}")

        if symbol == key.LEFT:
            self.del_a += .001
            print(f"Aleron Deflection: {self.del_a:.3}")
        if symbol == key.RIGHT:
            self.del_a -= .001
            print(f"Aleron Deflection: {self.del_a:.3}")

        if symbol == key.A:
            self.del_r -= .001
            print(f'Rudder Deflection: {self.del_r:.3}')
        if symbol == key.D:
            self.del_r += .001
            print(f'Rudder Deflection: {self.del_r:.3}')

        self.u = np.array([[self.del_e,self.del_t,self.del_a,self.del_r]]).T
