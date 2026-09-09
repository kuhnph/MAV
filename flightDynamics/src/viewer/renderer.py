import numpy as np
import moderngl
from sim.rotations import body_to_inertial

def euler_to_rotation(phi,theta,psi):
    """Backward-compatible wrapper for body->inertial DCM."""
    return body_to_inertial(phi,theta,psi,dtype=np.float32)

def model_matrix_from_state(pn, pe, pd, phi, theta, psi):
    # NED: x=north, y=east, z=down
    rotation=euler_to_rotation(phi,theta,psi)
    translation=np.array([pn,pe,pd],dtype=np.float32)
    matrix=np.eye(4,dtype=np.float32)
    matrix[:3,:3]=rotation
    matrix[:3,3]=translation
    return matrix

def perspective(fovy_deg, aspect, znear, zfar):
    f = 1.0 / np.tan(np.deg2rad(fovy_deg) / 2.0)
    matrix=np.zeros((4,4),dtype=np.float32)
    matrix[0,0]=f/aspect
    matrix[1,1]=f
    matrix[2,2]=(zfar+znear)/(znear-zfar)
    matrix[2,3]=(2*zfar*znear)/(znear-zfar)
    matrix[3,2]=-1.0
    return matrix

def look_at(eye, target, up):
    eye = np.array(eye, dtype=np.float32)
    target = np.array(target, dtype=np.float32)
    up = np.array(up, dtype=np.float32)

    f = target - eye
    f /= (np.linalg.norm(f) + 1e-9)
    r = np.cross(f, up)
    r /= (np.linalg.norm(r) + 1e-9)
    u = np.cross(r, f)

    matrix=np.eye(4,dtype=np.float32)
    matrix[0,:3]=r
    matrix[1,:3]=u
    matrix[2,:3]=-f
    matrix[:3,3]=-matrix[:3,:3]@eye
    return matrix

def build_ground_grid(extent=100000.0, minor_spacing=50.0, major_spacing=250.0, down=0.0):
    minor_lines = []
    major_lines = []

    coords = np.arange(-extent, extent + 0.5 * minor_spacing, minor_spacing, dtype=np.float32)

    for c in coords:
        is_major = np.isclose(np.mod(abs(c), major_spacing), 0.0, atol=1e-5)
        target = major_lines if is_major else minor_lines

        # constant north, varying east
        target.append([c, -extent, down])
        target.append([c,  extent, down])

        # constant east, varying north
        target.append([-extent, c, down])
        target.append([ extent, c, down])

    minor_vertices = np.array(minor_lines, dtype=np.float32) if minor_lines else np.zeros((0, 3), dtype=np.float32)
    major_vertices = np.array(major_lines, dtype=np.float32) if major_lines else np.zeros((0, 3), dtype=np.float32)

    return minor_vertices, major_vertices

class Renderer:
    def __init__(self, ctx: moderngl.Context, vertices, indices):
        self.ctx = ctx
        self.ctx.enable(moderngl.DEPTH_TEST)

        self.prog = self.ctx.program(
            vertex_shader="""
                #version 330
                uniform mat4 u_mvp;
                in vec3 in_pos;
                void main() {
                    gl_Position = u_mvp * vec4(in_pos, 1.0);
                }
            """,
            fragment_shader="""
                #version 330
                out vec4 f_color;
                void main() {
                    f_color = vec4(0.6, 0.75, 0.95, 1.0);
                }
            """,
        )

        vbo = self.ctx.buffer(vertices.tobytes())
        ibo = self.ctx.buffer(indices.tobytes())

        self.vao = self.ctx.vertex_array(self.prog, [(vbo, "3f", "in_pos")], index_buffer=ibo)
        self.u_mvp = self.prog["u_mvp"]

        self.up = np.array([0, 0, -1], dtype=np.float32)
        self.fovy = 60.0
        self.znear = 0.1
        self.zfar = 5000.0
        self.camera_mode = "chase"
        self.chase_offset = np.array([-250, -20, -30], dtype=np.float32)
        self.fixed_eye = np.array([0, -600, 0], dtype=np.float32)
        self.fixed_target = np.array([0, 0, 0], dtype=np.float32)

        # ---------- Trail ----------
        self.max_trail=1200
        self.trail=np.zeros((self.max_trail,3),dtype=np.float32)
        self.trail_count=0
        self.trail_vbo = self.ctx.buffer(reserve=self.trail.nbytes)
        self.trail_program = self.ctx.program(
            vertex_shader="""
            #version 330
            uniform mat4 u_mvp;
            in vec3 in_pos;
            void main(){
                gl_Position = u_mvp * vec4(in_pos, 1.0);
            }
        """,
        fragment_shader="""
            #version 330
            out vec4 f_color;
            void main() {
                f_color = vec4(1.0, 1.0, 0.3, 1.0);
            }
        """,
        )
        self.trail_u_mvp=self.trail_program["u_mvp"]
        self.trail_vao=self.ctx.vertex_array(self.trail_program,[(self.trail_vbo,"3f","in_pos")])

        # ---------- Ground Grid ----------
        self.grid_extent = 2000.0
        self.grid_minor_spacing = 50.0
        self.grid_major_spacing = 250.0
        self.grid_down = 0.0

        minor_grid, major_grid = build_ground_grid(
            extent=self.grid_extent,
            minor_spacing=self.grid_minor_spacing,
            major_spacing=self.grid_major_spacing,
            down=self.grid_down,
        )

        self.grid_minor_count = len(minor_grid)
        self.grid_major_count = len(major_grid)

        self.grid_prog = self.ctx.program(
            vertex_shader="""
                #version 330
                uniform mat4 u_mvp;
                in vec3 in_pos;
                void main() {
                    gl_Position = u_mvp * vec4(in_pos, 1.0);
                }
            """,
            fragment_shader="""
                #version 330
                uniform vec4 u_color;
                out vec4 f_color;
                void main() {
                    f_color = u_color;
                }
            """,
        )

        self.grid_u_mvp = self.grid_prog["u_mvp"]
        self.grid_u_color = self.grid_prog["u_color"]

        self.grid_minor_vbo = self.ctx.buffer(minor_grid.tobytes()) if self.grid_minor_count > 0 else None
        self.grid_major_vbo = self.ctx.buffer(major_grid.tobytes()) if self.grid_major_count > 0 else None

        self.grid_minor_vao = (
            self.ctx.vertex_array(self.grid_prog, [(self.grid_minor_vbo, "3f", "in_pos")])
            if self.grid_minor_vbo is not None else None
        )
        self.grid_major_vao = (
            self.ctx.vertex_array(self.grid_prog, [(self.grid_major_vbo, "3f", "in_pos")])
            if self.grid_major_vbo is not None else None
        )

    def _append_trail(self,pos):
        self.trail[self.trail_count%self.max_trail,:]=pos
        self.trail_count+=1
        n=min(self.trail_count,self.max_trail)
        if self.trail_count<self.max_trail:
            data = self.trail[:n]
        else:
            start=self.trail_count%self.max_trail
            data = np.vstack((self.trail[start:], self.trail[:start]))
        self.trail_vbo.write(data.tobytes())
        return n

    def _camera_matrices(self, pos, aspect):
        if self.camera_mode == 'chase':
            eye = pos + self.chase_offset
            target = pos
        elif self.camera_mode == "fixed":
            eye = self.fixed_eye
            target = self.fixed_target
        else:
            eye = self.fixed_eye
            target = pos

        view=look_at(eye,target,self.up)
        projection=perspective(self.fovy,aspect,self.znear,self.zfar)
        return projection,view

    def draw(self, width, height, model_mat):
        fb_w, fb_h = self.ctx.screen.size
        self.ctx.viewport = (0, 0, int(fb_w), int(fb_h))

        aspect = width / max(height, 1)
        pos = model_mat[:3, 3]

        trail_length=self._append_trail(pos)
        projection,view=self._camera_matrices(pos,aspect)
        mvp_world=projection@view@np.eye(4,dtype=np.float32)

        # Draw ground grid
        self.grid_u_mvp.write(mvp_world.T.tobytes())

        if self.grid_minor_vao is not None and self.grid_minor_count > 0:
            self.grid_u_color.value = (0.22, 0.26, 0.30, 1.0)
            self.grid_minor_vao.render(mode=moderngl.LINES, vertices=self.grid_minor_count)

        if self.grid_major_vao is not None and self.grid_major_count > 0:
            self.grid_u_color.value = (0.38, 0.42, 0.48, 1.0)
            self.grid_major_vao.render(mode=moderngl.LINES, vertices=self.grid_major_count)

        # Draw trail in world space
        self.trail_u_mvp.write(mvp_world.T.tobytes())
        self.trail_vao.render(mode=moderngl.LINE_STRIP,vertices=trail_length)

        # Draw aircraft mesh
        mvp=projection@view@model_mat
        self.u_mvp.write(mvp.T.tobytes())
        self.vao.render()
