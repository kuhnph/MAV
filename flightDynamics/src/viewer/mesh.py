# src/viewer/mesh.py - 2/15/2026
"""viewer.mesh

Defines simple hard-coded aircraft mesh vertices/indices used by the GPU renderer.

""" 
import numpy as np

def aircraft_model_mesh(scale=5.0):
    # Copy the same model-space points from your drawPlane.py (before rotation/translation)
    fuse_l1, fuse_l2, fuse_l3 = 2.0, 0.1, 5.0
    fuse_w, fuse_h = 1.0, 1.0
    wing_l, wing_w = 1.5, 7.0
    tail_h = 1.5
    tailwing_w, tailwing_l = 3.3, 0.75

    vertices = scale*np.array([
        [ fuse_l1, 0, 0],                         # 0
        [ fuse_l2,  fuse_w/2, -fuse_h/2],          # 1
        [ fuse_l2, -fuse_w/2, -fuse_h/2],          # 2
        [ fuse_l2, -fuse_w/2,  fuse_h/2],          # 3
        [ fuse_l2,  fuse_w/2,  fuse_h/2],          # 4
        [-fuse_l3, 0, 0],                          # 5
        [0,  0.5*wing_w, 0],                       # 6
        [-wing_l,  0.5*wing_w, 0],                 # 7
        [-wing_l, -0.5*wing_w, 0],                 # 8
        [0, -0.5*wing_w, 0],                       # 9
        [-(fuse_l3-tailwing_l),  0.5*tailwing_w, 0], # 10
        [-fuse_l3,  0.5*tailwing_w, 0],            # 11
        [-fuse_l3, -0.5*tailwing_w, 0],            # 12
        [-(fuse_l3-tailwing_l), -0.5*tailwing_w, 0], # 13
        [-(fuse_l3-tailwing_l), 0, 0],             # 14
        [-fuse_l3, 0, -tail_h],                    # 15
    ], dtype=np.float32)

    # Faces from your drawPlane.py, but now as vertex indices.
    # We'll triangulate polygons for the GPU.
    faces = [
        [0, 1, 2], [0, 2, 3], [0, 1, 4], [0, 4, 3],  # nose-ish
        [1, 2, 5], [2, 3, 5], [3, 4, 5], [1, 4, 5],  # fuselage
        [6, 7, 8], [6, 8, 9],                        # wing quad -> 2 tris
        [10, 11, 12], [10, 12, 13],                  # tail quad -> 2 tris
        [5, 14, 15],                                 # vertical stab tri
    ]
    indices=np.array(faces,dtype=np.uint32).reshape(-1)
    return vertices,indices
