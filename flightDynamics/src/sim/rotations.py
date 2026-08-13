"""sim.rotations

Shared rotation/transform helpers used across the simulation and viewer.

- Provides body-to-inertial direction cosine matrix (DCM) for 3-2-1 (roll-pitch-yaw) Euler angles
  in an NED convention (x=north, y=east, z=down).

"""  # GPT FIX
from __future__ import annotations  # GPT FIX

import numpy as np  # GPT FIX


def body_to_inertial(phi: float,theta: float,psi: float,*,dtype=np.float32) -> np.ndarray:
    """Return the 3x3 body->inertial DCM for 3-2-1 Euler angles (phi, theta, psi)."""  # GPT FIX
    c_phi = np.cos(phi)
    s_phi = np.sin(phi)
    c_theta = np.cos(theta)
    s_theta = np.sin(theta)
    c_psi = np.cos(psi)
    s_psi = np.sin(psi)

    rotation = np.array(
        [
            [c_theta * c_psi, s_phi * s_theta * c_psi - c_phi * s_psi, c_phi * s_theta * c_psi + s_phi * s_psi],
            [c_theta * s_psi, s_phi * s_theta * s_psi + c_phi * c_psi, c_phi * s_theta * s_psi - s_phi * c_psi],
            [-s_theta,        s_phi * c_theta,                        c_phi * c_theta],
        ],
        dtype=dtype,
    )
    return rotation


def model_matrix_from_pose(pn: float, pe: float, pd: float, phi: float, theta: float, psi: float, *, dtype=np.float32) -> np.ndarray:
    """Return 4x4 model matrix for rendering from pose in NED coordinates."""  # GPT FIX
    rotation = body_to_inertial(phi,theta,psi,dtype=dtype)
    matrix = np.eye(4,dtype=dtype)
    matrix[:3,:3] = rotation
    matrix[:3,3] = np.array([pn,pe,pd],dtype=dtype)
    return matrix
