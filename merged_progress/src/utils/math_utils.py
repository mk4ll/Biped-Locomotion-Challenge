"""Small math helpers: rotations, orientation error."""
import numpy as np
import mujoco


def quat_to_mat(q):
    """MuJoCo quaternion [w,x,y,z] -> 3x3 rotation matrix."""
    m = np.zeros(9)
    mujoco.mju_quat2Mat(m, q)
    return m.reshape(3, 3)


def orientation_error(R, R_des):
    """Rotation-vector error so that w_des ~ Kp * err drives R -> R_des.

    err = vee(skew part) of R_des R^T, i.e. the world-frame rotation vector
    rotating current orientation toward the desired one.
    """
    Rerr = R_des @ R.T
    # axis-angle of Rerr via mju
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, Rerr.reshape(9))
    # rotation vector = 2 * (w>0 ? xyz : -xyz) scaled by angle; use mju_quat2Vel-like
    angle = 2.0 * np.arctan2(np.linalg.norm(quat[1:]), quat[0])
    axis = quat[1:]
    n = np.linalg.norm(axis)
    if n < 1e-9:
        return np.zeros(3)
    return axis / n * angle
