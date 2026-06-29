"""Floating-base EKF: fuses simulated IMU + contact-FK for base state estimation.

State: x = [pos (3), vel (3), rpy (3)] = 9 states  (simplified orientation as Euler)
IMU: accelerometer a_meas (3, body frame) + gyroscope w_meas (3, body frame)

This is a simulation-ready EKF that adds realistic IMU noise so downstream
controllers see estimated rather than ground-truth base state.
"""

import numpy as np


# IMU noise parameters (realistic MEMS IMU)
SIGMA_ACCEL = 0.05        # m/s^2 per axis
SIGMA_GYRO = 0.005        # rad/s per axis
SIGMA_CONTACT_POS = 0.003 # m (foot FK position noise)

GRAVITY_WORLD = np.array([0.0, 0.0, -9.81])


def _rmat_x(angle: float) -> np.ndarray:
    """Rotation matrix about X axis (roll)."""
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[1, 0, 0],
                     [0, c, -s],
                     [0, s,  c]])


def _rmat_y(angle: float) -> np.ndarray:
    """Rotation matrix about Y axis (pitch)."""
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[ c, 0, s],
                     [ 0, 1, 0],
                     [-s, 0, c]])


def _rmat_z(angle: float) -> np.ndarray:
    """Rotation matrix about Z axis (yaw)."""
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0],
                     [s,  c, 0],
                     [0,  0, 1]])


def rpy_to_rotmat(rpy: np.ndarray) -> np.ndarray:
    """Rotation matrix from roll-pitch-yaw Euler angles.

    R = Rz(yaw) @ Ry(pitch) @ Rx(roll)
    Transforms vectors from body frame to world frame.
    """
    roll, pitch, yaw = rpy
    return _rmat_z(yaw) @ _rmat_y(pitch) @ _rmat_x(roll)


def skew(v: np.ndarray) -> np.ndarray:
    """Skew-symmetric matrix for cross product: skew(v) @ u = v x u."""
    return np.array([[ 0,    -v[2],  v[1]],
                     [ v[2],  0,    -v[0]],
                     [-v[1],  v[0],  0   ]])


class BaseEKF:
    """Extended Kalman Filter for floating-base pose and velocity estimation.

    Fuses simulated IMU measurements with contact kinematics (foot FK) to
    estimate the robot base state without relying on ground-truth simulation
    data.  The EKF is PASSIVE — it does not modify the MuJoCo simulation.

    State vector (9-dimensional):
        x[0:3]  = position p (world frame, m)
        x[3:6]  = velocity v (world frame, m/s)
        x[6:9]  = orientation rpy = [roll, pitch, yaw] (rad, ZYX Euler)

    Parameters
    ----------
    dt : float
        Integration time step matching the simulation dt (seconds).
    """

    def __init__(self, dt: float = 0.002) -> None:
        self.dt = dt

        # Process noise covariance Q (9x9)
        # Driven by IMU noise; position noise from velocity integration is
        # captured implicitly through the Jacobian propagation.
        self.Q = np.diag(
            [1e-6] * 3            # position process noise (small, dominated by FK)
            + [SIGMA_ACCEL ** 2] * 3   # velocity noise from accelerometer
            + [SIGMA_GYRO ** 2] * 3    # orientation noise from gyroscope
        ) * dt

        # Measurement noise covariance R_meas (3x3, contact position)
        self.R_meas = np.diag([SIGMA_CONTACT_POS ** 2] * 3)

        # State and covariance (initialised to zero/identity; call reset())
        self._x = np.zeros(9)
        self._P = np.eye(9)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def reset(self,
              pos0: np.ndarray,
              vel0: np.ndarray,
              rpy0: np.ndarray) -> None:
        """Reset filter state.

        Parameters
        ----------
        pos0 : array (3,)  Initial position in world frame.
        vel0 : array (3,)  Initial velocity in world frame.
        rpy0 : array (3,)  Initial roll-pitch-yaw (rad).
        """
        self._x = np.concatenate([pos0, vel0, rpy0]).astype(float)
        # Initial covariance: tighter on position/orientation, looser on velocity
        self._P = np.diag([0.01] * 3 + [0.1] * 3 + [0.01] * 3)

    def predict(self, a_imu: np.ndarray, w_imu: np.ndarray) -> None:
        """Propagate state with IMU measurements.

        Parameters
        ----------
        a_imu : array (3,)
            Accelerometer measurement in body frame (m/s^2).
            For a stationary robot this should be approximately [0, 0, 9.81]
            because the sensor measures the reaction force opposing gravity.
        w_imu : array (3,)
            Gyroscope measurement in body frame (rad/s).
        """
        dt = self.dt
        x = self._x.copy()

        p = x[0:3]
        v = x[3:6]
        rpy = x[6:9]

        R = rpy_to_rotmat(rpy)

        # Specific force (gravity subtracted; IMU reports -g when stationary)
        # World-frame acceleration: R @ a_body + g_world
        # When sensor reads [0,0,9.81] body-frame and R~I: a_world = [0,0,9.81] + [0,0,-9.81] = 0
        a_world = R @ a_imu + GRAVITY_WORLD

        # Euler integration (process model)
        p_new = p + v * dt
        v_new = v + a_world * dt
        rpy_new = rpy + w_imu * dt  # small-angle first-order gyro integration

        self._x[0:3] = p_new
        self._x[3:6] = v_new
        self._x[6:9] = rpy_new

        # ---- Linearise process model to get Jacobian F (9x9) ----
        # df_p/dv = I*dt
        # df_v/drpy: derivative of (R(rpy) @ a_imu) w.r.t. rpy
        #   dR/droll  @ a_imu, dR/dpitch @ a_imu, dR/dyaw @ a_imu
        dR_droll, dR_dpitch, dR_dyaw = self._drpy_jacobians(rpy)

        F = np.eye(9)
        # dp/dv
        F[0:3, 3:6] = np.eye(3) * dt
        # dv/drpy
        F[3:6, 6:9] = np.column_stack([
            dR_droll @ a_imu,
            dR_dpitch @ a_imu,
            dR_dyaw @ a_imu,
        ]) * dt

        # Covariance propagation: P = F P F^T + Q
        self._P = F @ self._P @ F.T + self.Q

    def update_contact(self,
                       foot_world_fk: np.ndarray,
                       foot_local_pos: np.ndarray) -> None:
        """Correct state using foot position from forward kinematics.

        The contact kinematic constraint is: foot_world = base_pos + R @ foot_local.
        Rearranged, this gives an indirect measurement of base position:
            z_base_implied = foot_world_fk - R @ foot_local
        which should equal p.  We use this as the measurement for the EKF:
            z_meas = foot_world_fk - R(rpy) @ foot_local  (implied base position)
            h(x)   = p
            residual = z_meas - p

        This form keeps the Jacobian simple (H[:, 0:3] = I) and correctly drives
        the base position estimate so that p ≈ foot_world_fk - R @ foot_local.

        Parameters
        ----------
        foot_world_fk : array (3,)
            Foot position in world frame computed from FK (measurement z).
        foot_local_pos : array (3,)
            Foot position in base (body) frame from nominal kinematics.
        """
        x = self._x
        p = x[0:3]
        rpy = x[6:9]
        R = rpy_to_rotmat(rpy)

        foot_local = foot_local_pos

        # Contact kinematic constraint: foot_world = p + R @ foot_local
        # Predicted foot world position
        h_full = p + R @ foot_local

        # Full 3D residual
        y_full = foot_world_fk - h_full

        # ---- Measurement Jacobian H (3x9) ----
        dR_droll, dR_dpitch, dR_dyaw = self._drpy_jacobians(rpy)

        # Use only x and y components of the contact constraint.
        # The z-constraint (base height) is handled separately via the IMU
        # and terrain height prior.  This avoids corrupting the z estimate
        # when foot_local_z encodes a nominal leg length rather than
        # the current ground height.
        H = np.zeros((3, 9))
        H[0:3, 0:3] = np.eye(3)
        H[0:3, 6:9] = np.column_stack([
            dR_droll @ foot_local,
            dR_dpitch @ foot_local,
            dR_dyaw @ foot_local,
        ])

        # Zero out z row so the update only corrects horizontal position
        H[2, :] = 0.0
        y = y_full.copy()
        y[2] = 0.0

        # Kalman gain: K = P H^T (H P H^T + R)^{-1}
        S = H @ self._P @ H.T + self.R_meas
        K = self._P @ H.T @ np.linalg.inv(S)

        # State update
        self._x = x + K @ y

        # Covariance update (Joseph form for numerical stability)
        I_KH = np.eye(9) - K @ H
        self._P = I_KH @ self._P @ I_KH.T + K @ self.R_meas @ K.T

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    @property
    def pos(self) -> np.ndarray:
        """Estimated base position in world frame (3,)."""
        return self._x[0:3].copy()

    @property
    def vel(self) -> np.ndarray:
        """Estimated base velocity in world frame (3,)."""
        return self._x[3:6].copy()

    @property
    def rpy(self) -> np.ndarray:
        """Estimated base orientation as roll-pitch-yaw (3,)."""
        return self._x[6:9].copy()

    @property
    def state(self) -> np.ndarray:
        """Full 9-dimensional state vector (copy)."""
        return self._x.copy()

    @property
    def covariance(self) -> np.ndarray:
        """9x9 state covariance matrix (copy)."""
        return self._P.copy()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _drpy_jacobians(rpy: np.ndarray):
        """Partial derivatives of rotation matrix R(rpy) w.r.t. roll, pitch, yaw.

        Returns dR/droll, dR/dpitch, dR/dyaw each as 3x3 matrices.
        """
        roll, pitch, yaw = rpy
        cr, sr = np.cos(roll), np.sin(roll)
        cp, sp = np.cos(pitch), np.sin(pitch)
        cy, sy = np.cos(yaw), np.sin(yaw)

        # R = Rz(yaw) @ Ry(pitch) @ Rx(roll)
        # Precompute Rz @ Ry and Rz for chain rule

        # dRx/droll
        dRx = np.array([[0,  0,  0],
                         [0, -sr, -cr],
                         [0,  cr, -sr]])

        Ry = _rmat_y(pitch)
        Rz = _rmat_z(yaw)

        dR_droll = Rz @ Ry @ dRx

        # dRy/dpitch
        dRy = np.array([[-sp, 0,  cp],
                         [ 0,  0,  0],
                         [-cp, 0, -sp]])
        Rx = _rmat_x(roll)
        dR_dpitch = Rz @ dRy @ Rx

        # dRz/dyaw
        dRz = np.array([[-sy, -cy, 0],
                         [ cy, -sy, 0],
                         [  0,   0, 0]])
        dR_dyaw = dRz @ Ry @ Rx

        return dR_droll, dR_dpitch, dR_dyaw
