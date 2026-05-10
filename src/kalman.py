
# A 4-state constant-velocity Kalman filter used to smooth the noisy GPS
# stream coming back from the drone before we feed it into the planner.
#
# State vector x = [px, py, vx, vy]^T  (position & velocity in metres)
# Measurement   z = [px, py]^T          (noisy GPS converted to metres)
#
# Standard linear KF equations:
#
#     Predict
#         x_hat   = F @ x
#         P_hat   = F @ P @ F.T + Q
#     Update
#         y       = z - H @ x_hat              # innovation
#         S       = H @ P_hat @ H.T + R
#         K       = P_hat @ H.T @ inv(S)        # Kalman gain
#         x       = x_hat + K @ y
#         P       = (I - K @ H) @ P_hat


from __future__ import annotations
import numpy as np


class KalmanFilter2D:
    """Constant-velocity 2-D Kalman filter."""

    def __init__(
        self,
        dt: float = 1.0,
        process_var: float = 0.5,
        meas_var: float = 4.0,
        initial_state: np.ndarray | None = None,
    ) -> None:
        self.dt = dt

        # State transition F: position += v * dt
        self.F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0,  dt],
            [0, 0, 1,  0],
            [0, 0, 0,  1],
        ], dtype=float)

        # Measurement H: we observe position only
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
        ], dtype=float)

        # Process noise (small disturbances in velocity)
        q = process_var
        self.Q = q * np.array([
            [dt**4 / 4, 0,         dt**3 / 2, 0],
            [0,         dt**4 / 4, 0,         dt**3 / 2],
            [dt**3 / 2, 0,         dt**2,     0],
            [0,         dt**3 / 2, 0,         dt**2],
        ])

        # Measurement noise (GPS jitter, ~ meas_var m^2)
        self.R = meas_var * np.eye(2)

        # State + covariance
        self.x = (np.zeros(4) if initial_state is None
                  else np.asarray(initial_state, dtype=float))
        self.P = np.eye(4) * 50.0

    # ------------------------------------------------------------------
    def predict(self) -> np.ndarray:
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x.copy()

    def update(self, measurement: np.ndarray) -> np.ndarray:
        z = np.asarray(measurement, dtype=float).reshape(2)
        y = z - self.H @ self.x                         # innovation
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)        # Kalman gain
        self.x = self.x + K @ y
        self.P = (np.eye(4) - K @ self.H) @ self.P
        return self.x.copy()

    # ------------------------------------------------------------------
    @property
    def position(self) -> np.ndarray:
        return self.x[:2].copy()

    @property
    def velocity(self) -> np.ndarray:
        return self.x[2:].copy()