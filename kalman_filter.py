"""Kalman filters for any number of state variables and any sensors.

All filters share one recipe:
    predict()          time passed: move the guess with the motion model f
    update(z, h, R)    a sensor reading z arrived (h = what that sensor sees
                       for a given state, R = its noise covariance): correct
They differ only in how they deal with f and h being curvy (nonlinear).

    LinearKF      plain Kalman filter: uses slopes from the first step forever
    EKF           Extended: re-computes slopes at the current guess every step
    IteratedEKF   EKF that re-computes slopes a few times within each update
    UKF           Unscented: pushes sample points through f and h, no slopes
"""
from statistics import NormalDist

import numpy as np


def jacobian(func, x, eps=1e-6):
    # Slope of every output with respect to every input, found by nudging
    # each input a tiny bit up and down. Row i, column j = ∂ out[i] / ∂ x[j].
    columns = []
    for i in range(x.size):
        dx = np.zeros_like(x)
        dx[i] = eps
        columns.append((func(x + dx) - func(x - dx)) / (2 * eps))
    return np.column_stack(columns)


def gate_threshold(probability, size):
    # How surprising a reading may be before we call it an outlier.
    # (Chi-squared quantile, Wilson-Hilferty approximation.)
    z = NormalDist().inv_cdf(probability)
    a = 2 / (9 * size)
    return size * (1 - a + z * a ** 0.5) ** 3


class KalmanFilter:
    name = "base"

    def __init__(self, f, x0, P0, Q, gate=None):
        self.f = f                          # motion model: state -> next state
        self.x = np.asarray(x0, float)      # mean of our belief (best guess)
        self.P = np.asarray(P0, float)      # covariance of our belief (uncertainty)
        self.Q = np.asarray(Q, float)       # uncertainty the world adds each step
        self.gate = gate                    # e.g. 0.999 to reject outliers, None = off
        self.rejected = 0

    def step(self, readings):
        self.predict()
        for z, h, R in readings:
            self.update(np.asarray(z, float), h, R)

    def correct(self, y, S, C):
        # The last part of every update, shared by all filters.
        #   y  innovation: the reading minus what we expected to read
        #   S  how big we expected y to be (covariance of y)
        #   C  how the state and the reading vary together
        if self.gate and y @ np.linalg.solve(S, y) > gate_threshold(self.gate, y.size):
            self.rejected += 1  # far too surprising: probably slip or a glitch
            return
        K = C @ np.linalg.inv(S)            # Kalman gain
        self.x = self.x + K @ y
        self.P = self.P - K @ S @ K.T
        self.P = (self.P + self.P.T) / 2    # undo rounding drift; P must stay symmetric


class EKF(KalmanFilter):
    name = "EKF"

    def slopes(self, func, x):
        return jacobian(func, x)

    def predict(self):
        F = self.slopes(self.f, self.x)     # slopes at the old guess
        self.x = self.f(self.x)             # the guess moves with the real model
        self.P = F @ self.P @ F.T + self.Q  # the uncertainty moves with the slopes

    def update(self, z, h, R):
        H = self.slopes(h, self.x)
        S = H @ self.P @ H.T + R
        self.correct(z - h(self.x), S, self.P @ H.T)


class LinearKF(EKF):
    # The plain Kalman filter assumes f and h are straight lines, so it works
    # out their slopes once and reuses them. Exact if they really are linear.
    name = "KF"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.saved_slopes = {}

    def slopes(self, func, x):
        if func not in self.saved_slopes:
            self.saved_slopes[func] = jacobian(func, x)
        return self.saved_slopes[func]


class IteratedEKF(EKF):
    # The EKF takes slopes at the guess from BEFORE the reading. If the reading
    # moves the guess a lot, those slopes were taken at the wrong place. So:
    # update, re-take the slopes at the new guess, redo the update, repeat.
    name = "IEKF"
    iterations = 5

    def update(self, z, h, R):
        x_lin = self.x                      # where we take the slopes
        for _ in range(self.iterations):
            H = jacobian(h, x_lin)
            S = H @ self.P @ H.T + R
            y = z - h(x_lin) - H @ (self.x - x_lin)
            x_lin = self.x + self.P @ H.T @ np.linalg.solve(S, y)
        self.correct(y, S, self.P @ H.T)


class UKF(KalmanFilter):
    # Instead of slopes, pick 2n+1 sample states ("sigma points") spread
    # around the guess according to P, push each through the real function,
    # and measure the mean and spread of where they land.
    name = "UKF"
    alpha = 0.5  # how far out the points sit (1 = wide, small = close, EKF-like)

    def sigma_points(self):
        n = self.x.size
        spread = self.alpha ** 2 * n                     # points sit √spread σ's out
        L = np.linalg.cholesky(spread * self.P)          # "matrix square root" of P
        points = np.vstack([self.x, self.x + L.T, self.x - L.T])  # one point per row
        w_mean = np.full(2 * n + 1, 1 / (2 * spread))    # weights for averaging points
        w_mean[0] = 1 - n / spread
        w_cov = w_mean.copy()                            # weights for measuring spread
        w_cov[0] += 3 - self.alpha ** 2                  # (standard tweak for bell curves)
        return points, w_mean, w_cov

    def predict(self):
        points, w_mean, w_cov = self.sigma_points()
        moved = np.array([self.f(p) for p in points])
        self.x = w_mean @ moved
        d = moved - self.x
        self.P = d.T @ (w_cov[:, None] * d) + self.Q

    def update(self, z, h, R):
        points, w_mean, w_cov = self.sigma_points()
        seen = np.array([h(p) for p in points])  # what each point says the sensor reads
        z_pred = w_mean @ seen
        dz, dx = seen - z_pred, points - self.x
        S = dz.T @ (w_cov[:, None] * dz) + R
        self.correct(z - z_pred, S, dx.T @ (w_cov[:, None] * dz))


FILTERS = {"kf": LinearKF, "ekf": EKF, "iekf": IteratedEKF, "ukf": UKF}
