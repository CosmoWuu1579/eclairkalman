"""The simulated world: the true state, and what the sensors report.

Works for any number of state variables and any sensors. States and readings
are plain numpy arrays. Knows nothing about Kalman filters.
"""
from dataclasses import dataclass

import numpy as np


class Sensor:
    def __init__(self, name, measure, noise_std):
        self.name = name
        self.measure = measure                        # state -> what this sensor sees
        self.noise_std = np.asarray(noise_std, float)  # one σ per measured value

    @property
    def R(self):
        # Noise covariance: each value's σ² on the diagonal (errors independent).
        return np.diag(self.noise_std ** 2)

    def read(self, state, rng):
        return self.measure(state) + rng.normal(0, self.noise_std)


@dataclass
class Snapshot:
    time: float
    truth: np.ndarray   # only the simulator knows this; filters must not use it
    readings: list      # (sensor, reading) pairs


class World:
    def __init__(self, motion, x0, process_noise_std, sensors, dt, seed=None):
        self.motion = motion                                 # state -> next state
        self.x = np.asarray(x0, float)
        self.process_noise_std = np.asarray(process_noise_std, float)
        self.sensors = sensors
        self.dt = dt
        self.time = 0.0
        self.rng = np.random.default_rng(seed)  # same seed -> same run

    def step(self):
        # Move the truth with the motion model plus random disturbances
        # (bumps, slip...) the model doesn't know about, then read every sensor.
        self.x = self.motion(self.x) + self.rng.normal(0, self.process_noise_std)
        self.time += self.dt
        readings = [(s, s.read(self.x, self.rng)) for s in self.sensors]
        return Snapshot(self.time, self.x.copy(), readings)
