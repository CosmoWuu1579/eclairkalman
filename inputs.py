"""The building blocks for describing a robot and scenario.

Build a Scenario out of Variables and sensors, then pass it to
main.run_scenario(). scenarios.py has examples.
"""
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np


@dataclass
class Variable:
    # One thing the filter estimates (e.g. x, heading, yaw rate).
    name: str
    initial: float                      # true value at t = 0
    uncertainty: float                  # σ of the filter's first guess
    process_noise: float                # σ of random drift per step
    motion: Optional[Callable] = None   # (s, dt) -> value one step later; None = stays the same

    def next(self, s, dt):
        return s[self.name] if self.motion is None else self.motion(s, dt)


class SensorInput:
    # A sensor: a name, one noise σ per value it reads, and measure(s), which
    # says what it would read if the state were exactly s.
    def __init__(self, name, noise):
        self.name = name
        self.noise = list(noise)

    def measure(self, s):
        raise NotImplementedError


class DirectSensor(SensorInput):
    # Reads some state variables directly, e.g. an IMU or a GPS.
    # noise: {variable name: σ}, one entry per variable it reads.
    def __init__(self, name, noise):
        super().__init__(name, noise.values())
        self.reads = list(noise)

    def measure(self, s):
        return [s[name] for name in self.reads]


class WheelEncoders(SensorInput):
    # Left and right wheel speeds (rad/s) of a differential drive.
    def __init__(self, wheel_radius, track_width, noise, name="encoders"):
        super().__init__(name, [noise, noise])
        self.wheel_radius = wheel_radius
        self.track_width = track_width

    def measure(self, s):
        # Inverse kinematics: body velocity -> how fast each wheel must spin.
        # Turning left (positive omega) speeds up the right wheel.
        turn = s["omega"] * self.track_width / 2
        return [(s["vx"] - turn) / self.wheel_radius,
                (s["vx"] + turn) / self.wheel_radius]


@dataclass
class Scenario:
    variables: list                     # Variable objects, in state order
    sensors: list                       # SensorInput objects
    dt: float = 0.1                     # s, time step
    duration: float = 30.0              # s
    filters: list = field(default_factory=lambda: ["ekf"])  # "kf", "ekf", "iekf", "ukf"
    gate: Optional[float] = None        # e.g. 0.999 to ignore readings with < 0.1% chance of being real
    seed: int = 0                       # same seed -> same simulated world

    # Per-variable settings gathered into lists, in state order.
    @property
    def names(self):
        return [v.name for v in self.variables]

    @property
    def initial(self):
        return [v.initial for v in self.variables]

    @property
    def uncertainty(self):
        return [v.uncertainty for v in self.variables]

    @property
    def process_noise(self):
        return [v.process_noise for v in self.variables]

    # The filters work on plain arrays; Variables and sensors talk in names
    # (s["x"]). These translate between the two.
    def as_dict(self, x):
        return dict(zip(self.names, x))

    def motion(self, x):
        s = self.as_dict(x)
        return np.array([v.next(s, self.dt) for v in self.variables])

    def measure_fn(self, sensor):
        return lambda x: np.asarray(sensor.measure(self.as_dict(x)), float)

    def check(self):
        # Fail early, with a clear message, if the pieces don't fit together.
        if len(set(self.names)) != len(self.names):
            raise ValueError(f"variable names must be unique: {self.names}")
        start = self.as_dict(self.initial)
        for sensor in self.sensors:
            try:
                size = len(sensor.measure(start))
            except KeyError as missing:
                raise ValueError(f"sensor '{sensor.name}' uses variable {missing}, "
                                 f"which isn't one of {self.names}") from None
            if size != len(sensor.noise):
                raise ValueError(f"sensor '{sensor.name}' reads {size} values "
                                 f"but has {len(sensor.noise)} noise σ's")
