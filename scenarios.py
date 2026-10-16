"""Specific robots and scenarios, built from the classes in inputs.py.

Each function returns a Scenario. Run one with
    python main.py --scenario differential_drive
"""
import math

from inputs import DirectSensor, Scenario, Variable, WheelEncoders

WHEEL_RADIUS = 0.10        # m
TRACK_WIDTH = 0.60         # m, distance between the left and right wheels
DT = 0.1                   # s
DURATION = 30.0            # s
ENCODER_NOISE = 0.15       # rad/s, σ of each wheel-speed reading
IMU_YAW_RATE_NOISE = 0.05  # rad/s


def differential_drive(moving=False):
    # moving=False: every variable just keeps its value (placeholder motion).
    # moving=True: x, y and heading follow the current velocities.
    def drive_x(s, dt):
        return s["x"] + (s["vx"] * math.cos(s["theta"]) - s["vy"] * math.sin(s["theta"])) * dt

    def drive_y(s, dt):
        return s["y"] + (s["vx"] * math.sin(s["theta"]) + s["vy"] * math.cos(s["theta"])) * dt

    def turn(s, dt):
        return s["theta"] + s["omega"] * dt

    variables = [
        #        name     start  uncertainty  process noise  motion (None = stays the same)
        Variable("x",     0.0,   0.5,         0.01,          drive_x if moving else None),
        Variable("y",     0.0,   0.5,         0.01,          drive_y if moving else None),
        Variable("theta", 0.0,   0.2,         0.005,         turn if moving else None),
        Variable("vx",    0.5,   0.3,         0.02),
        Variable("vy",    0.0,   0.3,         0.02),
        Variable("omega", 0.2,   0.3,         0.02),
    ]

    sensors = [
        WheelEncoders(WHEEL_RADIUS, TRACK_WIDTH, noise=ENCODER_NOISE),
        # IMU noise per variable it reads; placeholders except omega.
        DirectSensor("imu", {"x": 0.05, "y": 0.05, "theta": 0.02,
                             "vx": 0.05, "vy": 0.05, "omega": IMU_YAW_RATE_NOISE}),
    ]

    return Scenario(variables, sensors, dt=DT, duration=DURATION)


def driving():
    return differential_drive(moving=True)
