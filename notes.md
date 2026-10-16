# Sensor fusion: how this code works

This folder simulates a differential-drive robot (two driven wheels, steering by
spinning them at different speeds). It fuses noisy sensor readings into an
estimate of the robot's state using a choice of Kalman filters, then plots each
estimate against the truth.

```
python main.py                                   # placeholder scenario (nothing moves)
python main.py --scenario driving                # robot actually drives
python main.py --scenario driving --filter kf ekf iekf ukf   # compare filters
python main.py --gate 0.999                      # reject outlier readings
python main.py --seed 3                          # a different random world
python main.py --no-plot                         # table only
```

---

## 1. The big picture

| File | Job | Imports |
|---|---|---|
| [inputs.py](inputs.py) | Building blocks for describing a robot: `Variable`, sensor classes, `Scenario` | nothing from here |
| [scenarios.py](scenarios.py) | Specific robots built from those blocks (your numbers live here) | `inputs` |
| [sim_world.py](sim_world.py) | The fake world: true state + noisy sensor readings | nothing from here |
| [kalman_filter.py](kalman_filter.py) | The filters: KF, EKF, IEKF, UKF | nothing from here |
| [main.py](main.py) | Connects everything, runs, prints a table, plots | all of the above |

`sim_world.py` and `kalman_filter.py` don't know about each other or about the
robot. They only work with plain arrays and functions. On a real robot you would
delete `sim_world.py` and feed real readings into the same filters.

### What happens each time step

```
            ┌──────────── sim_world.World ────────────┐
            │ truth = motion(truth) + random drift    │
            │ reading = sensor_function(truth) + noise│
            └───────────────────┬─────────────────────┘
                                │ readings only (never the truth)
                                ▼
            ┌──────────── kalman_filter ──────────────┐
            │ predict: move guess with motion model   │
            │ update:  compare reading vs. expected,  │
            │          correct the guess              │
            └───────────────────┬─────────────────────┘
                                ▼
            main.py logs truth + guess + uncertainty → table and plot
```

**All data is simulated.** There's no real robot or dataset. The seed
(default 0) makes every run repeatable.

---

## 2. Key ideas

| Term | Meaning | In the code |
|---|---|---|
| **State** | What we want to know but can't see directly (x, y, heading, speeds...) | `Variable`s; `kf.x` |
| **Reading / measurement** | The number a sensor gives you. Known, but noisy | `z` |
| **Motion model** | Rule for how the state changes over one time step | `Variable.motion`, `f` |
| **Sensor function** | Rule: "if the state were *this*, the sensor would read *that*." A fixed formula you write from physics. It doesn't change and isn't estimated | `SensorInput.measure`, `h` |
| **Mean** | The filter's best guess of the state | `kf.x` |
| **Covariance** | How unsure the filter is, and how errors in different variables are linked | `kf.P` |
| **σ (sigma)** | Standard deviation, the typical size of an error. Variance = σ² | |
| **Process noise (Q)** | How much the real world drifts away from the motion model each step (bumps, slip) | `Variable.process_noise` |
| **Sensor noise (R)** | How noisy each sensor is | `SensorInput.noise` |
| **Innovation** | Reading minus what the filter expected to read. "How surprised am I?" | `y` |
| **Kalman gain (K)** | How far to move the guess toward the reading (0 = ignore it, 1 = copy it) | `K` |
| **Slopes (Jacobian)** | How much each output changes when each input is nudged slightly | `jacobian()`, `F`, `H` |

### The sensor function: an example

You want to know how many coins are in a jar, but you can only weigh it.
Physics gives you the rule `weight = 500 g + 5 g × coins`. That rule is the
sensor function.

1. The filter's guess is 100 coins.
2. The sensor function predicts the scale should read 500 + 5 × 100 = **1000 g**.
3. The scale actually reads **1020 g**, so the innovation is +20 g.
4. The slope (5 g per coin) translates that back: about 4 more coins than I thought.
5. The new guess is about 104 coins. It moves only part of the way, by the gain K,
   because the scale is noisy.

The formula never changes. Only the guess does.

For the robot, the coin count is like the robot's speed (vx, omega), the scale is
like the wheel encoders, and the formula is the wheel kinematics.

### What the slopes are for

The left encoder's sensor function is `left = (vx − omega·0.30) / 0.10 = 10·vx − 3·omega`.
Nudging each state variable gives its slope row: `[x:0, y:0, theta:0, vx:10, vy:0, omega:−3]`.

The filter uses the slopes two ways:
1. **State uncertainty → reading uncertainty.** If vx is uncertain by ±0.1, the
   left reading is uncertain by about ±1 (`H P Hᵀ`).
2. **Surprise → which variables to blame.** A reading only changes the
   variables it has a non-zero slope for (`P Hᵀ` in the gain). If the left wheel
   is fast and the right isn't, the slopes say "the robot is turning", not
   "the robot is speeding up".

For a straight-line function like the encoders, the slopes are the same everywhere.
For a curved one (e.g. distance to a landmark, `cos(theta)` in the driving motion)
they depend on the current state. That difference is why the filter types exist.

### Reading the uncertainty

Each filter's belief is a bell curve: a mean and a σ. About 95% of a bell curve
lies within ±2σ of the mean, so the shaded band on the plot is where the filter
is about 95% sure the truth is.

- After the start, the band shrinks as readings arrive.
- If nothing measures a variable, its band grows over time (drift).
- **"In band" ≈ 95%:** the filter's confidence is honest.
- **Much lower:** overconfident. The model is wrong or Q is too small. This is
  the dangerous case: confident and wrong.
- **Close to 100%:** a bit cautious, which is harmless.

---

## 3. The files

### [inputs.py](inputs.py): building blocks

- **`Variable(name, initial, uncertainty, process_noise, motion=None)`**: one state
  variable.
  - `initial` is the true value at t = 0.
  - `uncertainty` is σ of the filter's first guess.
  - `process_noise` is σ of random drift per step.
  - `motion(s, dt)` returns the value one step later, where `s` is the current
    state as a dict (`s["x"]`). Leave it as `None` and the variable stays the same.
- **`SensorInput(name, noise)`**: base class. Subclasses write `measure(s)`, which
  returns a list of values, one noise σ per value.
- **`DirectSensor(name, {variable: σ})`**: reads the listed variables directly.
  Use it for gyro, GPS, compass.
- **`WheelEncoders(wheel_radius, track_width, noise)`**: left/right wheel speeds
  in rad/s, from inverse kinematics: `(vx ∓ omega·track_width/2) / wheel_radius`.
- **`Scenario(variables, sensors, dt, duration, filters, gate, seed)`**: the object
  you pass to `main.run_scenario()`. It also:
  - collects per-variable settings into lists (`names`, `initial`, ...),
  - translates between names (`s["x"]`) and the plain arrays the filters use
    (`as_dict`, `motion`, `measure_fn`),
  - checks everything fits before running (`check()`: unique names, sensors
    only use existing variables, one noise σ per sensor value).

### [scenarios.py](scenarios.py): your robot

Your constants (wheel radius 0.10 m, track width 0.60 m, dt 0.1 s, 30 s,
encoder σ 0.15 rad/s, IMU yaw-rate σ 0.05 rad/s), plus functions that build
`Scenario`s.

- **`differential_drive()`**: placeholder motion. x, y and theta have no motion
  function, so they stay put apart from random drift. **This is the default, and
  why `python main.py` shows a robot that doesn't go anywhere.**
- **`driving()`**: the same robot with `drive_x`, `drive_y` and `turn` switched on:
  - `x += (vx·cos θ − vy·sin θ)·dt`
  - `y += (vx·sin θ + vy·cos θ)·dt`
  - `θ += omega·dt`

  vx and vy are in the robot's own frame (forward, sideways). cos and sin rotate
  them onto the map.

Sensors:
- `WheelEncoders`
- `DirectSensor("imu", ...)` reading **all six** variables (see section 5).

### [sim_world.py](sim_world.py): the fake world

- **`Sensor(name, measure, noise_std)`**: `read()` returns `measure(state)` plus
  bell-curve noise. `R` is the noise covariance, σ² on the diagonal.
- **`World.step()`**:
  1. Moves the truth with the motion model plus random drift.
  2. Has every sensor read it.
  3. Returns a `Snapshot(time, truth, readings)`.
- A seeded random generator makes runs repeatable.

### [kalman_filter.py](kalman_filter.py): the filters

Every filter does `step(readings)`: `predict()` once, then `update(z, h, R)` once
per sensor. Fusing another sensor is just another update. The filter weights
each one by its R.

- **`jacobian(func, x)`**: works out slopes numerically by nudging each input
  ±0.000001. You never write derivatives by hand.
- **`correct(y, S, C)`**: the final step of *every* filter's update:
  - `K = C S⁻¹` (gain)
  - `x += K y` (move the guess)
  - `P −= K S Kᵀ` (less uncertain now)
  - With a gate set, it first rejects readings that are too surprising.

  The filters differ only in how they compute `y`, `S` and `C`.
- **`gate_threshold`**: the cut-off for "too surprising" at a given probability.

| Filter | How it handles curved functions | Notes |
|---|---|---|
| `LinearKF` (`kf`) | Computes slopes once at the start and reuses them forever | Exact if f and h are straight lines. Goes stale and overconfident when they curve (e.g. once the robot turns) |
| `EKF` (`ekf`) | Re-computes slopes at the current guess every step | The industry default |
| `IteratedEKF` (`iekf`) | Re-computes the sensor slopes up to 5× within one update, each time at the improved guess | Only differs from the EKF when a *sensor* function curves |
| `UKF` (`ukf`) | No slopes. Pushes 2n+1 sample points through the real functions and measures the spread | `alpha = 0.5` sets how far out the points sit. Best when uncertainty is large relative to the curvature |

The EKF's core equations:
```
predict:  F = slopes of f      x = f(x)       P = F P Fᵀ + Q
update:   H = slopes of h      y = z − h(x)   S = H P Hᵀ + R    C = P Hᵀ   → correct()
```

### [main.py](main.py): running everything

- **`run(scenario, filter_name)`**: builds the world and filter, then loops
  `duration / dt` steps.
  - The filter's first guess is deliberately off by a random amount matching
    each variable's `uncertainty`.
  - The filter is given readings, sensor functions and R. Never the truth.
  - Logs truth, mean and σ each step.
- **`report`**: prints RMS error and "in band" % per variable, plus how many
  readings the gate rejected.
- **`plot`**: one panel per variable, with the truth (dashed), each filter's mean
  (line) and its ±2σ band (shaded).
- **`run_scenario(scenario)`**: check, run every filter in `scenario.filters` with
  the same seed (so they see the identical world), report, plot.
- **`main()`**: command-line options. `--scenario` picks a function from
  `scenarios.py`. `--filter`, `--gate` and `--seed` override the scenario's
  settings.

---

## 4. How to change things

**Pass in your own scenario from Python:**
```python
from inputs import Scenario, Variable, DirectSensor
from main import run_scenario

run_scenario(Scenario(
    variables=[
        Variable("pos", initial=0.0, uncertainty=1.0, process_noise=0.01,
                 motion=lambda s, dt: s["pos"] + s["vel"] * dt),
        Variable("vel", initial=1.0, uncertainty=0.5, process_noise=0.05),
    ],
    sensors=[DirectSensor("gps", {"pos": 0.5})],
    filters=["kf", "ukf"],
))
```

**Add your real motion functions:** in `scenarios.py`, give a `Variable` a
`motion=` function taking `(s, dt)`. Every variable's function sees the same
*old* state, so they all update at once.

**Add a sensor type:** subclass `SensorInput` and write `measure(s)`. Example: distance
to a landmark at (5, 3):
```python
import math
from inputs import SensorInput

class LandmarkRange(SensorInput):
    def __init__(self, lx, ly, noise):
        super().__init__("range", [noise])
        self.lx, self.ly = lx, ly

    def measure(self, s):
        return [math.hypot(self.lx - s["x"], self.ly - s["y"])]
```

**Add a scenario:** write a function in `scenarios.py` that returns a `Scenario`,
then run `python main.py --scenario your_function_name`.

**Tune:**
- `process_noise` too small → overconfident, lags real changes.
- `process_noise` too large → jittery.
- Sensor `noise` should match the real sensor's datasheet or calibration.

---

## 5. Placeholders and limitations

- **The IMU reads all six variables.** Only omega's σ (0.05) is a real number. A
  real IMU can't measure x, y, vx or vy. Its gyro measures yaw rate, and its
  accelerometer measures acceleration (not in the state yet). Because every state
  is measured directly, the problem is easy and all four filters give identical
  results. A realistic version is `DirectSensor("imu", {"omega": 0.05})`. Then
  theta, x and y are only inferred from motion, their bands widen over time, and
  vy isn't observed at all.
- **The default motion is a placeholder** (nothing moves). Use `--scenario driving`.
- **Angles aren't wrapped.** Heading counts past 2π in both the truth and the
  estimate, which is consistent. If your functions wrap heading to (−π, π], the
  innovation for heading must be wrapped too.
- **The same motion model drives the truth and the filter**, and the same
  `process_noise` sets both the world's drift and the filter's Q. The filter
  knows the world perfectly, which is ideal for testing but optimistic for reality.
- **Encoders are used as measurements.** Many real robots instead feed wheel
  speeds into the predict step as a control input. That would need a `u`
  argument for the motion functions.
- **A wrong sensor function gives a wrong filter.** If the real wheel radius is
  0.098 m but the code says 0.10, every speed estimate is biased by about 2%.
  Calibrate it, or add it as a state variable so the filter estimates it.

### What was observed in testing

- **Your scenario as given:** every filter scores 95–98% in band, with identical
  numbers.
- **A harder test** (not saved here): driving with fast turns, IMU reads only yaw rate,
  noisy GPS, heading almost unknown at the start. Averaged over 5 seeds:

  | Filter | x error | heading error | heading in band |
  |---|---|---|---|
  | KF | 6.07 m | 1.84 rad | 15% |
  | EKF / IEKF | 0.24 m | 1.25 rad | 71% |
  | UKF | 0.24 m | 0.39 rad | 81% |

- **Outlier gate** (`--gate 0.999`): with wheel-slip spikes on 3% of encoder
  readings, it cut the EKF's yaw-rate error from 0.105 to 0.021 rad/s.
