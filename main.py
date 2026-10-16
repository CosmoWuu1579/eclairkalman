"""Run a Scenario through one or more Kalman filters.

From the command line (scenarios come from scenarios.py):
    python main.py                               # differential_drive, its default filters
    python main.py --scenario driving --filter ekf ukf
    python main.py --gate 0.999 --seed 3

From Python, pass any Scenario object:
    from main import run_scenario
    run_scenario(my_scenario)
"""
import argparse
import dataclasses

import matplotlib.pyplot as plt
import numpy as np

import scenarios
from kalman_filter import FILTERS
from sim_world import Sensor, World


def run(scenario, filter_name):
    sensors = [Sensor(s.name, scenario.measure_fn(s), s.noise) for s in scenario.sensors]
    world = World(scenario.motion, scenario.initial, scenario.process_noise,
                  sensors, scenario.dt, scenario.seed)

    # The filter doesn't know the true start: its first guess is off by a
    # random amount matching each variable's uncertainty.
    start_error = np.random.default_rng(scenario.seed + 1).normal(0, scenario.uncertainty)
    kf = FILTERS[filter_name](
        f=scenario.motion,
        x0=np.asarray(scenario.initial) + start_error,
        P0=np.diag(np.square(scenario.uncertainty)),
        Q=np.diag(np.square(scenario.process_noise)),
        gate=scenario.gate,
    )

    log = {"time": [], "truth": [], "mean": [], "std": []}
    for _ in range(round(scenario.duration / scenario.dt)):
        snap = world.step()
        # The filter gets readings, each sensor's model and noise -- never the truth.
        kf.step([(z, sensor.measure, sensor.R) for sensor, z in snap.readings])
        log["time"].append(snap.time)
        log["truth"].append(snap.truth)
        log["mean"].append(kf.x.copy())
        log["std"].append(np.sqrt(np.diag(kf.P)))
    return kf, {key: np.array(values) for key, values in log.items()}


def report(scenario, results):
    print(f"{'':8}" + "".join(f"{name:>16}" for name in scenario.names))
    for kf, log in results:
        error = log["mean"] - log["truth"]
        rms = np.sqrt(np.mean(error ** 2, axis=0))
        in_band = np.mean(np.abs(error) <= 2 * log["std"], axis=0)
        cells = "".join(f"{r:9.4f} ({b:4.0%})" for r, b in zip(rms, in_band))
        print(f"{kf.name:8}{cells}   rejected readings: {kf.rejected}")
    print("Each cell: RMS error (share of steps the truth was inside mean ± 2σ; ~95% is honest)")


def plot(scenario, results):
    n = len(scenario.names)
    rows = (n + 1) // 2
    fig, axes = plt.subplots(rows, 2, figsize=(12, 2.6 * rows), sharex=True, squeeze=False)
    t = results[0][1]["time"]
    for i, (ax, name) in enumerate(zip(axes.flat, scenario.names)):
        ax.plot(t, results[0][1]["truth"][:, i], "k--", label="True")
        for kf, log in results:
            mean, band = log["mean"][:, i], 2 * log["std"][:, i]
            line, = ax.plot(t, mean, label=kf.name)
            ax.fill_between(t, mean - band, mean + band, color=line.get_color(), alpha=0.2)
        ax.set_ylabel(name)
        ax.grid(True)
    for ax in axes.flat[n:]:
        ax.set_visible(False)
    for ax in axes[-1]:
        ax.set_xlabel("Time (s)")
    axes.flat[0].legend()
    fig.suptitle("Filter estimate (line) and mean ± 2σ (band)")
    fig.tight_layout()
    plt.show()


def run_scenario(scenario, show_plot=True):
    scenario.check()
    # Every filter gets the same seed, so they all see the exact same world.
    results = [run(scenario, name) for name in scenario.filters]
    report(scenario, results)
    if show_plot:
        plot(scenario, results)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--scenario", default="differential_drive",
                        help="name of a function in scenarios.py")
    parser.add_argument("--filter", nargs="+", choices=FILTERS, help="filter(s) to run")
    parser.add_argument("--gate", type=float, help="outlier gate probability, e.g. 0.999")
    parser.add_argument("--seed", type=int, help="random seed (same seed = same world)")
    parser.add_argument("--no-plot", action="store_true", help="print the table only")
    args = parser.parse_args()

    scenario = getattr(scenarios, args.scenario)()
    # Command-line options replace the scenario's own settings when given.
    overrides = {"filters": args.filter, "gate": args.gate, "seed": args.seed}
    scenario = dataclasses.replace(scenario, **{k: v for k, v in overrides.items() if v is not None})
    run_scenario(scenario, show_plot=not args.no_plot)


if __name__ == "__main__":
    main()
