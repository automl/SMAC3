from __future__ import annotations

# We don't want to create a "real" package here so we just work with this hack
import sys

sys.path.insert(0, ".")

import logging

logging.disable(9999)

from typing import Any

import numpy as np
import json
import socket  # noqa: E402
from collections import defaultdict
from pathlib import Path

import pandas as pd
from src.tasks import TASKS  # noqa: E402
from src.utils.exceptions import NotSupportedError  # noqa: E402
from src.utils.styled_plot import plt  # noqa: E402
from src.wrappers.v140 import Version140  # noqa: E402
from src.wrappers.v200a3 import Version200a3  # noqa: E402
from src.wrappers.wrapper import Wrapper  # noqa: E402

SEEDS = [0, 50, 100, 150, 200]
WRAPPERS = [Version140, Version200a3]
RAW_FILENAME = Path("report/raw.json")


class Benchmark:
    """Selects the right wrapper (based on the environment), runs, and saves the src."""

    def __init__(self, overwrite: bool = False) -> None:
        import smac

        self._version: str | None = None
        for version in ["__version__", "version"]:
            if hasattr(smac, version):
                version = getattr(smac, version)
                break

        if version is None:
            raise RuntimeError("Could not find version of SMAC.")
        else:
            version = version.replace("v", "")

        self._wrapper: Wrapper | None = None
        for wrapper in WRAPPERS:
            if version in wrapper.supported_versions:
                self._wrapper = wrapper
                break

        if self._wrapper is None:
            raise RuntimeError(f"Could not find a wrapper for version {version}.")

        print(f"Found wrapper for version {version}.")

        self._version: str = version
        self._computer: str = str(socket.gethostname())
        self._data: Any = {}
        self._overwrite: bool = overwrite

    def _load_data(self) -> None:
        """Loads data from the file to object cache."""
        print("Loading data...")
        RAW_FILENAME.parent.mkdir(parents=True, exist_ok=True)

        # First, we load all the previous data
        if RAW_FILENAME.exists() and not self._overwrite:
            with open(RAW_FILENAME, "r") as f:
                self._data = json.load(f)
        else:
            self._data = {}

    def _save_data(self) -> None:
        """Saves the internal data to the file."""
        print("Saving data...")
        with open(str(RAW_FILENAME), "w") as f:
            json.dump(self._data, f, indent=4)

    def _fill_keys(self) -> None:
        """Fill data with keys based on computer name, tasks, and selected version."""
        print("Filling keys...")
        if self._computer not in self._data:
            self._data[self._computer] = {}

        for task in TASKS:
            if task.id not in self._data[self._computer]:
                self._data[self._computer][task.id] = {}

            if "name" not in self._data[self._computer][task.id]:
                self._data[self._computer][task.id]["name"] = task.name

            if "objective" not in self._data[self._computer][task.id]:
                self._data[self._computer][task.id]["objective"] = task.objectives[0]

            if "trajectory_trials" not in self._data[self._computer][task.id]:
                self._data[self._computer][task.id]["trajectory_trials"] = {}

            if "trajectory_walltime" not in self._data[self._computer][task.id]:
                self._data[self._computer][task.id]["trajectory_walltime"] = {}

            if self._version not in self._data[self._computer][task.id]["trajectory_trials"]:
                self._data[self._computer][task.id]["trajectory_trials"][self._version] = {}

            if self._version not in self._data[self._computer][task.id]["trajectory_walltime"]:
                self._data[self._computer][task.id]["trajectory_walltime"][self._version] = {}

        self._save_data()

    def _get_mean_std(self, values: list[tuple[list[float], list[float]]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        # First we have to get all x values
        X: list[float] = []
        for x_values, _ in values:
            for x in x_values:
                if x not in X:
                    X.append(x)

        # Let's sort the x values
        X.sort()
        
        if len(X) == 0:
            return np.array([]), np.array([]), np.array([])

        # Now we need to look for corresponding y values
        # Each x is associated with #seed y values
        Y: list[list[float]] = []
        for x in X:
            y = []

            # Iterate over the values again
            for x_values, y_values in values:
                # Findest closest value to x
                idx = min(range(len(x_values)), key=lambda i: abs(x_values[i] - x))
                y.append(y_values[idx])

            Y.append(y)

        # Make numpy arrays
        Y_array = np.array(Y)

        mean = np.mean(Y_array, axis=1)
        std = np.std(Y_array, axis=1)

        return np.array(X), mean, std

    def run(self) -> None:
        # Get name of the current computer (for comparison purposes)
        assert self._wrapper is not None

        # First, we load all the previous data
        self._load_data()

        # Fill keys of the data object
        self._fill_keys()

        # Now iterate over all tasks
        for task in TASKS:
            for seed in SEEDS:
                print(f"--- Start {task.name} with seed {seed}")

                # Some variables
                d = self._data[self._computer][task.id]

                # Check if we already calculated the data before
                if (
                    str(seed) in d["trajectory_trials"][self._version]
                    and str(seed) in d["trajectory_walltime"][self._version]
                ):
                    print(f"Already done {task.name}. Skipping.")
                    continue
                else:
                    print("Not found in the data. Running...")

                task_wrapper = self._wrapper(task, seed)
                try:
                    task_wrapper.run()
                except NotSupportedError:
                    print("Not supported. Skipping.")
                    continue
                except Exception as e:
                    print("Something went wrong:")
                    print(e)
                    continue

                for sort_by in ["trials", "walltime"]:
                    # Now we collect some metrics
                    try:
                        X, Y = task_wrapper.get_trajectory(sort_by=sort_by)
                    except NotSupportedError:
                        continue

                    d[f"trajectory_{sort_by}"][self._version][str(seed)] = (X, Y)

                self._save_data()

        # Plot things
        self._plot_trajectory()

        # Write things
        self._write_table()

    def _plot_trajectory(self) -> None:
        d = self._data[self._computer]

        for sort_by in ["trials", "walltime"]:
            len_tasks = len(d.keys())

            traj = f"trajectory_{sort_by}"
            filename = Path(f"report/{traj}.png")

            plt.figure(rows=len_tasks)
            plt.tight_layout()
            plt.subplots_adjust(wspace=0.6)

            i = 0
            for task in TASKS:
                objective = d[task.id]["objective"]

                plt.subplot(len_tasks, 1, i + 1)
                plt.title(task.name)

                for version, seeds in d[task.id][traj].items():
                    X, Y_mean, Y_std = self._get_mean_std(seeds.values())

                    plt.plot(X, Y_mean, label=version, linewidth=0.75)
                    plt.fill_between(X, Y_mean - Y_std, Y_mean + Y_std, alpha=0.25)
                    # plt.scatter(X, Y_mean, s=3, marker="x")

                plt.legend()
                plt.ylabel(objective)

                if task.y_log_scale:
                    plt.yscale("log")
                    pass
                else:
                    if task.y_limit is not None:
                        plt.ylim(task.y_limit[0], task.y_limit[1])

                if task.x_log_scale:
                    plt.xscale("log")
                else:
                    if sort_by == "trials" and task.n_trials < 1000:
                        plt.xlim(0, task.n_trials)

                    if sort_by == "walltime" and task.walltime_limit < 1000:
                        plt.xlim(0, task.walltime_limit)

                if i == len_tasks - 1:
                    plt.xlabel(sort_by)

                i += 1

            plt.save_figure(str(filename))

    def _write_table(self) -> None:
        """Writes a table with the best result found so far."""
        d = self._data[self._computer]

        versions = set()
        for task in TASKS:
            for version in d[task.id]["trajectory_walltime"].keys():
                versions.add(version)

        # For each version a column
        data = defaultdict(list)
        for task in TASKS:
            data["tasks"].append(task.name)
            data["n_trials"].append(str(task.n_trials))
            data["walltime"].append(str(task.walltime_limit))
            data["instances"].append(str(task.use_instances))

            for version in versions:
                not_found = False
                for seed in SEEDS:
                    if seed not in d[task.id]["trajectory_walltime"][version]:
                        not_found = True
                        break

                if not_found:
                    data[f"{version} (cost / time)"].append("N/A")
                    continue

                values = d[task.id]["trajectory_walltime"][version].values()
                X, Y_mean, Y_std = self._get_mean_std(values)
                data[f"{version} (cost / time)"].append(
                    f"{round(Y_mean[-1], 2)} +- {round(Y_std[-1], 2)} / {round(X[-1], 2)}"
                )

        # Now make a dataframe
        df = pd.DataFrame(data=data)
        df.to_csv("report/table.csv", index=False)


if __name__ == "__main__":
    benchmark = Benchmark(overwrite=False)
    benchmark.run()
