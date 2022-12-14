from __future__ import annotations

# We don't want to create a "real" package here so we just work with this hack
import sys
sys.path.insert(0, ".")

import logging
logging.disable(9999)

import json
from collections import defaultdict
from pathlib import Path

import pandas as pd
from typing import Any
from benchmark.wrappers.v200a3 import Version200a3  # noqa: E402
from benchmark.wrappers.v140 import Version140  # noqa: E402
from benchmark.tasks import tasks  # noqa: E402
from benchmark.utils.exceptions import NotSupportedError  # noqa: E402
import socket  # noqa: E402
from benchmark.utils.styled_plot import plt  # noqa: E402


WRAPPERS = [Version140, Version200a3]


class Benchmark:
    """Selects the right wrapper (based on the environment), runs, and saves the benchmark."""

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

        self._wrapper = None
        for wrapper in WRAPPERS:
            if version in wrapper.supported_versions:
                self._wrapper = wrapper
                break

        if self._wrapper is None:
            raise RuntimeError(f"Could not find a wrapper for version {version}.")
        
        print(f"Found wrapper for version {version}.")

        self._version = version
        self._computer = socket.gethostname()
        self._data: Any = {}
        self._overwrite = overwrite

    def run(self) -> None:
        # Get name of the current computer (for comparison purposes)
        assert self._wrapper is not None

        nested_dict = lambda: defaultdict(nested_dict)  # type: ignore
        data = nested_dict()

        # Now we get the old data
        filename = Path("benchmark/report/raw.json")
        previous_data: Any = {self._computer: {}}
        if filename.exists() and not self._overwrite:
            with open(filename, "r") as f:
                d = json.load(f)
                previous_data.update(d)

        for task in tasks:
            print(f"--- Start {task.name}")
            if task.name in previous_data[self._computer]:
                print("Already done. Skipping.")
                continue

            task_wrapper = self._wrapper(task)
            try:
                task_wrapper.run()
            except NotSupportedError:
                print("Not supported. Skipping.")
                continue

            data[self._computer][task.name]["objective"] = task.objectives[0]
            d = data[self._computer][task.name]

            for sort_by in ["trials", "walltime"]:
                # Now we collect some metrics
                try:
                    X, Y = task_wrapper.get_trajectory(sort_by=sort_by)
                except NotSupportedError:
                    continue

                d[f"trajectory_{sort_by}"][self._version] = (X, Y)

        previous_data.update(data)
        data = previous_data

        # .. and save the new data
        with open(filename, "w") as f:
            json.dump(data, f, indent=4)

        # Save globally
        self._data = data

        # Plot things
        self._plot_trajectory()
        self._write_table()

    def _plot_trajectory(self) -> None:
        d = self._data[self._computer]

        for sort_by in ["trials", "walltime"]:
            traj = f"trajectory_{sort_by}"
            filename = Path(f"benchmark/report/{traj}.png")
            plt.figure(rows=len(tasks))
            plt.tight_layout()
            plt.subplots_adjust(wspace=0.6)

            for i, task in enumerate(tasks):
                objective = d[task.name]["objective"]

                plt.subplot(len(tasks), 1, i + 1)
                plt.title(task.name)
                for version, (X, Y) in d[task.name][traj].items():
                    plt.plot(X, Y, label=version, linewidth=0.5)
                    plt.scatter(X, Y, s=3)

                plt.legend()
                plt.ylabel(objective)

                if i == len(tasks) - 1:
                    plt.xlabel(sort_by)

            plt.save_figure(str(filename))

    def _write_table(self) -> None:
        """Writes a table with the best result found so far."""
        d = self._data[self._computer]

        # For each version a column
        data = defaultdict(list)
        for task in tasks:
            data["tasks"].append(task.name)
            for version, (X, Y) in d[task.name]["trajectory_trials"].items():
                data[version].append(Y[-1])

        # Now make a dataframe
        df = pd.DataFrame(data=data)
        df.to_csv("benchmark/report/table.csv", index=False)


if __name__ == "__main__":
    benchmark = Benchmark(overwrite=False)
    benchmark.run()
