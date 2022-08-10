from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Union

import json
import os
from pathlib import Path

import numpy as np
import regex as re
from ConfigSpace.read_and_write import json as csjson

from smac.runhistory.runhistory import RunHistory


def glob_re(pattern: str, strings: List[str]) -> filter:
    """
    Filter strings according to pattern.

    Parameters
    ----------
    pattern: str
        Regex pattern
    strings: List[str]
        List of strings to filter.

    Returns
    -------
    filter[str]
    """
    return filter(re.compile(pattern).match, strings)


def get_rundirs(pattern: str, path: Union[str, Path]) -> Sequence[str]:
    """
    Get SMAC run dirs, often starting with `run_`.

    Parameters
    ----------
    pattern: str
        Regex expresssion.
    path: Union[str, Path]
        Path to folder containing single SMAC rundirs

    Returns
    -------
    Sequence[str]
        Single SMAC rundirs

    """
    subdirs = list(glob_re(pattern, os.listdir(path)))
    rundirs = [os.path.join(path, sd) for sd in subdirs]
    return rundirs


class ResultMerger:
    def __init__(
        self,
        output_dir: Optional[Union[str, Path]] = None,
        rundir_pattern: str = r"run_*\d$",
        rundirs: Optional[List[Union[str, Path]]] = None,
    ):
        """
        Merge runhistories from different SMAC runs.

        Parameters
        ----------
        output_dir : Optional[Union[str, Path]]
            Output directory containing single SMAC run folders. The rundirs are inside
            and collected via the pattern `rundir_pattern`.
        rundir_pattern : str
            Regex expression to find single rundirs in `output_dir`.
        rundirs : Optional[List[Union[str, Path]]]
            Paths to all SMAC output folders.
            If not specified, please specify `output_dir`.
        """
        self.output_dir = output_dir
        self.run_dirs: Sequence[Union[str, Path]]
        if rundirs:
            self.run_dirs = rundirs
        else:
            if self.output_dir is None:
                raise ValueError("Please provide either `rundirs` or `output_dir` with" " an optional pattern.")
            self.run_dirs = get_rundirs(pattern=rundir_pattern, path=self.output_dir)

        cs_fn = Path(self.run_dirs[0]) / "configspace.json"
        with open(cs_fn, "r") as fh:
            json_string = fh.read()
            self.configuration_space = csjson.read(json_string)

    def get_runhistory(self) -> RunHistory:
        """
        Get runhistory

        For this, merge all runhistories in pSMAC subfolders.

        Returns
        -------
        RunHistory
            Empty, if `self.run_dirs` is None.

        """
        runhistory = RunHistory()
        if self.run_dirs:
            runhistory_filenames = [os.path.join(d, "runhistory.json") for d in self.run_dirs]
            for fn in runhistory_filenames:
                runhistory.update_from_json(fn=fn, cs=self.configuration_space)
        return runhistory

    def get_trajectory(self) -> Optional[List[Dict[str, Any]]]:
        """
        Get trajectory

        For this, extract trajectory from merged runhistories.
        Return trajectories in json format.

        Returns
        -------
        Optional[List[Dict[str, Any]]
            - None, if `self.run_dirs` is None.
            - List of trajectory entries. Each trajectory entry is a dict with keys
                ['cpu_time', 'wallclock_time', 'evaluations', 'cost', 'incumbent', 'budget', 'origin'].

        """
        trajectory = None
        if self.run_dirs is None:
            return trajectory
        rh = self.get_runhistory()

        # Sort configurations chronologically by starttime
        rvals = rh.values()
        starttimes = np.array([rv.starttime for rv in rvals])
        ids = np.argsort(starttimes)
        rhitems = list(rh.items())
        rhitems = [rhitems[i] for i in ids]

        # Find incumbents
        # Incumbent = cost is lower than alltime cost
        trajectory = []

        # Inject first trajectory entry from file from first rundir
        rundir = self.run_dirs[0]
        traj_fn = Path(rundir) / "traj.json"
        with open(traj_fn, "r") as file:
            line = file.readline()
        traj_entry = json.loads(line)
        trajectory.append(traj_entry)

        # Populate from merged runhistory
        cost = np.inf
        for i, (rk, rv) in enumerate(rhitems):
            if rv.cost < cost:
                cost = rv.cost
                # traj_entry = TrajEntry(
                #     rv.cost,  # train_perf
                #     rk.config_id,  # incumbent_id
                #     rh.ids_config[rk.config_id],  # incumbent
                #     i + 1,  # ta_runs
                #     rv.time,  # ta_time_used
                #     rv.starttime,  # wallclock_time
                #     rk.budget,  # budget
                # )  # TODO return traj_entry as TrajEntry and convert to json for write_trajectory
                incumbent = rh.ids_config[rk.config_id]
                traj_entry = {
                    "cpu_time": rv.time,
                    "wallclock_time": rv.starttime,
                    "evaluations": i + 1,
                    "cost": rv.cost,
                    "incumbent": incumbent.get_dictionary(),
                    "budget": rk.budget,
                    "origin": incumbent.origin,
                }
                trajectory.append(traj_entry)

        return trajectory

    def write_trajectory(self) -> None:
        """
        Write trajectory to traj.json

        Returns
        -------
        None

        """
        if self.output_dir is not None:
            traj_fn = Path(self.output_dir) / "traj.json"
            traj = self.get_trajectory()

            traj_fn.open("w")
            if traj is not None:
                for traj_entry in traj:
                    with open(traj_fn, "a") as fp:  # TODO: write or append?
                        json.dump(traj_entry, fp)
                        fp.write("\n")

    def write_runhistory(self) -> None:
        """
        Write runhistory to runhistory.json

        Returns
        -------
        None

        """
        if self.output_dir is not None:
            rh_fn = Path(self.output_dir) / "runhistory.json"
            rh = self.get_runhistory()
            rh.save_json(fn=str(rh_fn), save_external=True)
