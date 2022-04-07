from typing import Dict, List, Optional, Union

import collections
import json
import logging
import os

import numpy as np
from ConfigSpace.configuration_space import Configuration, ConfigurationSpace
from ConfigSpace.hyperparameters import (
    CategoricalHyperparameter,
    Constant,
    FloatHyperparameter,
    IntegerHyperparameter,
)

from smac.stats.stats import Stats
from smac.utils.logging import format_array

__author__ = "Marius Lindauer"
__copyright__ = "Copyright 2016, ML4AAD"
__license__ = "3-clause BSD"

TrajEntry = collections.namedtuple(
    "TrajEntry",
    [
        "train_perf",
        "incumbent_id",
        "incumbent",
        "ta_runs",
        "ta_time_used",
        "wallclock_time",
        "budget",
    ],
)


class TrajLogger(object):
    """
    Writes trajectory logs files and creates output directory if not exists already

    Parameters
    ----------
    output_dir: str
    directory for logging (or None to disable logging)
    stats: Stats()
    Stats object

    Attributes
    ----------
    stats
    logger
    output_dir
    aclib_traj_fn
    old_traj_fn
    trajectory
    """

    def __init__(self, output_dir: Optional[str], stats: Stats) -> None:
        self.stats = stats
        self.logger = logging.getLogger(self.__module__ + "." + self.__class__.__name__)

        self.output_dir = output_dir
        if output_dir is None or output_dir == "":
            self.output_dir = None
            self.logger.info(
                "No output directory for trajectory logging " "specified -- trajectory will not be logged."
            )

        else:
            if not os.path.isdir(output_dir):
                try:
                    os.makedirs(output_dir)
                except OSError:
                    e = OSError("Could not make output directory: {}.".format(output_dir))
                    self.logger.error("Could not make output directory.", exc_info=e)
                    raise e

            self.old_traj_fn = os.path.join(output_dir, "traj_old.csv")
            if not os.path.isfile(self.old_traj_fn):
                with open(self.old_traj_fn, "w") as fp:
                    fp.write(
                        '"CPU Time Used","Estimated Training Performance",'
                        '"Wallclock Time","Incumbent ID",'
                        '"Automatic Configurator (CPU) Time",'
                        '"Configuration..."\n'
                    )

            self.aclib_traj_fn = os.path.join(output_dir, "traj_aclib2.json")
            self.alljson_traj_fn = os.path.join(output_dir, "traj.json")

        self.trajectory = []  # type: List[TrajEntry]

    def add_entry(
        self,
        train_perf: Union[float, np.ndarray],
        incumbent_id: int,
        incumbent: Configuration,
        budget: float = 0,
    ) -> None:
        """Adds entries to trajectory files (several formats) with using the
        same timestamps for each entry

        Parameters
        ----------
        train_perf: float or np.ndarray
            estimated performance on training (sub)set
        incumbent_id: int
            id of incumbent
        incumbent: Configuration()
            current incumbent configuration
        budget: float
            budget used in intensifier to limit TA (default: 0)
        """
        perf = format_array(train_perf)

        finished_ta_runs = self.stats.finished_ta_runs
        ta_time_used = self.stats.ta_time_used
        wallclock_time = self.stats.get_used_wallclock_time()
        self.trajectory.append(
            TrajEntry(
                perf,
                incumbent_id,
                incumbent,
                finished_ta_runs,
                ta_time_used,
                wallclock_time,
                budget,
            )
        )
        if self.output_dir is not None:
            self._add_in_old_format(perf, incumbent_id, incumbent, ta_time_used, wallclock_time)
            self._add_in_aclib_format(perf, incumbent_id, incumbent, ta_time_used, wallclock_time)
            self._add_in_alljson_format(perf, incumbent_id, incumbent, budget, ta_time_used, wallclock_time)

    def _add_in_old_format(
        self,
        train_perf: Union[float, List[float]],
        incumbent_id: int,
        incumbent: Configuration,
        ta_time_used: float,
        wallclock_time: float,
    ) -> None:
        """Adds entries to old SMAC2-like trajectory file

        Parameters
        ----------
        train_perf: float or list of floats
            Estimated performance on training (sub)set
        incumbent_id: int
            Id of incumbent
        incumbent: Configuration()
            Current incumbent configuration
        ta_time_used: float
            CPU time used by the target algorithm
        wallclock_time: float
            Wallclock time used so far
        """
        conf = []
        for p in incumbent:
            if not incumbent.get(p) is None:
                conf.append("%s='%s'" % (p, repr(incumbent[p])))
        if isinstance(train_perf, float):
            # Make it compatible with old format
            with open(self.old_traj_fn, "a") as fp:
                fp.write(
                    f"{ta_time_used:f}, {train_perf:f}, {wallclock_time:f}, {incumbent_id:d}, "
                    f"{wallclock_time - ta_time_used:f}, {','.join(conf):s}\n"
                )
        else:
            # We recommend to use pandas to read this csv file
            with open(self.old_traj_fn, "a") as fp:
                fp.write(
                    f"{ta_time_used:f}, {train_perf}, {wallclock_time:f}, {incumbent_id:d}, "
                    f"{wallclock_time - ta_time_used:f}, {','.join(conf):s}\n"
                )

    def _add_in_aclib_format(
        self,
        train_perf: Union[float, List[float]],
        incumbent_id: int,
        incumbent: Configuration,
        ta_time_used: float,
        wallclock_time: float,
    ) -> None:
        """Adds entries to AClib2-like trajectory file

        Parameters
        ----------
        train_perf: float or list of floats
            Estimated performance on training (sub)set
        incumbent_id: int
            Id of incumbent
        incumbent: Configuration()
            Current incumbent configuration
        ta_time_used: float
            CPU time used by the target algorithm
        wallclock_time: float
            Wallclock time used so far
        """
        conf = []
        for p in incumbent:
            if not incumbent.get(p) is None:
                conf.append("%s='%s'" % (p, repr(incumbent[p])))

        traj_entry = {
            "cpu_time": ta_time_used,
            "wallclock_time": wallclock_time,
            "evaluations": self.stats.finished_ta_runs,
            "cost": format_array(train_perf, False),
            "incumbent": conf,
            "origin": incumbent.origin,
        }

        with open(self.aclib_traj_fn, "a") as fp:
            json.dump(traj_entry, fp)
            fp.write("\n")

    def _add_in_alljson_format(
        self,
        train_perf: Union[float, List[float]],
        incumbent_id: int,
        incumbent: Configuration,
        budget: float,
        ta_time_used: float,
        wallclock_time: float,
    ) -> None:
        """Adds entries to AClib2-like (but with configs as json) trajectory file

        Parameters
        ----------
        train_perf: float or list of floats
            Estimated performance on training (sub)set
        incumbent_id: int
            Id of incumbent
        incumbent: Configuration()
            Current incumbent configuration
        budget: float
            budget (cutoff) used in intensifier to limit TA (default: 0)
        ta_time_used: float
            CPU time used by the target algorithm
        wallclock_time: float
            Wallclock time used so far
        """
        traj_entry = {
            "cpu_time": ta_time_used,
            "wallclock_time": wallclock_time,
            "evaluations": self.stats.finished_ta_runs,
            "cost": train_perf,
            "incumbent": incumbent.get_dictionary(),
            "budget": budget,
            "origin": incumbent.origin,
        }

        with open(self.alljson_traj_fn, "a") as fp:
            json.dump(traj_entry, fp)
            fp.write("\n")

    @staticmethod
    def read_traj_alljson_format(
        fn: str,
        cs: ConfigurationSpace,
    ) -> List[Dict[str, Union[float, int, Configuration]]]:
        """Reads trajectory from file

        Parameters
        ----------
        fn: str
            Filename with saved runhistory in self._add_in_alljson_format format
        cs: ConfigurationSpace
            Configuration Space to translate dict object into Confiuration object

        Returns
        -------
        trajectory: list
            Each entry in the list is a dictionary of the form
            {
            "cpu_time": float,
            "wallclock_time": float,
            "evaluations": int
            "cost": float or list of floats,
            "budget": budget,
            "incumbent": Configuration
            }
        """
        trajectory = []
        with open(fn) as fp:
            for line in fp:
                entry = json.loads(line)
                entry["incumbent"] = Configuration(cs, entry["incumbent"])
                trajectory.append(entry)

        return trajectory

    @staticmethod
    def read_traj_aclib_format(
        fn: str,
        cs: ConfigurationSpace,
    ) -> List[Dict[str, Union[float, int, Configuration]]]:
        """Reads trajectory from file

        Parameters
        ----------
        fn: str
            Filename with saved runhistory in self._add_in_aclib_format format
        cs: ConfigurationSpace
            Configuration Space to translate dict object into Confiuration object

        Returns
        -------
        trajectory: list
            Each entry in the list is a dictionary of the form
            {
            "cpu_time": float,
            "wallclock_time": float,
            "evaluations": int
            "cost": float or list of floats,
            "incumbent": Configuration
            }
        """
        trajectory = []
        with open(fn) as fp:
            for line in fp:
                entry = json.loads(line)
                entry["incumbent"] = TrajLogger._convert_dict_to_config(entry["incumbent"], cs=cs)
                trajectory.append(entry)

        return trajectory

    @staticmethod
    def _convert_dict_to_config(config_list: List[str], cs: ConfigurationSpace) -> Configuration:
        """Since we save a configurations in a dictionary str->str we have to
        try to figure out the type (int, float, str) of each parameter value

        Parameters
        ----------
        config_list: List[str]
            Configuration as a list of "str='str'"
        cs: ConfigurationSpace
            Configuration Space to translate dict object into Confiuration object
        """
        config_dict = {}
        v = ""  # type: Union[str, float, int, bool]
        for param in config_list:
            k, v = param.split("=")
            v = v.strip("'")
            hp = cs.get_hyperparameter(k)
            if isinstance(hp, FloatHyperparameter):
                v = float(v)
            elif isinstance(hp, IntegerHyperparameter):
                v = int(v)
            elif isinstance(hp, (CategoricalHyperparameter, Constant)):
                # Checking for the correct type requires jumping some hoops
                # First, we gather possible interpretations of our string
                interpretations = [v]  # type: List[Union[str, bool, int, float]]
                if v in ["True", "False"]:
                    # Special Case for booleans (assuming we support them)
                    # This is important to avoid false positive warnings triggered by 1 == True or "False" == True
                    interpretations.append(True if v == "True" else False)
                else:
                    for t in [int, float]:
                        try:
                            interpretations.append(t(v))
                        except ValueError:
                            continue

                # Second, check if it's in the choices / the correct type.
                legal = {interpretation for interpretation in interpretations if hp.is_legal(interpretation)}

                # Third, issue warnings if the interpretation is ambigious
                if len(legal) != 1:
                    logging.getLogger("smac.trajlogger").warning(
                        "Ambigous or no interpretation of value {} for hp {} found ({} possible interpretations). "
                        "Passing string, but this will likely result in an error".format(v, hp.name, len(legal))
                    )
                else:
                    v = legal.pop()

            config_dict[k] = v

        config = Configuration(configuration_space=cs, values=config_dict)
        config.origin = "External Trajectory"

        return config
