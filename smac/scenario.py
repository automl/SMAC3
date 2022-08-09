from __future__ import annotations

from typing import Any, Mapping

import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path

import ConfigSpace
import numpy as np
from ConfigSpace.read_and_write import json as cs_json

from smac.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class Scenario:
    """
    Replaces the scenario class in the original code.
    """

    configspace: ConfigSpace

    # If no name is used, SMAC generates a hash based on the meta data (basically from the config and the arguments of the components)
    # You can use `name` to directly identify your run.
    name: str | None = None
    output_directory: Path = Path("smac3_output")  # ./smac3_output/name/seed/... in the end.
    deterministic: bool = True  # Whether the target algorithm is determinstic or not.

    objectives: str | list[str] = "cost"
    crash_cost: float | list[float] = np.inf

    # Limitations
    walltime_limit: float = np.inf
    cputime_limit: float = np.inf
    memory_limit: int | None = None
    algorithm_walltime_limit: float | None = None
    n_runs: int = 100

    # Other time things
    intensify_percentage: float = 0.5

    # always_race_default
    # How to deal with instances? Have them here too? It's not really a config, rather data
    # So they probably should go to the cli directory.
    # However, we need an interface to accept instances/test instances in the main python code.

    # Algorithm Configuration
    instances: list[str] | None = None  # e.g. if the algorithm should optimized across multiple datasets, seeds, ...
    instance_features: dict[
        str, list[int | float]
    ] | None = None  # Each instance can be associated with features. Those features are incorporated in the runhistory transformer.
    instance_order: str | None = "shuffle_once"  # Can be "shuffle_once", "shuffle" or None

    # What we want to have is:
    # instances: dict[str, list[int | float]] | None = None
    # or
    # instances: list[str] | None = None
    # instance_features: dict[str, list[int | float]] | None = None

    # For multi-fidelity and instance optimization
    min_budget: float | None = None
    max_budget: float | None = None
    # eta: float = 3

    # Others
    seed: int = 0  # TODO: Document if seed is set to -1, we use a random seed.
    n_workers: int = 1  # Parallelization is automatically applied if n_workers > 1 using dask

    def __post_init__(self) -> None:
        """Checks whether the config is valid."""
        # transform_y_options = [None, "log", "log_scaled", "inverse_scaled"]
        # if self.transform_y not in transform_y_options:
        #    raise RuntimeError(f"`transform_y` must be one of `{transform_y_options}`")

        # Intensify percentage must be between 0 and 1
        assert self.intensify_percentage >= 0.0 and self.intensify_percentage <= 1.0

        # Use random seed if seed is -1
        if self.seed == -1:
            seed = random.randint(0, 999999)
            object.__setattr__(self, "seed", seed)

        # Change directory wrt name and seed
        self._change_output_directory()

        # Set hashes
        object.__setattr__(self, "_meta", {})

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Scenario):
            # When using __dict__, we make sure to include the meta data
            return self.__dict__ == other.__dict__

        raise RuntimeError("Can only compare scenario objects.")

    def _change_output_directory(self) -> None:
        # Create output directory
        if self.name is not None:
            new = Path(self.name) / str(self.seed)
            if not str(self.output_directory).endswith(str(new)):
                object.__setattr__(self, "output_directory", self.output_directory / new)

    def _set_meta(self, meta: dict[str, dict[str, Any]]) -> None:
        object.__setattr__(self, "_meta", meta)

        # We overwrite name with the hash of the meta (if no name is passed)
        if self.name is None:
            hash = hashlib.md5(str(self.__dict__).encode("utf-8")).hexdigest()
            object.__setattr__(self, "name", hash)
            self._change_output_directory()

    def get_meta(self) -> dict[str, str]:
        return self._meta  # type: ignore

    def count_objectives(self) -> int:
        if isinstance(self.objectives, list):
            return len(self.objectives)

        return 1

    def count_instance_features(self) -> int:
        # Check whether key of instance features exist
        n_features = 0
        if self.instance_features is not None:
            for k, v in self.instance_features.items():
                if k not in self.instances:
                    raise RuntimeError(f"Instance {k} is not specified in instances.")

                if n_features == 0:
                    n_features = len(v)
                else:
                    if len(v) != n_features:
                        raise RuntimeError("Instances must have the same number of features.")

        return n_features

    def save(self) -> None:
        if self.name is None:
            raise RuntimeError(
                "Please specify meta data for generating a name. Alternatively, you can specify a name manually."
            )

        self.output_directory.mkdir(parents=True, exist_ok=True)

        data = {}
        for k, v in self.__dict__.items():
            if k in ["configspace", "output_directory"]:
                continue

            data[k] = v

        # Convert `output_directory`
        data["output_directory"] = str(self.output_directory)

        # Save everything
        filename = self.output_directory / "scenario.json"
        with open(filename, "w") as fh:
            json.dump(data, fh, indent=4)

        # Save configspace on its own
        configspace_filename = self.output_directory / "configspace.json"
        with open(configspace_filename, "w") as f:
            f.write(cs_json.write(self.configspace))

    @staticmethod
    def load(path: Path) -> Scenario:
        filename = path / "scenario.json"
        with open(filename, "r") as fh:
            data = json.load(fh)

        # Convert `output_directory` to path object again
        data["output_directory"] = Path(data["output_directory"])
        meta = data["_meta"]
        del data["_meta"]

        # Read configspace
        configspace_filename = path / "configspace.json"
        with open(configspace_filename, "r") as f:

            configspace = cs_json.read(f.read())

        data["configspace"] = configspace

        scenario = Scenario(**data)
        scenario._set_meta(meta)

        return scenario
