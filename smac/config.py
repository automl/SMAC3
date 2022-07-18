from __future__ import annotations

from typing import Any, Mapping

import hashlib
import json
import random
from dataclasses import dataclass, field
from pathlib import Path

import ConfigSpace
import numpy as np
from ConfigSpace.read_and_write import json as cs_json

from smac.utils.logging import get_logger
from smac.utils.others import recursively_compare_dicts

logger = get_logger(__name__)


@dataclass(frozen=True)
class Config:
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
    crash_cost: float = np.inf
    # transform_y: str | None = None  # Whether y should be transformed (different runhistory2epm)

    # Limitations
    walltime_limit: float = np.inf
    cputime_limit: float = np.inf
    memory_limit: float | None = None
    algorithm_walltime_limit: float | None = None
    n_runs: int = 20

    # Other time things
    intensify_percentage: float = 0.5

    # always_race_default
    # How to deal with instances? Have them here too? It's not really a config, rather data
    # So they probably should go to the cli directory.
    # However, we need an interface to accept instances/test instances in the main python code.

    # Algorithm Configuration
    instances: np.array | None = None
    instance_features: np.array | None = None
    instance_specifics: Mapping[str, str] | None = None

    # Others
    seed: int = 0  # TODO: Document if seed is set to -1, we use a random seed.
    n_workers: int = 1

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
        filename = self.output_directory / "config.json"
        with open(filename, "w") as fh:
            json.dump(data, fh, indent=4)

        # Save configspace on its own
        configspace_filename = self.output_directory / "configspace.json"
        with open(configspace_filename, "w") as f:
            f.write(cs_json.write(self.configspace))

    @staticmethod
    def load(path: Path) -> Config:
        filename = path / "config.json"
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

        config = Config(**data)
        config.set_meta(meta)

        return config

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Config):
            # When using __dict__, we make sure to include the meta data
            return self.__dict__ == other.__dict__

        raise RuntimeError("Can only compare config objects.")
