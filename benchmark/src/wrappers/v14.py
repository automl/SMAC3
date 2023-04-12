from __future__ import annotations

from typing import Any
from src.tasks.task import Task
from src.utils.exceptions import NotSupportedError
from src.wrappers.wrapper import Wrapper


class Version14(Wrapper):
    supported_versions: list[str] = ["1.4.0"]

    def __init__(self, task: Task, seed: int) -> None:
        super().__init__(task, seed)

        from smac.facade.smac_ac_facade import SMAC4AC

        self._smac: SMAC4AC | None = None

    def run(self) -> None:
        from smac.facade.smac_bb_facade import SMAC4BB
        from smac.facade.smac_hpo_facade import SMAC4HPO
        from smac.facade.smac_mf_facade import SMAC4MF
        from smac.facade.smac_ac_facade import SMAC4AC
        from smac.scenario.scenario import Scenario

        if self.task.n_workers > 1 and self.task.optimization_type != "mf":
            raise NotSupportedError("SMAC 1.4 does not support parallel execution natively.")

        if self.task.retrain_after > 1:
            raise NotSupportedError("SMAC 1.4 does not support ``retrain_after`.")

        # Get instances
        if self.task.use_instances:
            dataset = self.task.model.dataset
            assert dataset is not None
            all_feat = dataset.get_instance_features()

            instances = []
            instance_features = {}

            # We map the instance to its index (idk why they implemented it this way)
            for i, instance in enumerate(dataset.get_instances()):
                instances.append([instance])
                instance_features[instance] = all_feat[instance]
        else:
            instances = [[None]]
            instance_features = {}

        multi_objectives: str | list[str]
        if len(self.task.objectives) > 1:
            multi_objectives = self.task.objectives
        elif len(self.task.objectives) == 1:
            multi_objectives = self.task.objectives[0]

        scenario = Scenario(
            {
                "run_obj": "quality",  # we optimize quality (alternatively runtime)
                "runcount-limit": self.task.n_trials,  # max. number of function evaluations
                "cs": self.task.model.configspace,  # configuration space
                "wallclock_limit": self.task.walltime_limit,
                "deterministic": self.task.deterministic,
                "min_budget": self.task.min_budget,
                "max_budget": self.task.max_budget,
                "minR": 1,
                "maxR": self.task.max_config_calls,
                "instances": instances,
                "feature_dict": instance_features,
                "multi_objectives": multi_objectives,
                "seed": self.seed,
            }
        )

        intensifier_kwargs: dict[Any, Any] = {}
        facade_kwargs: dict[Any, Any] = {}

        # Create facade
        if self.task.optimization_type == "bb":
            facade_object = SMAC4BB
            intensifier_kwargs["maxR"] = self.task.max_config_calls

        elif self.task.optimization_type == "hpo":
            facade_object = SMAC4HPO
            intensifier_kwargs["maxR"] = self.task.max_config_calls

        elif self.task.optimization_type == "mf":
            facade_object = SMAC4MF

            n_seeds = self.task.n_seeds
            assert n_seeds is not None
            intensifier_kwargs["n_seeds"] = n_seeds
            intensifier_kwargs["initial_budget"] = self.task.min_budget
            intensifier_kwargs["max_budget"] = self.task.max_budget

            inc_selection = self.task.incumbent_selection
            if inc_selection == "highest_observed_budget":
                inc_selection = "highest_budget"

            intensifier_kwargs["incumbent_selection"] = inc_selection
            facade_kwargs["n_jobs"] = self.task.n_workers
        elif self.task.optimization_type == "ac":
            facade_object = SMAC4AC
            intensifier_kwargs["maxR"] = self.task.max_config_calls

        else:
            raise RuntimeError("Unknown optimization type.")

        if self.task.intensifier is None:
            intensifier = None
        else:
            if self.task.intensifier == "successive_halving":
                from smac.intensification.successive_halving import SuccessiveHalving

                intensifier = SuccessiveHalving
            else:
                raise RuntimeError("Unsupported intensifier.")

        smac = facade_object(
            scenario=scenario,
            tae_runner=self.task.model.train,
            intensifier=intensifier,
            intensifier_kwargs=intensifier_kwargs,
            **facade_kwargs,
        )
        smac.optimize()

        self._smac = smac

    def get_trajectory(self, sort_by: str = "trials") -> tuple[list[float], list[float]]:

        if len(self.task.objectives) > 1:
            raise NotSupportedError

        assert self._smac is not None
        trajectory = self._smac.trajectory
        X: list[int | float] = []
        Y: list[float] = []

        for (cost, _, _, trial, _, walltime, _) in trajectory:
            if cost > 1e6:
                continue

            if sort_by == "trials":
                X.append(trial)
            elif sort_by == "walltime":
                X.append(walltime)
            else:
                raise RuntimeError("Unknown sort_by.")

            Y.append(cost)

        return X, Y
