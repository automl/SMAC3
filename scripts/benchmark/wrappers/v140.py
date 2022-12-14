from __future__ import annotations
from benchmark.tasks.task import Task
from benchmark.utils.exceptions import NotSupportedError
from benchmark.wrappers.wrapper import Wrapper


class Version140(Wrapper):
    supported_versions: list[str] = ["1.4.0"]

    def __init__(self, task: Task) -> None:
        super().__init__(task)

        from smac.facade.smac_ac_facade import SMAC4AC

        self._smac: SMAC4AC | None = None

    def run(self) -> None:
        from smac.facade.smac_hpo_facade import SMAC4HPO
        from smac.facade.smac_bb_facade import SMAC4BB
        from smac.facade.smac_mf_facade import SMAC4MF
        from smac.scenario.scenario import Scenario
        
        if self.task.n_workers > 1:
            raise NotSupportedError("SMAC 1.4 does not support parallel execution natively.")
        
        if self.task.retrain_after > 1:
            raise NotSupportedError("SMAC 1.4 does not support ``retrain_after`.")

        # Get instances
        instances = None
        instance_features = None

        if self.task.use_instances:
            instances = self.task.model.dataset.get_instances()
            instance_features = self.task.model.dataset.get_instance_features()

        # Create scenario
        # scenario = Scenario(
        #     self.model.configspace,
        #     n_trials=self.task.n_trials,
        #     walltime_limit=self.task.walltime_limit,
        #     deterministic=self.task.deterministic,
        #     instances=instances,
        #     instance_features=instance_features,
        #     min_budget=self.task.min_budget,
        #     max_budget=self.task.max_budget,
        #     n_workers=self.task.n_workers,
        # )
        
        scenario = Scenario(
            {
                "run_obj": "quality",  # we optimize quality (alternatively runtime)
                "runcount-limit": self.task.n_trials,  # max. number of function evaluations
                "cs": self.task.model.configspace,  # configuration space
                "deterministic": self.task.deterministic,
                "min_budget": self.task.min_budget,
                "max_budget": self.task.max_budget,
                "minR": 1,
                "maxR": self.task.max_config_calls,
            }
        )

        intensifier_kwargs = {}

        # Create facade
        if self.task.optimization_type == "bb":
            facade_object = SMAC4BB
        elif self.task.optimization_type == "hpo":
            facade_object = SMAC4HPO
        elif self.task.optimization_type == "mf":
            facade_object = SMAC4MF
            intensifier_kwargs["n_seeds"] = self.task.n_seeds
            intensifier_kwargs["incumbent_selection"] = self.task.incumbent_selection
        else:
            raise RuntimeError("Unknown optimization type.")

        smac = facade_object(
            scenario=scenario,
            tae_runner=self.task.model.train,
            intensifier_kwargs=intensifier_kwargs,
        )
        smac.optimize()

        self._smac = smac

    def get_trajectory(self, sort_by: str = "trials") -> tuple[list[float], list[float]]:

        if len(self.task.objectives) > 1:
            raise NotSupportedError

        assert self._smac is not None
        rh = self._smac.runhistory
        trajectory = self._smac.trajectory
        X: list[int | float] = []
        Y: list[float] = []

        for (cost, _, _, trial, _, walltime, _) in trajectory:
            if sort_by == "trials":
                X.append(trial)
            elif sort_by == "walltime":
                X.append(walltime)
            else:
                raise RuntimeError("Unknown sort_by.")

            Y.append(cost)

        return X, Y
