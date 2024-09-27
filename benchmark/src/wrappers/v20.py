from __future__ import annotations

from src.tasks.task import Task
from src.utils.exceptions import NotSupportedError
from src.wrappers.wrapper import Wrapper


class Version20(Wrapper):
    supported_versions: list[str] = ["2.1.0"]

    def __init__(self, task: Task, seed: int) -> None:
        super().__init__(task, seed)

        from smac.facade import AbstractFacade

        self._smac: AbstractFacade | None = None

    def run(self) -> None:
        from smac import (
            BlackBoxFacade,
            HyperparameterOptimizationFacade,
            MultiFidelityFacade,
            Scenario,
            AlgorithmConfigurationFacade
        )

        # Get instances
        instances = None
        instance_features = None

        if self.task.use_instances:
            instances = self.task.model.dataset.get_instances()
            instance_features = self.task.model.dataset.get_instance_features()

        # Create scenario
        scenario = Scenario(
            self.model.configspace,
            n_trials=self.task.n_trials,
            walltime_limit=self.task.walltime_limit,
            deterministic=self.task.deterministic,
            instances=instances,
            instance_features=instance_features,
            min_budget=self.task.min_budget,
            max_budget=self.task.max_budget,
            n_workers=self.task.n_workers,
            seed=self.seed,
        )

        intensifier_kwargs = {}

        # Create facade
        if self.task.optimization_type == "bb":
            facade_object = BlackBoxFacade
            intensifier_kwargs["max_config_calls"] = self.task.max_config_calls
        elif self.task.optimization_type == "hpo":
            facade_object = HyperparameterOptimizationFacade
            intensifier_kwargs["max_config_calls"] = self.task.max_config_calls
        elif self.task.optimization_type == "mf":
            facade_object = MultiFidelityFacade
            intensifier_kwargs["n_seeds"] = self.task.n_seeds
            intensifier_kwargs["incumbent_selection"] = self.task.incumbent_selection
        elif self.task.optimization_type == "ac":
            facade_object = AlgorithmConfigurationFacade
            intensifier_kwargs["max_config_calls"] = self.task.max_config_calls
        else:
            raise RuntimeError("Unknown optimization type.")

        if self.task.intensifier is None:
            intensifier = facade_object.get_intensifier(scenario, **intensifier_kwargs)
        else:
            if self.task.intensifier == "successive_halving":
                from smac.intensifier.successive_halving import SuccessiveHalving

                intensifier = SuccessiveHalving(scenario, **intensifier_kwargs)
            else:
                raise RuntimeError("Unsupported intensifier.")

        config_selector = facade_object.get_config_selector(scenario, retrain_after=self.task.retrain_after)

        smac = facade_object(
            scenario,
            self.task.model.train,
            intensifier=intensifier,
            config_selector=config_selector,
            logging_level=99999,
            overwrite=True,
        )
        smac.optimize()

        self._smac = smac

    def get_trajectory(self, sort_by: str = "trials") -> tuple[list[float], list[float]]:
        if len(self.task.objectives) > 1:
            raise NotSupportedError

        assert self._smac is not None
        rh = self._smac.runhistory
        trajectory = self._smac.intensifier.trajectory
        X: list[int | float] = []
        Y: list[float] = []

        for traj in trajectory:
            assert len(traj.config_ids) == 1
            config_id = traj.config_ids[0]
            config = rh.get_config(config_id)

            cost = rh.get_cost(config)
            if cost > 1e6:
                continue

            if sort_by == "trials":
                X.append(traj.trial)
            elif sort_by == "walltime":
                X.append(traj.walltime)
            else:
                raise RuntimeError("Unknown sort_by.")

            Y.append(cost)

        return X, Y
