from __future__ import annotations
from benchmark.tasks.task import Task
from benchmark.wrappers.wrapper import Wrapper


class Version200a3(Wrapper):
    @property
    def version(self) -> str:
        return "v2.0.0a3"

    def __init__(self, task: Task) -> None:
        super().__init__(task)
        self._smac = None

    def run(self) -> None:
        from smac import Scenario, MultiFidelityFacade, HyperparameterOptimizationFacade

        # Create scenario
        scenario = Scenario(
            self.model.configspace,
            n_trials=self.task.n_trials,
        )

        # Create facade
        if self.task.optimization_type == "hyperparameter_optimization":
            facade_object = HyperparameterOptimizationFacade
        elif self.task.optimization_type == "multi_fidelity":
            facade_object = MultiFidelityFacade
        else:
            raise RuntimeError("Unknown optimization type.")

        smac = facade_object(scenario, self.task.model.train)
        smac.optimize()

        self._smac = smac

    def get_trajectory(self) -> list[float]:
        pass

    def plot(self) -> None:
        pass

    def used_walltime(self) -> float:
        pass

    def finished_trials(self) -> float:
        pass
