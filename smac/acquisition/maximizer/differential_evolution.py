from __future__ import annotations

import inspect

import numpy as np
from ConfigSpace import Configuration, ConfigurationSpace
from scipy.optimize._differentialevolution import DifferentialEvolutionSolver

from smac.acquisition.function.abstract_acquisition_function import (
    AbstractAcquisitionFunction,
)
from smac.acquisition.maximizer import AbstractAcquisitionMaximizer
from smac.utils.configspace import transform_continuous_designs

__copyright__ = "Copyright 2025, Leibniz University Hanover, Institute of AI"
__license__ = "3-clause BSD"


def check_kwarg(cls: type, kwarg_name: str) -> bool:
    """
    Checks if a given class accepts a specific keyword argument in its __init__ method.

    Parameters
    ----------
        cls (type): The class to inspect.
        kwarg_name (str): The name of the keyword argument to check.

    Returns
    -------
        bool: True if the class's __init__ method accepts the keyword argument,
              otherwise False.
    """
    # Get the signature of the class's __init__ method
    init_signature = inspect.signature(cls.__init__)  # type: ignore[misc]

    # Check if the kwarg_name is present in the signature as a parameter
    for param in init_signature.parameters.values():
        if param.name == kwarg_name and param.default != inspect.Parameter.empty:
            return True  # It accepts the kwarg
    return False  # It does not accept the kwarg


class DifferentialEvolution(AbstractAcquisitionMaximizer):
    """Get candidate solutions via `DifferentialEvolutionSolvers` from scipy.

    According to scipy 1.9.2 documentation:

    'Finds the global minimum of a multivariate function.
    Differential Evolution is stochastic in nature (does not use gradient methods) to find the minimum,
    and can search large areas of candidate space, but often requires larger numbers of function
    evaluations than conventional gradient-based techniques.
    The algorithm is due to Storn and Price [1].'

    [1] Storn, R and Price, K, Differential Evolution - a Simple and Efficient Heuristic for Global
     Optimization over Continuous Spaces, Journal of Global Optimization, 1997, 11, 341 - 359.

    Parameters
    ----------
    configspace : ConfigurationSpace
    acquisition_function : AbstractAcquisitionFunction
    challengers : int, defaults to 50000
        Number of challengers.
    max_iter: int | None, defaults to None
        Maximum number of iterations that the DE will perform.
    strategy: str, defaults to "best1bin"
        The strategy to use for the DE.
    polish: bool, defaults to True
        Whether to polish the final solution using L-BFGS-B.
    mutation: tuple[float, float], defaults to (0.5, 1.0)
        The mutation constant.
    recombination: float, defaults to 0.7
        The recombination constant.
    seed : int, defaults to 0
    """

    def __init__(
        self,
        configspace: ConfigurationSpace,
        acquisition_function: AbstractAcquisitionFunction | None = None,
        max_iter: int = 1000,
        challengers: int = 50000,
        strategy: str = "best1bin",
        polish: bool = True,
        mutation: tuple[float, float] = (0.5, 1.0),
        recombination: float = 0.7,
        seed: int = 0,
    ):
        super().__init__(configspace, acquisition_function, challengers, seed)
        # raise NotImplementedError("DifferentialEvolution is not yet implemented.")
        self.max_iter = max_iter
        self.strategy = strategy
        self.polish = polish
        self.mutation = mutation
        self.recombination = recombination

    def _maximize(
        self,
        previous_configs: list[Configuration],
        n_points: int,
    ) -> list[tuple[float, Configuration]]:
        # n_points is not used here, but is required by the interface

        configs: list[tuple[float, Configuration]] = []

        def func(x: np.ndarray) -> np.ndarray:
            assert self._acquisition_function is not None
            if len(x.shape) == 1:
                return -self._acquisition_function(
                    [
                        transform_continuous_designs(
                            design=np.expand_dims(x, axis=0),
                            origin="Diffrential Evolution",
                            configspace=self._configspace,
                        )[0]
                    ]
                )
            return -self._acquisition_function(
                transform_continuous_designs(design=x.T, origin="Diffrential Evolution", configspace=self._configspace)
            )

        accepts_seed = check_kwarg(DifferentialEvolutionSolver, "seed")
        if accepts_seed:
            kwargs = {"seed": self._rng.randint(1000)}
        else:
            kwargs = {"rng": self._rng.randint(1000)}
        ds = DifferentialEvolutionSolver(
            func,
            bounds=[[0, 1] for _ in range(len(self._configspace))],
            args=(),
            strategy=self.strategy,
            maxiter=self.max_iter,
            popsize=self._challengers // self.max_iter,
            tol=0.01,
            mutation=self.mutation,
            recombination=self.recombination,
            polish=self.polish,
            callback=None,
            disp=False,
            init="latinhypercube",
            atol=0,
            vectorized=True,
            **kwargs,
        )

        _ = ds.solve()
        for pop, val in zip(ds.population, ds.population_energies):
            rc = transform_continuous_designs(
                design=np.expand_dims(pop, axis=0),
                origin="Acquisition Function Maximizer: Differential Evolution",
                configspace=self._configspace,
            )[0]
            configs.append((-val, rc))

        configs.sort(key=lambda t: t[0])
        configs.reverse()

        return configs
