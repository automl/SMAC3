from __future__ import annotations

from typing import Any, Iterator

import copy
from types import SimpleNamespace

import numpy as np
from ConfigSpace import Configuration, ConfigurationSpace, Constant
from ConfigSpace.exceptions import ForbiddenValueError
from ConfigSpace.hyperparameters import Hyperparameter
from ConfigSpace.util import deactivate_inactive_hyperparameters

from smac.acquisition.function.abstract_acquisition_function import (
    AbstractAcquisitionFunction,
)
from smac.acquisition.maximizer.helpers import PseudoConstant
from smac.acquisition.maximizer.random_search import RandomSearch
from smac.model.abstract_model import AbstractModel
from smac.model.gaussian_process.abstract_gaussian_process import (
    AbstractGaussianProcess,
)
from smac.random_design.abstract_random_design import AbstractRandomDesign
from smac.runhistory.runhistory import RunHistory
from smac.utils.configspace import convert_configurations_to_array
from smac.utils.logging import get_logger
from smac.utils.multi_objective import normalize_costs

__copyright__ = "Copyright 2025, Leibniz University Hanover, Institute of AI"
__license__ = "3-clause BSD"

logger = get_logger(__name__)


class HPIRandomSearch(RandomSearch):
    """Sorted random search that dynamically restricts the configuration space to the currently most important
    hyperparameters, estimated via HyperSHAP (Wever et al., 2026) on a Gaussian process fitted on the observed costs
    and sets the other hyperparameters to a constant value. Combined with ``ParEGO`` as the
    ``multi_objective_algorithm``, this implements HPI-ParEGO (Theodorakopoulos et al., 2026), concentrating the
    search on the hyperparameters that matter most for the current scalarization.

    Note
    ----
    Requires the optional ``hypershap`` dependency, install it with ``pip install smac[hpi]``.

    The observed costs are read from ``runhistory``, which is set by the ``HPIFacade``. Without it, the surrogate
    model is used instead.

    For HPI-ParEGO, construct ``ParEGO`` with ``reweigh=10`` (``ParEGO(scenario, reweigh=10)``), to give HPI-ParEGO
    the chance to exploit a scalarization before resampling.

    Parameters
    ----------
    configspace : ConfigurationSpace
        Configuration space used for sampling.
    acquisition_function : AbstractAcquisitionFunction | None, defaults to None
        Acquisition function to maximize.
    challengers : int, defaults to 5000
        Number of configurations sampled during the optimization process. Details depend on the used maximizer.
        Also, the number of configurations that is returned by calling `maximize`.
    seed : int, defaults to 0
    n_trials : int | None, defaults to None
        Total optimization budget. Used to derive the reduction schedule when ``threshold`` is a list.
    threshold : float | list[float] | None, defaults to None
        Cumulative contribution that the selected hyperparameters must jointly explain between 0 and 1. 0 disables
        the configuration space reduction. ``None`` uses the three-phase schedule ``[0.0, 0.8, 0.8]``, which
        requires ``n_trials``. If a list, ``n_trials`` is split into ``len(threshold)`` equal-width
        phases, and ``threshold[i]`` is applied during phase ``i``. It is recommended to use a warm-up phase for
        the surrogate model to work well. For example:
        * Reduction in middle and last third of trials: ``[0.0, 0.8, 0.8]`` (default)
        * Reduction only in the second half: ``[0.0, 0.8]``
        * Reduction all the time: 0.8
    fixing_strategy : str, defaults to "incumbent"
        Value unimportant hyperparameters are fixed to, and HyperSHAP's reference configuration. One of
        "incumbent" (current best configuration), "default" (default configuration), "random" (a random configuration).
        For multi-objective optimization, "incumbent" refers to the incumbent of the current scalarization.
    random_prob : float, defaults to 0.1
        To avoid premature convergence, with probability ``random_prob`` the maximizer evaluates a random configuration
        from the original configuration space at random instead of using the reduced space.
    """

    def __init__(
        self,
        configspace: ConfigurationSpace = ConfigurationSpace(),
        acquisition_function: AbstractAcquisitionFunction | None = None,
        challengers: int = 5000,
        seed: int = 0,
        n_trials: int | None = None,
        threshold: float | list[float] | None = None,
        fixing_strategy: str = "incumbent",
        random_prob: float = 0.1,
    ) -> None:
        super().__init__(
            configspace,
            acquisition_function=acquisition_function,
            challengers=challengers,
            seed=seed,
        )

        if threshold is None:
            threshold = [0.0, 0.8, 0.8]

        if fixing_strategy not in ("incumbent", "default", "random"):
            raise ValueError(
                f"Unknown fixing_strategy '{fixing_strategy}'. Must be one of 'incumbent', 'default', 'random'."
            )

        if isinstance(threshold, list):
            if n_trials is not None and n_trials < len(threshold):
                raise ValueError("n_trials must be at least the length of the threshold list.")
            if any(not 0 <= t <= 1 for t in threshold):
                raise ValueError("Every value in threshold must lie in [0, 1].")
        elif not 0 <= threshold <= 1:
            raise ValueError("threshold must lie in [0, 1].")

        if not 0 <= random_prob <= 1:
            raise ValueError("random_prob must lie in [0, 1].")

        self._fixing_strategy = fixing_strategy
        self._threshold = threshold
        self._n_trials = n_trials
        self._random_prob = random_prob

        self._original_cs = copy.deepcopy(configspace)

        self._configspace.seed(seed)
        self._original_cs.seed(seed)

        # Separate generators, so that the random branch and the selection do not shift each other's draws.
        self._random_branch_rng = np.random.RandomState(seed)
        self._hpi_rng = np.random.RandomState(seed)

        self.runhistory: RunHistory | None = None
        self._gp: AbstractGaussianProcess | None = None

        # Used by `_incumbent_value_for` to fix inactive conditional hyperparameters.
        self._context_configs: list[Configuration] = []
        self._context_Y: np.ndarray = np.empty((0, 1))

    @property
    def meta(self) -> dict[str, Any]:
        """Return the meta-data of the created object."""
        meta = super().meta
        meta.update(
            {
                "n_trials": self._n_trials,
                "threshold": self._threshold,
                "fixing_strategy": self._fixing_strategy,
                "random_prob": self._random_prob,
            }
        )

        return meta

    def maximize(
        self,
        previous_configs: list[Configuration],
        n_points: int | None = None,
        random_design: AbstractRandomDesign | None = None,
        n_evaluated_trials: int | None = None,
    ) -> Iterator[Configuration]:
        """Maximizes the acquisition function, see ``AbstractAcquisitionMaximizer.maximize``. The random design
        samples from the (possibly reduced) configuration space, so its configurations are mapped back to the original
        configuration space.
        """
        for config in super().maximize(previous_configs, n_points, random_design, n_evaluated_trials):
            yield self._to_space(config, self._original_cs)

    def _maximize(
        self,
        previous_configs: list[Configuration],
        n_points: int,
        _sorted: bool = True,
    ) -> list[tuple[float, Configuration]]:
        """Maximizes the acquisition function via random search on a configuration space that is dynamically
        reduced to the currently most important hyperparameters.

        Parameters
        ----------
        previous_configs : list[Configuration]
            Previously evaluated configurations. Used to derive the current position in the reduction schedule,
            and, if ``fixing_strategy == "incumbent"``, used for the reference configuration.
        n_points : int
            Number of configurations to return.
        _sorted : bool, optional
            If True, sort candidates by their acquisition value (descending), by default True.

        Returns
        -------
        list[tuple[float, Configuration]]
            Candidates with their acquisition function value. (acq value, candidate)
        """
        if self._random_branch_rng.rand() < self._random_prob:
            logger.debug("Sampling configurations from the original configuration space at random.")
            random_configs = (
                self._original_cs.sample_configuration(n_points)
                if n_points > 1
                else [self._original_cs.sample_configuration()]
            )
            for config in random_configs:
                config.origin = "Acquisition Function Maximizer: HPI Random Search (random)"
            return [(0, cfg) for cfg in random_configs if cfg not in previous_configs]

        threshold = self._get_current_threshold()

        if threshold <= 0:
            self._configspace = self._original_cs
            return super()._maximize(previous_configs, n_points, _sorted=_sorted)

        reference_config = self._get_reference_configuration(previous_configs)
        important_hps = self._compute_important_hps(previous_configs, reference_config, threshold)

        if len(important_hps) > 0:
            self._reduce_configspace(important_hps, reference_config)
        else:
            self._configspace = self._original_cs

        seen = set(previous_configs)
        configs = (
            self._configspace.sample_configuration(n_points)
            if n_points > 1
            else [self._configspace.sample_configuration()]
        )
        # Constants of the reduced space are encoded differently, which would corrupt the surrogate model's inputs.
        configs = self._drop_forbidden([self._to_space(config, self._original_cs) for config in configs])

        if len(configs) < min(2, n_points):
            logger.info(
                "The reduced configuration space did not yield enough new configurations. Falling back to the "
                "original configuration space for this iteration."
            )

        configs = [c for c in configs if c not in seen]
        attempts = 0
        while len(configs) < min(2, n_points) and attempts < 30:
            configs = (
                self._original_cs.sample_configuration(n_points)
                if n_points > 1
                else [self._original_cs.sample_configuration()]
            )
            configs = [c for c in configs if c not in seen]
            attempts += 1

        for config in configs:
            config.origin = "Acquisition Function Maximizer: HPI Random Search (sorted)"

        if not _sorted:
            return [(0, config) for config in configs]

        return self._sort_by_acquisition_value(configs)

    def _get_current_threshold(self) -> float:
        """Determines the threshold for the current iteration. If ``self._threshold`` is a plain
        value, it is applied constantly. If it is a list, ``self._n_trials`` is split into
        ``len(self._threshold)`` equal-width phases, and ``self._threshold[i]`` is applied during phase ``i``.

        Returns
        -------
        float
            The threshold to apply for the current iteration.
        """
        if not isinstance(self._threshold, list):
            return self._threshold

        assert self._n_trials is not None
        phase_length = max(1, self._n_trials // len(self._threshold))
        position = min(self._n_evaluated_trials // phase_length, len(self._threshold) - 1)

        return self._threshold[position]

    def _get_reference_configuration(self, previous_configs: list[Configuration]) -> Configuration:
        """Determines the reference configuration that unimportant hyperparameters are fixed to, and that
        HyperSHAP measures tunability against.

        Parameters
        ----------
        previous_configs : list[Configuration]
            Previously evaluated configurations, used to determine the incumbent.

        Returns
        -------
        Configuration
        """
        if self._fixing_strategy == "default":
            return self._original_cs.get_default_configuration()

        if self._fixing_strategy == "random":
            return self._original_cs.sample_configuration()

        costs = self._observed_costs(previous_configs)
        if costs is None:
            assert self._acquisition_function is not None and self._acquisition_function.model is not None
            X = convert_configurations_to_array(previous_configs)
            costs = self._acquisition_function.model.predict_marginalized(X)[0]

        self._context_Y = costs
        self._context_configs = previous_configs

        return previous_configs[int(np.argmin(self._context_Y))]

    def _compute_important_hps(
        self, previous_configs: list[Configuration], reference_config: Configuration, threshold: float
    ) -> list[str]:
        """Estimates the HyperSHAP tunability of each hyperparameter relative to ``reference_config`` on the
        importance model, and returns the smallest set of hyperparameters that jointly explain at least
        ``threshold`` of the cumulative tunability gain. Constants are excluded from the game. A non-empty selection
        is filled up with random tunable hyperparameters to at least two.

        Parameters
        ----------
        previous_configs : list[Configuration]
            Previously evaluated configurations, used to fit the importance model.
        reference_config : Configuration
        threshold : float

        Returns
        -------
        list[str]
            Names of the important hyperparameters.
        """
        try:
            from hypershap import ExplanationTask, HyperSHAP
        except ImportError as e:
            raise ImportError(
                "HPIRandomSearch requires hypershap to estimate hyperparameter importance. "
                "Install it with `pip install smac[hpi]`."
            ) from e

        # Constants would only double the number of coalitions.
        reduced_cs = ConfigurationSpace()
        reduced_cs.add([hp for hp in self._original_cs.values() if not isinstance(hp, Constant)])
        reduced_cs.add(self._original_cs.conditions)

        # The model is explained directly, as `ExplanationTask.from_function` would fit a proxy forest on uniform
        # samples, which loses the structure around the incumbent.
        model = _NegatedModel(self._importance_model(previous_configs), self._original_cs, reduced_cs)
        explanation_task = ExplanationTask.from_base_model(reduced_cs, model)
        hyper_shap = HyperSHAP(explanation_task, approximation_budget=10_000)
        settings: dict[str, Any] = {"order": 1, "index": "SV", "n_samples": 2000, "fast_mode": True}

        try:
            interaction_values = hyper_shap.tunability(
                baseline_config=self._to_space(reference_config, reduced_cs), **settings
            )
        except Exception:
            logger.debug("HyperSHAP failed with the reference configuration as baseline; using the default.")
            interaction_values = hyper_shap.tunability(
                baseline_config=reduced_cs.get_default_configuration(), **settings
            )

        shapley_values = hyper_shap.get_interaction_values_with_names(interaction_values)
        shapley_values = {hp[0]: value for hp, value in shapley_values.items() if len(hp) == 1}
        selected_hps = self._select_important_hps(shapley_values, threshold)

        if len(selected_hps) == 1:
            tunable_hps = [
                hp.name
                for hp in self._original_cs.values()
                if not isinstance(hp, Constant)
                and len(getattr(hp, "choices", (0, 1))) > 1
                and hp.name not in selected_hps
            ]
            if len(tunable_hps) > 0:
                selected_hps.append(str(self._hpi_rng.choice(tunable_hps)))

        return selected_hps

    def _observed_costs(self, configs: list[Configuration]) -> np.ndarray | None:
        """Observed costs of ``configs``, averaged over their trials. With several objectives, they are normalized
        and scalarized as the surrogate model's training data, e.g., with the current ParEGO weights. Configurations
        without a finite cost (e.g., crashed or running) get an infinite cost.

        Parameters
        ----------
        configs : list[Configuration]

        Returns
        -------
        np.ndarray | None
            Costs of ``configs``, or None if the runhistory is not available or no cost is finite.
        """
        if self.runhistory is None:
            return None

        mo = self.runhistory.multi_objective_algorithm
        costs = []
        for config in configs:
            cost = np.mean(np.asarray(self.runhistory.get_costs(config) or [np.inf], dtype=float), axis=0)
            if not np.all(np.isfinite(cost)):
                cost = np.inf
            elif np.size(cost) > 1:
                if mo is None:
                    return None
                cost = mo(normalize_costs(cost.tolist(), self.runhistory.objective_bounds))
            costs.append(float(cost))

        costs_array = np.array(costs, dtype=float)
        return costs_array if np.isfinite(costs_array).any() else None

    def _importance_model(self, configs: list[Configuration]) -> AbstractModel:
        """The model HyperSHAP explains. If the surrogate model is a Gaussian process, it is used directly. Otherwise,
        a Gaussian process is fitted on the observed costs, as tree-based models predict their best training point
        almost exactly and rarely predict an improvement near it. Falls back to the surrogate model without observed
        costs.

        Parameters
        ----------
        configs : list[Configuration]

        Returns
        -------
        AbstractModel
        """
        assert self._acquisition_function is not None and self._acquisition_function.model is not None

        model = self._acquisition_function.model
        costs = self._observed_costs(configs)
        if isinstance(model, AbstractGaussianProcess) or costs is None:
            return model

        if self._gp is None:
            # Imported here to avoid a circular import.
            from smac.facade.blackbox_facade import BlackBoxFacade

            # Reused across iterations, so the kernel's hyperparameter optimization starts from the last fit.
            scenario = SimpleNamespace(configspace=self._original_cs, seed=self._seed)
            self._gp = BlackBoxFacade.get_model(scenario)  # type: ignore[arg-type]

        valid = np.isfinite(costs)
        self._gp.train(convert_configurations_to_array(configs)[valid], costs[valid].reshape(-1, 1))

        return self._gp

    @staticmethod
    def _to_space(config: Configuration, configspace: ConfigurationSpace) -> Configuration:
        """Maps ``config`` to ``configspace``, e.g., from the reduced to the original configuration space. Values of
        hyperparameters that are not in ``configspace`` or inactive are dropped.
        """
        if config.config_space is configspace:
            return config

        values = {name: value for name, value in dict(config).items() if name in configspace}
        mapped_config = deactivate_inactive_hyperparameters(values, configspace)
        mapped_config.origin = config.origin
        return mapped_config

    def _select_important_hps(self, shapley_values: dict[str, float], threshold: float) -> list[str]:
        """Greedily selects hyperparameters by descending Shapley value until their cumulative share of the total
        tunability gain reaches ``threshold``, ignoring hyperparameters whose individual contribution is
        negligible.

        Parameters
        ----------
        shapley_values : dict[str, float]
            Shapley value per hyperparameter name.
        threshold : float

        Returns
        -------
        list[str]
            Names of the selected, important hyperparameters.
        """
        # Shapley values of zero come out as tiny positive or negative numbers due to rounding, which must not count
        # as an improvement.
        tolerance = max(1e-12, 1e-9 * max((abs(v) for v in shapley_values.values()), default=0.0))
        contributions = sorted(((value, hp) for hp, value in shapley_values.items() if value > tolerance), reverse=True)

        total = sum(value for value, _ in contributions)
        cumulative_target = threshold * total
        min_contribution = min(0.05, 1 - threshold) * total

        selected_hps: list[str] = []
        cum_sum = 0.0
        for value, hp in contributions:
            if cum_sum >= cumulative_target:
                break

            if value > min_contribution:
                selected_hps.append(hp)
                cum_sum += value
        logger.info(f"Important hyperparameters: {selected_hps}.")
        return selected_hps

    def _incumbent_value_for(self, hp: Hyperparameter) -> Any:
        """Best value for `hp` among previously evaluated configs that satisfy its parent conditions.

        Generalizes "freeze at the incumbent" to conditional hyperparameters: the global incumbent only
        has a value for the region where it's active, so `hp` would otherwise always fall back to
        `hp.default_value` when active elsewhere. Falls back to it here too, only if `hp` was never active
        in any evaluated config.

        Parameters
        ----------
        hp : Hyperparameter

        Returns
        -------
        The best value for `hp` among previously evaluated configs that satisfy its parent conditions, or
        `hp.default_value` if none of the evaluated configs satisfy the conditions.
        """
        conditions = self._original_cs.parent_conditions_of[hp.name]
        best_value, best_y = None, float("inf")
        for cfg, y in zip(self._context_configs, self._context_Y):
            y = float(np.asarray(y).reshape(-1)[0])
            cfg_dict = dict(cfg)
            if hp.name in cfg_dict and all(cond.satisfied_by_value(cfg_dict) for cond in conditions) and y < best_y:
                best_value, best_y = cfg_dict[hp.name], y
        return best_value if best_value is not None else hp.default_value

    def _reduce_configspace(self, important_hps: list[str], reference_config: Configuration) -> None:
        """Builds a reduced configuration space in which unimportant hyperparameters are fixed to their
        value in ``reference_config``. Parents of important, conditional hyperparameters are kept
        important themselves so that the condition remains.

        Note
        ----
        Forbidden clauses are deliberately not transferred because fixing hyperparameters can turn a forbidden
        clause into one that is always (or never) violated. Forbidden clauses are enforced by filtering out
        violating candidates after sampling instead, see :meth:`_drop_forbidden`.

        Parameters
        ----------
        important_hps : list[str]
        reference_config : Configuration

        """
        changed = True
        while changed:
            changed = False
            for cond in self._original_cs.conditions:
                if cond.child.name not in important_hps:
                    continue
                for parent_name in self._parent_names(cond):
                    if parent_name not in important_hps:
                        important_hps.append(parent_name)
                        changed = True

        conditions = [
            cond
            for cond in self._original_cs.conditions
            if all(parent_name in important_hps for parent_name in self._parent_names(cond))
        ]

        reduced_cs = ConfigurationSpace()
        reduced_cs.random.set_state(self._configspace.random.get_state())

        for hp in self._original_cs.values():
            if hp.name in important_hps:
                try:
                    reduced_cs.add(hp)
                except Exception:
                    reduced_cs.add_hyperparameter(hp)
            else:
                if self._fixing_strategy == "incumbent":
                    fixed_value = self._incumbent_value_for(hp)
                else:
                    fixed_value = reference_config[hp.name] if hp.name in reference_config else hp.default_value
                new_hp: Hyperparameter
                try:
                    new_hp = PseudoConstant(hp.name, fixed_value)
                except Exception:
                    logger.debug(f"Could not fix hp '{hp.name}', keeping it tunable.")
                    new_hp = hp
                try:
                    reduced_cs.add(new_hp)
                except Exception:
                    reduced_cs.add_hyperparameter(new_hp)

        reduced_cs.add(conditions)
        self._configspace = reduced_cs

    @staticmethod
    def _parent_names(condition: Any) -> list[str]:
        """Names of all parents of a condition, including those of ``AndConjunction`` and ``OrConjunction``."""
        if hasattr(condition, "components"):
            return [name for component in condition.components for name in HPIRandomSearch._parent_names(component)]

        return [condition.parent.name]

    def _drop_forbidden(self, configs: list[Configuration]) -> list[Configuration]:
        """Removes configurations that violate a forbidden clause of the original (unreduced) configuration space.

        Parameters
        ----------
        configs : list[Configuration]
            Candidates, sampled from either the reduced or the original configuration space.

        Returns
        -------
        list[Configuration]
            The subset of ``configs`` that does not violate any forbidden clause.
        """
        if not self._original_cs.forbidden_clauses:
            return configs

        valid_configs = []
        for config in configs:
            try:
                self._original_cs.check_configuration_vector_representation(config.get_array())
            except ForbiddenValueError:
                continue

            valid_configs.append(config)

        return valid_configs


class _NegatedModel:
    """Wraps a model for HyperSHAP, which maximizes, so ``predict`` returns the negated predicted mean. It takes arrays
    of ``reduced_cs`` and fills the remaining columns of ``configspace`` with the default configuration's values, as
    the model expects the full width.
    """

    def __init__(self, model: AbstractModel, configspace: ConfigurationSpace, reduced_cs: ConfigurationSpace):
        self._model = model
        names = list(configspace.keys())
        self._columns = [names.index(name) for name in reduced_cs.keys()]
        self._template = configspace.get_default_configuration().get_array()

    def predict(self, X: np.ndarray) -> np.ndarray:  # noqa: D102
        full = np.tile(self._template, (len(X), 1))
        full[:, self._columns] = X
        return (-1) * self._model.predict(full)[0][:, 0]
