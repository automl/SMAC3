from __future__ import annotations

from typing import Any

import copy
import numpy as np
from ConfigSpace import Configuration, ConfigurationSpace
from ConfigSpace.exceptions import ForbiddenValueError

from smac.acquisition.function.abstract_acquisition_function import (
    AbstractAcquisitionFunction,
)
from smac.acquisition.maximizer.helpers import PseudoConstant
from smac.acquisition.maximizer.random_search import RandomSearch
from smac.utils.configspace import convert_configurations_to_array
from smac.utils.logging import get_logger
import random

__copyright__ = "Copyright 2025, Leibniz University Hanover, Institute of AI"
__license__ = "3-clause BSD"

logger = get_logger(__name__)


class HPIRandomSearch(RandomSearch):
    """Sorted random search that dynamically restricts the configuration space to the currently most important
    hyperparameters, estimated via HyperSHAP (Wever et al., 2026) on the trained surrogate model and sets the
    other hyperparameters to a constant value. Combined with ``ParEGO`` as the ``multi_objective_algorithm``,
    this implements HPI-ParEGO (Theodorakopoulos et al., 2026), concentrating the search on the hyperparameters
    that matter most for the current scalarization.

    Note
    ----
    Requires the optional ``hypershap`` dependency, install it with ``pip install smac[hpi]``.

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
    threshold : float | list[float], defaults to 0.8
        Cumulative contribution that the selected hyperparameters must jointly explain between 0 and 1. 0 disables
        the configuration space reduction. If a list, ``n_trials`` is split into ``len(threshold)`` equal-width
        phases, and ``threshold[i]`` is applied during phase ``i``. It is recommended to use a warm-up phase for
        the surrogate model to work well. For example:
        * Reduction in the middle third of trials only: ``[0.0, 0.8, 0.0]`` (default)
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
        threshold: float | list[float] = [0.0, 0.8, 0.0],
        fixing_strategy: str = "incumbent",
        random_prob: float = 0.1,
    ) -> None:
        super().__init__(
            configspace,
            acquisition_function=acquisition_function,
            challengers=challengers,
            seed=seed,
        )

        if fixing_strategy not in ("incumbent", "default", "random"):
            raise ValueError(
                f"Unknown fixing_strategy '{fixing_strategy}'. Must be one of 'incumbent', 'default', 'random'."
            )

        if isinstance(threshold, list):
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
        self.incumbent = self._original_cs.sample_configuration()

        self._configspace.seed(seed)
        self._original_cs.seed(seed)

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
        if random.random() < self._random_prob:
            logger.debug("Sampling configurations from the original configuration space at random.")
            random_configs = self._original_cs.sample_configuration(n_points)
            for config in random_configs:
                config.origin = "Acquisition Function Maximizer: HPI Random Search (random)"
            random_configs = [(0, cfg) for cfg in random_configs if cfg not in previous_configs]
            return random_configs

        threshold = self._get_current_threshold()

        if threshold <= 0:
            return super()._maximize(previous_configs, n_points, _sorted=_sorted)

        reference_config = self._get_reference_configuration(previous_configs)
        important_hps = self._compute_important_hps(reference_config, threshold)

        if len(important_hps) > 0:
            self._reduce_configspace(important_hps, reference_config)

        try:
            configs = (
                self._configspace.sample_configuration(n_points)
                if n_points > 1
                else [self._configspace.sample_configuration()]
            )
            configs = self._drop_forbidden(configs)
        except:
            configs = []

        if len(configs) < min(2, n_points):
            logger.info(
                "The reduced configuration space did not yield enough new configurations. Falling back to the "
                "original configuration space for this iteration."
            )

        attempts = 0
        while len(configs) < min(2, n_points) and attempts < 30:
            configs = (
                self._original_cs.sample_configuration(n_points)
                if n_points > 1
                else [self._original_cs.sample_configuration()]
            )
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
        assert self._n_evaluated_trials is not None
        phase_length = self._n_trials // len(self._threshold)
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

        X = convert_configurations_to_array(previous_configs)
        self._context_Y = self._acquisition_function.model.predict_marginalized(X)[0]
        self._context_configs = previous_configs

        return previous_configs[int(np.argmin(self._context_Y))]

    def _compute_important_hps(self, reference_config: Configuration, threshold: float) -> list[str]:
        """Estimates the HyperSHAP tunability of each hyperparameter relative to ``reference_config`` on the
        trained surrogate model, and returns the smallest set of hyperparameters that jointly explain at least
        ``threshold`` of the cumulative tunability gain.

        Parameters
        ----------
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

        explanation_task = ExplanationTask.from_function(config_space=self._original_cs, function=self._predict_config)
        hyper_shap = HyperSHAP(explanation_task, n_workers=4)

        try:
            interaction_values = hyper_shap.tunability(baseline_config=reference_config, order=1, index="SV")
        except Exception:
            logger.debug("HyperSHAP failed with the reference configuration as baseline; using the default.")
            interaction_values = hyper_shap.tunability(
                baseline_config=self._original_cs.get_default_configuration(), order=1, index="SV"
            )

        shapley_values = hyper_shap.get_interaction_values_with_names(interaction_values)
        shapley_values = {hp: value for hp, value in shapley_values.items()}

        del explanation_task, hyper_shap
        return self._select_important_hps(shapley_values, threshold)

    def _predict_config(self, config: Configuration) -> float:
        """Predicts the (negated) cost of ``config`` using the trained surrogate model.

        HyperSHAP maximizes, so the costs need to be negated.

        Parameters
        ----------
        config : Configuration

        Returns
        -------
        float
        """
        arr = config.get_array()
        return (-1) * self._acquisition_function.model.predict(np.array([arr]))[0][0]

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
        contributions = sorted(((value, hp) for hp, value in shapley_values.items() if value > 0), reverse=True)

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

        return selected_hps

    def _incumbent_value_for(self, hp):
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

    def _reduce_configspace(self, important_hps: list[str], reference_config: Configuration) -> ConfigurationSpace:
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
                if cond.child.name in important_hps and cond.parent.name not in important_hps:
                    important_hps.append(cond.parent.name)
                    changed = True

        conditions = [cond for cond in self._original_cs.conditions if cond.parent.name in important_hps]

        reduced_cs = ConfigurationSpace()
        reduced_cs.random.set_state(self._original_cs.random.get_state())

        for hp in self._original_cs.values():
            if hp.name in important_hps:
                try:
                    reduced_cs.add(hp)
                except:
                    reduced_cs.add_hyperparameter(hp)
            else:
                if self._fixing_strategy == "incumbent":
                    fixed_value = self._incumbent_value_for(hp)
                else:
                    fixed_value = reference_config[hp.name] if hp.name in reference_config else hp.default_value
                try:
                    new_hp = PseudoConstant(hp.name, fixed_value)
                except Exception:
                    logger.debug(f"Could not fix hp '{hp.name}'; keeping it tunable.")
                    new_hp = hp
                try:
                    reduced_cs.add(new_hp)
                except:
                    reduced_cs.add_hyperparameter(new_hp)

        reduced_cs.add(conditions)
        self._configspace = reduced_cs

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
