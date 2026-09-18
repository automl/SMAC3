from __future__ import annotations

from typing import Any, Iterator, Mapping

import copy
import time

import numpy as np
from ConfigSpace import Configuration, ConfigurationSpace

from smac.acquisition.function.abstract_acquisition_function import (
    AbstractAcquisitionFunction,
)
from smac.acquisition.function.weighted_acquisition_function import (
    WeightedAcquisitionFunction,
)
from smac.acquisition.maximizer.abstract_acquisition_maximizer import (
    AbstractAcquisitionMaximizer,
)
from smac.acquisition.weight.abstract_weight import AbstractAcquisitionWeight
from smac.acquisition.weight.acceptance import (
    AbstractPriorAcceptancePolicy,
    AcceptAllPriors,
)
from smac.acquisition.weight.composite import PriorEnsemble
from smac.acquisition.weight.decay import DecaySchedule, PolynomialDecay
from smac.acquisition.weight.prior import (
    AbstractInputPrior,
    ConfigSpacePrior,
    PriorWeight,
)
from smac.callback.callback import Callback
from smac.initial_design import AbstractInitialDesign
from smac.main.exceptions import ConfigurationSpaceExhaustedException
from smac.model.abstract_model import AbstractModel
from smac.random_design.abstract_random_design import AbstractRandomDesign
from smac.runhistory.encoder.abstract_encoder import AbstractRunHistoryEncoder
from smac.runhistory.runhistory import RunHistory
from smac.scenario import Scenario
from smac.utils.configspace import create_prior_configspace_copy
from smac.utils.logging import get_logger

__copyright__ = "Copyright 2025, Leibniz University Hanover, Institute of AI"
__license__ = "3-clause BSD"


logger = get_logger(__name__)


class ConfigSelector:
    """The config selector handles the surrogate model and the acquisition function. Based on both components, the next
    configuration is selected.

    Parameters
    ----------
    retrain_after : int | None, defaults to 8
        How many configurations should be returned before the surrogate model is retrained.
    retrain_wallclock_ratio: float | None, default to None
        How much time of the total elapsed wallclock time should be spend on retraining the surrogate model
        and the acquisition function look. Example ratio of 0.1 would result in that only 10% of the wallclock time is
        spend on retraining.
    min_configurations: int, defaults to 1
        The minimum number of configurations that need to yield before retraining can occur. Should be lower or equal to
        retrain_after.
    max_new_config_tries : int, defaults to 8
        How often to retry receiving a new configuration before giving up.
    min_trials: int, defaults to 1
        How many samples are required to train the surrogate model. If budgets are involved,
        the highest budgets are checked first. For example, if min_trials is three, but we find only
        two trials in the runhistory for the highest budget, we will use trials of a lower budget
        instead.
    """

    def __init__(
        self,
        scenario: Scenario,
        *,
        retrain_after: int | None = 8,
        retrain_wallclock_ratio: float | None = None,
        min_configurations: int = 1,
        max_new_config_tries: int = 16,
        min_trials: int = 1,
    ) -> None:
        # Those are the configs sampled from the passed initial design
        # Selecting configurations from initial design
        self._initial_design_configs: list[Configuration] = []

        # Set classes globally
        self._scenario = scenario
        self._runhistory: RunHistory | None = None
        self._runhistory_encoder: AbstractRunHistoryEncoder | None = None
        self._model: AbstractModel | None = None
        self._acquisition_maximizer: AbstractAcquisitionMaximizer | None = None
        self._acquisition_function: AbstractAcquisitionFunction | None = None
        self._random_design: AbstractRandomDesign | None = None
        self._callbacks: list[Callback] = []

        # And other variables
        self._retrain_after = retrain_after
        self._retrain_wallclock_ratio = retrain_wallclock_ratio
        self._min_configurations = min_configurations
        self._previous_entries = -1
        self._predict_x_best = True

        # Set when the acquisition function is changed from outside, so that it is brought up to date even
        # though no new trial has been reported since the last update.
        self._acquisition_needs_update = True
        self._force_retrain = False

        # Judges a user belief stated during a run. Accepts everything, so that nobody pays for a safeguard they
        # did not ask for.
        self._prior_acceptance_policy: AbstractPriorAcceptancePolicy = AcceptAllPriors()
        self._min_trials = min_trials
        self._considered_budgets: list[float | int | None] = [None]

        # How often to retry receiving a new configuration
        # (counter increases if the received config was already returned before)
        self._max_new_config_tries = max_new_config_tries
        self._counter = 0

        self._wallclock_start_time: float = time.time()
        self._acquisition_training_times: list[float] = []

        # Processed configurations should be stored here; this is important to not return the same configuration twice
        self._processed_configs: list[Configuration] = []

        # Check if there is at least one retrain condition
        if self._retrain_after is None and self._retrain_wallclock_ratio is None:
            raise ValueError("No retrain condition specified!")

        if self._retrain_after is not None:
            if self._retrain_after < self._min_configurations:
                raise ValueError("retrain_after should be higher or equal to min_configurations")

    def _set_components(
        self,
        initial_design: AbstractInitialDesign,
        runhistory: RunHistory,
        runhistory_encoder: AbstractRunHistoryEncoder,
        model: AbstractModel,
        acquisition_maximizer: AbstractAcquisitionMaximizer,
        acquisition_function: AbstractAcquisitionFunction,
        random_design: AbstractRandomDesign,
        callbacks: list[Callback] = None,
    ) -> None:
        self._runhistory = runhistory
        self._runhistory_encoder = runhistory_encoder
        self._model = model
        self._acquisition_maximizer = acquisition_maximizer
        self._acquisition_function = acquisition_function
        self._random_design = random_design
        self._callbacks = callbacks if callbacks is not None else []

        self._initial_design_configs = initial_design.select_configurations()
        if len(self._initial_design_configs) == 0:
            # raise RuntimeError("SMAC needs initial configurations to work.")
            logger.warning("No initial configurations were sampled.")

    @property
    def meta(self) -> dict[str, Any]:
        """Returns the meta data of the created object."""
        return {
            "name": self.__class__.__name__,
            "retrain_after": self._retrain_after,
            "retrain_wallclock_ratio": self._retrain_wallclock_ratio,
            "min_configurations": self._min_configurations,
            "max_new_config_tries": self._max_new_config_tries,
            "min_trials": self._min_trials,
        }

    def __iter__(self) -> Iterator[Configuration]:
        """This method returns the next configuration to evaluate. It ignores already processed configurations, i.e.,
        the configurations from the runhistory, if the runhistory is not empty.
        The method (after yielding the initial design configurations) trains the surrogate model, maximizes the
        acquisition function and yields ``n`` configurations. After the ``n`` configurations, the surrogate model is
        trained again, etc. The program stops if ``retries`` was reached within each iteration. A configuration
        is ignored, if it was used already before.

        Note
        ----
        When SMAC continues a run, processed configurations from the runhistory are ignored. For example, if the
        initial design configurations already have been processed, they are ignored here. After the run is
        continued, however, the surrogate model is trained based on the runhistory in all cases.

        Returns
        -------
        next_config : Iterator[Configuration]
            The next configuration to evaluate.
        """
        assert self._runhistory is not None
        assert self._runhistory_encoder is not None
        assert self._model is not None
        assert self._acquisition_maximizer is not None
        assert self._acquisition_function is not None
        assert self._random_design is not None

        self._processed_configs = self._runhistory.get_configs()

        # We add more retries because there could be a case in which the processed configs are sampled again
        self._max_new_config_tries += len(self._processed_configs)

        logger.debug("Search for the next configuration...")
        self._call_callbacks_on_start()

        # Configurations that are already in the runhistory at this point can be related
        # to the initial design's own configurations in different ways depending on the warmstart mode.
        initial_design_configs = self._initial_design_configs
        n_warmstarted = len(self._processed_configs)

        if n_warmstarted > 0:
            mode = self._scenario.initial_design_warmstart_mode
            if mode == "reduce_budget":
                # The already-evaluated configs count against the initial design's budget: we only
                # propose as many (still missing) configs as are needed to reach the original budget.
                initial_design_configs = initial_design_configs[n_warmstarted:]
            elif mode == "replace":
                # The already-evaluated configs *are* the complete initial design.
                initial_design_configs = []
            # "additional" (default): the initial design is not affected at all.

        # First: We return the initial configurations
        for config in initial_design_configs:
            if config not in self._processed_configs:
                self._processed_configs.append(config)
                self._call_callbacks_on_end(config)
                yield config
                self._call_callbacks_on_start()

        # We want to generate configurations endlessly
        while True:
            # Cost value of incumbent configuration (required for acquisition function).
            # If not given, it will be inferred from runhistory or predicted.
            # If not given and runhistory is empty, it will raise a ValueError.
            incumbent_value: float | None = None

            # Everytime we re-train the surrogate model, we also update our multi-objective algorithm
            if (mo := self._runhistory_encoder.multi_objective_algorithm) is not None:
                mo.update_on_iteration_start()

            X, Y, X_configurations = self._collect_data()
            previous_configs = self._runhistory.get_configs()

            if X.shape[0] == 0:
                # Only return a single point to avoid an overly high number of random search iterations.
                # We got rid of random search here and replaced it with a simple configuration sampling from
                # the configspace.
                logger.debug("No data available to train the model. Sample a random configuration.")

                config = self._scenario.configspace.sample_configuration()
                self._call_callbacks_on_end(config)
                yield config
                self._call_callbacks_on_start()

                # Important to continue here because we still don't have data available
                continue

            # Check if X/Y differs from the last run, otherwise use cached results
            train_start_time = time.time()
            data_changed = self._previous_entries != Y.shape[0]

            if data_changed:
                self._model.train(X, Y)

            # The acquisition function is also updated when it was changed from outside without new data
            # arriving, which is how a user prior supplied during a run takes effect.
            if data_changed or self._acquisition_needs_update:
                x_best_array: np.ndarray | None = None
                if incumbent_value is not None:
                    best_observation = incumbent_value
                else:
                    if self._runhistory.empty():
                        raise ValueError("Runhistory is empty and the cost value of the incumbent is unknown.")

                    x_best_array, best_observation = self._get_x_best(X_configurations)

                self._acquisition_function.update(
                    model=self._model,
                    eta=best_observation,
                    incumbent_array=x_best_array,
                    num_data=self._current_num_data(),
                    X=X_configurations,
                    incumbents=self._runhistory.incumbents,
                    runhistory=self._runhistory,
                    runhistory_encoder=self._runhistory_encoder,
                )

                self._acquisition_needs_update = False

            # We want to cache how many entries we used because if we have the same number of entries
            # we don't need to train the next time
            self._previous_entries = Y.shape[0]

            # Now we maximize the acquisition function
            challengers = self._acquisition_maximizer.maximize(
                previous_configs,
                # n_points=self._retrain_after, #TODO MERGE check
                random_design=self._random_design,
            )

            if self._retrain_wallclock_ratio is not None:
                # TODO: CB: What does this actually do? Delete/clear the iterator?
                #  --> JG: To easily measure the time needed to perform maximise, difficult otherwise due to yield
                len(list(challengers))  # Forces actual computation of the acquisition function maximizer

            self._acquisition_training_times.append(time.time() - train_start_time)

            retrain = False
            failed_counter = 0
            for config in challengers:
                if config not in self._processed_configs:
                    self._counter += 1
                    self._processed_configs.append(config)
                    self._call_callbacks_on_end(config)
                    yield config
                    retrain = self._check_for_retrain()
                    self._call_callbacks_on_start()

                    # We break to enforce a new iteration of the while loop (i.e. we retrain the surrogate model)
                    if retrain:
                        self._counter = 0
                        break
                else:
                    failed_counter += 1

                    # We exit the loop if we have tried to add the same configuration too often
                    if failed_counter == self._max_new_config_tries:
                        logger.warning(f"Could not return a new configuration after {failed_counter} retries.")
                        break

            # if we don't have enough configurations, we want to sample random configurations
            if not retrain:
                logger.warning(
                    "Did not find enough configurations from the acquisition function. Sampling random configurations."
                )
                random_configs_retries = 0
                while not retrain and random_configs_retries < self._max_new_config_tries:
                    config = self._scenario.configspace.sample_configuration()
                    if config not in self._processed_configs:
                        self._counter += 1
                        config.origin = "Random Search (max retries, no candidates)"
                        self._processed_configs.append(config)
                        self._call_callbacks_on_end(config)
                        yield config
                        retrain = self._check_for_retrain()
                        self._call_callbacks_on_start()
                    else:
                        random_configs_retries += 1

                    if random_configs_retries == self._max_new_config_tries:
                        logger.warning(f"Could not return a new configuration after {random_configs_retries} retries.")
                        raise ConfigurationSpaceExhaustedException()

    def _check_for_retrain(self) -> bool:
        if self._force_retrain:
            # The challengers still queued were ranked by an acquisition function which has since changed, so
            # they no longer reflect what the search should try next.
            self._force_retrain = False
            logger.debug("The acquisition function changed. Start a new iteration and rank the challengers again.")

            return True

        if self._retrain_after is not None:
            if self._counter >= self._retrain_after:
                logger.debug(
                    f"Yielded {self._counter} configurations. Start new iteration and retrain surrogate model."
                )
                return True

        if self._retrain_wallclock_ratio is not None:
            if self._counter < self._min_configurations:
                # Force a minimum number of configurations to be yielded despite the ratio
                return False

            # Total elapsed wallcock time
            elapsed_time = time.time() - self._wallclock_start_time

            # Total time spend on getting configurations with the surrogate model
            acquisition_training_time = sum(self._acquisition_training_times)

            # Retrain when more time has been spend
            if acquisition_training_time / elapsed_time < self._retrain_wallclock_ratio:
                logger.debug(
                    f"Less than {self._retrain_wallclock_ratio:.2%} "  # noqa: E231
                    f"({acquisition_training_time / elapsed_time:.2f}) "  # noqa: E231
                    f"of the elapsed wallclock time ({elapsed_time:.2f}s) has "  # noqa: E231
                    "been spend on finding new configurations "
                    f"with the surrogate model. Start new iteration and retrain surrogate model."
                )
                return True

        return False

    def _call_callbacks_on_start(self) -> None:
        for callback in self._callbacks:
            callback.on_next_configurations_start(self)

    def _call_callbacks_on_end(self, config: Configuration) -> None:
        """Calls ``on_next_configurations_end`` of the registered callbacks."""
        # For safety reasons: Return a copy of the config
        if len(self._callbacks) > 0:
            config = copy.deepcopy(config)

        for callback in self._callbacks:
            callback.on_next_configurations_end(self, config)

    def _collect_data(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Collects the data from the runhistory to train the surrogate model. In the case of budgets, the data
        collection strategy is as follows: Looking from highest to lowest budget, return those observations
        that support at least ``self._min_trials`` points.

        If no budgets are used, this is equivalent to returning all observations.
        """
        assert self._runhistory is not None
        assert self._runhistory_encoder is not None

        # If we use a float value as a budget, we want to train the model only on the highest budget
        unique_budgets: set[float] = {run_key.budget for run_key in self._runhistory if run_key.budget is not None}

        available_budgets: list[float] | list[None]
        if len(unique_budgets) > 0:
            # Sort available budgets from highest to lowest budget
            available_budgets = sorted(unique_budgets, reverse=True)
        else:
            available_budgets = [None]

        # Get #points per budget and if there are enough samples, then build a model
        for b in available_budgets:
            X, Y = self._runhistory_encoder.transform(budget_subset=[b])

            if X.shape[0] >= self._min_trials:
                self._considered_budgets = [b]

                # Possible add running configs?
                configs_array = self._runhistory_encoder.get_configurations(budget_subset=self._considered_budgets)

                return X, Y, configs_array

        return (
            np.empty(shape=[0, 0]),
            np.empty(
                shape=[
                    0,
                ]
            ),
            np.empty(shape=[0, 0]),
        )

    def add_prior(
        self,
        prior: PriorWeight | AbstractInputPrior | ConfigurationSpace | Mapping[str, Any],
        *,
        key: str | None = None,
        decay: DecaySchedule | None = None,
        acceptance_policy: AbstractPriorAcceptancePolicy | None = None,
        sampling_weight: float | None = None,
    ) -> str | None:
        """Adds a user belief about where the optimum lies to the acquisition function.

        Safe to call at any time, including from a callback and between ``ask`` and ``tell``. The belief is
        anchored at the current trial count, so one supplied late in a run arrives at full strength rather than
        inheriting the exponent an older belief has already decayed to.

        Parameters
        ----------
        prior : PriorWeight | AbstractInputPrior | ConfigurationSpace | Mapping[str, Any]
            The belief. A mapping of hyperparameter names to believed values is turned into a configuration space
            carrying a prior; a configuration space is read as one directly.
        key : str | None, defaults to None
            Key to register the belief under, so that it can be replaced or removed later. Generated if not given.
        decay : DecaySchedule | None, defaults to None
            How the belief fades. Defaults to the piBO schedule with a decay factor of ``n_trials`` / 10. Ignored
            when a `PriorWeight` is passed, which carries its own.
        acceptance_policy : AbstractPriorAcceptancePolicy | None, defaults to None
            Judges whether the belief is plausible enough to act on. Defaults to the policy set on this config
            selector, which accepts everything unless it has been changed.
        sampling_weight : float | None, defaults to None
            Share of the acquisition function maximizer's candidates to draw from the belief, relative to the
            search space itself.

        Returns
        -------
        str | None
            The key the belief is registered under, or `None` if the acceptance policy rejected it, in which case
            nothing was changed.
        """
        assert self._acquisition_function is not None
        assert self._acquisition_maximizer is not None

        weight = self._as_prior_weight(prior, decay)

        if weight.prior is not None:
            weight.prior.validate_against(self._scenario.configspace)

        policy = acceptance_policy if acceptance_policy is not None else self._prior_acceptance_policy
        if not policy.accept(
            weight,
            configspace=self._scenario.configspace,
            model=self._model,
            runhistory=self._runhistory,
            incumbent=self._incumbent(),
            rng=self._acquisition_maximizer.rng,
        ):
            return None

        weight.anchor(self._current_num_data())

        weighted = self._ensure_weighted()

        ensemble = weighted.get_weight(PriorEnsemble)
        if ensemble is None:
            ensemble = PriorEnsemble()
            weighted.add_weight(ensemble)

        key = ensemble.add(weight, key=key)

        self._add_sampling_space(key, weight, sampling_weight)
        self.invalidate_acquisition()

        logger.info(f"Added the user prior {key!r}, anchored at trial {weight.t0}.")

        return key

    @property
    def prior_acceptance_policy(self) -> AbstractPriorAcceptancePolicy:
        """Judges whether a stated belief is plausible enough to act on. Accepts everything by default."""
        return self._prior_acceptance_policy

    @prior_acceptance_policy.setter
    def prior_acceptance_policy(self, policy: AbstractPriorAcceptancePolicy) -> None:
        self._prior_acceptance_policy = policy

    def _incumbent(self) -> Configuration | None:
        """The best configuration so far, if there is one."""
        if self._runhistory is None or len(self._runhistory.incumbents) == 0:
            return None

        return self._runhistory.incumbents[0]

    def remove_prior(self, key: str) -> None:
        """Removes a previously added user belief.

        Parameters
        ----------
        key : str
            The key the belief was registered under.
        """
        assert self._acquisition_function is not None
        assert self._acquisition_maximizer is not None

        ensemble = self.priors_ensemble
        if ensemble is None:
            raise KeyError(f"No user prior is registered under the key {key!r}.")

        ensemble.remove(key)

        if self._acquisition_maximizer.supports_sampling_spaces:
            try:
                self._acquisition_maximizer.remove_sampling_space(key)
            except KeyError:
                # The belief could not be sampled from, so it never contributed candidates.
                pass

        self.invalidate_acquisition()

        logger.info(f"Removed the user prior {key!r}.")

    @property
    def priors(self) -> Mapping[str, AbstractAcquisitionWeight]:
        """The user beliefs currently weighting the acquisition function, by key."""
        ensemble = self.priors_ensemble

        return {} if ensemble is None else ensemble.members

    @property
    def priors_ensemble(self) -> PriorEnsemble | None:
        """The weight collecting the user beliefs, if there is one."""
        if not isinstance(self._acquisition_function, WeightedAcquisitionFunction):
            return None

        return self._acquisition_function.get_weight(PriorEnsemble)

    def _as_prior_weight(
        self,
        prior: PriorWeight | AbstractInputPrior | ConfigurationSpace | Mapping[str, Any],
        decay: DecaySchedule | None,
    ) -> PriorWeight:
        """Turns whatever the caller supplied into a prior weight."""
        if isinstance(prior, PriorWeight):
            if decay is not None:
                raise ValueError("A PriorWeight carries its own decay schedule; do not pass one as well.")

            return prior

        if decay is None:
            # The empirically founded default from piBO.
            decay = PolynomialDecay(beta=self._scenario.n_trials / 10)

        if isinstance(prior, ConfigurationSpace):
            return PriorWeight(ConfigSpacePrior(prior), decay)

        if isinstance(prior, AbstractInputPrior):
            return PriorWeight(prior, decay)

        if isinstance(prior, Mapping):
            configspace = create_prior_configspace_copy(self._scenario.configspace, prior)

            return PriorWeight(ConfigSpacePrior(configspace), decay)

        raise TypeError(f"Cannot read {prior!r} as a user prior over the optimum.")

    def _add_sampling_space(self, key: str, weight: PriorWeight, sampling_weight: float | None) -> None:
        """Lets the acquisition function maximizer draw part of its candidates from the belief."""
        assert self._acquisition_maximizer is not None

        if weight.prior is None or weight.prior.sample(1) is None:
            logger.debug(f"The user prior {key!r} cannot be sampled from; it only weights the acquisition function.")

            return

        if not self._acquisition_maximizer.supports_sampling_spaces:
            logger.warning(
                f"{self._acquisition_maximizer.__class__.__name__} cannot sample from a user prior, so the "
                f"prior {key!r} only weights the acquisition function. A sharply peaked prior may then never be "
                "reached by the candidates that are ranked."
            )

            return

        self._acquisition_maximizer.add_sampling_space(key, weight.prior, sampling_weight)

    def _ensure_weighted(self) -> WeightedAcquisitionFunction:
        """Returns the acquisition function as a weighted one, wrapping it in place if it is not one yet.

        A belief may be stated during a run that started without one, so the wrapper cannot be required to have
        been installed up front. The maximizer is re-pointed at the wrapper, because it scores the object it was
        given rather than looking it up.
        """
        assert self._acquisition_function is not None
        assert self._acquisition_maximizer is not None

        if isinstance(self._acquisition_function, WeightedAcquisitionFunction):
            return self._acquisition_function

        weighted = WeightedAcquisitionFunction(self._acquisition_function)

        if self._model is not None:
            weighted.model = self._model

        self._acquisition_function = weighted
        self._acquisition_maximizer.acquisition_function = weighted

        return weighted

    def invalidate_acquisition(self, force_retrain: bool = True) -> None:
        """Marks the acquisition function as out of date.

        Call this after changing the acquisition function from outside, so that it is updated on the next
        iteration even though no new trial has been reported. With ``force_retrain``, the challengers already
        ranked are discarded as well and the maximizer runs again, so that the change takes effect on the very
        next configuration rather than up to ``retrain_after`` configurations later.

        Note an intensifier holding trials of its own may still hand out a queued one first.

        Parameters
        ----------
        force_retrain : bool, defaults to True
            Whether to also discard the challengers ranked before the change.
        """
        self._acquisition_needs_update = True

        if force_retrain:
            self._force_retrain = True

    def _current_num_data(self) -> int:
        """The number of finished trials, as the acquisition function counts them.

        This is the quantity handed to ``update`` as ``num_data``, so anything anchored against it - the decay of
        a user prior, say - is measured in the same units as the decay itself.
        """
        return len(self._get_evaluated_configs())

    def _get_evaluated_configs(self) -> list[Configuration]:
        assert self._runhistory is not None
        return self._runhistory.get_configs_per_budget(budget_subset=self._considered_budgets)

    def _get_x_best(self, X: np.ndarray) -> tuple[np.ndarray, float]:
        """Get value, configuration, and array representation of the *best* configuration.

        The definition of best varies depending on the argument ``predict``. If set to `True`,
        this function will return the stats of the best configuration as predicted by the model,
        otherwise it will return the stats for the best observed configuration.

        Parameters
        ----------
        predict : bool
            Whether to use the predicted or observed best.

        Returns
        -------
        float
        np.ndarry
        Configuration
        """
        if self._predict_x_best:
            model = self._model
            assert model is not None

            means, _ = model.predict_marginalized(X)
            costs = means[:, 0]
            best_index = int(np.argmin(costs))
            x_best_array = X[best_index]
            best_observation = float(costs[best_index])

        # else:
        #    all_configs = self._runhistory.get_configs_per_budget(budget_subset=self._considered_budgets)
        #    x_best = self._incumbent
        #    x_best_array = convert_configurations_to_array(all_configs)
        #    best_observation = self._runhistory.get_cost(x_best)
        #    best_observation_as_array = np.array(best_observation).reshape((1, 1))

        #    # It's unclear how to do this for inv scaling and potential future scaling.
        #    # This line should be changed if necessary
        #    best_observation = self._runhistory_encoder.transform_response_values(best_observation_as_array)
        #    best_observation = best_observation[0][0]

        return x_best_array, best_observation
