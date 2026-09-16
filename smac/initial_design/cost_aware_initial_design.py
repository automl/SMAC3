from __future__ import annotations

from typing import Any, Iterator

import logging
from collections import OrderedDict

import numpy as np
from ConfigSpace import Configuration
from scipy.spatial.distance import cdist

from smac.initial_design.abstract_initial_design import AbstractInitialDesign
from smac.initial_design.sobol_design import SobolInitialDesign
from smac.model.abstract_model import AbstractModel
from smac.runhistory.dataclasses import TrialKey
from smac.runhistory.runhistory import RunHistory
from smac.scenario import Scenario


class CostAwareInitialDesign(AbstractInitialDesign):
    """
    Selects initial configurations by balancing exploration and cost.

    This strategy aims to pick a set of starting configurations that are diverse
    (spread out in the configuration space) and cheap, staying within a given
    initial budget. It implements the "Cost-effective initial design" algorithm.

    The algorithm works by running an elimination process in each round to select
    one new configuration. This process repeatedly eliminates candidate
    configurations that are either too expensive or too similar to points already
    chosen, until only one suitable candidate remains.
    """

    def __init__(
        self,
        scenario: Scenario,
        cost_model: AbstractModel,
        initial_budget: float,
        runhistory: RunHistory | None = None,
        candidate_pool_size: int = 1000,
        candidate_generator: type[AbstractInitialDesign] = SobolInitialDesign,
        n_bootstrap_points: int = 1,
        **kwargs: Any,
    ):
        super().__init__(scenario=scenario, n_configs=None, **kwargs)
        self._scenario = scenario
        self._cost_model = cost_model
        self._initial_budget = initial_budget
        self._candidate_pool_size = candidate_pool_size
        self._logger = logging.getLogger(self.__class__.__name__)
        self._candidate_generator = candidate_generator
        self._n_bootstrap_points = n_bootstrap_points
        self._runhistory = runhistory

    def _select_configurations(self):  # type: ignore
        pass

    def select_configurations(self) -> Iterator[Configuration]:  # type: ignore
        """
        Generates a set of cost-aware initial configurations following Algorithm 1.
        This method is a generator, yielding one configuration at a time.

        The main logic:
        1.  Initializes cumulative time and the list for the initial design.
        2.  A large pool of random candidate configurations is sampled.
        3.  It then loops until the initial budget is spent:
            a. An elimination process begins on the set of available candidates.
               This process repeatedly removes the most expensive candidate (based on model prediction)
               and the candidate closest to the already-selected design points.
            b. The single remaining candidate is chosen and yielded.
            c. The predicted cost of this new configuration is added to the total. The cost model
               is expected to be retrained externally after evaluation.
        """

        def get_initial_design_cost() -> float:
            """Calculates the true cost of the initial design from the runhistory."""
            if self._runhistory is None:
                return 0.0

            initial_design_origins = {"Sampling", "Initial design", "Cost Aware Initial Design"}
            cost = 0.0
            processed_configs = set()

            for config in self._runhistory.get_configs():
                config_id = self._runhistory.get_config_id(config)
                if config.origin in initial_design_origins:
                    if config_id in processed_configs:
                        continue

                    trial_infos = self._runhistory.get_trials(config, highest_observed_budget_only=False)
                    trial_keys = [TrialKey(config_id, info.instance, info.seed, info.budget) for info in trial_infos]

                    # CHANGED: Read resource cost from additional_info instead of time field
                    resource_costs = [
                        self._runhistory[key].additional_info.get("resource_cost", self._runhistory[key].time)
                        for key in trial_keys
                        if key in self._runhistory
                    ]

                    if resource_costs:
                        cost += np.mean(resource_costs)
                        processed_configs.add(config_id)
            return cost

        selected_arrays: list[np.ndarray] = []

        # Step 3: Discretize Ω into ˜Ω using the specified candidate generator.
        generator = self._candidate_generator(
            scenario=self._scenario,
            n_configs=self._candidate_pool_size,
            max_ratio=1.0,  # This ensures the pool size is not reduced.
        )
        candidate_pool_raw = generator.select_configurations()

        discretized_space = list(
            OrderedDict((tuple(config.get_array()), config) for config in candidate_pool_raw).values()
        )

        # --- Bootstrap phase: Yield random points to train the model ---
        if self._n_bootstrap_points > 0 and discretized_space:
            self._logger.info(f"Yielding {self._n_bootstrap_points} random point(s) to bootstrap the cost model.")
            available_candidates = list(discretized_space)
            n_to_sample = min(self._n_bootstrap_points, len(available_candidates))
            sample_indices = self._rng.choice(len(available_candidates), n_to_sample, replace=False)

            configs_to_remove_from_pool_indices = sorted(sample_indices, reverse=True)

            for idx in configs_to_remove_from_pool_indices:
                bootstrap_config = available_candidates[idx]
                bootstrap_config.origin = "Sampling"
                selected_arrays.append(bootstrap_config.get_array())
                yield bootstrap_config

            # Remove the sampled configs from the main discretized space
            # Create a set of arrays to remove for efficient lookup
            arrays_to_remove = {tuple(discretized_space[i].get_array()) for i in sample_indices}
            discretized_space = [c for c in discretized_space if tuple(c.get_array()) not in arrays_to_remove]

        self._logger.info(
            f"Generated {len(candidate_pool_raw)} candidates, resulting in "
            f"{len(discretized_space)} unique configurations for cost-aware selection."
        )

        if not discretized_space:
            return

        all_config_arrays = np.array([c.get_array() for c in discretized_space])
        remaining_indices = set(range(len(discretized_space)))

        # Step 4: Main loop `while ct < τinit do`
        iteration = 0
        while remaining_indices:
            cumulative_time = get_initial_design_cost()
            if cumulative_time >= self._initial_budget:
                self._logger.info(
                    f"Initial design budget of {self._initial_budget: .2f} reached or exceeded. "
                    f"Actual cost: {cumulative_time: .2f}. Stopping."
                )
                break

            iteration += 1
            self._logger.info(
                f"Iteration {iteration}: Budget used {cumulative_time: .2f}/{self._initial_budget: .2f}, "
                f"Candidates remaining: {len(remaining_indices)}, seed = {self._seed}"
            )

            if len(remaining_indices) > 1:
                current_indices_list = list(remaining_indices)
                current_arrays = all_config_arrays[current_indices_list]
                current_costs, _ = self._cost_model.predict(current_arrays)
                costs_dict = dict(zip(current_indices_list, current_costs.flatten()))

                if selected_arrays:
                    selected_matrix = np.array(selected_arrays)
                    distances = cdist(current_arrays, selected_matrix)
                    min_distances = np.min(distances, axis=1)
                    distances_dict = dict(zip(current_indices_list, min_distances))
                else:
                    distances_dict = {}
            else:
                # If only one candidate is left, we select it directly
                chosen_idx = next(iter(remaining_indices))
                chosen_config = discretized_space[chosen_idx]

                chosen_config.origin = "Cost Aware Initial Design"
                yield chosen_config
                break  # End of initial design

            # Step 5: Inner loop `while size Xcand > 1 do`
            candidates = set(remaining_indices)
            while len(candidates) > 1:
                # Step 6: Remove the most expensive candidate.
                if len(candidates) > 1:
                    most_expensive_idx = max(candidates, key=lambda i: costs_dict[i])
                    candidates.remove(most_expensive_idx)

                if len(candidates) <= 1:
                    break

                # Step 7: Remove the candidate closest to our already-selected points.
                if distances_dict:
                    filtered_distances = {k: v for k, v in distances_dict.items() if k in candidates}
                    if filtered_distances:
                        closest_idx = min(filtered_distances, key=lambda k: filtered_distances[k])
                        candidates.remove(closest_idx)
                    else:
                        candidates.pop()
                else:
                    pass  # First iteration, no selected points yet

            # Step 8: `end while`
            if candidates:
                chosen_idx = next(iter(candidates))
                chosen_config = discretized_space[chosen_idx]
                selected_arrays.append(chosen_config.get_array())
                remaining_indices.remove(chosen_idx)

                chosen_config.origin = "Cost Aware Initial Design"
                yield chosen_config
            else:
                self._logger.warning("No candidates left after elimination process. Stopping.")
                break

        final_cost = get_initial_design_cost()
        self._logger.info(
            "Cost-aware initial design finished." f"Final actual cost: {final_cost: .2f}/{self._initial_budget: .2f}."
        )
