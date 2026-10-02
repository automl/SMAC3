from __future__ import annotations

from typing import Iterator

from collections import defaultdict

import numpy as np

from smac.callback.callback import Callback
from smac.constants import VERY_SMALL_NUMBER
from smac.main.smbo import SMBO
from smac.runhistory import StatusType, TrialInfo, TrialValue
from smac.utils.cost_transformer import CostTransformer
from smac.utils.logging import get_logger

__copyright__ = "Copyright 2025, Leibniz University Hanover, Institute of AI"
__license__ = "3-clause BSD"

logger = get_logger(__name__)

# Some facades (e.g. RandomFacade, MultiObjectiveFacade, AlgorithmConfigurationFacade) use a
# DefaultInitialDesign, which proposes exactly one configuration. Diagnosing -- and possibly
# aborting -- based on a single configuration would be wrong: a default configuration that happens
# to be invalid for the given dataset is a common and recoverable situation. If the initial design
# is smaller than this, the diagnosis waits for (and includes) the configurations evaluated next.
MIN_CONFIGS_FOR_DIAGNOSIS = 3


class InitialDesignDiagnosticsCallback(Callback):
    """Runs early diagnostics on the initial design evaluations.

    Once every initial design configuration has been evaluated at least once, the initial design
    evaluations are checked for degeneracies that leave the surrogate model without any feedback
    signal:

    * **All configurations failed.** Not a single configuration produced a successful evaluation.
      The target function or the pipeline around it is probably broken. There is nothing to build
      the surrogate model on, since all failed trials share the same penalty cost
      (``Scenario.crash_cost``).
    * **Some configurations failed.** Partial failures are a normal part of hyperparameter
      optimization and never abort the run, but the number of failed configurations is reported so
      that the user can react.
    * **All successful configurations share the same cost.** The metric or the search space is
      likely misconfigured. This may also be a huge plateau, but in either case the surrogate model
      has no feedback signal to learn from, so there is nothing to be gained from continuing.

    The unit of all three checks is the *configuration*, not the trial: a configuration may be
    evaluated several times (on other instances, seeds or budgets), and it only counts as failed if
    none of its evaluations succeeded.

    The check runs exactly once per optimization.

    Parameters
    ----------
    mode : str
        One of ``"warn"`` (log a warning and continue) or ``"abort"`` (additionally stop the
        optimization gracefully if the initial design carries no feedback signal, i.e. if all
        configurations failed or all of them perform identically). ``"off"`` disables the callback
        entirely and is handled by the facade, which then does not create this callback at all.
    atol : float, defaults to ``VERY_SMALL_NUMBER``
        Absolute tolerance used to decide whether all costs are identical. The costs are compared via
        their peak-to-peak range, so that near-constant objectives are detected as well.
    """

    def __init__(self, mode: str = "warn", atol: float = VERY_SMALL_NUMBER) -> None:
        self._mode = mode
        self._atol = atol
        self._done = False
        self._expected_configs: list = []
        self._seen_config_ids: set[int] = set()

    @property
    def mode(self) -> str:
        """The configured diagnostics mode."""
        return self._mode

    def on_start(self, smbo: SMBO) -> None:
        """Remembers which configurations belong to the initial design."""
        config_selector = smbo.intensifier._config_selector
        assert config_selector is not None
        if isinstance(config_selector._initial_design_configs, Iterator):
            self._expected_configs = []
            return

        self._expected_configs = list(config_selector._initial_design_configs)

    def on_tell_end(self, smbo: SMBO, info: TrialInfo, value: TrialValue) -> bool | None:
        """Checks the initial design evaluations once they are complete.

        Returns ``False`` if the optimization should be stopped, which is the established way for a
        callback to request a graceful shutdown.
        """
        if self._done:
            return None

        if not self._is_initial_design_complete(smbo, info):
            return None

        self._done = True
        return self._diagnose(smbo)

    def _is_initial_design_complete(self, smbo: SMBO, info: TrialInfo) -> bool:
        """Whether enough initial design evaluations have been collected to draw a conclusion.

        Two conditions have to hold:

        * Every initial design configuration has been evaluated at least once. A configuration may
          be evaluated repeatedly (on other instances, seeds or budgets); only the first evaluation
          of each configuration is relevant here.
        * At least ``MIN_CONFIGS_FOR_DIAGNOSIS`` distinct configurations have been evaluated. This
          also covers the cases in which there is no initial design to wait for at all, for example
          because ``n_configs=0`` was requested or because the warmstart mode ``"replace"`` consumed
          it.
        """
        # Book-keeping first: the trial we were just told about may well be an initial design
        # configuration, even if we are not ready to diagnose yet.
        for config in self._expected_configs:
            if info.config == config:
                self._seen_config_ids.add(smbo.runhistory.get_config_id(config))
                break

        if self._count_evaluated_configs(smbo) < MIN_CONFIGS_FOR_DIAGNOSIS:
            return False

        return len(self._seen_config_ids) >= len(self._expected_configs)

    @staticmethod
    def _count_evaluated_configs(smbo: SMBO) -> int:
        """Number of distinct configurations that have at least one finished trial."""
        return len({trial_key.config_id for trial_key in smbo.runhistory})

    def _collect_initial_design_costs(self, smbo: SMBO) -> tuple[int, list[list[float]]]:
        """Groups the initial design evaluations by configuration.

        The issue this implements is phrased in terms of configurations, not trials: a single
        configuration may be evaluated several times (on other instances, seeds or budgets). A
        configuration counts as failed only if *none* of its evaluations succeeded, and its cost is
        the mean over its successful evaluations -- the same aggregation SMAC uses elsewhere.

        Returns
        -------
        n_configs : int
            Number of initial design configurations that were evaluated.
        costs : list[list[float]]
            One cost vector per configuration that produced at least one successful evaluation.
        """
        runhistory = smbo.runhistory

        # Restrict the diagnosis to the initial design, as the issue asks for. If there is no
        # initial design of its own -- ``n_configs=0``, or the warmstart mode ``"replace"``, where
        # the already-evaluated configurations *are* the initial design -- we fall back to
        # everything the runhistory holds at this point.
        config_ids = [
            runhistory.get_config_id(config) for config in self._expected_configs if runhistory.has_config(config)
        ]

        # A single configuration is not enough evidence to judge a whole run, so if the initial
        # design is smaller than `MIN_CONFIGS_FOR_DIAGNOSIS` we include the configurations that
        # were evaluated next, in the order in which they were evaluated.
        if len(config_ids) < MIN_CONFIGS_FOR_DIAGNOSIS:
            for trial_key in runhistory:
                if trial_key.config_id not in config_ids:
                    config_ids.append(trial_key.config_id)
                if len(config_ids) >= MIN_CONFIGS_FOR_DIAGNOSIS:
                    break

        successful_costs: dict[int, list] = defaultdict(list)
        for trial_key in runhistory:
            if trial_key.config_id not in config_ids:
                continue

            trial_value = runhistory[trial_key]
            if trial_value.status == StatusType.SUCCESS:
                successful_costs[trial_key.config_id].append(trial_value.cost)

        costs = []
        for config_id in config_ids:
            if config_id not in successful_costs:
                continue

            mean_cost = CostTransformer.mean(successful_costs[config_id])
            costs.append(mean_cost if isinstance(mean_cost, list) else [mean_cost])

        return len(config_ids), costs

    def _diagnose(self, smbo: SMBO) -> bool | None:
        """Performs the actual checks and reports the findings."""
        n_configs, costs = self._collect_initial_design_costs(smbo)

        if n_configs == 0:
            return None

        n_successful = len(costs)

        # Case 1: Not a single configuration produced a result.
        if n_successful == 0:
            logger.warning(
                f"All {n_configs} configurations of the initial design failed. Your target function "
                "or the pipeline around it is probably broken. All failed trials are assigned the "
                "same penalty cost (`Scenario.crash_cost`), so there is no data to build the "
                "surrogate model on."
            )

            if self._mode == "abort":
                logger.warning("Aborting the optimization because the initial design did not produce any result.")
                return False

            return None

        # Case 2: Some -- but not all -- configurations failed. This is not a reason to abort, since
        # partial failures are a normal part of hyperparameter optimization, but the user has to
        # know about them in order to be able to react.
        n_failed = n_configs - n_successful
        if n_failed > 0:
            logger.warning(f"{n_failed} of {n_configs} initial design configurations failed.")

        # Case 3: All successful configurations share the same cost.
        cost_matrix = np.array(costs, dtype=float)

        constant_objectives = [
            objective
            for objective in range(cost_matrix.shape[1])
            if float(np.ptp(cost_matrix[:, objective])) <= self._atol
        ]

        if len(constant_objectives) > 0 and n_successful > 1:
            objectives = smbo._scenario.objectives
            if not isinstance(objectives, list):
                objectives = [objectives]

            names = ", ".join(str(objectives[objective]) for objective in constant_objectives)
            logger.warning(
                f"All {n_successful} successful configurations of the initial design have the same cost "
                f"for the following objective(s): {names}. Your metric or your search space is likely "
                "misconfigured. This may also be a huge plateau -- in either case the surrogate model "
                "has no feedback signal to learn from."
            )

            if self._mode == "abort":
                logger.warning(
                    "Aborting the optimization because the initial design carries no feedback signal "
                    "for the surrogate model."
                )
                return False

        return None
