import logging
import typing

import numpy as np

from smac.configspace import Configuration
from smac.configspace.util import convert_configurations_to_array
from smac.epm.rf_with_instances import RandomForestWithInstances
from smac.optimizer.acquisition import AbstractAcquisitionFunction
from smac.optimizer.ei_optimization import AcquisitionFunctionMaximizer, \
    RandomSearch
from smac.optimizer.random_configuration_chooser import RandomConfigurationChooser, ChooserNoCoolDown
from smac.runhistory.runhistory import RunHistory
from smac.runhistory.runhistory2epm import AbstractRunHistory2EPM
from smac.scenario.scenario import Scenario
from smac.stats.stats import Stats

__copyright__ = "Copyright 2021, AutoML.org Freiburg-Hannover"
__license__ = "3-clause BSD"


class EPMChooser(object):
    """
    Interface to train the EPM and generate next configurations

    Parameters
    ----------

    scenario: smac.scenario.scenario.Scenario
        Scenario object
    stats: smac.stats.stats.Stats
        statistics object with configuration budgets
    runhistory: smac.runhistory.runhistory.RunHistory
        runhistory with all runs so far
    model: smac.epm.rf_with_instances.RandomForestWithInstances
        empirical performance model (right now, we support only
        RandomForestWithInstances)
    acq_optimizer: smac.optimizer.ei_optimization.AcquisitionFunctionMaximizer
        Optimizer of acquisition function.
    restore_incumbent: Configuration
        incumbent to be used from the start. ONLY used to restore states.
    rng: np.random.RandomState
        Random number generator
    random_configuration_chooser:
        Chooser for random configuration -- one of

        * ChooserNoCoolDown(modulus)
        * ChooserLinearCoolDown(start_modulus, modulus_increment, end_modulus)
    predict_x_best: bool
        Choose x_best for computing the acquisition function via the model instead of via the observations.
    min_samples_model: int
        Minimum number of samples to build a model
    """
    def __init__(self,
                 scenario: Scenario,
                 stats: Stats,
                 runhistory: RunHistory,
                 runhistory2epm: AbstractRunHistory2EPM,
                 model: RandomForestWithInstances,
                 acq_optimizer: AcquisitionFunctionMaximizer,
                 acquisition_func: AbstractAcquisitionFunction,
                 rng: np.random.RandomState,
                 restore_incumbent: Configuration = None,
                 random_configuration_chooser: typing.Union[RandomConfigurationChooser] = ChooserNoCoolDown(2.0),
                 predict_x_best: bool = True,
                 min_samples_model: int = 1
                 ):
        self.logger = logging.getLogger(
            self.__module__ + "." + self.__class__.__name__)
        self.incumbent = restore_incumbent

        self.scenario = scenario
        self.stats = stats
        self.runhistory = runhistory
        self.rh2EPM = runhistory2epm
        self.model = model
        self.acq_optimizer = acq_optimizer
        self.acquisition_func = acquisition_func
        self.rng = rng
        self.random_configuration_chooser = random_configuration_chooser

        self._random_search = RandomSearch(
            acquisition_func,
            self.scenario.cs,  # type: ignore[attr-defined] # noqa F821
            rng,
        )

        self.initial_design_configs = []  # type: typing.List[Configuration]

        self.predict_x_best = predict_x_best

        self.min_samples_model = min_samples_model
        self.currently_considered_budgets = [0.0, ]

    def _collect_data_to_train_model(self) -> typing.Tuple[np.ndarray, np.ndarray, np.ndarray]:
        # if we use a float value as a budget, we want to train the model only on the highest budget
        available_budgets = []
        for run_key in self.runhistory.data.keys():
            available_budgets.append(run_key.budget)

        # Sort available budgets from highest to lowest budget
        available_budgets = sorted(list(set(available_budgets)), reverse=True)

        # Get #points per budget and if there are enough samples, then build a model
        for b in available_budgets:
            X, Y = self.rh2EPM.transform(self.runhistory, budget_subset=[b, ])
            if X.shape[0] >= self.min_samples_model:
                self.currently_considered_budgets = [b, ]
                configs_array = self.rh2EPM.get_configurations(
                    self.runhistory, budget_subset=self.currently_considered_budgets)
                return X, Y, configs_array

        return np.empty(shape=[0, 0]), np.empty(shape=[0, ]), np.empty(shape=[0, 0])

    def _get_evaluated_configs(self) -> typing.List[Configuration]:
        return self.runhistory.get_all_configs_per_budget(budget_subset=self.currently_considered_budgets)

    def choose_next(self, incumbent_value: float = None) -> typing.Iterator[Configuration]:
        """Choose next candidate solution with Bayesian optimization. The
        suggested configurations depend on the argument ``acq_optimizer`` to
        the ``SMBO`` class.

        Parameters
        ----------
        incumbent_value: float
            Cost value of incumbent configuration (required for acquisition function);
            If not given, it will be inferred from runhistory or predicted;
            if not given and runhistory is empty, it will raise a ValueError.

        Returns
        -------
        Iterator
        """

        self.logger.debug("Search for next configuration")
        X, Y, X_configurations = self._collect_data_to_train_model()

        if X.shape[0] == 0:
            # Only return a single point to avoid an overly high number of
            # random search iterations
            return self._random_search.maximize(
                runhistory=self.runhistory, stats=self.stats, num_points=1
            )
        self.model.train(X, Y)

        if incumbent_value is not None:
            best_observation = incumbent_value
            x_best_array = None  # type: typing.Optional[np.ndarray]
        else:
            if self.runhistory.empty():
                raise ValueError("Runhistory is empty and the cost value of "
                                 "the incumbent is unknown.")
            x_best_array, best_observation = self._get_x_best(self.predict_x_best, X_configurations)

        self.acquisition_func.update(
            model=self.model,
            eta=best_observation,
            incumbent_array=x_best_array,
            num_data=len(self._get_evaluated_configs()),
            X=X_configurations,
        )

        challengers = self.acq_optimizer.maximize(
            runhistory=self.runhistory,
            stats=self.stats,
            num_points=self.scenario.acq_opt_challengers,  # type: ignore[attr-defined] # noqa F821
            random_configuration_chooser=self.random_configuration_chooser
        )
        return challengers

    def _get_x_best(self, predict: bool, X: np.ndarray) -> typing.Tuple[float, np.ndarray]:
        """Get value, configuration, and array representation of the "best" configuration.

        The definition of best varies depending on the argument ``predict``. If set to ``True``,
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
        if predict:
            costs = list(map(
                lambda x: (
                    self.model.predict_marginalized_over_instances(x.reshape((1, -1)))[0][0][0],
                    x,
                ),
                X,
            ))
            costs = sorted(costs, key=lambda t: t[0])
            x_best_array = costs[0][1]
            best_observation = costs[0][0]
            # won't need log(y) if EPM was already trained on log(y)
        else:
            all_configs = self.runhistory.get_all_configs_per_budget(budget_subset=self.currently_considered_budgets)
            x_best = self.incumbent
            x_best_array = convert_configurations_to_array(all_configs)
            best_observation = self.runhistory.get_cost(x_best)
            best_observation_as_array = np.array(best_observation).reshape((1, 1))
            # It's unclear how to do this for inv scaling and potential future scaling.
            # This line should be changed if necessary
            best_observation = self.rh2EPM.transform_response_values(best_observation_as_array)
            best_observation = best_observation[0][0]

        return x_best_array, best_observation
