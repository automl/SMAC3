import logging
import typing

import numpy as np

import smac
from smac.configspace import Configuration
from smac.configspace.util import convert_configurations_to_array
from smac.epm.rf_with_instances import RandomForestWithInstances
from smac.optimizer.acquisition import AbstractAcquisitionFunction
from smac.optimizer.ei_optimization import AcquisitionFunctionMaximizer, \
    RandomSearch
from smac.optimizer.random_configuration_chooser import ChooserNoCoolDown, \
    ChooserLinearCoolDown
from smac.runhistory.runhistory import RunHistory
from smac.runhistory.runhistory2epm import AbstractRunHistory2EPM
from smac.scenario.scenario import Scenario
from smac.stats.stats import Stats


class EPMChooser(object):
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
                 random_configuration_chooser: typing.Union[
                     ChooserNoCoolDown, ChooserLinearCoolDown]=ChooserNoCoolDown(2.0),
                 predict_incumbent: bool = True):
        """
        Interface to train the EPM and generate next configurations

        Parameters
        ----------
        scenario: smac.scenario.scenario.Scenario
            Scenario object
        stats: Stats
            statistics object with configuration budgets
        runhistory: RunHistory
            runhistory with all runs so far
        model: RandomForestWithInstances
            empirical performance model (right now, we support only
            RandomForestWithInstances)
        acq_optimizer: AcquisitionFunctionMaximizer
            Optimizer of acquisition function.
        restore_incumbent: Configuration
            incumbent to be used from the start. ONLY used to restore states.
        rng: np.random.RandomState
            Random number generator
        random_configuration_chooser
            Chooser for random configuration -- one of
            * ChooserNoCoolDown(modulus)
            * ChooserLinearCoolDown(start_modulus, modulus_increment, end_modulus)
        predict_incumbent: bool
            Use predicted performance of incumbent instead of observed performance
        """

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
            acquisition_func, self.scenario.cs, rng
        )

        self.initial_design_configs = []

        self.predict_incumbent = predict_incumbent

    def choose_next(self, incumbent_value: float = None):
        """Choose next candidate solution with Bayesian optimization. The
        suggested configurations depend on the argument ``acq_optimizer`` to
        the ``SMBO`` class.

        Parameters
        ----------
        incumbent_value: float
            Cost value of incumbent configuration
            (required for acquisition function);
            if not given, it will be inferred from runhistory;
            if not given and runhistory is empty,
            it will raise a ValueError

        Returns
        -------
        Iterable
        """

        self.logger.debug("Search for next configuration")

        X, Y = self.rh2EPM.transform(self.runhistory)

        if X.shape[0] == 0:
            # Only return a single point to avoid an overly high number of
            # random search iterations
            return self._random_search.maximize(
                runhistory=self.runhistory, stats=self.stats, num_points=1
            )

        self.model.train(X, Y)

        if incumbent_value is None:
            if self.runhistory.empty():
                raise ValueError("Runhistory is empty and the cost value of "
                                 "the incumbent is unknown.")
            incumbent, incumbent_array, incumbent_value = self._get_incumbent()
        else:
            incumbent = None
            incumbent_array = None

        self.acquisition_func.update(
            model=self.model,
            eta=incumbent_value,
            incumbent=incumbent,
            incumbent_array=incumbent_array,
            num_data=len(self.runhistory.data),
            X=X,
        )

        challengers = self.acq_optimizer.maximize(
            runhistory=self.runhistory,
            stats=self.stats,
            num_points=self.scenario.acq_opt_challengers,
            random_configuration_chooser=self.random_configuration_chooser
        )
        return challengers

    def _get_incumbent(self) -> typing.Tuple[float, np.ndarray, Configuration]:
        """Get incumbent value, configuration, and array representation.

        This is retrieved either from the runhistory or from best predicted
        performance on configs in runhistory (depends on self.predict_incumbent)

        Return
        ------
        float
        np.ndarry
        Configuration
        """
        all_configs = self.runhistory.get_all_configs()
        if self.predict_incumbent:
            configs_array = convert_configurations_to_array(all_configs)
            costs = list(map(
                lambda input_: (
                    self.model.predict_marginalized_over_instances(input_[0].reshape((1, -1)))[0][0][0],
                    input_[0], input_[1],
                ),
                zip(configs_array, all_configs),
            ))
            costs = sorted(costs, key=lambda t: t[0])
            incumbent = costs[0][2]
            incumbent_array = costs[0][1]
            incumbent_value = costs[0][0]
            # won't need log(y) if EPM was already trained on log(y)
        else:
            if self.runhistory.empty():
                raise ValueError("Runhistory is empty and the cost value of "
                                 "the incumbent is unknown.")
            incumbent = self.incumbent
            incumbent_array = convert_configurations_to_array([all_configs])
            incumbent_value = self.runhistory.get_cost(incumbent)
            incumbent_value_as_array = np.array(incumbent_value).reshape((1, 1))
            # It's unclear how to do this for inv scaling and potential future scaling.
            # This line should be changed if necessary
            incumbent_value = self.rh2EPM.transform_response_values(incumbent_value_as_array)
            incumbent_value = incumbent_value[0][0]

        return incumbent, incumbent_array, incumbent_value
