import typing

import numpy as np

from smac.configspace import Configuration
from smac.epm.random_forest.rf_with_instances import RandomForestWithInstances
from smac.epm.utils import get_types
from smac.optimizer.acquisition import TS, AbstractAcquisitionFunction
from smac.optimizer.acquisition.maximizer import AcquisitionFunctionMaximizer
from smac.optimizer.configuration_chooser.epm_chooser import EPMChooser
from smac.optimizer.configuration_chooser.random_chooser import (
    ChooserNoCoolDown,
    RandomChooser,
)
from smac.optimizer.subspaces.turbo_subspace import TuRBOSubSpace
from smac.runhistory.runhistory import RunHistory
from smac.runhistory.runhistory2epm import AbstractRunHistory2EPM
from smac.scenario.scenario import Scenario
from smac.stats.stats import Stats


class TurBOChooser(EPMChooser):
    """
    Interface to train the EPM and generate next configurations with TurBO:
        D. Eriksson et al. Scalable Global Optimization via Local Bayesian Optimization
        https://papers.nips.cc/paper/2019/file/6c990b7aca7bc7058f5e98ea909e924b-Paper.pdf

    Parameters
    ----------
    length_init: float
        Initialized length after restarting
    length_min: float
        If the subspace length is smaller than length_min, TurBO will restart
    length_max: float
        Maximum length of subspace
    success_tol: int
        Number of successful suggestions (suggested points become incumbent) required for expanding subspace
    failure_tol_min: int
        Minimum number of failure suggestions (suggested points fails to become incumbent) required for shrinking
        subspace
    n_init_x_params: int
        how many configurations will be used at most in the initial design (X*D). Used for restarting the subspace
    n_candidate_max: int
        Maximal Number of points used as candidates
    """

    def __init__(
        self,
        scenario: Scenario,
        stats: Stats,
        runhistory: RunHistory,
        runhistory2epm: AbstractRunHistory2EPM,
        model: RandomForestWithInstances,
        acq_optimizer: AcquisitionFunctionMaximizer,
        acquisition_func: AbstractAcquisitionFunction,
        rng: np.random.RandomState,
        restore_incumbent: Configuration = None,
        random_configuration_chooser: RandomChooser = ChooserNoCoolDown(2.0),
        predict_x_best: bool = False,
        min_samples_model: int = 1,
        length_init: float = 0.8,
        length_min: float = 0.5**8,
        length_max: float = 1.6,
        success_tol: int = 3,
        failure_tol_min: int = 4,
        n_init_x_params: int = 2,
        n_candidate_max: int = 5000,
    ):
        super(TurBOChooser, self).__init__(
            scenario=scenario,
            stats=stats,
            runhistory=runhistory,
            runhistory2epm=runhistory2epm,
            model=model,
            acquisition_func=acquisition_func,
            acq_optimizer=acq_optimizer,
            restore_incumbent=restore_incumbent,
            rng=rng,
            random_configuration_chooser=random_configuration_chooser,
            predict_x_best=predict_x_best,
            min_samples_model=min_samples_model,
        )
        cs = self.scenario.cs  # type: ignore
        types, bounds = get_types(cs, instance_features=None)

        self.turbo = TuRBOSubSpace(
            config_space=cs,
            bounds=bounds,
            hps_types=types,
            model_local=model,
            acq_func_local=TS,
            length_init=length_init,
            length_min=length_min,
            length_max=length_max,
            success_tol=success_tol,
            failure_tol_min=failure_tol_min,
            n_init_x_params=n_init_x_params,
            n_candidate_max=n_candidate_max,
        )

    def choose_next(self, incumbent_value: float = None) -> typing.Iterator[Configuration]:
        """
        Choose next candidate solution with TuRBO

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

        num_new_bservations = 1  # here we only consider batch size = 1

        new_observations = Y[-num_new_bservations:]
        if len(self.turbo.init_configs) > 0:
            self.turbo.add_new_observations(X[-num_new_bservations:], Y[-num_new_bservations:])
            return self.turbo.generate_challengers()

        self.turbo.adjust_length(new_observations)

        self.turbo.add_new_observations(X[-num_new_bservations:], Y[-num_new_bservations:])

        challengers = self.turbo.generate_challengers()
        return challengers
