import typing
import copy

import numpy as np

from smac.configspace import Configuration
from smac.configspace.util import convert_configurations_to_array
from smac.epm.base_epm import AbstractEPM
from smac.epm.rf_with_instances import RandomForestWithInstances
from smac.epm.partial_sparse_gaussian_process import PartialSparseGaussianProcess
from smac.epm.util_funcs import get_types
from smac.optimizer.acquisition import AbstractAcquisitionFunction, TS, EI
from smac.optimizer.ei_optimization import AcquisitionFunctionMaximizer
from smac.optimizer.epm_configuration_chooser import EPMChooser
from smac.optimizer.random_configuration_chooser import RandomConfigurationChooser, ChooserNoCoolDown
from smac.runhistory.runhistory import RunHistory
from smac.scenario.scenario import Scenario
from smac.stats.stats import Stats
from smac.utils.constants import MAXINT
from smac.optimizer.local_bo.turbo_subspace import TurBOSubSpace
from smac.optimizer.local_bo.boing_subspace import BOinGSubspace
from smac.optimizer.local_bo.rh2epm_boing import RunHistory2EPM4CostWithRaw


class EPMChooserBOinG(EPMChooser):
    def __init__(self,
                 scenario: Scenario,
                 stats: Stats,
                 runhistory: RunHistory,
                 runhistory2epm: RunHistory2EPM4CostWithRaw,
                 model: RandomForestWithInstances,
                 acq_optimizer: AcquisitionFunctionMaximizer,
                 acquisition_func: AbstractAcquisitionFunction,
                 rng: np.random.RandomState,
                 restore_incumbent: Configuration = None,
                 random_configuration_chooser: typing.Union[RandomConfigurationChooser] = ChooserNoCoolDown(2.0),
                 predict_x_best: bool = True,
                 min_samples_model: int = 1,
                 model_local: AbstractEPM = PartialSparseGaussianProcess,
                 acquisition_func_local: AbstractAcquisitionFunction = EI,
                 model_local_kwargs: typing.Optional[typing.Dict] = None,
                 acquisition_func_local_kwargs: typing.Optional[typing.Dict] = None,
                 max_configs_local_fracs: float = 0.5,
                 min_configs_local: typing.Optional[int] = None,
                 do_switching: bool = False,
                 turbo_kwargs: typing.Optional[typing.Dict] = None,
                 ):
        """
        Interface to train the EPM and generate next configurations with outer loop level and inner loop level

        Parameters
        ----------
        scenario: smac.scenario.scenario.Scenario
            Scenario object
        stats: smac.stats.stats.Stats
            statistics object with configuration budgets
        runhistory: smac.runhistory.runhistory.RunHistory
            runhistory with all runs so far
        runhistory2epm: RunHistory2EPM4CostWithRaw,
            a transformer to transform rh to vectors, needs also to provide the raw values for optimizer in different
            stages
        model: smac.epm.rf_with_instances.RandomForestWithInstances
            empirical performance model (right now, we support only
            RandomForestWithInstances) in the outer loop
        acq_optimizer: smac.optimizer.ei_optimization.AcquisitionFunctionMaximizer
            Optimizer of acquisition function in the outer loop
        model_local: AbstractEPM,
            local empirical performance model
        model_local_kwargs: typing.Optional[typing.Dict] = None,
            parameters for initializing a local model
        acquisition_func_local: AbstractAcquisitionFunction,
            local acquisition function
        acquisition_func_local_kwargs: typing.Optional[typing.Dict] = None,
            parameters for initializing a acquisition function
        restore_incumbent: Configuration
            incumbent to be used from the start. ONLY used to restore states.
        rng: np.random.RandomState
            Random number generator
        random_configuration_chooser
            Chooser for random configuration -- one of
            * ChooserNoCoolDown(modulus)
            * ChooserLinearCoolDown(start_modulus, modulus_increment, end_modulus)

        predict_x_best: bool
            Choose x_best for computing the acquisition function via the model instead of via the observations.
        max_configs_local_fracs : float
            Maximal number of fractions of samples to be included in the inner loop. If the number of samples in the
            subsapce is beyond this value and n_min_config_inner, the subspace will be cropped to fit the requirement
        min_configs_local: int,
            Minimum number of samples included in the inner loop model
        do_switching: bool
           if we want to switch between turbo and boing
        turbo_kwargs: typing.Optional[typing.Dict] = None
           parameters for building a turbo subspace
        """
        # initialize the original EPM_Chooser
        super(EPMChooserBOinG, self).__init__(scenario=scenario,
                                              stats=stats,
                                              runhistory=runhistory,
                                              runhistory2epm=runhistory2epm,
                                              model=model,
                                              acq_optimizer=acq_optimizer,
                                              acquisition_func=acquisition_func,
                                              rng=rng,
                                              restore_incumbent=restore_incumbent,
                                              random_configuration_chooser=random_configuration_chooser,
                                              predict_x_best=predict_x_best,
                                              min_samples_model=min_samples_model)
        self.model_local = model_local
        self.model_local_kwargs = model_local_kwargs
        self.acquisition_func_local = acquisition_func_local
        self.acquisition_func_local_kwargs = acquisition_func_local_kwargs

        self.max_configs_local_fracs = max_configs_local_fracs
        self.min_configs_local = min_configs_local if min_configs_local is not None \
            else 5 * len(scenario.cs.get_hyperparameters())

        types, bounds = get_types(self.scenario.cs, instance_features=None)

        self.types = types
        self.bounds = bounds
        self.cat_dims = np.where(np.array(types) != 0)[0]
        self.cont_dims = np.where(np.array(types) == 0)[0]
        self.config_space = scenario.cs

        self.frac_to_start_bi = 0.8
        self.split_count = np.zeros(len(types))
        self.do_switching = do_switching
        self.random_search_upper_log = 1

        self.optimal_value = np.inf
        self.optimal_config = None

        self.ss_threshold = 0.1 ** len(self.scenario.cs.get_hyperparameters())
        if self.do_switching:
            self.run_turBO = False
            self.failcount_BOinG = 0
            self.failcount_TurBO = 0

            turbo_model = copy.deepcopy(model_local)
            turbo_acq = TS(turbo_model)
            turbo_opt_kwargs = dict(config_space=scenario.cs,
                                    bounds=bounds,
                                    hps_types=types,
                                    model_local=turbo_model,
                                    model_local_kwargs=copy.deepcopy(model_local_kwargs),
                                    acq_func_local=turbo_acq,
                                    rng=rng,
                                    )
            self.turbo_kwargs = turbo_opt_kwargs
            if turbo_kwargs is not None:
                turbo_opt_kwargs.update(turbo_kwargs)
            self.turbo_optimizer = TurBOSubSpace(**turbo_opt_kwargs)

    def restart_TurBOinG(self, X, Y, Y_raw, train_model=False):
        """
        Restart a new TurBO Optimizer, the bounds of the TurBO Optimizer is selected by a RF 
        """
        if train_model:
            self.model.train(X, Y)
        num_samples = 20
        union_ss = []
        union_indices = []
        rand_samples = self.config_space.sample_configuration(num_samples)
        for sample in rand_samples:
            sample_array = sample.get_array()
            union_bounds_cont, _, ss_data_indices = subspace_extraction(X=X,
                                                                        challenger=sample_array,
                                                                        model=self.model,
                                                                        num_min=self.min_configs_local,
                                                                        num_max=MAXINT,
                                                                        bounds=self.bounds,
                                                                        cont_dims=self.cont_dims,
                                                                        cat_dims=self.cat_dims)
            union_ss.append(union_bounds_cont)
            union_indices.append(ss_data_indices)
        union_ss = np.asarray(union_ss)
        volume_ss = np.product(union_ss[:, :, 1] - union_ss[:, :, 0], axis=1)
        ss_idx = np.argmax(volume_ss)
        ss_turbo = union_ss[ss_idx]
        ss_data_indices = union_indices[ss_idx]

        # we only consder numerical(continuous) hyperparameters here
        self.turbo_optimizer = TurBOSubSpace(**self.turbo_kwargs,
                                             bounds_ss_cont=ss_turbo,
                                             initial_data=(X[ss_data_indices], Y_raw[ss_data_indices]))

    def choose_next(self, incumbent_value: float = None) -> typing.Iterator[Configuration]:
        """Choose next candidate solution with Bayesian optimization. The
        suggested configurations depend on the argument ``acq_optimizer_outer`` and ''acq_optimizer_inner'' to
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
        X, Y, Y_raw, X_configurations = self._collect_data_to_train_model()
        # X, Y, X_configurations = self._collect_data_to_train_model()
        # Y_raw = Y
        if self.do_switching:
            if self.run_turBO:
                X, Y, Y_raw, X_configurations = self._collect_data_to_train_model()

                num_new_bservations = 1 # here we only consider batch_size ==1

                new_observations = Y_raw[-num_new_bservations:]

                if len(self.turbo_optimizer.init_configs) > 0:
                    self.turbo_optimizer.add_new_observations(X[-num_new_bservations:],
                                                              Y_raw[-num_new_bservations:])
                    return self.turbo_optimizer.generate_challengers()

                self.turbo_optimizer.adjust_length(new_observations)

                if self.turbo_optimizer.length < self.turbo_optimizer.length_min:
                    optimal_turbo = np.min(self.turbo_optimizer.ss_y)
                    # self.optimal_value = Y_raw[-1].item()
                    # self.optimal_config = X[-1]

                    self.logger.debug(f'Best Found value by TurBO: {optimal_turbo}')

                    increment = optimal_turbo - self.optimal_value
                    if increment < 0:
                        min_idx = np.argmin(Y_raw)
                        self.optimal_value = Y_raw[min_idx].item()
                        # compute the distance between the previous incumbent and new incumbent
                        cfg_diff = X[min_idx] - self.optimal_config
                        self.optimal_config = X[min_idx]
                        # we avoid sticking to a local minimum too often
                        if increment < -1e-3 * np.abs(self.optimal_value) or np.abs(
                                np.product(cfg_diff)) >= self.ss_threshold:
                            self.failcount_TurBO -= 1
                            # switch to BOinG as TurBO found a better model and we could do a exploration
                            self.failcount_BOinG = self.failcount_BOinG // 2
                            self.run_turBO = False
                            self.logger.debug('Optimizer switches to BOinG!')

                    else:
                        self.failcount_TurBO += 1

                    if self.failcount_TurBO < 4:
                        prob_to_BOinG = 0.5 ** (4 - self.failcount_TurBO)
                    else:
                        prob_to_BOinG = 1 - 0.5 ** (self.failcount_TurBO - 4)

                    self.logger.debug(f'failure count TurBO :{self.failcount_TurBO}')
                    rand_value = self.rng.random()
                    if rand_value < prob_to_BOinG:
                        self.failcount_BOinG = self.failcount_BOinG // 2
                        self.run_turBO = False
                        self.logger.debug('Swich to BOinG!')
                    else:
                        self.restart_TurBOinG(X=X, Y=Y, Y_raw=Y_raw, train_model=True)
                        return self.turbo_optimizer.generate_challengers()

                self.turbo_optimizer.add_new_observations(X[-num_new_bservations:],
                                                          Y_raw[-num_new_bservations:])

                return self.turbo_optimizer.generate_challengers()

        if X.shape[0] == 0:
            # Only return a single point to avoid an overly high number of
            # random search iterations
            return self._random_search.maximize(
                runhistory=self.runhistory, stats=self.stats, num_points=1
            )
        # if the number of points is not big enough, we simply build one subspace and
        # fits the model with that single model
        if X.shape[0] < (self.min_configs_local / self.frac_to_start_bi):
            ss = BOinGSubspace(config_space=self.scenario.cs,
                               bounds=self.bounds,
                               hps_types=self.types,
                               model_local=self.model_local,
                               model_local_kwargs=self.model_local_kwargs,
                               acq_func_local=self.acquisition_func_local,
                               acq_func_local_kwargs=self.acquisition_func_local_kwargs,
                               rng=self.rng,
                               initial_data=(X, Y_raw),
                               incumbent_array=None,
                               )
            return ss.generate_challengers()

        # train the outer model
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
        list_sub_space = []

        if self.do_switching:
            # check if we need to switch to turbo
            self.failcount_BOinG += 1
            increment = Y_raw[-1].item() - self.optimal_value
            if increment < 0:
                if self.optimal_config is not None:
                    cfg_diff = X[-1] - self.optimal_config
                    if increment < -1e-2 * np.abs(self.optimal_value) or np.abs(
                            np.product(cfg_diff)) >= self.ss_threshold:
                        self.failcount_BOinG -= X.shape[-1]
                    self.optimal_value = Y_raw[-1].item()
                    self.optimal_config = X[-1]
                else:
                    # restart
                    idx_min = np.argmin(Y_raw)
                    self.logger.debug(f"Better value found by BOiNG, continue BOinG")
                    self.optimal_value = Y_raw[idx_min].item()
                    self.optimal_config = X[idx_min]
                    self.failcount_BOinG = 0

            amplify_param = self.failcount_BOinG // (X.shape[-1] * 1)

            if self.failcount_BOinG % (X.shape[-1] * 1) == 0:
                if amplify_param > 4:
                    prob_to_TurBO = 1 - 0.5 ** min(amplify_param - 4, 3)
                else:
                    if amplify_param < 1:
                        prob_to_TurBO = 0
                    else:
                        prob_to_TurBO = 0.5 ** max(4 - amplify_param, self.random_search_upper_log)

                rand_value = self.rng.random()
                if rand_value < prob_to_TurBO:
                    # self.failcount_BOinG = 0
                    self.run_turBO = True
                    self.logger.debug('Switch To TurBO')
                    self.failcount_TurBO = self.failcount_TurBO // 2
                    self.restart_TurBOinG(X=X, Y=Y, Y_raw=Y_raw, train_model=False)

        challengers = self.acq_optimizer.maximize(
            runhistory=self.runhistory,
            stats=self.stats,
            num_points=self.scenario.acq_opt_challengers,  # type: ignore[attr-defined] # noqa F821
            random_configuration_chooser=self.random_configuration_chooser
        )

        cfg_challenger = next(challengers)
        challenger_global = cfg_challenger.get_array()

        num_max_configs = int(X.shape[0] * self.max_configs_local_fracs)

        num_max = MAXINT if num_max_configs <= 2 * self.min_configs_local else num_max_configs,

        bounds_ss_cont, bounds_ss_cat, ss_data_indices = subspace_extraction(X=X,
                                                                             challenger=challenger_global,
                                                                             model=self.model,
                                                                             num_min=self.min_configs_local,
                                                                             num_max=num_max,
                                                                             bounds=self.bounds,
                                                                             cont_dims=self.cont_dims,
                                                                             cat_dims=self.cat_dims)

        self.logger.debug('contained {0} data of {1}'.format(sum(ss_data_indices), Y_raw.size))

        ss = BOinGSubspace(config_space=self.scenario.cs,
                           bounds=self.bounds,
                           hps_types=self.types,
                           bounds_ss_cont=bounds_ss_cont,
                           bounds_ss_cat=bounds_ss_cat,
                           model_local=self.model_local,
                           model_local_kwargs=self.model_local_kwargs,
                           acq_func_local=self.acquisition_func_local,
                           acq_func_local_kwargs=self.acquisition_func_local_kwargs,
                           rng=self.rng,
                           initial_data=(X, Y_raw),
                           incumbent_array=challenger_global,
                           )
        return ss.generate_challengers()

    def _get_x_best(self, predict: bool, X: np.ndarray, use_inner_model: bool = False) \
            -> typing.Tuple[float, np.ndarray]:
        """Get value, configuration, and array representation of the "best" configuration.

        The definition of best varies depending on the argument ``predict``. If set to ``True``,
        this function will return the stats of the best configuration as predicted by the model,
        otherwise it will return the stats for the best observed configuration.

        Parameters
        ----------
        predict : bool
            Whether to use the predicted or observed best.

        Return
        ------
        float
        np.ndarry
        Configuration
        """
        if predict:
            if use_inner_model:
                costs = list(map(
                    lambda x: (
                        self.model_inner.predict_marginalized_over_instances(x.reshape((1, -1)))[0][0][0],
                        x,
                    ),
                    X,
                ))
            else:
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

    def _collect_data_to_train_model(self) -> typing.Tuple[np.ndarray, np.ndarray, np.ndarray]:
        # if we use a float value as a budget, we want to train the model only on the highest budget
        available_budgets = []
        for run_key in self.runhistory.data.keys():
            available_budgets.append(run_key.budget)

        # Sort available budgets from highest to lowest budget
        available_budgets = sorted(list(set(available_budgets)), reverse=True)

        # Get #points per budget and if there are enough samples, then build a model
        for b in available_budgets:
            X, Y, Y_raw = self.rh2EPM.transform(self.runhistory, budget_subset=[b, ])
            if X.shape[0] >= self.min_samples_model:
                self.currently_considered_budgets = [b, ]
                configs_array = self.rh2EPM.get_configurations(
                    self.runhistory, budget_subset=self.currently_considered_budgets)
                return X, Y, Y_raw, configs_array

        return np.empty(shape=[0, 0]), np.empty(shape=[0, ]), np.empty(shape=[0, ]), np.empty(shape=[0, 0])


def subspace_extraction(X: np.ndarray,
                        challenger: np.ndarray,
                        model: RandomForestWithInstances,
                        num_min: int,
                        num_max: int,
                        bounds: np.ndarray,
                        cat_dims: np.ndarray,
                        cont_dims: np.ndarray):
    """
    extract a subspace that contains at least num_min but no more than num_max
    Parameters
    ----------
    X: points used to train the model
    challenger: the challenger where the subspace would grow
    model: a rf model
    num_min: minimal number of points to be included in the subspace
    num_max: maximal number of points to be included in the subspace
    bounds: bounds of the entire space
    cat_dims: categorical dimensions
    cont_dims: continuous dimensions

    Returns
    -------
    union_bounds_cont: np.ndarray, the continuous bounds of the subregion
    union_bounds_cat, List[Tuple], the categorical bounds of the subregion
    in_ss_dims: indices of the points that lie inside the subregion
    """
    trees = model.rf.get_all_trees()
    num_trees = len(trees)
    node_indices = [0] * num_trees

    indices_trees = np.arange(num_trees)
    np.random.shuffle(indices_trees)
    ss_indices = np.full(X.shape[0], True)

    stop_update = [False] * num_trees

    ss_bounds = np.array(bounds)

    if cat_dims.size == 0:
        ss_bounds_cat = [()]
    else:
        ss_bounds_cat = [() for _ in range(len(cat_dims))]
        for i, cat_dim in enumerate(cat_dims):
            ss_bounds_cat[i] = np.arange(ss_bounds[cat_dim][0])

    if cont_dims.size == 0:
        ss_bounds_cont = np.array([])
    else:
        ss_bounds_cont = ss_bounds[cont_dims]

    def traverse_forest(check_num_min=True):
        nonlocal ss_indices
        np.random.shuffle(indices_trees)
        for i in indices_trees:
            if stop_update[i]:
                continue
            tree = trees[int(i)]
            node_idx = node_indices[i]
            node = tree.get_node(node_idx)

            if node.is_a_leaf():
                stop_update[i] = True
                continue

            feature_idx = node.get_feature_index()
            cont_feature_idx = np.where(feature_idx == cont_dims)[0]
            if cont_feature_idx.size == 0:
                cat_feature_idx = np.where(feature_idx == cat_dims)[0][0]
                split_value = node.get_cat_split()
                intersect = np.intersect1d(ss_bounds_cat[cat_feature_idx], split_value, assume_unique=True)

                if len(intersect) == len(ss_bounds_cat[cat_feature_idx]):
                    temp_child_idx = 0
                    node_indices[i] = node.get_child_index(temp_child_idx)
                elif len(intersect) == 0:
                    temp_child_idx = 1
                    node_indices[i] = node.get_child_index(temp_child_idx)
                else:
                    if challenger[feature_idx] in intersect:
                        temp_child_idx = 0
                        temp_node_indices = ss_indices & np.in1d(X[:, feature_idx], split_value)
                        temp_bound_ss = intersect
                    else:
                        temp_child_idx = 1
                        temp_node_indices = ss_indices & np.in1d(X[:, feature_idx], split_value, invert=True)
                        temp_bound_ss = np.setdiff1d(ss_bounds_cat[cat_feature_idx], split_value)
                    if sum(temp_node_indices) > num_min:
                        # number of points inside subspace is still greater than num_min
                        ss_bounds_cat[cat_feature_idx] = temp_bound_ss
                        ss_indices = temp_node_indices
                        node_indices[i] = node.get_child_index(temp_child_idx)
                    else:
                        if check_num_min:
                            stop_update[i] = True
                        else:
                            node_indices[i] = node.get_child_index(temp_child_idx)
            else:
                split_value = node.get_num_split_value()
                cont_feature_idx = cont_feature_idx.item()
                if ss_bounds_cont[cont_feature_idx][0] <= split_value <= ss_bounds_cont[cont_feature_idx][1]:
                    # the subspace can be further split
                    if challenger[feature_idx] >= split_value:
                        temp_bound_ss = np.array([split_value, ss_bounds_cont[cont_feature_idx][1]])
                        temp_node_indices = ss_indices & (X[:, feature_idx] >= split_value)
                        temp_child_idx = 1
                    else:
                        temp_bound_ss = np.array([ss_bounds_cont[cont_feature_idx][0], split_value])
                        temp_node_indices = ss_indices & (X[:, feature_idx] <= split_value)
                        temp_child_idx = 0
                    if sum(temp_node_indices) > num_min:
                        # number of points inside subspace is still greater than num_min
                        ss_bounds_cont[cont_feature_idx] = temp_bound_ss
                        ss_indices = temp_node_indices
                        node_indices[i] = node.get_child_index(temp_child_idx)
                    else:
                        if check_num_min:
                            stop_update[i] = True
                        else:
                            node_indices[i] = node.get_child_index(temp_child_idx)
                else:
                    temp_child_idx = 1 if challenger[feature_idx] >= split_value else 0
                    node_indices[i] = node.get_child_index(temp_child_idx)

    while sum(stop_update) < num_trees:
        traverse_forest()

    if sum(ss_indices) > num_max:
        stop_update = [False] * num_trees
        while sum(stop_update) < num_trees:
            traverse_forest(False)

    return ss_bounds_cont, ss_bounds_cat, ss_indices
