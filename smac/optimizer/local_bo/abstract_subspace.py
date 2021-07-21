import typing
import copy
import inspect
import math
import logging
from abc import ABC, abstractmethod

import numpy as np

from ConfigSpace.hyperparameters import CategoricalHyperparameter, OrdinalHyperparameter, \
    UniformIntegerHyperparameter, Constant, UniformFloatHyperparameter

from smac.configspace import Configuration, ConfigurationSpace
from smac.epm.base_epm import AbstractEPM
from smac.optimizer.acquisition import AbstractAcquisitionFunction
from smac.epm.partial_sparse_gaussian_process import PartialSparseGaussianProcess
from smac.optimizer.acquisition import TS


class AbstractSubspace(ABC):
    def __init__(self,
                 config_space: ConfigurationSpace,
                 bounds: typing.List[typing.Tuple[float, float]],
                 hps_types: typing.List[int],
                 bounds_ss_cont: typing.Optional[np.ndarray] = None,
                 bounds_ss_cat: typing.Optional[typing.List[typing.Tuple]] = None,
                 model_local: AbstractEPM = PartialSparseGaussianProcess,
                 model_local_kwargs: typing.Optional[typing.Dict] = None,
                 acq_func_local: AbstractAcquisitionFunction = TS,
                 acq_func_local_kwargs: typing.Optional[typing.Dict] = None,
                 rng: typing.Optional[np.random.RandomState] = None,
                 initial_data: typing.Optional[typing.Tuple[np.ndarray, np.ndarray]] = None,
                 activate_dims: typing.Optional[np.ndarray] = None,
                 incumbent_array: typing.Optional[np.ndarray] = None,
                 ):
        """
        A subspace that is designed for local Bayesian Optimization, if bounds_ss_cont and bounds_ss_cat are not given,
        this subspace is equivalent to the original configuration space which manage its own runhistory that allows to
        implement its own optimizer in self._generate_challengers(), for example, turbo. Alternatively, this subspace
        support local BO that only focus on a subset of the dimensions, where the missing values are filled by the
        corresponding values in incumbent_array
        Parameters
        ----------
        config_space: ConfigurationSpace
            raw Configuration space
        bounds: typing.List[typing.Tuple[float, float]]
            raw bounds of the Configuration space, notice that here bounds denotes the bounds of the entire space
        hps_types: typing.List[int],
            types of the hyperparameters
        bounds_ss_cont: np.ndarray(D_cont, 2)
            subspaces bounds of continuous hyperparameters, its length is the number of continuous hyperparameters
        bounds_ss_cat: typing.List[typing.Tuple]
            subspaces bounds of categorical hyperparameters, its length is the number of categorical hyperparameters
        rng: np.random.RandomState
            random state
        model_local: ~smac.epm.base_epm.AbstractEPM
            model in subspace
        model_local_kwargs: typing.Optional[typing.Dict]
            argument for subspace model
        acq_func_local: ~smac.optimizer.ei_optimization.AbstractAcquisitionFunction
            local acquisition function
        acq_func_local_kwargs: typing.Optional[typing.Dict]
            argument for acquisition function
        activate_dims: typing.Optional[np.ndarray]
            activate dimensions in the subspace, if it is None, we preserve all the dimensions
        incumbent_array: typing.Optional[np.ndarray]
            incumbent array, used when activate_dims has less dimension and this value is used to complementary the
            resulted configurations
        """
        self.logger = logging.getLogger(
            self.__module__ + "." + self.__class__.__name__)
        self.cs_global = config_space
        if rng is None:
            self.rng = np.random.RandomState(1)
        else:
            self.rng = np.random.RandomState(rng.randint(0, 2 ** 20))

        n_hypers = len(config_space.get_hyperparameters())
        model_types = copy.deepcopy(hps_types)
        model_bounds = copy.deepcopy(bounds)

        cat_dims = np.where(np.array(hps_types) != 0)[0]
        cont_dims = np.where(np.array(hps_types) == 0)[0]

        if activate_dims is None:
            activate_dims = np.arange(n_hypers)
            activate_dims_cont = cont_dims
            activate_dims_cat = cat_dims
            self.activate_dims = np.arange(n_hypers)
        else:
            activate_dims_cont = np.intersect1d(activate_dims, cont_dims)
            activate_dims_cat = np.intersect1d(activate_dims, cat_dims)
            self.activate_dims = activate_dims

        self.activate_dims_cont = activate_dims_cont
        self.activate_dims_cat = activate_dims_cat

        lbs = np.full(n_hypers, 0.)
        scales = np.full(n_hypers, 1.)

        if bounds_ss_cont is None or bounds_ss_cat is None:
            # cs_inner is cs
            self.cs_local = config_space
            self.new_config_space = False
            self.bounds_ss_cat = []
            self.bounds_ss_cont = np.array([[0, 1]]).repeat(len(self.activate_dims_cont), axis=0)
            self.lbs = lbs
            self.scales = scales
            self.new_config = False

        else:
            self.new_config = True
            # we normalize the non-CategoricalHyperparameter by x = (x-lb)*scale

            hps = config_space.get_hyperparameters()

            # deal with categorical hyperaprameters
            for i, cat_idx in enumerate(cat_dims):
                hp_cat = hps[cat_idx]
                parents = config_space.get_parents_of(hp_cat.name)
                if len(parents) == 0:
                    can_be_inactive = False
                else:
                    can_be_inactive = True
                n_cats = len(bounds_ss_cat[i])
                if can_be_inactive:
                    n_cats = n_cats + 1
                model_types[cat_idx] = n_cats
                model_bounds[cat_idx] = (int(n_cats), np.nan)

            # store the dimensions of numerical hyperparameters, UniformFloatHyperparameter and UniformIntegerHyperparameter
            dims_cont_num = []
            idx_cont_num = []
            dims_cont_ord = []
            idx_cont_ord = []
            # deal with ordinary hyperaprameters
            for i, cont_idx in enumerate(cont_dims):
                param = hps[cont_idx]
                if isinstance(param, OrdinalHyperparameter):
                    parents = config_space.get_parents_of(param.name)
                    if len(parents) == 0:
                        can_be_inactive = False
                    else:
                        can_be_inactive = True
                    n_cats = bounds_ss_cont[i][1] - bounds_ss_cont[i][0] + 1
                    if can_be_inactive:
                        model_bounds[cont_idx] = (0, int(n_cats))
                    else:
                        model_bounds[cont_idx] = (0, int(n_cats) - 1)
                    lbs[cont_idx] = bounds_ss_cont[i][0]  # in subapce, it should start from 0
                    dims_cont_ord.append(cont_idx)
                    idx_cont_ord.append(i)
                else:
                    dims_cont_num.append(cont_idx)
                    idx_cont_num.append(i)

            self.bounds_ss_cat = bounds_ss_cat
            self.bounds_ss_cont = np.array([[0, 1]]).repeat(len(self.activate_dims_cont), axis=0)

            lbs[dims_cont_num] = bounds_ss_cont[idx_cont_num, 0]
            # rescale numerical hyperparameters to [0., 1.]
            scales[dims_cont_num] = 1. / (bounds_ss_cont[idx_cont_num, 1] - bounds_ss_cont[idx_cont_num, 0])

            self.lbs = lbs
            self.scales = scales

            self.cs_local = ConfigurationSpace()
            hp_list = []
            idx_cont = 0
            idx_cat = 0

            hps = config_space.get_hyperparameters()

            for idx in self.activate_dims:
                param = hps[idx]
                if isinstance(param, CategoricalHyperparameter):
                    choices = [param.choices[int(choice_idx)] for choice_idx in bounds_ss_cat[idx_cat]]
                    # cat_freq_arr = np.array((cats_freq[idx_cat]))
                    # weights = cat_freq_arr / np.sum(cat_freq_arr)
                    hp_new = CategoricalHyperparameter(param.name, choices=choices)  # , weights=weights)
                    idx_cat += 1

                elif isinstance(param, OrdinalHyperparameter):
                    hp_new = OrdinalHyperparameter(param.name, sequence=np.arrange(model_bounds[idx]))
                    idx_cont += 1

                elif isinstance(param, Constant):
                    hp_new = copy.deepcopy(param)

                elif isinstance(param, UniformFloatHyperparameter):
                    lower = param.lower
                    upper = param.upper
                    if param.log:
                        lower_log = np.log(lower)
                        upper_log = np.log(upper)
                        hp_new_lower = np.exp((upper_log - lower_log) * bounds_ss_cont[idx_cont][0] + lower_log)
                        hp_new_upper = np.exp((upper_log - lower_log) * bounds_ss_cont[idx_cont][1] + lower_log)
                        hp_new = UniformFloatHyperparameter(name=param.name,
                                                            lower=max(hp_new_lower, lower),
                                                            upper=min(hp_new_upper, upper),
                                                            log=True)
                    else:
                        hp_new_lower = (upper - lower) * bounds_ss_cont[idx_cont][0] + lower
                        hp_new_upper = (upper - lower) * bounds_ss_cont[idx_cont][1] + lower
                        hp_new = UniformFloatHyperparameter(name=param.name,
                                                            lower=max(hp_new_lower, lower),
                                                            upper=min(hp_new_upper, upper),
                                                            log=False)
                    idx_cont += 1
                elif isinstance(param, UniformIntegerHyperparameter):
                    lower = param.lower
                    upper = param.upper
                    if param.log:
                        lower_log = np.log(lower)
                        upper_log = np.log(upper)
                        hp_new_lower = int(
                            math.floor(np.exp((upper_log - lower_log) * bounds_ss_cont[idx_cont][0] + lower_log)))
                        hp_new_upper = int(
                            math.ceil(np.exp((upper_log - lower_log) * bounds_ss_cont[idx_cont][1] + lower_log)))
                        hp_new = UniformIntegerHyperparameter(name=param.name,
                                                              lower=max(hp_new_lower, lower),
                                                              upper=min(hp_new_upper, upper),
                                                              log=True)
                    else:
                        hp_new_lower = int(math.floor((upper - lower) * bounds_ss_cont[idx_cont][0])) + lower
                        hp_new_upper = int(math.ceil((upper - lower) * bounds_ss_cont[idx_cont][1])) + lower
                        hp_new = UniformIntegerHyperparameter(name=param.name,
                                                              lower=max(hp_new_lower, lower),
                                                              upper=min(hp_new_upper, upper),
                                                              log=False)
                    idx_cont += 1
                else:
                    raise ValueError(f"Unsupported type of Hyperparameter: {type(param)}")
                hp_list.append(hp_new)

            # TODO Consider how to deal with subspace with reduced dimensions here, e.g. some of the conditions
            #  and forbidden clauses might become invalid
            self.cs_local.add_hyperparameters(hp_list)
            self.cs_local.add_conditions(config_space.get_conditions())
            self.cs_local.add_forbidden_clauses(config_space.get_forbiddens())

        model_kwargs = dict(configspace=self.cs_local,
                            types=model_types,
                            bounds=model_bounds,
                            bounds_cont=np.array([[0, 1.] for _ in range(len(activate_dims_cont))]),
                            bounds_cat=bounds_ss_cat,
                            seed=np.random.randint(0, 2 ** 20))

        if inspect.isclass(model_local):
            if model_local_kwargs is not None:
                model_kwargs.update(copy.deepcopy(model_local_kwargs))
            self.model = model_local(**model_kwargs)
        else:
            self.model = model_local

        if inspect.isclass(acq_func_local):
            acq_func_kwargs = {"model": self.model}
            if acq_func_local_kwargs is not None:
                acq_func_kwargs.update(acq_func_local_kwargs)
            self.acquisition_function = acq_func_local(**acq_func_kwargs)
        else:
            self.acquisition_function = acq_func_local

        self.incumbent_array = incumbent_array

        self.model_x = np.empty([0, len(activate_dims)])
        self.ss_x = np.empty([0, len(activate_dims)])
        self.model_y = np.empty([0, 1])
        self.ss_y = np.empty([0, 1])

        if initial_data is not None:
            X = initial_data[0]
            y = initial_data[1]
            X = X[:, activate_dims]

            self.add_new_observations(X, y)

        self.config_origin = "subspace"

    def update_model(self, predict_x_best: bool = True, update_incumbent_array: bool = False):
        """
        update the model and acquisition function parameters
        Parameters
        ----------
        predict_x_best: bool,
            if the incumbent is acquired by the prediceted mean of a surrogate model
        update_incumbent_array: bool
            if the incumbent_array of this subspaced is replace with the newly updated incumbent
        """
        self.model.train(self.model_x, self.model_y)

        acq_func_kwargs = {'model': self.model,
                           'num_data': len(self.ss_x)}

        if predict_x_best:
            mu, _ = self.model.predict(self.ss_x)
            idx_eta = np.argmin(mu)
            incumbent_array = self.ss_x[idx_eta]
            acq_func_kwargs.update({'incumbent_array': incumbent_array, 'eta': mu[idx_eta]})
        else:
            idx_eta = np.argmin(self.ss_y)
            incumbent_array = self.ss_x[idx_eta]
            acq_func_kwargs.update({'incumbent_array': incumbent_array, 'eta': self.ss_y[idx_eta]})
        if update_incumbent_array:
            self.incumbent_array = self.ss_x[idx_eta]

        self.acquisition_function.update(**acq_func_kwargs)

    def add_new_observations(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        add new observations to the subspace
        Parameters
        ----------
        X: np.ndarray(N,D),
            new feature vector of the observations, constructed by the global configuration space
        y: np.ndarray(N)
           new performances of the observations
        Return
        ----------
        indices_in_ss:np.ndarray(N)
            indices of data that included in subspaces
        """
        if len(X.shape) == 1:
            X = X[np.newaxis, :]
        if len(X.shape) == 1:
            y = y[np.newaxis, :]
        X = self.normalize_input(X=X)

        self.model_x = np.vstack([self.model_x, X])
        self.model_y = np.vstack([self.model_y, y])

        ss_indices = self.check_points_in_ss(X=X,
                                        cont_dims=self.activate_dims_cont,
                                        cat_dims=self.activate_dims_cat,
                                        bounds_cont=self.bounds_ss_cont,
                                        bounds_cat=self.bounds_ss_cat)
        self.ss_x = np.vstack([self.ss_x, X[ss_indices]])
        self.ss_y = np.vstack([self.ss_y, y[ss_indices]])

    def update_incumbent_array(self, new_incumbent):
        self.incumbent_array = self.normalize_input(X=new_incumbent)

    def generate_challengers(self, **optimizer_kwargs):
        challengers = self._generate_challengers(**optimizer_kwargs)
        return ChallengerListLocal(cs_local=self.cs_local,
                                   cs_global=self.cs_global,
                                   challengers=challengers,
                                   config_origin=self.config_origin,
                                   incumbent_array=self.incumbent_array)

    @abstractmethod
    def _generate_challengers(self, **optimizer_kwargs) -> typing.List[typing.Tuple[float, Configuration]]:
        """
        generate new challengers list for this subspace
        """
        raise NotImplementedError

    def normalize_input(self, X: np.ndarray):
        """
        normalize X to fit the local configuration space
        Parameters
        ----------
        X: np.ndarray(N,D)
            input X, configurations arrays
        Returns
        -------
        X_normalized: np.ndarray(N,D)
            normalized input X
        """
        if not self.new_config:
            return X

        if len(X.shape) == 1:
            X = X[np.newaxis, :]

        # normalize X
        X_normalized = (X - self.lbs) * self.scales
        # normalize categorical function, for instance, if bounds_subspace[i] is a categorical bound contains elements
        # [1, 3, 5], then we map 1->0, 3->1, 5->2
        for cat_idx, cat_bound in zip(self.activate_dims_cat, self.bounds_ss_cat):
            X_i = X_normalized[:, cat_idx]
            cond_list = [X_i == cat for cat in cat_bound]
            choice_list = np.arange(len(cat_bound))
            X_i = np.select(cond_list, choice_list)
            X_normalized[:, cat_idx] = X_i

        return X_normalized

    def check_points_in_ss(self, X: np.ndarray):
        """
        check which points will be included in this subspace, unlike the implementation in smac.epm.util_funcs, here
        the bounds for continuous hyperparameters are more strict, e.g., we do not expand the subspace to contian more
        points
        Parameters
        ----------
        X: np.ndarray(N,D),
            points to be checked
        Return
        ----------
        indices_in_ss:np.ndarray(N)
            indices of data that included in subspaces
        """
        if len(X.shape) == 1:
            X = X[np.newaxis, :]

        if self.activate_dims_cont.size != 0:
            data_in_ss = np.all(X[:, self.activate_dims_cont] <= self.bounds_ss_cont[:, 1], axis=1) & \
                         np.all(X[:, self.activate_dims_cont] >= self.bounds_ss_cont[:, 0], axis=1)

        else:
            data_in_ss = np.ones(X.shape[-1], dtype=bool)

        for bound_cat, cat_dim in zip(self.bounds_ss_cat, self.activate_dims_cat):
            data_in_ss &= np.in1d(X[:, cat_dim], bound_cat)

        data_in_ss = np.where(data_in_ss)[0]
        return data_in_ss


class ChallengerListLocal(typing.Iterator):
    def __init__(
            self,
            cs_local: ConfigurationSpace,
            cs_global: ConfigurationSpace,
            challengers: typing.List[typing.Tuple[float, Configuration]],
            config_origin: str,
            incumbent_array: typing.Optional[np.ndarray] = None,
    ):
        """
        A Challenger list to convert the configuration from local configuration space to global configuration space
        Parameters
        ----------
        cs_local: ConfigurationSpace
            local configuration space
        cs_global: ConfigurationSpace
            global configuration space
        challengers: typing.List[typing.Tuple[float, Configuration]],
            challenger lists
        config_origin: str
            configuration origin
        incumbent_array: typing.Optional[np.ndarray] = None,
            global incumbent array, used when cs_local and cs_global have different number of dimensions and we need to
            supplement the missing values.
        """
        self.cs_local = cs_local
        self.challengers = challengers
        self.cs_global = cs_global
        self._index = 0
        self.config_origin = config_origin
        # In case cs_in and cs_out have different dimensions
        self.expand_dims = (len(cs_global.get_hyperparameters()) != len(cs_local.get_hyperparameters()))
        self.incumbent_array = incumbent_array

        if self.expand_dims and self.incumbent_array is None:
            raise ValueError("Incumbent array must be provided if global configuration space has more hyperparameters"
                             "then local configuration space")

    def __next__(self) -> Configuration:
        if self.challengers is not None and self._index == len(self.challengers):
            raise StopIteration
        challenger = self.challengers[self._index][1]
        self._index += 1
        value = challenger.get_dictionary()
        if self.expand_dims:
            incumbent_array = Configuration(configuration_space=self.cs_local,
                                            vector=self.incumbent_array).get_dictionary()
            # we replace the cooresponding value in incumbent array with the value suggested by our optimizer
            for k in value.keys():
                incumbent_array[k] = value[k]
            config = Configuration(configuration_space=self.cs_global, values=incumbent_array)
        else:
            config = Configuration(configuration_space=self.cs_global, values=value)
        config.origin = self.config_origin
        return config

    def __len__(self) -> int:
        if self.challengers is None:
            self.challengers = []
        return len(self.challengers) - self._index
