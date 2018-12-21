from typing import Dict, Optional, Tuple

from ConfigSpace import (
    Configuration,
    ConfigurationSpace,
    Constant,
    CategoricalHyperparameter,
    UniformFloatHyperparameter,
    UniformIntegerHyperparameter,
    InCondition,
)
import george
import numpy as np
from scipy.stats import spearmanr
# TODO replace by group shuffle split to take instances into account?
from sklearn.model_selection import ShuffleSplit

from smac.utils.constants import MAXINT
from smac.epm.base_epm import AbstractEPM
from smac.epm.random_epm import RandomEPM
from smac.epm.rf_with_instances import RandomForestWithInstances
from smac.epm.gaussian_process_mcmc import GaussianProcessMCMC
from smac.epm.gp_default_priors import DefaultPrior
from smac.optimizer.acquisition import AbstractAcquisitionFunction, EI, LogEI, LCB, PI
from smac.runhistory.runhistory import RunHistory
from smac.runhistory.runhistory2epm import AbstractRunHistory2EPM, \
    RunHistory2EPM4LogCost, RunHistory2EPM4Cost, \
    RunHistory2EPM4LogScaledCost, RunHistory2EPM4SqrtScaledCost, \
    RunHistory2EPM4InvScaledCost, RunHistory2EPM4ScaledCost
from smac.scenario.scenario import Scenario
from smac.utils.util_funcs import get_types
from smac.tae.execute_ta_run import StatusType

class LossFunction(object):
    """Abstract loss function for internal model and acquisition function maximization.

    Paramaters
    ----------
    X_train : np.ndarray

    y_train : np.ndarray

    X_test : np.ndarray

    y_test : np.ndarray

    acq : AbstractAcquisitionFunction

    model : AbstractEPM

    config_space : ConfigurationSpace

    Returns
    -------
    float
    """

    def __call__(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        acq: AbstractAcquisitionFunction,
        model: AbstractEPM,
        config_space: ConfigurationSpace,
    ) -> float:

        raise NotImplementedError()


class SpearmanLossFunction(LossFunction):
    """Computes the spearman rank coefficient between the test data and the acquisition function values.

    The returned loss is the negative spearman rank coefficient.

    Paramaters
    ----------
    X_train : np.ndarray

    y_train : np.ndarray

    X_test : np.ndarray

    y_test : np.ndarray

    acq : AbstractAcquisitionFunction

    model : AbstractEPM

    config_space : ConfigurationSpace

    Returns
    -------
    float
    """

    def __call__(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        acq: AbstractAcquisitionFunction,
        model: AbstractEPM,
        config_space: ConfigurationSpace,
    ):

        model.train(X_train, y_train)
        acq.update(model=model, eta=np.min(y_train), num_data=len(X_train))
        configs = [Configuration(config_space, vector=x) for x in X_test]
        acquivals = acq(configurations=configs).flatten()
        return 1 - spearmanr(y_test, -acquivals)[0]
    
class RandomLossFunction(LossFunction):
    """Computes random loss function (as a baseline)

    Paramaters
    ----------
    X_train : np.ndarray

    y_train : np.ndarray

    X_test : np.ndarray

    y_test : np.ndarray

    acq : AbstractAcquisitionFunction

    model : AbstractEPM

    config_space : ConfigurationSpace

    Returns
    -------
    float
    """

    def __call__(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        acq: AbstractAcquisitionFunction,
        model: AbstractEPM,
        config_space: ConfigurationSpace,
    ):
        return np.random.rand()

class TwoStepLookbackBOLossFunction(LossFunction):
    """Perform two steps of Bayesian optimization on the test data.

    The returned loss is the minimum of the two points selected by Bayesian optimization.

    Paramaters
    ----------
    X_train : np.ndarray

    y_train : np.ndarray

    X_test : np.ndarray

    y_test : np.ndarray

    acq : AbstractAcquisitionFunction

    model : AbstractEPM

    config_space : ConfigurationSpace

    Returns
    -------
    float
    """

    def __call__(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        acq: AbstractAcquisitionFunction,
        model: AbstractEPM,
        config_space: ConfigurationSpace,
    ):

        losses_for_split = []

        # First iteration of Bayesian optimization
        model.train(X_train, y_train)
        acq.update(model=model, eta=np.min(y_train), num_data=len(X_train))
        configs = [Configuration(config_space, vector=x) for x in X_test]
        acquivals = acq(configurations=configs).flatten()

        # Query test data and move queried data point to training data
        argmax = np.nanargmax(acquivals)
        x = X_test[argmax]
        y_ = y_test[argmax]
        losses_for_split.append(y_)
        X_test = np.delete(X_test, argmax, axis=0)
        y_test = np.delete(y_test, argmax, axis=0)
        X_train = np.concatenate((X_train, x.reshape((1, -1))), axis=0)
        y_train = np.concatenate((y_train, y_.reshape((1, 1))), axis=0)

        # Second iteration of Bayesian optimization
        # TODO only update the model here (important for a GP to only sample the hyperparameters once)
        model.train(X_train, y_train)
        acq.update(model=model, eta=np.min(y_train), num_data=len(X_train))
        configs = [Configuration(config_space, vector=x) for x in X_test]
        acquivals = acq(configurations=configs).flatten()

        # Query test data
        argmax = np.nanargmax(acquivals)
        y_ = y_test[argmax] - np.min(y_test)

        losses_for_split.append(y_)
        return np.min(losses_for_split)


class AbstractComponentSelection(object):

    def __init__(
        self,
        rng: np.random.RandomState,
        config_space: ConfigurationSpace,
        scenario: Scenario,
    ):
        self.rng = rng
        self.config_space = config_space
        self.scenario = scenario

    def select(
        self,
        runhistory: RunHistory,
        runhistory2epm: AbstractRunHistory2EPM,
        default_model: AbstractEPM,
        default_acquisition_function: AbstractAcquisitionFunction,
    ) -> Tuple[AbstractEPM, AbstractAcquisitionFunction]:
        raise NotImplementedError()


class DefaultComponentSelection(AbstractComponentSelection):

    def select(
        self,
        runhistory: RunHistory,
        runhistory2epm: AbstractRunHistory2EPM,
        default_model: AbstractEPM,
        default_acquisition_function: AbstractAcquisitionFunction,
    ) -> Tuple[AbstractEPM, AbstractAcquisitionFunction]:

        return default_model, default_acquisition_function


class AdaptiveComponentSelection(AbstractComponentSelection):

    def __init__(
        self,
        rng: np.random.RandomState,
        config_space: ConfigurationSpace,
        scenario: Scenario,
        loss_function: Optional[LossFunction] = None,
        sampling_based: bool = True,
        min_test_size: int = 5,
        test_fraction: float = 0.2,
        min_runhistory_length: int = 10,
        n_splits: int = 20,
        n_random_configurations: int = 50,
    ):
        self.rng = rng
        self.config_space = config_space
        self.scenario = scenario
        self.loss_function = loss_function
        self.sampling_based = sampling_based
        self.min_test_size = min_test_size
        self.test_fraction = test_fraction
        self.min_runhistory_length = min_runhistory_length
        self.n_splits = n_splits
        self.n_random_configurations = n_random_configurations

    def select(
        self,
        runhistory: RunHistory,
        runhistory2epm: AbstractRunHistory2EPM,
        default_model: AbstractEPM,
        default_acquisition_function: AbstractAcquisitionFunction,

    ) -> Tuple[AbstractEPM, AbstractAcquisitionFunction]:
        """Select a model and an acquisition function given the runhistory data.

        Parameters
        ----------
        runhistory : RunHistory

        runhistory2epm : AbstractRunHistory2EPM

        default_model : AbstractEPM

        default_acquisition_function : AbstractAcquisitionFunction

        loss_function : LossFunction
            Uses the ``SpearmanLossFunction`` if no loss function is given.
        sampling_based : bool

        min_test_size : int

        test_fraction : float

        min_runhistory_length : int

        n_splits : int

        n_random_configurations : int

        Returns
        -------
        AbstractEPM

        AbstractAcquisitionFunction
        """

        if len(runhistory.data) < self.min_runhistory_length:
            return default_model, default_acquisition_function

        model_configurations = self._get_acm_cs().sample_configuration(self.n_random_configurations)
        # Always consider default
        model_configurations.append(self._get_acm_cs().get_default_configuration())
        # Remove duplicates
        model_configurations = list(set(model_configurations))
        # TODO remember old combinations of models and acquisition functions!
        combinations = [self._component_builder(conf.get_dictionary()) for conf in model_configurations]
        # Add random search
        random_model = RandomEPM(
            rng=self.rng,
            types=default_model.types.copy(),
            bounds=default_model.bounds,
        )
        
        combinations.append((random_model, EI(model=random_model),
                             RunHistory2EPM4Cost(scenario=self.scenario, 
                                num_params=len(self.scenario.cs.get_hyperparameters()), 
                                success_states=[StatusType.SUCCESS, StatusType.CRASHED], 
                                impute_censored_data=False, 
                                impute_state=None)))

        if self.loss_function is None:
            loss_function = SpearmanLossFunction()
        else:
            loss_function = self.loss_function

        X, y = runhistory2epm.transform(runhistory)
        test_size = max(self.min_test_size, int(len(X) * self.test_fraction))

        combination_losses = []
        splitter = ShuffleSplit(n_splits=self.n_splits, test_size=test_size, random_state=1)
        for train_indices, test_indices in splitter.split(X, y):
            
            losses = []
            for model, acq, rh2epm in combinations:
                X, y = rh2epm.transform(runhistory)
                X_train = X[train_indices]
                y_train = y[train_indices]
                X_test = X[test_indices]
                y_test = y[test_indices]
                
                loss = loss_function(
                    X_train=X_train.copy(), y_train=y_train.copy(), X_test=X_test.copy(), y_test=y_test.copy(),
                    acq=acq, model=model, config_space=runhistory2epm.scenario.cs,
                )
                losses.append(loss)
            combination_losses.append(losses)

        combination_losses = np.array(combination_losses)
        if self.sampling_based:
            wins = np.zeros((len(combinations), ))
            for cl in combination_losses:
                try:
                    min_value = np.nanmin(cl)
                    minima = min_value == cl
                    minima_indices = np.where(minima)[0]
                    if len(minima_indices) == len(cl):
                        # This split is not informative -> ignore it
                        print("FAIL")
                        continue
                    for midx in minima_indices:
                        wins[midx] += 1 / len(minima_indices)
                # In case the whole row is empty (i.e. full of NaNs)
                except ValueError:
                    continue
            if np.sum(wins) == 0:
                wins[-1] += 1
            wins = wins / np.sum(wins)
            choice = self.rng.choice(len(combinations), p=wins)
        else:
            combination_losses = np.nanmean(combination_losses, axis=0)
            assert len(combination_losses) == len(combinations)
            choice = np.nanargmin(combination_losses)
            for l, c in zip(combination_losses,combinations):
                print(l,list(map(str,c)))

        print("Adaptive Selection: %s, %s, %s" %(combinations[choice][0], combinations[choice][1], combinations[choice][2]))
        return combinations[choice][0], combinations[choice][1], combinations[choice][2]

    def _component_builder(self, conf: Dict) -> Tuple[AbstractEPM, 
                                                      AbstractAcquisitionFunction,
                                                      AbstractRunHistory2EPM]:
        """
            builds new Acquisition function object
            and EPM object and returns these

            Parameters
            ----------
            conf: typing.Union[Configuration, dict]
                configuration specifying "model", "acq_func" and "y_transform"

            Returns
            -------
            typing.Tuple[AbstractAcquisitionFunction, 
                        AbstractEPM,
                        AbstractRunHistory2EPM]

        """
        types, bounds = get_types(self.config_space, instance_features=self.scenario.feature_array)
        if conf["model"] == "RF":
            model = RandomForestWithInstances(
                types=types,
                bounds=bounds,
                instance_features=self.scenario.feature_array,
                seed=self.rng.randint(MAXINT),
                pca_components=conf.get("pca_dim", self.scenario.PCA_DIM),
                # TODO add log-space to the inner hyperparameter sampling procedure
                log_y=conf.get("log_y", self.scenario.transform_y in ["LOG", "LOGS"]),
                num_trees=conf.get("num_trees", self.scenario.rf_num_trees),
                do_bootstrapping=conf.get("do_bootstrapping", self.scenario.rf_do_bootstrapping),
                ratio_features=conf.get("ratio_features", self.scenario.rf_ratio_features),
                min_samples_split=int(conf.get("min_samples_to_split", self.scenario.rf_min_samples_split)),
                min_samples_leaf=int(conf.get("min_samples_in_leaf", self.scenario.rf_min_samples_leaf)),
                max_depth=int(conf.get("max_depth", self.scenario.rf_max_depth)),
            )

        elif conf["model"] == "GP":
            cov_amp = 2
            n_dims = len(types)
            initial_ls = np.ones([n_dims])
            exp_kernel = george.kernels.Matern52Kernel(initial_ls, ndim=n_dims)
            kernel = cov_amp * exp_kernel
            prior = DefaultPrior(len(kernel) + 1, rng=self.rng)
            n_hypers = 3 * len(kernel)
            if n_hypers % 2 == 1:
                n_hypers += 1
            model = GaussianProcessMCMC(
                types=types,
                bounds=bounds,
                kernel=kernel,
                prior=prior,
                n_hypers=n_hypers,
                chain_length=200,
                burnin_steps=100,
                normalize_input=True,
                normalize_output=True,
                rng=self.rng,
            )
        else:
            raise ValueError(conf['model'])

        if conf["acq_func"] == "EI":
            acq = EI(model=model, par=conf.get("par_ei", 0))
        elif conf["acq_func"] == "LCB":
            acq = LCB(model=model, par=conf.get("par_lcb", 0.05))
        elif conf["acq_func"] == "PI":
            acq = PI(model=model, par=conf.get("par_pi", 0))
        elif conf["acq_func"] == "LogEI":
            # par value should be in log-space
            acq = LogEI(model=model, par=conf.get("par_logei", 0))
        else:
            raise ValueError(conf['acq_func'])

        num_params = len(self.scenario.cs.get_hyperparameters())
        success_states = [StatusType.SUCCESS, StatusType.CRASHED]
        #TODO: only designed for black box problems without instances
        if conf["y_transform"] == "y":
            rh2epm = RunHistory2EPM4Cost(scenario=self.scenario, 
                                num_params=num_params, 
                                success_states=success_states, 
                                impute_censored_data=False, 
                                impute_state=None)
        elif conf["y_transform"] == "log_scaled":
            rh2epm = RunHistory2EPM4LogScaledCost(scenario=self.scenario, 
                                num_params=num_params, 
                                success_states=success_states, 
                                impute_censored_data=False, 
                                impute_state=None)
        elif conf["y_transform"] == "inv_scaled":
            rh2epm = RunHistory2EPM4InvScaledCost(scenario=self.scenario, 
                                num_params=num_params, 
                                success_states=success_states, 
                                impute_censored_data=False, 
                                impute_state=None)
        else:
            raise ValueError(conf["y_transform"])
        
        return model, acq, rh2epm

 #==============================================================================
 #    def _get_acm_cs(self) -> ConfigurationSpace:
 #        """returns a configuration space designed for querying ~smac.optimizer.smbo._component_builder
 # 
 #        Returns
 #        -------
 #        ConfigurationSpace
 #        """
 # 
 #        cs = ConfigurationSpace()
 #        cs.seed(self.rng.randint(0, 2 ** 20))
 # 
 #        # Temporarily disable the GP
 #        model = CategoricalHyperparameter("model", choices=("RF",))  # "GP"))
 # 
 #        num_trees = Constant("num_trees", value=10)
 #        bootstrap = CategoricalHyperparameter("do_bootstrapping", choices=(True, False), default_value=True)
 #        ratio_features = CategoricalHyperparameter("ratio_features", choices=(3 / 6, 4 / 6, 5 / 6, 1), default_value=1)
 #        min_split = UniformIntegerHyperparameter("min_samples_to_split", lower=2, upper=10, default_value=2)
 #        min_leaves = UniformIntegerHyperparameter("min_samples_in_leaf", lower=1, upper=10, default_value=1)
 # 
 #        cs.add_hyperparameters([model, num_trees, bootstrap, ratio_features, min_split, min_leaves])
 # 
 #        inc_num_trees = InCondition(num_trees, model, ["RF"])
 #        inc_bootstrap = InCondition(bootstrap, model, ["RF"])
 #        inc_ratio_features = InCondition(ratio_features, model, ["RF"])
 #        inc_min_split = InCondition(min_split, model, ["RF"])
 #        inc_min_leavs = InCondition(min_leaves, model, ["RF"])
 # 
 #        cs.add_conditions([inc_num_trees, inc_bootstrap, inc_ratio_features, inc_min_split, inc_min_leavs])
 # 
 #        acq = CategoricalHyperparameter("acq_func", choices=("EI", "LCB", "PI", "LogEI"))
 #        par_ei = UniformFloatHyperparameter("par_ei", lower=-10, upper=10)
 #        par_pi = UniformFloatHyperparameter("par_pi", lower=-10, upper=10)
 #        par_logei = UniformFloatHyperparameter("par_logei", lower=0.001, upper=10, log=True)
 #        par_lcb = UniformFloatHyperparameter("par_lcb", lower=0.0001, upper=0.9999)
 # 
 #        cs.add_hyperparameters([acq, par_ei, par_pi, par_logei, par_lcb])
 # 
 #        inc_par_ei = InCondition(par_ei, acq, ["EI"])
 #        inc_par_pi = InCondition(par_pi, acq, ["PI"])
 #        inc_par_logei = InCondition(par_logei, acq, ["LogEI"])
 #        inc_par_lcb = InCondition(par_lcb, acq, ["LCB"])
 # 
 #        cs.add_conditions([inc_par_ei, inc_par_pi, inc_par_logei, inc_par_lcb])
 # 
 #        return cs
 #==============================================================================
    
    def _get_acm_cs(self) -> ConfigurationSpace:
        """returns a configuration space designed for querying ~smac.optimizer.smbo._component_builder
 
        Returns
        -------
        ConfigurationSpace
        """
 
        cs = ConfigurationSpace()
        cs.seed(self.rng.randint(0, 2 ** 20))
 
        # EPM
        model = CategoricalHyperparameter("model", choices=["RF"], default_value="RF")#,"GP")) 
        bootstrap = CategoricalHyperparameter("do_bootstrapping", choices=[True, False])#, False), default_value=True)
        num_trees = Constant("num_trees", value=10)
        ratio_features = CategoricalHyperparameter("ratio_features", choices=(3 / 6, 4 / 6, 5 / 6, 1), default_value=1)
        min_split = UniformIntegerHyperparameter("min_samples_to_split", lower=2, upper=10, default_value=2)
        min_leaves = UniformIntegerHyperparameter("min_samples_in_leaf", lower=1, upper=10, default_value=1)
        
        # Acquisition
        acq = CategoricalHyperparameter("acq_func", 
                                        choices=["EI"],# "LCB", "PI"],# "LogEI"), 
                                        default_value="EI")
 
        # y_transform
        y_trans = CategoricalHyperparameter("y_transform", 
                                            #choices=["y","log_scaled", "inv_scaled"],
                                            choices=["log_scaled","y"],
                                            default_value="log_scaled")
        
        cs.add_hyperparameters([model, acq, ratio_features, bootstrap, y_trans, min_split, min_leaves])
 
        return cs
    
 #==============================================================================
 #    def _get_acm_cs(self) -> ConfigurationSpace:
 #        """returns a configuration space designed for querying ~smac.optimizer.smbo._component_builder
 # 
 #        Returns
 #        -------
 #        ConfigurationSpace
 #        """
 # 
 #        cs = ConfigurationSpace()
 #        cs.seed(self.rng.randint(0, 2 ** 20))
 # 
 #        model = CategoricalHyperparameter("model", choices=("RF",))  # "GP"))
 #        ratio_features = CategoricalHyperparameter("ratio_features", choices=[1], default_value=1)
 #        cs.add_hyperparameters([model,ratio_features])
 # 
 #        acq = CategoricalHyperparameter("acq_func", choices=("EI", "LCB", "PI", "LogEI"))
 #        par_ei = UniformFloatHyperparameter("par_ei", lower=-10, upper=10)
 #        par_pi = UniformFloatHyperparameter("par_pi", lower=-10, upper=10)
 #        par_logei = UniformFloatHyperparameter("par_logei", lower=0.001, upper=10, log=True)
 #        par_lcb = UniformFloatHyperparameter("par_lcb", lower=0.0001, upper=0.9999)
 #  
 #        cs.add_hyperparameters([acq, par_ei, par_pi, par_logei, par_lcb])
 #  
 #        inc_par_ei = InCondition(par_ei, acq, ["EI"])
 #        inc_par_pi = InCondition(par_pi, acq, ["PI"])
 #        inc_par_logei = InCondition(par_logei, acq, ["LogEI"])
 #        inc_par_lcb = InCondition(par_lcb, acq, ["LCB"])
 #  
 #        cs.add_conditions([inc_par_ei, inc_par_pi, inc_par_logei, inc_par_lcb])
 # 
 #        return cs
 #==============================================================================
