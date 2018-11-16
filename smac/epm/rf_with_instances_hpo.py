import logging
import typing

from ConfigSpace import (
    CategoricalHyperparameter,
    UniformIntegerHyperparameter,
    Constant,
    ConfigurationSpace,
    Configuration,
)
import numpy as np
from pyrfr import regression
from sklearn.model_selection import KFold
import scipy.stats.distributions as scst

from smac.epm.rf_with_instances import RandomForestWithInstances
from smac.utils.constants import N_TREES


MAX_NUM_NODES = 2 ** 20
MAX_DEPTH = 2 ** 20
EPSILON_IMPURITY = 1e-8
N_POINTS_PER_TREE = -1


class RandomForestWithInstancesHPO(RandomForestWithInstances):
    """Random forest that takes instance features into account and performs automatic hyperparameter optimization.

    Attributes
    ----------
    rf_opts : pyrfr.regression.rf_opts
        Random forest hyperparameter
    n_points_per_tree : int
    rf : regression.binary_rss_forest
        Only available after training
    hypers: list
        List of random forest hyperparameters
    unlog_y: bool
    seed : int
    types : list
    bounds : list
    rng : np.random.RandomState
    logger : logging.logger
    """

    def __init__(
        self,
        types: np.ndarray,
        bounds: typing.List[typing.Tuple[float, float]],
        log_y: bool=False,
        bootstrap: bool=False,
        n_iters: int=50,
        n_splits: int=10,
        seed: int=42,
    ):
        """Parameters
        ----------
        types : np.ndarray (D)
            Specifies the number of categorical values of an input dimension where
            the i-th entry corresponds to the i-th input dimension. Let's say we
            have 2 dimension where the first dimension consists of 3 different
            categorical choices and the second dimension is continuous than we
            have to pass np.array([2, 0]). Note that we count starting from 0.
        bounds : np.ndarray (D, 2)
            Specifies the bounds for continuous features.
        log_y: bool
            y values (passed to this RF) are expected to be log(y) transformed;
            this will be considered during predicting
        bootstrap : bool
            Turns on / off bootstrapping in the random forest.
        n_iters : int
            Number of iterations for random search.
        n_splits : int
            Number of cross-validation splits.
        seed : int
            The seed that is passed to the random_forest_run library.
        """
        super().__init__(
            types,
            bounds,
            log_y,
            num_trees=N_TREES,
            do_bootstrapping=bootstrap,
            n_points_per_tree=N_POINTS_PER_TREE,
            ratio_features=5/6,
            min_samples_split=3,
            min_samples_leaf=3,
            max_depth=MAX_DEPTH,
            eps_purity=EPSILON_IMPURITY,
            max_num_nodes=MAX_NUM_NODES,
            seed=seed,
        )

        self.types = types
        self.bounds = bounds
        self.log_y = log_y
        self.n_iters = n_iters
        self.n_splits = n_splits
        self.rng = regression.default_random_engine(seed)
        self.rs = np.random.RandomState(seed)
        self.bootstrap = bootstrap

        self.rf_opts = regression.forest_opts()
        self.rf_opts.num_trees = N_TREES
        self.rf_opts.compute_oob_error = True
        self.rf_opts.do_bootstrapping = self.bootstrap
        self.rf_opts.tree_opts.max_features = int(types.shape[0])
        self.rf_opts.tree_opts.min_samples_to_split = 2
        self.rf_opts.tree_opts.min_samples_in_leaf = 1
        self.rf_opts.tree_opts.max_depth = MAX_DEPTH
        self.rf_opts.tree_opts.epsilon_purity = EPSILON_IMPURITY
        self.rf_opts.tree_opts.max_num_nodes = MAX_NUM_NODES
        self.rf_opts.compute_law_of_total_variance = False

        self.rf = None  # type: regression.binary_rss_forest

        # This list will be read out by save_iteration() in the solver
        self._set_hypers(self._get_configuration_space().get_default_configuration())
        self.seed = seed

        self.logger = logging.getLogger(self.__module__ + "." +
                                        self.__class__.__name__)

    def _train(self, X: np.ndarray, y: np.ndarray) -> 'RandomForestWithInstancesHPO':
        """Trains the random forest on X and y.

        Parameters
        ----------
        X : np.ndarray [n_samples, n_features (config + instance features)]
            Input data points.
        y : np.ndarray [n_samples, ]
            The corresponding target values.

        Returns
        -------
        self
        """

        self.X = X
        self.y = y.flatten()

        cfg = self._get_configuration_space()

        # Draw 50 random samples and use best according to 10 CV
        best_error = None
        best_config = None
        if X.shape[0] > 3:
            for i in range(self.n_iters):
                if i == 0:
                    configuration = cfg.get_default_configuration()
                else:
                    configuration = cfg.sample_configuration()
                n_splits = min(X.shape[0], self.n_splits)
                kf = KFold(n_splits=n_splits)
                error = 0.0
                for train_index, test_index in kf.split(X):
                    error += self._eval_rf(
                        c=configuration,
                        X=X[train_index, :],
                        y=y[train_index],
                        X_test=X[test_index, :],
                        y_test=y[test_index],
                    )
                self.logger.debug(error)
                if best_error is None or error < best_error:
                    best_config = configuration
                    best_error = error
        else:
            best_config = cfg.get_default_configuration()

        self.rf_opts = self._set_conf(
            c=best_config, n_features=self.X.shape[1], num_data_points=X.shape[0],
        )
        self._set_hypers(best_config)

        self.logger.debug("Use %s" % str(self.rf_opts))
        self.rf = regression.binary_rss_forest()
        self.rf.options = self.rf_opts
        data = self._init_data_container(self.X, self.y)
        self.rf.fit(data, rng=self.rng)

        return self

    def _eval_rf(
        self,
        c: Configuration,
        X: np.ndarray,
        y: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ) -> float:
        """Evaluate random forest configuration on train/test data.

        Parameters
        ----------
        c : Configuration
            Random forest configuration to evaluate on the train/test data
        X : np.ndarray [n_samples, n_features (config + instance features)]
            Training features
        y : np.ndarray [n_samples, ]
            Training targets
        X_test : np.ndarray [n_samples, n_features (config + instance features)]
            Validation features
        y_test : np.ndarray [n_samples, ]
            Validation targets

        Returns
        -------
        float
        """
        opts = self._set_conf(c, n_features=X.shape[1], num_data_points=X.shape[0])
        rng = regression.default_random_engine(1)
        rf = regression.binary_rss_forest()
        rf.options = opts
        data = self._init_data_container(X, y)
        rf.fit(data, rng=rng)

        loss = 0
        for row, lab in zip(X_test, y_test):
            m, v = rf.predict_mean_var(row)
            std = max(1e-8, np.sqrt(v))
            nllh = -scst.norm(loc=m, scale=std).logpdf(lab)
            loss += nllh

        return loss

    def _set_conf(
        self,
        c: Configuration,
        n_features: int,
        num_data_points: int,
    ) -> regression.forest_opts:
        """Transform a Configuration object a forest_opts object.

        Parameters
        ----------
        c : Configuration
            Hyperparameter configurations
        n_features : int
            Number of features used to calculate the feature subset in the random forest.
        num_data_points : int
            Number of data points (required by the random forest).

        Returns
        -------
        pyrfr.regression.rf_opts
        """

        rf_opts = regression.forest_opts()
        rf_opts.num_trees = c["num_trees"]
        rf_opts.do_bootstrapping = c["do_bootstrapping"]
        rf_opts.tree_opts.max_num_nodes = 2 ** 20

        rf_opts.tree_opts.max_features = max(1, int(np.rint(n_features * c["max_features"])))
        rf_opts.tree_opts.min_samples_to_split = int(
            c["min_samples_to_split"])
        rf_opts.tree_opts.min_samples_in_leaf = c["min_samples_in_leaf"]
        rf_opts.tree_opts.max_depth = MAX_DEPTH
        rf_opts.tree_opts.max_num_nodes = MAX_NUM_NODES

        if N_POINTS_PER_TREE <= 0:
            rf_opts.num_data_points_per_tree = num_data_points
        else:
            raise ValueError()

        return rf_opts

    def _set_hypers(self, c: Configuration) -> None:
        """Set hyperparameters array.

        Parameters
        ----------
        c : Configuration
        """

        self.hypers = [
            int(c["num_trees"]),
            MAX_NUM_NODES,
            c["do_bootstrapping"],
            N_POINTS_PER_TREE,
            c["max_features"],
            c["min_samples_to_split"],
            c["min_samples_in_leaf"],
            MAX_DEPTH,
            EPSILON_IMPURITY,
            self.seed,
        ]

    def _get_configuration_space(self) -> ConfigurationSpace:
        """Get the configuration space for the random forest.

        Returns
        -------
        ConfigurationSpace
        """
        cfg = ConfigurationSpace()
        cfg.seed(int(self.rs.randint(0, 1000)))

        num_trees = Constant("num_trees", value=N_TREES)
        bootstrap = CategoricalHyperparameter(
            "do_bootstrapping", choices=(self.bootstrap,), default_value=self.bootstrap,
        )
        max_feats = CategoricalHyperparameter("max_features", choices=(3 / 6, 4 / 6, 5 / 6, 1), default_value=1)
        min_split = UniformIntegerHyperparameter("min_samples_to_split", lower=1, upper=10, default_value=2)
        min_leavs = UniformIntegerHyperparameter("min_samples_in_leaf", lower=1, upper=10, default_value=1)
        cfg.add_hyperparameters([num_trees, bootstrap, max_feats, min_split, min_leavs])
        return cfg
