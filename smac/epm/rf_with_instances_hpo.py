import numpy as np
import logging

from pyrfr import regression
from sklearn.model_selection import KFold
import scipy.stats.distributions as scst

from ConfigSpace import (
    CategoricalHyperparameter,
    UniformIntegerHyperparameter,
    Constant,
    ConfigurationSpace,
    Configuration,
)

from smac.epm.rf_with_instances import RandomForestWithInstances

__author__ = "Aaron Klein"
__copyright__ = "Copyright 2015, ML4AAD"
__license__ = "3-clause BSD"
__maintainer__ = "Aaron Klein"
__email__ = "kleinaa@cs.uni-freiburg.de"
__version__ = "0.0.1"


MAX_NUM_NODES = 2 ** 20
MAX_DEPTH = 2 ** 20
EPSILON_IMPURITY = 1e-8
N_POINTS_PER_TREE = -1


class RandomForestWithInstancesHPO(RandomForestWithInstances):
    """Interface to the random forest that takes instance features
    into account.

    Attributes
    ----------
    rf_opts :
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

    def __init__(self, types: np.ndarray,
                 bounds: np.ndarray,
                 log_y: bool=False,
                 seed: int=42):
        """Constructor

        Parameters
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
        num_trees : int
            The number of trees in the random forest.
        do_bootstrapping : bool
            Turns on / off bootstrapping in the random forest.
        n_points_per_tree : int
            Number of points per tree. If <= 0 X.shape[0] will be used
            in _train(X, y) instead
        ratio_features : float
            The ratio of features that are considered for splitting.
        min_samples_split : int
            The minimum number of data points to perform a split.
        min_samples_leaf : int
            The minimum number of data points in a leaf.
        max_depth : int
            The maximum depth of a single tree.
        eps_purity : float
            The minimum difference between two target values to be considered
            different
        max_num_nodes : int
            The maxmimum total number of nodes in a tree
        seed : int
            The seed that is passed to the random_forest_run library.
        """
        super().__init__(
            types,
            bounds,
            log_y,
            num_trees=10,
            do_bootstrapping=True,
            n_points_per_tree=N_POINTS_PER_TREE,
            ratio_features=5/6,
            min_samples_split=3,
            min_samples_leaf=3,
            max_depth=MAX_DEPTH,
            eps_purity=EPSILON_IMPURITY,
            max_num_nodes=MAX_NUM_NODES,
            seed=seed
        )

        self.types = types
        self.bounds = bounds
        self.log_y = log_y
        self.rng = regression.default_random_engine(seed)
        self.rs = np.random.RandomState(seed)

        self.rf_opts = regression.forest_opts()
        self.rf_opts.num_trees = 10
        self.rf_opts.do_bootstrapping = True
        self.rf_opts.tree_opts.max_features = int(types.shape[0])
        self.rf_opts.tree_opts.min_samples_to_split = 2
        self.rf_opts.tree_opts.min_samples_in_leaf = 1
        self.rf_opts.tree_opts.max_depth = MAX_DEPTH
        self.rf_opts.tree_opts.epsilon_purity = EPSILON_IMPURITY
        self.rf_opts.tree_opts.max_num_nodes = MAX_NUM_NODES
        self.rf_opts.compute_law_of_total_variance = False

        self.rf = None  # type: regression.binary_rss_forest

        # This list will be read out by save_iteration() in the solver
        self.set_hypers(self._get_configuration_space().get_default_configuration())
        self.seed = seed

        self.logger = logging.getLogger(self.__module__ + "." +
                                        self.__class__.__name__)

    def _train(self, X: np.ndarray, y: np.ndarray, **kwargs):
        """Trains the random forest on X and y.

        Parameters
        ----------
        X : np.ndarray [n_samples, n_features (config + instance features)]
            Input data points.
        Y : np.ndarray [n_samples, ]
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
            for i in range(50):
                if i == 0:
                    configuration = cfg.get_default_configuration()
                else:
                    configuration = cfg.sample_configuration()
                n_splits = min(X.shape[0], 10)
                kf = KFold(n_splits=n_splits)
                error = 0
                for train_index, test_index in kf.split(X):
                    error += self.eval_rf(c=configuration,
                                          x=X[train_index, :], y=y[train_index],
                                          x_test=X[test_index, :], y_test=y[test_index])
                self.logger.debug(error)
                if best_error is None or error < best_error:
                    best_config = configuration
                    best_error = error
        else:
            best_config = cfg.get_default_configuration()

        self.rf_opts = self.set_conf(best_config, self.X.shape)
        self.set_hypers(best_config)

        self.logger.debug("Use %s" % str(self.rf_opts))
        self.rf = regression.binary_rss_forest()
        self.rf.options = self.rf_opts
        data = self._init_data_container(self.X, self.y)
        self.rf.fit(data, rng=self.rng)

        return self

    def eval_rf(self, c: Configuration, x: np.ndarray, y: np.ndarray, x_test: np.ndarray, y_test: np.ndarray):
        opts = self.set_conf(c, x.shape)
        rng = regression.default_random_engine(1)
        rf = regression.binary_rss_forest()
        rf.options = opts
        data = self._init_data_container(x, y)
        rf.fit(data, rng=rng)

        loss = 0
        for row, lab in zip(x_test, y_test):
            m, v = rf.predict_mean_var(row)
            std = max(1e-8, np.sqrt(v))
            nllh = -scst.norm(loc=m, scale=std).logpdf(lab)
            loss += nllh
            # m = rf.predict(row)
            # loss += np.sqrt(mean_squared_error(y_true=lab, y_pred=m))

        return loss

    def set_conf(self, c: Configuration, x_shape: int):
        rf_opts = regression.forest_opts()
        rf_opts.num_trees = int(c["num_trees"])
        rf_opts.do_bootstrapping = c["do_bootstrapping"]
        rf_opts.tree_opts.max_num_nodes = 2 ** 20

        rf_opts.tree_opts.max_features = max(1, int(np.rint(x_shape[1] * c["max_features"])))
        rf_opts.tree_opts.min_samples_to_split = int(
            c["min_samples_to_split"])
        rf_opts.tree_opts.min_samples_in_leaf = c["min_samples_in_leaf"]
        rf_opts.tree_opts.max_depth = MAX_DEPTH
        rf_opts.tree_opts.max_num_nodes = MAX_NUM_NODES

        if N_POINTS_PER_TREE <= 0:
            rf_opts.num_data_points_per_tree = self.X.shape[0]
        else:
            rf_opts.num_data_points_per_tree = N_POINTS_PER_TREE

        return rf_opts

    def set_hypers(self, c: Configuration) -> None:
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
        cfg = ConfigurationSpace()
        cfg.seed(self.seed)

        num_trees = Constant("num_trees", value=10)
        # lower=10, upper=100, default_value=10, log=True)
        bootstrap = CategoricalHyperparameter("do_bootstrapping", choices=(True,), default_value=True)
        # bootstrap = CategoricalHyperparameter("do_bootstrapping", choices=(True, False), default_value=True)
        max_feats = CategoricalHyperparameter("max_features", choices=(3 / 6, 4 / 6, 5 / 6, 1), default_value=1)
        min_split = UniformIntegerHyperparameter("min_samples_to_split", lower=1, upper=10, default_value=2)
        min_leavs = UniformIntegerHyperparameter("min_samples_in_leaf", lower=1, upper=10, default_value=1)
        cfg.add_hyperparameters([num_trees, bootstrap, max_feats, min_split, min_leavs])
        return cfg
