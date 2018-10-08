import numpy as np
import logging

from pyrfr import regression

from smac.epm.base_epm import AbstractEPM
from smac.epm.rf_with_instances import RandomForestWithInstances
from smac.utils.constants import N_TREES

__author__ = "Aaron Klein"
__copyright__ = "Copyright 2015, ML4AAD"
__license__ = "3-clause BSD"
__maintainer__ = "Aaron Klein"
__email__ = "kleinaa@cs.uni-freiburg.de"
__version__ = "0.0.1"


class ReinterpolationRF(RandomForestWithInstances):
    """Reinterpolation by Forrester:

    http://www.soton.ac.uk/~nwb/lectures/AeroCFD/Other/AlexForrester_AIAA-20068-629.pdf

    Reduces noise at observed values (noise can be due to RF hyperparameters, too).
    """

    def __init__(self, types: np.ndarray,
                 bounds: np.ndarray,
                 model: AbstractEPM,
                 bootstrap: bool,
                 seed: int = 42):
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
        model : AbstractEPM

        seed : int
            The seed that is passed to the random_forest_run library.
        """
        super().__init__(
            types=types,
            bounds=bounds,
            instance_features=None,
            seed=1,
            pca_components=7,
            log_y=False,
            num_trees=N_TREES,
            do_bootstrapping=True,
            ratio_features=1,
            min_samples_split=2,
            min_samples_leaf=1,
            max_depth=2 ** 20,
        )

        self.types = types
        self.bounds = bounds
        self.model = model
        self.rng = regression.default_random_engine(seed)

        self.rf_opts = regression.forest_opts()
        self.rf_opts.num_trees = N_TREES
        self.rf_opts.do_bootstrapping = bootstrap
        self.rf_opts.tree_opts.max_features = types.shape[0]
        self.rf_opts.tree_opts.min_samples_to_split = 2
        self.rf_opts.tree_opts.min_samples_in_leaf = 1
        self.rf_opts.tree_opts.max_depth = 2 ** 20
        self.rf_opts.tree_opts.epsilon_purity = 1e-8
        self.rf_opts.tree_opts.max_num_nodes = 2 ** 20
        self.rf_opts.compute_law_of_total_variance = False

        self.rf = None  # type: regression.binary_rss_forest

        # This list well be read out by save_iteration() in the solver
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

        self.rf = regression.binary_rss_forest()
        self.rf_opts.num_data_points_per_tree = self.X.shape[0]
        self.rf.options = self.rf_opts

        reinterpolate = True

        # In case the model contains HPO this must be done in advance to know
        # whether it is necessary to do a reinterpolation or not
        self.model.train(X, y)

        if isinstance(self.model, RandomForestWithInstances):
            # No reinterpolation if the hyperparameters are equal!!!
            if self.model.hypers[:-1] == self.hypers[:-1]:
                print('Not reinterpolating the data!')
                print(self.model.hypers[:-1], self.hypers[:-1])
                reinterpolate = False
                data = self._init_data_container(self.X, y)
                self.rf.fit(data, rng=self.rng)

        if reinterpolate:
            y_new = np.array([self.model.predict(x.reshape((1, -1)))[0] for x in self.X]).flatten()
            assert len(X) == len(y_new)
            self.y_new = y_new
            reinterpolation_data = self._init_data_container(self.X, y_new)
            self.rf.fit(reinterpolation_data, rng=self.rng)

        return self
