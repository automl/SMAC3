import logging
import numpy
import scipy.stats

import smac.epm.base_imputor
import pyrfr.regression

__author__ = "Katharina Eggensperger"
__copyright__ = "Copyright 2015, ML4AAD"
__license__ = "BSD"
__maintainer__ = "Katharina Eggensperger"
__email__ = "eggenspk@cs.uni-freiburg.de"
__version__ = "0.0.1"


class RFRImputator(smac.epm.base_imputor.BaseImputor):
    """Uses an rfr to do imputation"""

    def __init__(self, cs, rs, cutoff, threshold, change_threshold=0.01, max_iter=10):
        """
        initialize imputator module

        Parameters
        ----------
        max_iter : maximum number of iteration
        cs : config space object
        rs : random state generator
        cutoff : int
            cutoff value used for this scenario
        threshold : float
            highest possible values (e.g. cutoff * par)
        change_threshold : stop imputation if change is less than this
        -------
        """

        super(RFRImputator, self).__init__()
        self.logger = logging.getLogger("RFRImputor")
        self.max_iter = max_iter
        self.change_threshold = change_threshold
        self.cutoff = cutoff
        self.threshold = threshold
        self.seed = rs.random_integer(low=0, high=1000)

        # TODO find out correct syntax of this
        self.type = numpy.array(cs._cat_size, dtype=numpy.uint64)
        self.std_threshold = 10**-5

        # Hyperparameter for random forest, mostly defaults
        self.do_bootstrapping = True
        self.num_data_points_per_tree = 0
        self.max_features = 2
        self.max_features_per_split = 0
        self.min_samples_to_split = 2
        self.min_samples_in_leaf = 1
        self.max_depth = 0
        self.epsilon_purity = 1e-8
        self.num_trees = 50

    def _get_model(self, X, y):
        data1 = pyrfr.regression.numpy_data_container(X, y, self.types)
        model = pyrfr.regression.binary_rss()
        model.seed = self.seed
        model.do_bootstrapping = self.do_bootstrapping
        model.num_data_points_per_tree = self.num_data_points_per_tree
        model.max_features = self.max_features_per_split
        model.min_samples_to_split = self.min_samples_to_split
        model.min_samples_in_leaf = self.min_samples_in_leaf
        model.max_depth = self.max_depth
        model.epsilon_purity = self.epsilon_purity
        model.num_trees = self.num_trees

        model.fit(data1)
        return model

    def _predict(self, m, X):
        """
        wrap rfr predict method to predict for multiple X's and returns y's
        Parameters
        ----------
        m : pyrfr.regression.binary_rss
        x : array
        -------
        """
        if len(X.shape) > 1:
            pred = numpy.array([m.predict(x) for x in X])
            mean = pred[:, 0]
            std = pred[:, 1]
            std[std < self.std_threshold] = self.std_threshold
            std[numpy.isnan(std)] = self.std_threshold

            mean = numpy.array(mean)
            std = numpy.array(std)
        else:
            mean, std = self.model.predict(X)
            if std < self.std_threshold:
                self.logger.debug("Standard deviation is small, capping to 10^-5")
                std = self.std_threshold
            std = numpy.array([std, ])
            mean = numpy.array([mean, ])

        return mean, std

    def impute(self, censored_X, censored_y, uncensored_X, uncensored_y):
        """
        impute runs and returns imputed y values

        Parameters
        ----------
        censored_X : array
            Object that keeps complete run_history
        censored_y : list
        uncensored_X : array
        uncensored_y : list
        """

        # first learn model without censored data
        m = self._get_model(uncensored_X, uncensored_y)

        self.logger.debug("Going to impute y-values with %s" % str(m))

        imputed_y = None  # define this, if imputation fails

        # Define variables
        y = None

        it = 0
        change = 0
        while True:
            self.logger.debug("Iteration %d of %d" % (it, self.max_iter))

            # predict censored y values
            y_mean, y_stdev = self._predict(m, censored_X)
            del m

            imputed_y = \
                [scipy.stats.truncnorm.stats(a=(censored_y[index] -
                                                y_mean[index]) / y_stdev[index],
                                             b=(numpy.inf - y_mean[index]) /
                                               y_stdev[index],
                                             loc=y_mean[index],
                                             scale=y_stdev[index],
                                             moments='m')
                 for index in range(len(censored_y))]
            imputed_y = numpy.array(imputed_y)

            # Replace all nans with threshold
            self.logger.critical("Going to replace %d nan-value(s) with "
                                 "threshold" %
                                 sum(numpy.isfinite(imputed_y) == False))
            imputed_y[numpy.isfinite(imputed_y) == False] = self.threshold

            if it > 0:
                # Calc mean difference between imputed values this and last
                # iteration, assume imputed values are always concatenated
                # after uncensored values
                change = numpy.mean(abs(imputed_y - y[uncensored_y.shape[0]:]) /
                                    y[uncensored_y.shape[0]:])

            # lower all values that are higher than threshold
            imputed_y[imputed_y >= self.threshold] = self.threshold

            self.logger.debug("Change: %f" % change)

            X = numpy.concatenate((uncensored_X, censored_X))
            y = numpy.concatenate((uncensored_y, imputed_y)).flatten()

            if change > self.change_threshold or it == 0:
                m = self._get_model(X, y)
            else:
                break

            it += 1
            if it > self.max_iter:
                break

        self.logger.info("Imputation used %d/%d iterations, last_change=%f)" %
                         (it, self.max_iter, change))

        new_ys = numpy.array(imputed_y, dtype=numpy.float)
        new_ys[new_ys >= self.threshold] = self.threshold

        if not numpy.isfinite(new_ys).all():
            self.logger.critical("Imputed values are not finite, %s" %
                                 str(imputed_y))
        return imputed_y