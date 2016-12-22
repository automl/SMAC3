import logging
import warnings
import numpy as np
from scipy.stats import truncnorm

import smac.epm.base_imputor

from ConfigSpace.hyperparameters import UniformIntegerHyperparameter, \
    CategoricalHyperparameter, UniformFloatHyperparameter

__author__ = "Katharina Eggensperger"
__copyright__ = "Copyright 2015, ML4AAD"
__license__ = "3-clause BSD"
__maintainer__ = "Katharina Eggensperger"
__email__ = "eggenspk@cs.uni-freiburg.de"
__version__ = "0.0.1"


class RFRImputator(smac.epm.base_imputor.BaseImputor):

    """Uses an rfr to do imputation"""

    def __init__(self, rs, cutoff, threshold,
                 model,
                 change_threshold=0.01,
                 max_iter=2):
        """
        initialize imputator module

        Parameters
        ----------
        rs : random state generator
        cutoff : float
            cutoff value used for this scenario
        threshold : float
            highest possible values (e.g. cutoff * par)
        model:
            epm model (i.e. RandomForestWithInstances)
        change_threshold : float 
            stop imputation if change is less than this
        max_iter : maximum number of iteration            
        -------
        """

        super(RFRImputator, self).__init__()
        self.logger = logging.getLogger("RFRImputor")
        self.max_iter = max_iter
        self.change_threshold = change_threshold
        self.cutoff = cutoff
        self.threshold = threshold
        self.seed = rs.random_integers(low=0, high=1000)

        self.model = model

        # Never use a lower variance than this
        self.var_threshold = 10 ** -2

    def impute(self, censored_X, censored_y, uncensored_X, uncensored_y):
        """
        impute runs and returns imputed y values

        Parameters
        ----------
        censored_X : array
            X matrix of censored data
        censored_y : array
            y matrix of censored data
        uncensored_X : array
            X matrix of uncensored data
        uncensored_y : array
            y matrix of uncensored data
        """
        if censored_X.shape[0] == 0:
            self.logger.critical("Nothing to impute, return None")
            return None

        censored_y = censored_y.flatten()
        uncensored_y = uncensored_y.flatten()

        # first learn model without censored data
        self.model.train(uncensored_X, uncensored_y)

        self.logger.debug("Going to impute %d y-values with %s" %
                          (censored_X.shape[0], str(self.model)))

        imputed_y = None  # define this, if imputation fails

        # Define variables
        y = None

        it = 1
        change = 0

        while True:
            self.logger.debug("Iteration %d of %d" % (it, self.max_iter))

            # predict censored y values
            y_mean, y_var = self.model.predict(censored_X)
            y_var[y_var < self.var_threshold] = self.var_threshold
            y_stdev = np.sqrt(y_var)[:, 0]
            y_mean = y_mean[:, 0]

            # ignore the warnings of truncnorm.stats
            # since we handle them appropriately
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    'ignore', r'invalid value encountered in (subtract|true_divide).*')
                warnings.filterwarnings(
                    'ignore', r'divide by zero encountered in (true_divide|log).*')
                imputed_y = truncnorm.stats(a=(censored_y - y_mean) / y_stdev,
                                            b=(self.threshold - y_mean) /
                                            y_stdev,
                                            loc=y_mean,
                                            scale=y_stdev,
                                            moments='m')

            imputed_y = np.array(imputed_y)

            nans = ~np.isfinite(imputed_y)
            n_nans = sum(nans)
            if n_nans > 0:
                # Replace all nans with maximum of predicted perf and censored value
                # this happens if the prediction is far smaller than the
                # censored data point
                self.logger.debug("Going to replace %d nan-value(s) with "
                                  "max(captime, predicted mean)" % n_nans)
                imputed_y[nans] = np.max(
                    [censored_y[nans], y_mean[nans]], axis=0)

            if it > 1:
                # Calc mean difference between imputed values this and last
                # iteration, assume imputed values are always concatenated
                # after uncensored values

                change = np.mean(np.abs(imputed_y -
                                     y[uncensored_y.shape[0]:]) /
                                 y[uncensored_y.shape[0]:])

            # lower all values that are higher than threshold
            # should probably never happen
            imputed_y[imputed_y >= self.threshold] = self.threshold

            self.logger.debug("Change: %f" % change)

            X = np.concatenate((uncensored_X, censored_X))
            y = np.concatenate((uncensored_y, imputed_y))

            if change > self.change_threshold or it == 1:
                self.model.train(X, y)
            else:
                break

            it += 1
            if it > self.max_iter:
                break

        self.logger.debug("Imputation used %d/%d iterations, last_change=%f" %
                          (it - 1, self.max_iter, change))

        # replace all y > cutoff with PAR10 values (i.e., threshold)
        imputed_y = np.array(imputed_y, dtype=np.float)
        imputed_y[imputed_y >= self.cutoff] = self.threshold

        if not np.isfinite(imputed_y).all():
            self.logger.critical("Imputed values are not finite, %s" %
                                 str(imputed_y))
        return np.reshape(imputed_y, [imputed_y.shape[0], 1])
