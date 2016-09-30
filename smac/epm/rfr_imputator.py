import logging
import numpy
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
                 max_iter=10):
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

        # first learn model without censored data
        self.model.train(uncensored_X, uncensored_y)

        self.logger.debug("Going to impute %d y-values with %s" %
                          (censored_X.shape[0], str(self.model)))

        imputed_y = None  # define this, if imputation fails

        # Define variables
        y = None

        it = 0
        change = 0
        while True:
            self.logger.debug("Iteration %d of %d" % (it, self.max_iter))

            # predict censored y values
            y_mean, y_var = self.model.predict(censored_X)
            y_var[y_var < self.var_threshold] = self.var_threshold
            y_stdev = numpy.sqrt(y_var)

            imputed_y = \
                [truncnorm.stats(a=(censored_y[index] -
                                    y_mean[index]) / y_stdev[index],
                                 b=(self.cutoff * 10 - y_mean[index]) /
                                 y_stdev[index],
                                 loc=y_mean[index],
                                 scale=y_stdev[index],
                                 moments='m')
                 for index in range(len(censored_y))]
            imputed_y = numpy.array(imputed_y)

            if sum(numpy.isfinite(imputed_y) == False) > 0:
                # Replace all nans with threshold
                self.logger.debug("Going to replace %d nan-value(s) with "
                                  "threshold" %
                                  sum(numpy.isfinite(imputed_y) == False))
                imputed_y[numpy.isfinite(imputed_y) == False] = self.threshold

            if it > 0:
                # Calc mean difference between imputed values this and last
                # iteration, assume imputed values are always concatenated
                # after uncensored values

                change = numpy.mean(abs(imputed_y -
                                        y[uncensored_y.shape[0]:]) /
                                    y[uncensored_y.shape[0]:])

            # lower all values that are higher than threshold
            imputed_y[imputed_y >= self.threshold] = self.threshold

            self.logger.debug("Change: %f" % change)

            X = numpy.concatenate((uncensored_X, censored_X))
            y = numpy.concatenate((uncensored_y, imputed_y))

            if change > self.change_threshold or it == 0:
                self.model.train(X, y)
            else:
                break

            it += 1
            if it > self.max_iter:
                break

        self.logger.debug("Imputation used %d/%d iterations, last_change=%f" %
                          (it, self.max_iter, change))

        imputed_y = numpy.array(imputed_y, dtype=numpy.float)
        imputed_y[imputed_y >= self.threshold] = self.threshold

        if not numpy.isfinite(imputed_y).all():
            self.logger.critical("Imputed values are not finite, %s" %
                                 str(imputed_y))
        return imputed_y
