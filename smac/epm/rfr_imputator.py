import logging
import warnings
import numpy as np
from scipy.stats import truncnorm

import smac.epm.base_imputor
from smac.epm.base_epm import AbstractEPM


__author__ = "Katharina Eggensperger"
__copyright__ = "Copyright 2015, ML4AAD"
__license__ = "3-clause BSD"
__maintainer__ = "Katharina Eggensperger"
__email__ = "eggenspk@cs.uni-freiburg.de"
__version__ = "0.0.1"


class RFRImputator(smac.epm.base_imputor.BaseImputor):

    """Imputor using pyrfr's Random Forest regressor.

    **Note:** Sets var_threshold as the lower bound on the variance for the
    predictions of the random forest

    Attributes
    ----------
    logger : logging.Logger
    max_iter : int
    change_threshold : float
    cutoff : float
    threshold : float
    seed : int
        Created by drawing random int from rng
    model : AbstractEPM
        Predictive model (i.e. RandomForestWithInstances)
    var_threshold: float

    """

    def __init__(self, rng: np.random.RandomState, cutoff: float,
                 threshold: float, model: AbstractEPM,
                 change_threshold: float=0.01,
                 max_iter: int=2):
        """Constructor

        Parameters
        ----------
        rng : np.random.RandomState
            Will be used to draw a seed (currently not used)
        cutoff : float
            Cutoff value for this scenario (upper runnning time limit)
        threshold : float
            Highest possible values (e.g. cutoff * parX).
        model : AbstractEPM
            Predictive model (i.e. RandomForestWithInstances)
        change_threshold : float
            Stop imputation if change is less than this.
        max_iter : int
            Maximum number of imputation iterations.
        """
        super(RFRImputator, self).__init__()
        self.logger = logging.getLogger(self.__module__ + "." + self.__class__.__name__)
        self.max_iter = max_iter
        self.change_threshold = change_threshold
        self.cutoff = cutoff
        self.threshold = threshold
        self.seed = rng.random_integers(low=0, high=1000)

        self.model = model

        # Never use a lower variance than this
        self.var_threshold = 10 ** -2

    def impute(self, censored_X: np.ndarray, censored_y: np.ndarray,
               uncensored_X: np.ndarray, uncensored_y: np.ndarray):
        """
        Imputes censored runs and returns new y values.

        Parameters
        ----------
        censored_X : np.ndarray [N, M]
            Feature array of all censored runs.
        censored_y : np.ndarray [N, 1]
            Target values for all runs censored runs.
        uncensored_X : np.ndarray [N, M]
            Feature array of all non-censored runs.
        uncensored_y : np.ndarray [N, 1]
            Target values for all non-censored runs.

        Returns
        ----------
        imputed_y : np.ndarray
            Same shape as censored_y [N, 1]
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
