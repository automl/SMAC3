import copy as copy_module

import numpy as np
import scipy.sparse


def get_random_neighbor(configuration, copy=True):
    """Change one of the active parameters to a neighbor value.

    Neighbor values are defined as in Hutter et al. 2011:
    * Categorical: Each choice is a neighbor
    * Real: Rescale range to [0, 1] and sample four neighbors from a
      univariate Gaussian distribution with mean v and standard deviation 0.2
    * Integer: Same as for real
    * Ordinal: The ordinal value above or below.

    Parameters
    ----------
    configuration : HPOlibConfigSpace.configuration_space.Configuration

    copy : boolaen (default=True)
        Return a copy if True, otherwise change the configuration.
    """
    if copy:
        configuration = copy_module.deepcopy(configuration)

    # TODO change exactly one parameter as defined above

    return configuration


class OneHotEncoder(object):
    def __init__(self):
        """Perform one hot (a.k.a. 1-out-of-k) encoding for categorical
        parameters."""

    def fit(self, configurations, y=None):
        """Compability to scikit-learn."""
        return self

    def transform(self, configurations):
        """Transform configurations using one-hot encoding.

        Parameters
        ----------
        configurations : List of HPOlibConfigSpace.configuration_space.
                         Configuration

        Returns
        -------
        X : scipy.sparse matrix
        """
        np.random.seed(1)
        array = np.random.randint(0, 2, (100, 100))
        matrix = scipy.sparse.csr_matrix(array)
        return matrix


class Imputer(object):
    def __init__(self, strategy='default'):
        """Impute inactive parameters.

        Parameters
        ----------
        strategy : string, optional (default='default')
            The imputation strategy.

            - If 'default', replace inactive parameters by their default.
            - If 'outlier', replace inactive parameters by an outlier value
              which can be splitted apart by a tree-based model.
        """

    def fit(self, configurations, y=None):
        """Compability to scikit-learn."""
        return self

    def transform(self, configurations):
        """Impute inactive values of the configurations.

        Parameters
        ----------
        configurations : List of HPOlibConfigSpace.configuration_space.
                         Configuration

        Returns
        -------
        X : np.ndarray
        """
