import numpy as np


class BaseModel(object):

    def __init__(self):
        """
        Abstract base class for all models
        """
        self.X = None
        self.y = None

    def train(self, X: np.ndarray, y: np.ndarray):
        """
        Trains the model on the provided data.

        Parameters
        ----------
        X: np.ndarray (N, D)
            Input data points. The dimensionality of X is (N, D),
            with N as the number of points and D is the number of input dimensions.
        y: np.ndarray (N,)
            The corresponding target values of the input data points.
        """
        raise NotImplementedError()

    def update(self, X: np.ndarray, y: np.ndarray):
        """
        Update the model with the new additional data. Override this function if your
        model allows to do something smarter than simple retraining

        Parameters
        ----------
        X: np.ndarray (N, D)
            Input data points. The dimensionality of X is (N, D),
            with N as the number of points and D is the number of input dimensions.
        y: np.ndarray (N,)
            The corresponding target values of the input data points.
        """
        X = np.append(self.X, X, axis=0)
        y = np.append(self.y, y, axis=0)
        self.train(X, y)

    def predict(self, X_test: np.ndarray):
        """
        Predicts for a given set of test data points the mean and variance of its target values

        Parameters
        ----------
        X_test: np.ndarray (N, D)
            N Test data points with input dimensions D

        Returns
        ----------
        mean: ndarray (N,)
            Predictive mean of the test data points
        var: ndarray (N,)
            Predictive variance of the test data points
        """
        raise NotImplementedError()

    def predict_marginalized_over_instances(self, X_test: np.ndarray, **kwargs):
        """Predict mean and variance marginalized over all instances.

                Returns the predictive mean and variance marginalised over all
                instances for a set of configurations.

                Parameters
                ----------
                X : np.ndarray
                    [n_samples, n_features (config)]

                Returns
                -------
                means : np.ndarray of shape = [n_samples, 1]
                    Predictive mean
                vars : np.ndarray  of shape = [n_samples, 1]
                    Predictive variance
        """
        m, v = self.predict(X_test, **kwargs)
        return m.reshape(-1, 1), v.reshape(-1, 1)

    def _check_shapes_train(func):
        def func_wrapper(self, X, y, *args, **kwargs):
            assert X.shape[0] == y.shape[0]
            assert len(X.shape) == 2
            y = y.flatten()
            assert len(y.shape) == 1
            return func(self, X, y, *args, **kwargs)
        return func_wrapper

    def _check_shapes_predict(func):
        def func_wrapper(self, X, *args, **kwargs):
            assert len(X.shape) == 2
            return func(self, X, *args, **kwargs)

        return func_wrapper

    def get_json_data(self):
        """
        Json getter function'

        Returns
        ----------
            dictionary
        """
        json_data = {'X': self.X if self.X is None else self.X.tolist(),
                     'y': self.y if self.y is None else self.y.tolist(),
                     'hyperparameters': ""}
        return json_data

    def get_incumbent(self):
        """
        Returns the best observed point and its function value

        Returns
        ----------
        incumbent: ndarray (D,)
            current incumbent
        incumbent_value: ndarray (N,)
            the observed value of the incumbent
        """
        best_idx = np.argmin(self.y)
        return self.X[best_idx], self.y[best_idx]
