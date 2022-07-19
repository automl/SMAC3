import numpy as np

from smac.acquisition_function import AbstractAcquisitionFunction


class TS(AbstractAcquisitionFunction):
    def __init__(self, par: float = 0.0):
        r"""Do a Thompson Sampling for a given x over the best so far value as
        acquisition value.

        Warning
        -------
        Thompson Sampling can only be used together with
        smac.optimizer.ei_optimization.RandomSearch, please do not use
        smac.optimizer.ei_optimization.LocalAndSortedRandomSearch to optimize TS
        acquisition function!

        :math:`TS(X) ~ \mathcal{N}(\mu(\mathbf{X}),\sigma(\mathbf{X}))'
        Returns -TS(X) as the acquisition_function optimizer maximizes the acquisition value.

        Parameters
        ----------
        model : BaseEPM
            A model that implements at least
                 - predict_marginalized_over_instances(X)
        par : float, default=0.0
            TS does not require par here, we only wants to make it consistent with
            other acquisition functions.
        """
        super(TS, self).__init__()
        self.long_name = "Thompson Sampling"
        self.par = par
        self.num_data = None
        self._required_updates = ("model",)

    def _compute(self, X: np.ndarray) -> np.ndarray:
        """Sample a new value from a gaussian distribution whose mean and covariance values
        are given by model.

        Parameters
        ----------
        X: np.ndarray(N, D)
           Points to be evaluated where we could sample a value. N is the number of points and D the dimension
           for the points

        Returns
        -------
        np.ndarray(N,1)
            negative sample value of X
        """
        assert self.model

        if len(X.shape) == 1:
            X = X[:, np.newaxis]
        sample_function = getattr(self.model, "sample_functions", None)
        if callable(sample_function):
            return -sample_function(X, n_funcs=1)

        m, var_ = self.model.predict_marginalized_over_instances(X)
        rng = getattr(self.model, "rng", np.random.RandomState(self.model.seed))
        m = m.flatten()
        var_ = np.diag(var_.flatten())
        return -rng.multivariate_normal(m, var_, 1).T
