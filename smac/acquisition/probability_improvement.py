import numpy as np
from scipy.stats import norm
from smac.acquisition import AbstractAcquisitionFunction


class PI(AbstractAcquisitionFunction):
    def __init__(self, par: float = 0.0):
        r"""Computes the probability of improvement for a given x over the best so far value as acquisition value.

        :math:`P(f_{t+1}(\mathbf{X})\geq f(\mathbf{X^+}))` :math:`:= \Phi(\\frac{ \mu(\mathbf{X})-f(\mathbf{X^+}) }
        { \sigma(\mathbf{X}) })` with :math:`f(X^+)` as the incumbent and :math:`\Phi` the cdf of the standard normal

        Parameters
        ----------
        model : BaseEPM
            A model that implements at least
                 - predict_marginalized_over_instances(X)
        par : float, default=0.0
            Controls the balance between exploration and exploitation of the
            acquisition function.
        """
        super(PI, self).__init__()
        self.long_name = "Probability of Improvement"
        self.par = par
        self.eta = None
        self._required_updates = ("model", "eta")

    def _compute(self, X: np.ndarray) -> np.ndarray:
        """Computes the PI value.

        Parameters
        ----------
        X: np.ndarray(N, D)
           Points to evaluate PI. N is the number of points and D the dimension for the points

        Returns
        -------
        np.ndarray(N,1)
            Expected Improvement of X
        """
        if self.eta is None:
            raise ValueError(
                "No current best specified. Call update("
                "eta=<float>) to inform the acquisition function "
                "about the current best value."
            )

        if len(X.shape) == 1:
            X = X[:, np.newaxis]
        m, var_ = self.model.predict_marginalized_over_instances(X)
        std = np.sqrt(var_)

        return norm.cdf((self.eta - m - self.par) / std)
