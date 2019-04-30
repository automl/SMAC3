import numpy as np
import sklearn.gaussian_process.kernels

from smac.epm.base_epm import AbstractEPM
import smac.epm.gp_base_prior


class BaseModel(AbstractEPM):

    def __init__(self, types, bounds, seed, **kwargs):
        """
        Abstract base class for all Gaussian process models.
        """
        super().__init__(types=types, bounds=bounds, seed=seed, **kwargs)

        self.rng = np.random.RandomState(seed)

    def _normalize_y(self, y):
        self.mean_y_ = np.mean(y)
        self.std_y_ = np.std(y)
        if self.std_y_ == 0:
            self.std_y_ = 1
        return (y - self.mean_y_) / self.std_y_

    def _untransform_y(self, y, var=None):
        y = y * self.std_y_ + self.mean_y_
        if var is not None:
            var = var * self.std_y_ ** 2
            return y, var
        return y

    def _get_all_priors(self, add_bound_priors=True, add_soft_bounds=False):
        # Obtain a list of all priors for each tunable hyperparameter of the kernel
        all_priors = []
        to_visit = []
        to_visit.append(self.gp.kernel.k1)
        to_visit.append(self.gp.kernel.k2)
        while len(to_visit) > 0:
            current_param = to_visit.pop(0)
            if isinstance(current_param, sklearn.gaussian_process.kernels.KernelOperator):
                to_visit.insert(0, current_param.k1)
                to_visit.insert(1, current_param.k2)
                continue
            elif isinstance(current_param, sklearn.gaussian_process.kernels.Kernel):
                hps = current_param.hyperparameters
                assert len(hps) == 1
                hp = hps[0]
                if hp.fixed:
                    continue
                bounds = hps[0].bounds
                for i in range(hps[0].n_elements):
                    priors_for_hp = []
                    if current_param.prior is not None:
                        priors_for_hp.append(current_param.prior)
                    if add_bound_priors:
                        if add_bound_priors:
                            priors_for_hp.append(smac.epm.gp_base_prior.SoftTopHatPrior(
                               lower_bound=bounds[i][0], upper_bound=bounds[i][1], rng=self.rng,
                            ))
                        else:
                            priors_for_hp.append(smac.epm.gp_base_prior.TophatPrior(
                                lower_bound=bounds[i][0], upper_bound=bounds[i][1], rng=self.rng,
                            ))
                    all_priors.append(priors_for_hp)
        return all_priors
