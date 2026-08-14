from __future__ import annotations

from smac.facade.blackbox_facade import BlackBoxFacade
from smac.model.tabpfn.tabpfn_model import TabPFNModel
from smac.scenario import Scenario

__copyright__ = "Copyright 2025, Leibniz University Hanover, Institute of AI"
__license__ = "3-clause BSD"


class TabPFNFacade(BlackBoxFacade):
    """Facade using `TabPFNModel` as surrogate, for out-of-the-box comparison against
    SMAC's other surrogate models.

    This subclasses `BlackBoxFacade` (not `HyperparameterOptimizationFacade`) because
    `TabPFNModel`, like `GaussianProcess`, has no incremental fit and z-score
    normalizes targets rather than log-scaling them -- so it needs the same
    `retrain_after=1` config selector cadence, non-log-scaled runhistory encoder, and
    `EI(log=False)` acquisition function `BlackBoxFacade` already uses for GP, all
    inherited unchanged. Only `get_model` (and, since `TabPFNModel` isn't a
    `AbstractGaussianProcess` and does support instance features, `_validate`) differ
    from `BlackBoxFacade`.
    """

    def _validate(self) -> None:
        """Ensure that the SMBO configuration with all its (updated) dependencies is valid."""
        # Intentionally does not call `BlackBoxFacade._validate` -- its GP-only type check and
        # "instances are unsupported" restriction don't apply to `TabPFNModel`.
        assert self._acquisition_function == self._acquisition_maximizer._acquisition_function

        if not isinstance(self._model, TabPFNModel):
            raise ValueError("The TabPFN facade only works with `TabPFNModel`.")

    @staticmethod
    def get_model(  # type: ignore
        scenario: Scenario,
        *,
        pca_components: int | None = 7,
        normalize_y: bool = True,
        device: str = "auto",
        n_estimators: int = 8,
    ) -> TabPFNModel:
        """Returns a `TabPFNModel` surrogate.

        Parameters
        ----------
        scenario : Scenario
        pca_components : int | None, defaults to 7
            Number of components to keep when using PCA to reduce dimensionality of instance features.
        normalize_y : bool, defaults to True
            Zero mean unit variance normalization of the output values.
        device : str, defaults to "auto"
            Device passed to `TabPFNRegressor`. Resolves to the strongest available GPU
            accelerator and raises if none is available -- see `TabPFNModel`.
        n_estimators : int, defaults to 8
            Number of TabPFN ensemble members.
        """
        return TabPFNModel(
            configspace=scenario.configspace,
            instance_features=scenario.instance_features,
            pca_components=pca_components,
            normalize_y=normalize_y,
            device=device,
            n_estimators=n_estimators,
            seed=scenario.seed,
        )
