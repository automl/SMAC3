from __future__ import annotations

from typing import Any, cast

import numpy as np
import torch

from smac.acquisition.function import AbstractAcquisitionFunction
from smac.model.tabPFNv2 import TabPFNModel  # <--- Adjust path as necessary


class RiemannExpectedImprovement(AbstractAcquisitionFunction):
    """Expected Improvement computed from a discrete (Riemann) predictive distribution.

    This version is designed for TabPFN/PFNs4BO models that output discrete logits
    rather than Gaussian mean/variance pairs.
    """

    @property
    def name(self) -> str:  # noqa: D102
        return "RiemannExpectedImprovement"

    def _update(self, **kwargs: Any) -> None:
        """Called after the model is fitted. Updates current best (f_best)."""
        if self.model is None:
            raise ValueError("No model attached to acquisition function.")
        assert "eta" in kwargs
        self._eta = kwargs["eta"]

    def _compute(self, X: np.ndarray) -> np.ndarray:
        """Compute Riemann-based EI for given X."""
        if self.model is None:
            raise ValueError("Model not set for acquisition function.")

        model = cast(TabPFNModel, self.model)

        # Impute, transform, scale
        X_imputed = model._x_imputer.transform(X)
        X_transformed = model._x_pt.transform(X_imputed)
        X_scaled = model._x_scaler.transform(X_transformed)

        assert model._tabpfn is not None
        with torch.no_grad():
            pred = model._tabpfn.predict(X_scaled, output_type="full")

        # change sign because TabPFN maximizes by default
        ei = pred["criterion"].ei(pred["logits"], (-1) * self._eta)
        return ei.cpu().numpy().reshape(-1, 1)
