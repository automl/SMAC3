from __future__ import annotations

from typing import Any

import numpy as np
import torch
from ConfigSpace import ConfigurationSpace
from tabpfn import TabPFNRegressor

from smac.model.abstract_model import AbstractModel
from smac.model.surrogate_transformer import SurrogateTransformer
from smac.utils.logging import get_logger

__copyright__ = "Copyright 2025, Leibniz University Hanover, Institute of AI"
__license__ = "3-clause BSD"


logger = get_logger(__name__)


def _to_numpy(x: Any) -> np.ndarray:
    """Converts a torch tensor (or anything array-like) to a numpy array."""
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()

    return np.asarray(x)


# Accelerator kinds SMAC considers "GPU acceleration", each with its own torch
# availability check. CUDA (NVIDIA) and MPS (Apple Silicon) are both real GPU
# backends -- CPU is not, so it's deliberately absent from this map.
_ACCELERATOR_AVAILABLE = {
    "cuda": lambda: torch.cuda.is_available(),
    "mps": lambda: torch.backends.mps.is_available(),
}


def _resolve_device(device: str) -> str:
    """Resolves `device="auto"` to the strongest available GPU accelerator, and
    validates an explicitly requested accelerator is actually available.

    Unlike `tabpfn`'s own `device="auto"`, this never silently falls back to CPU --
    it raises instead, so callers who ask for GPU acceleration are told loudly when
    they don't have it. Pass `device="cpu"` explicitly to opt out of this check.
    """
    if device == "auto":
        for kind, is_available in _ACCELERATOR_AVAILABLE.items():
            if is_available():
                return kind

        raise RuntimeError(
            "TabPFNModel could not find an available GPU accelerator (checked CUDA and MPS). "
            "SMAC does not silently fall back to CPU -- pass device='cpu' explicitly if that's "
            "really what you want."
        )

    device_kind = device.split(":")[0]
    check = _ACCELERATOR_AVAILABLE.get(device_kind)
    if check is not None and not check():
        raise RuntimeError(
            f"TabPFNModel was configured with device={device!r} but no {device_kind.upper()} "
            "device is available. SMAC does not silently fall back to CPU -- pass device='cpu' "
            "explicitly if that's really what you want."
        )

    return device


class TabPFNModel(AbstractModel):
    """Surrogate model wrapping TabPFN v3 (`tabpfn.TabPFNRegressor`), a pretrained
    tabular foundation model, as an alternative to SMAC's RandomForest/GaussianProcess
    surrogates.

    Note
    ----
    TabPFN's model weights are released under the non-commercial TabPFN-3.0 License
    (see https://huggingface.co/Prior-Labs/tabpfn_3/blob/main/LICENSE); the `tabpfn`
    package code itself is Apache-2.0.

    Note
    ----
    TabPFN has no incremental fit -- every call to `train()` refits a fresh
    `TabPFNRegressor` on the entire (transformed) runhistory.

    Note
    ----
    Prefer `acquisition_function=EI(log=False)` over the `log=True` default most
    facades use (e.g. `HyperparameterOptimizationFacade.get_acquisition_function()`).
    `EI(log=True)`'s formula calls `exp()` on the model's (untransformed) prediction;
    a still-uncalibrated TabPFN posterior -- realistic early in optimization, with very
    few observed points -- can produce an outlier mean estimate that `exp()` then
    blows up into `inf`/`nan` (observed empirically: `RuntimeWarning: overflow
    encountered in exp`). `EI(log=False)`'s formula never exponentiates and stays
    numerically robust regardless of the model's calibration quality.

    Parameters
    ----------
    configspace : ConfigurationSpace
    instance_features : dict[str, list[int | float]] | None, defaults to None
        Features (list of int or floats) of the instances (str). The features are incorporated into the X data,
        on which the model is trained on.
    pca_components : int | None, defaults to 7
        Number of components to keep when using PCA to reduce dimensionality of instance features.
    seed : int
    normalize_y : bool, defaults to True
        Zero mean unit variance normalization of the output values.
    device : str, defaults to "auto"
        Device passed to `TabPFNRegressor`. Unlike `tabpfn`'s own `device="auto"`, this
        resolves to the strongest available GPU accelerator (CUDA, then MPS) and raises
        a `RuntimeError` if neither is available -- SMAC will not silently fall back to
        CPU. Pass `device="cpu"` explicitly if that's really what you want, or a specific
        accelerator (e.g. `"cuda"`, `"mps"`) to require it and fail loudly if unavailable.
    n_estimators : int, defaults to 8
        Number of TabPFN ensemble members, passed through to `TabPFNRegressor`.
    """

    def __init__(
        self,
        configspace: ConfigurationSpace,
        instance_features: dict[str, list[int | float]] | None = None,
        pca_components: int | None = 7,
        seed: int = 0,
        normalize_y: bool = True,
        device: str = "auto",
        n_estimators: int = 8,
    ) -> None:
        super().__init__(
            configspace=configspace,
            instance_features=instance_features,
            pca_components=pca_components,
            seed=seed,
        )

        self._device = _resolve_device(device)
        self._n_estimators = n_estimators

        # Columns with a nonzero type are exactly the categorical hyperparameter columns,
        # per the convention `get_types()` already establishes (smac/utils/configspace.py).
        # Instance-feature columns always have type 0. Reusing this instead of
        # recomputing categorical membership independently.
        self._categorical_features_indices = [i for i, t in enumerate(self._types) if t > 0]

        self.transformer = self.build_transformer(normalize_y)

    @property
    def meta(self) -> dict[str, Any]:  # noqa: D102
        meta = super().meta
        meta.update({"device": self._device, "n_estimators": self._n_estimators})

        return meta

    def build_transformer(self, normalize_y: bool = False) -> SurrogateTransformer:  # noqa: D102
        return SurrogateTransformer(
            n_hps=self._n_hps,
            n_features=self._n_features,
            instance_features=self._instance_features,
            impute_inactive=None,
            normalize_y=normalize_y,
            pca_components=self._pca_components,
        )

    def _get_tabpfn(self) -> TabPFNRegressor:
        return TabPFNRegressor(
            categorical_features_indices=self._categorical_features_indices,
            device=self._device,
            random_state=self._seed,
            model_path="auto",
            n_estimators=self._n_estimators,
        )

    def _train(self, X: np.ndarray, Y: np.ndarray) -> "TabPFNModel":
        self._tabpfn = self._get_tabpfn()
        self._tabpfn.fit(X, Y.ravel())

        return self

    def _predict(
        self,
        X: np.ndarray,
        covariance_type: str | None = "diagonal",
    ) -> tuple[np.ndarray, np.ndarray | None]:
        if covariance_type not in (None, "diagonal"):
            raise ValueError("`covariance_type` can only be None or `diagonal` for TabPFNModel.")

        if not hasattr(self, "_tabpfn"):
            raise Exception("Model has to be trained first!")

        if covariance_type is None:
            mean = _to_numpy(self._tabpfn.predict(X, output_type="mean")).reshape(-1)
            mean = self.transformer.untransform_y(mean)

            return mean, None

        output = self._tabpfn.predict(X, output_type="full")
        criterion, logits = output["criterion"], output["logits"]

        mean = _to_numpy(criterion.mean(logits)).reshape(-1)
        var = _to_numpy(criterion.variance(logits)).reshape(-1)
        var = np.clip(var, self._var_threshold, np.inf)

        mean, var = self.transformer.untransform_y(mean, var)

        return mean, var
