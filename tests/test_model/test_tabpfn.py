import numpy as np
import pytest
import torch
from ConfigSpace import (
    CategoricalHyperparameter,
    ConfigurationSpace,
    EqualsCondition,
    OrdinalHyperparameter,
    UniformFloatHyperparameter,
    UniformIntegerHyperparameter,
)

from smac import BlackBoxFacade, HyperparameterOptimizationFacade, Scenario
from smac.facade.tabpfn import TabPFNFacade
from smac.model.random_forest import RandomForest
from smac.model.tabpfn.tabpfn_model import TabPFNModel
from smac.utils.configspace import convert_configurations_to_array

__copyright__ = "Copyright 2025, Leibniz University Hanover, Institute of AI"
__license__ = "3-clause BSD"


def _get_cs(n_dimensions):
    configspace = ConfigurationSpace(seed=0)
    for i in range(n_dimensions):
        configspace.add(UniformFloatHyperparameter("x%d" % i, 0, 1))

    return configspace


# The heavier tests below run real TabPFN inference, so they should actually exercise
# GPU acceleration where one is available (CUDA on most machines, MPS on Apple Silicon)
# rather than always falling back to a slow CPU forward pass through the foundation model.
if torch.cuda.is_available():
    _ACCELERATOR_DEVICE = "cuda"
elif torch.backends.mps.is_available():
    _ACCELERATOR_DEVICE = "mps"
else:
    _ACCELERATOR_DEVICE = None

requires_accelerator = pytest.mark.skipif(
    _ACCELERATOR_DEVICE is None, reason="no GPU accelerator (CUDA or MPS) available"
)


def test_predict_wrong_X_dimensions():
    rs = np.random.RandomState(1)

    model = TabPFNModel(configspace=_get_cs(10), device=_ACCELERATOR_DEVICE)
    X = rs.rand(10)
    with pytest.raises(ValueError, match="Expected 2d array.*"):
        model.predict(X)

    X = rs.rand(10, 10, 10)
    with pytest.raises(ValueError, match="Expected 2d array.*"):
        model.predict(X)

    X = rs.rand(10, 5)
    with pytest.raises(ValueError, match="Feature mismatch: .*"):
        model.predict(X)


def test_cuda_requested_but_unavailable_raises(monkeypatch):
    """Expects
    -------
    * TabPFNModel must fail loudly (not silently fall back to CPU) if `device="cuda"`
      is requested but no CUDA device is available.
    """
    import smac.model.tabpfn.tabpfn_model as tabpfn_model

    monkeypatch.setattr(tabpfn_model.torch.cuda, "is_available", lambda: False)

    with pytest.raises(RuntimeError, match=".*CUDA.*"):
        TabPFNModel(configspace=_get_cs(3), device="cuda")


def test_cuda_available_does_not_raise(monkeypatch):
    import smac.model.tabpfn.tabpfn_model as tabpfn_model

    monkeypatch.setattr(tabpfn_model.torch.cuda, "is_available", lambda: True)
    TabPFNModel(configspace=_get_cs(3), device="cuda")


def test_mps_requested_but_unavailable_raises(monkeypatch):
    """Expects
    -------
    * Apple Silicon's MPS backend counts as a real GPU accelerator, not just CUDA --
      requesting `device="mps"` with none available must fail loudly too.
    """
    import smac.model.tabpfn.tabpfn_model as tabpfn_model

    monkeypatch.setattr(tabpfn_model.torch.backends.mps, "is_available", lambda: False)

    with pytest.raises(RuntimeError, match=".*MPS.*"):
        TabPFNModel(configspace=_get_cs(3), device="mps")


def test_mps_available_does_not_raise(monkeypatch):
    import smac.model.tabpfn.tabpfn_model as tabpfn_model

    monkeypatch.setattr(tabpfn_model.torch.backends.mps, "is_available", lambda: True)
    TabPFNModel(configspace=_get_cs(3), device="mps")


def test_auto_prefers_cuda_then_mps_then_raises(monkeypatch):
    """Expects
    -------
    * `device="auto"` resolves to CUDA if available, else MPS if available, else raises
      -- it must never silently resolve to CPU the way `tabpfn`'s own "auto" does.
    """
    import smac.model.tabpfn.tabpfn_model as tabpfn_model

    monkeypatch.setattr(tabpfn_model.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(tabpfn_model.torch.backends.mps, "is_available", lambda: False)
    assert TabPFNModel(configspace=_get_cs(3), device="auto")._device == "cuda"

    monkeypatch.setattr(tabpfn_model.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(tabpfn_model.torch.backends.mps, "is_available", lambda: True)
    assert TabPFNModel(configspace=_get_cs(3), device="auto")._device == "mps"

    monkeypatch.setattr(tabpfn_model.torch.backends.mps, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match=".*CUDA.*MPS.*"):
        TabPFNModel(configspace=_get_cs(3), device="auto")


def test_cpu_explicitly_requested_bypasses_accelerator_check(monkeypatch):
    """Expects
    -------
    * `device="cpu"` is a deliberate opt-out and never raises, regardless of whether a
      GPU accelerator happens to be available.
    """
    import smac.model.tabpfn.tabpfn_model as tabpfn_model

    monkeypatch.setattr(tabpfn_model.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(tabpfn_model.torch.backends.mps, "is_available", lambda: False)
    assert TabPFNModel(configspace=_get_cs(3), device="cpu")._device == "cpu"


def test_categorical_features_indices_reuse_get_types():
    """Expects
    -------
    * Categorical column indices are derived from `get_types()` (via `self._types`)
      rather than being recomputed independently, and match the hyperparameter order.
    """
    cs = ConfigurationSpace(seed=0)
    cs.add(UniformFloatHyperparameter("a", 0, 1))
    cs.add(CategoricalHyperparameter("b", ["x", "y", "z"]))
    cs.add(OrdinalHyperparameter("c", [0, 1, 2]))
    cs.add(CategoricalHyperparameter("d", ["cat", "dog"]))

    model = TabPFNModel(configspace=cs, device="cpu")
    assert model._categorical_features_indices == [1, 3]


def test_impute_inactive_hyperparameters_are_left_as_nan():
    """Expects
    -------
    * Phase-1 `TabPFNModel` does not impute inactive (conditional) hyperparameters --
      it relies on TabPFN's native NaN support, so `transform_configs` must leave NaNs
      untouched (unlike RandomForest's sentinel-based `_impute_inactive`).
    """
    cs = ConfigurationSpace(seed=0)
    a = CategoricalHyperparameter("a", [0, 1, 2])
    b = CategoricalHyperparameter("b", [0, 1])
    c = UniformFloatHyperparameter("c", 0, 1)
    d = OrdinalHyperparameter("d", [0, 1, 2])
    cs.add([a, b, c, d])
    cs.add([EqualsCondition(b, a, 1), EqualsCondition(c, a, 0), EqualsCondition(d, a, 2)])

    configs = cs.sample_configuration(size=50)
    config_array = convert_configurations_to_array(configs)
    assert np.isnan(config_array).any()

    model = TabPFNModel(configspace=cs, device="cpu")
    transformed = model.transformer.transform_configs(config_array)
    assert np.array_equal(np.isnan(config_array), np.isnan(transformed))


@pytest.mark.slow
@requires_accelerator
def test_predict():
    rs = np.random.RandomState(1)
    X = rs.rand(20, 10)
    Y = rs.rand(10, 1)
    model = TabPFNModel(configspace=_get_cs(10), device=_ACCELERATOR_DEVICE)
    model.train(X[:10], Y[:10])
    m_hat, v_hat = model.predict(X[10:])
    assert m_hat.shape == (10, 1)
    assert v_hat.shape == (10, 1)


@pytest.mark.slow
@requires_accelerator
def test_predict_none_covariance_matches_full_mean():
    rs = np.random.RandomState(1)
    X = rs.rand(20, 5)
    Y = rs.rand(10, 1)
    model = TabPFNModel(configspace=_get_cs(5), device=_ACCELERATOR_DEVICE)
    model.train(X[:10], Y[:10])

    mean_only, var_none = model.predict(X[10:], covariance_type=None)
    mean_full, _ = model.predict(X[10:], covariance_type="diagonal")

    assert var_none is None
    assert mean_only.shape == (10, 1)
    assert np.allclose(mean_only, mean_full, atol=1e-2)


@pytest.mark.slow
@requires_accelerator
def test_reproducibility_same_seed():
    """Expects
    -------
    * Two `TabPFNModel`s trained with the same seed on the same data produce the
      same predictions -- guards against `seed` being silently disconnected from
      the model's own randomness (this happened on the old TabPFN v2 branch).
    """
    rs = np.random.RandomState(1)
    X = rs.rand(20, 5)
    Y = rs.rand(20, 1)

    model1 = TabPFNModel(configspace=_get_cs(5), device=_ACCELERATOR_DEVICE, seed=42)
    model1.train(X, Y)
    mean1, var1 = model1.predict(X)

    model2 = TabPFNModel(configspace=_get_cs(5), device=_ACCELERATOR_DEVICE, seed=42)
    model2.train(X, Y)
    mean2, var2 = model2.predict(X)

    assert np.allclose(mean1, mean2)
    assert np.allclose(var1, var2)


def test_facade_get_config_selector_retrain_after_matches_gp():
    """Expects
    -------
    * `TabPFNFacade` retrains every trial (`retrain_after=1`), matching
      `BlackBoxFacade`'s GP cadence -- not `HyperparameterOptimizationFacade`'s
      `retrain_after=8`, which is tuned for RandomForest's much cheaper refit and
      would otherwise leave TabPFN (which, like GP, has no incremental fit) making
      decisions off a stale model for 7 out of every 8 trials.
    """
    scenario = Scenario(_get_cs(3), n_trials=10)

    assert TabPFNFacade.get_config_selector(scenario)._retrain_after == 1
    assert TabPFNFacade.get_config_selector(scenario)._retrain_after == BlackBoxFacade.get_config_selector(
        scenario
    )._retrain_after
    assert HyperparameterOptimizationFacade.get_config_selector(scenario)._retrain_after == 8


def test_facade_get_acquisition_function_is_non_log_ei():
    """Expects
    -------
    * `TabPFNFacade` inherits `BlackBoxFacade`'s `EI(log=False)` (not
      `HyperparameterOptimizationFacade`'s `EI(log=True)`) -- see `TabPFNModel`'s
      docstring for why `log=True` is not numerically robust here.
    """
    scenario = Scenario(_get_cs(3), n_trials=10)
    assert TabPFNFacade.get_acquisition_function(scenario)._log is False


def test_facade_validate_rejects_non_tabpfn_model():
    """Expects
    -------
    * `TabPFNFacade` raises if constructed with a non-`TabPFNModel` (mirroring how
      `BlackBoxFacade` rejects a non-GP model), rather than silently accepting one.
    """
    cs = _get_cs(3)
    scenario = Scenario(cs, n_trials=5)

    with pytest.raises(ValueError, match=".*TabPFNModel.*"):
        TabPFNFacade(
            scenario,
            lambda config, seed=0: 0.0,
            model=RandomForest(configspace=cs),
            overwrite=True,
        )


@pytest.mark.slow
@requires_accelerator
def test_facade_optimize_smoke(rosenbrock):
    """Expects
    -------
    * `TabPFNFacade` runs a short `optimize()` loop end to end out of the box -- no
      manually-constructed model or acquisition-function override needed -- with
      `device="auto"` resolving to a real GPU accelerator, not CPU.
    """
    scenario = Scenario(rosenbrock.configspace, n_trials=10)

    smac = TabPFNFacade(scenario, rosenbrock.train, overwrite=True)
    assert smac._model._device == _ACCELERATOR_DEVICE

    incumbent = smac.optimize()
    assert incumbent is not None
