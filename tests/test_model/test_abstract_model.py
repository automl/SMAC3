import numpy as np
import pytest

from smac.model.abstract_model import AbstractModel
from smac.utils.configspace import convert_configurations_to_array

__copyright__ = "Copyright 2021, AutoML.org Freiburg-Hannover"
__license__ = "3-clause BSD"


def get_X_y(cs, n_samples, n_instance_features):
    X = convert_configurations_to_array(cs.sample_configuration(n_samples))

    if n_instance_features is not None and n_instance_features > 0:
        X_inst = np.random.rand(n_samples, n_instance_features)
        X = np.hstack((X, X_inst))

    y = np.random.rand(n_samples)

    return X, y


def _train(X, Y):
    return None


def test_no_pca(configspace_small, make_scenario):
    n_instances = 100
    n_instance_features = 10
    n_samples = 5

    scenario = make_scenario(
        configspace_small,
        use_instances=True,
        n_instances=n_instances,
        n_instance_features=n_instance_features,
    )
    model = AbstractModel(configspace_small, scenario.instance_features, pca_components=7)
    # We just overwrite the function as mock here
    model._train = _train

    # No PCA
    X, y = get_X_y(configspace_small, n_samples, n_instance_features)
    model.train(X, y)
    assert not model._apply_pca

    X, y = get_X_y(configspace_small, n_samples, n_instance_features + 1)
    with pytest.raises(ValueError, match="Feature mismatch.*"):
        model.train(X, y)

    X_test, _ = get_X_y(configspace_small, n_samples, None)
    with pytest.raises(NotImplementedError):
        model.predict_marginalized(X_test)

    X_test, _ = get_X_y(configspace_small, n_samples, 10)
    with pytest.raises(ValueError, match="Feature mismatch.*"):
        model.predict_marginalized(X_test)


def test_pca(configspace_small, make_scenario):
    n_instances = 100
    n_instance_features = 10
    n_samples = 155

    scenario = make_scenario(
        configspace_small,
        use_instances=True,
        n_instances=n_instances,
        n_instance_features=n_instance_features,
    )
    model = AbstractModel(configspace_small, scenario.instance_features, pca_components=7)
    # We just overwrite the function as mock here
    model._train = _train

    # PCA
    X, y = get_X_y(configspace_small, n_samples, n_instance_features)
    model.train(X, y)
    assert model._apply_pca

    X, y = get_X_y(configspace_small, n_samples, n_instance_features + 1)
    with pytest.raises(ValueError, match="Feature mismatch.*"):
        model.train(X, y)

    X_test, _ = get_X_y(configspace_small, n_samples, None)
    with pytest.raises(NotImplementedError):
        model.predict_marginalized(X_test)

    X_test, _ = get_X_y(configspace_small, n_samples, 10)
    with pytest.raises(ValueError, match="Feature mismatch.*"):
        model.predict_marginalized(X_test)
