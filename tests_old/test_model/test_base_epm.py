import unittest
import unittest.mock

import numpy as np
from ConfigSpace import UniformFloatHyperparameter

import smac
import smac.configspace
from smac.model.abstract_model import AbstractModel
from smac.model.utils import get_types

__copyright__ = "Copyright 2021, AutoML.org Freiburg-Hannover"
__license__ = "3-clause BSD"


class TestRFWithInstances(unittest.TestCase):
    def _get_cs(self, n_dimensions):
        configspace = smac.configspace.ConfigurationSpace()
        for i in range(n_dimensions):
            configspace.add_hyperparameter(UniformFloatHyperparameter("x%d" % i, 0, 1))
        return configspace

    def test_apply_pca(self):
        cs = self._get_cs(5)
        instance_features = np.array([np.random.rand(10) for _ in range(5)])
        types, bounds = get_types(cs, instance_features)

        def get_X_y(num_samples, num_instance_features):
            X = smac.configspace.convert_configurations_to_array(cs.sample_configuration(num_samples))
            if num_instance_features:
                X_inst = np.random.rand(num_samples, num_instance_features)
                X = np.hstack((X, X_inst))
            y = np.random.rand(num_samples)
            return X, y

        with unittest.mock.patch.object(AbstractModel, "_train"):
            with unittest.mock.patch.object(AbstractModel, "_predict") as predict_mock:

                predict_mock.side_effect = lambda x, _: (x, x)

                epm = AbstractModel(
                    configspace=cs,
                    types=types,
                    bounds=bounds,
                    seed=1,
                    pca_components=7,
                    instance_features=instance_features,
                )

                X, y = get_X_y(5, 10)
                epm.train(X, y)
                self.assertFalse(epm._apply_pca)
                X_test, _ = get_X_y(5, None)
                epm.predict_marginalized_over_instances(X_test)

                # more data points than pca components
                X, y = get_X_y(8, 10)
                epm.train(X, y)
                self.assertTrue(epm._apply_pca)
                X_test, _ = get_X_y(5, None)
                epm.predict_marginalized_over_instances(X_test)

                # and less again - this ensures that the types array inside the epm is reverted
                # and the pca is disabled again
                X, y = get_X_y(5, 10)
                epm.train(X, y)
                self.assertFalse(epm._apply_pca)
                X_test, _ = get_X_y(5, None)
                epm.predict_marginalized_over_instances(X_test)
