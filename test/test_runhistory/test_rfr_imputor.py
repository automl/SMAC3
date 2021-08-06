import unittest
import unittest.mock
import logging
import numpy

from ConfigSpace import Configuration, ConfigurationSpace
from ConfigSpace.hyperparameters import UniformIntegerHyperparameter, \
    CategoricalHyperparameter, UniformFloatHyperparameter

from smac.tae import StatusType
from smac.runhistory import runhistory, runhistory2epm
from smac.scenario import scenario
from smac.epm import rfr_imputator
from smac.epm.rf_with_instances import RandomForestWithInstances
from smac.epm.util_funcs import get_types

__copyright__ = "Copyright 2021, AutoML.org Freiburg-Hannover"
__license__ = "3-clause BSD"


def generate_config(cs, rs):
    i = rs.randint(-10, 10)
    f = rs.rand(1)[0]
    seed = rs.randint(0, 10000)

    # 'a' occurs more often than 'b'
    c = 'a' if rs.binomial(1, 0.2) == 0 else 'b'

    # We have 100 instance, but prefer the middle ones
    instance_id = int(rs.normal(loc=50, scale=20, size=1)[0])
    instance_id = min(max(0, instance_id), 100)

    status = StatusType.SUCCESS
    runtime = 10**(numpy.sin(i) + f) + seed / 10000 - numpy.sin(instance_id)

    if runtime > 40:
        status = StatusType.TIMEOUT
        runtime = 40
    elif instance_id > 50 and runtime > 15:
        # This is a timeout with probability 0.5
        status = StatusType.TIMEOUT
        runtime /= 2.0

    config = Configuration(cs, values={'cat_a_b': c, 'float_0_1': f,
                                       'integer_0_100': i})

    return config, seed, runtime, status, instance_id


class Scen(scenario.Scenario):
    """
    DUMMY class to fake scenario
    """

    def __init__(self):
        self.run_obj = None
        self.overall_obj = None
        self.cutoff = None
        self.feature_dict = None
        self.n_features = 0
        self.par_factor = 1


class ImputorTest(unittest.TestCase):

    def setUp(self):
        logging.basicConfig(level=logging.DEBUG)

    def get_model(self, cs, instance_features=None):
        if instance_features:
            instance_features = numpy.array([instance_features[key] for key in instance_features])
        types, bounds = get_types(cs, instance_features)
        model = RandomForestWithInstances(
            configspace=cs,
            types=types,
            bounds=bounds,
            instance_features=instance_features,
            seed=1234567980,
            pca_components=7,
        )
        return model

    def get_runhistory(self, num_success, num_capped, num_timeout):
        cs = ConfigurationSpace()
        cs.add_hyperparameter(CategoricalHyperparameter(name="cat_a_b", choices=["a", "b"], default_value="a"))
        cs.add_hyperparameter(UniformFloatHyperparameter(name="float_0_1", lower=0, upper=1, default_value=0.5))
        cs.add_hyperparameter(UniformIntegerHyperparameter(name='integer_0_100',
                                                           lower=-10, upper=10, default_value=0))

        rh = runhistory.RunHistory()
        rs = numpy.random.RandomState(1)
        successes = 0
        capped = 0
        timeouts = 0
        while successes < num_success or capped < num_capped or timeouts < num_timeout:
            config, seed, runtime, status, instance_id = \
                generate_config(cs=cs, rs=rs)
            if status == StatusType.SUCCESS and successes < num_success:
                successes += 1
                add = True
            elif status == StatusType.TIMEOUT:
                if runtime < 40 and capped < num_capped:
                    capped += 1
                    add = True
                elif runtime == 40 and timeouts < num_timeout:
                    timeouts += 1
                    add = True
                else:
                    add = False
            else:
                add = False

            if add:
                rh.add(config=config, cost=runtime, time=runtime,
                       status=status, instance_id=instance_id,
                       seed=seed, additional_info=None)
        return cs, rh

    def get_scenario(self, instance_features=None):
        scen = Scen()
        scen.run_obj = "runtime"
        scen.overall_obj = "par10"
        scen.cutoff = 40
        if instance_features:
            scen.feature_dict = instance_features
            scen.n_features = len(list(instance_features.values())[0])
        return scen

    def testRandomImputation(self):
        rs = numpy.random.RandomState(1)

        for i in range(0, 150, 15):
            # First random imputation sanity check
            num_samples = max(1, i * 10)
            num_feat = max(1, i)
            num_censored = int(num_samples * 0.1)
            X = rs.rand(num_samples, num_feat)
            y = numpy.sin(X[:, 0:1])

            cutoff = max(y) * 0.9
            y[y > cutoff] = cutoff

            # We have some cen data
            cen_X = X[:num_censored, :]
            cen_y = y[:num_censored]
            uncen_X = X[num_censored:, :]
            uncen_y = y[num_censored:]

            cen_y /= 2

            cs = ConfigurationSpace()
            for i in range(num_feat):
                cs.add_hyperparameter(UniformFloatHyperparameter(name="a_%d" % i,
                                                                 lower=0, upper=1, default_value=0.5))

            imputor = rfr_imputator.RFRImputator(rng=rs,
                                                 cutoff=cutoff,
                                                 threshold=cutoff * 10,
                                                 change_threshold=0.01,
                                                 max_iter=5,
                                                 model=self.get_model(cs))

            imp_y = imputor.impute(censored_X=cen_X, censored_y=cen_y,
                                   uncensored_X=uncen_X,
                                   uncensored_y=uncen_y)

            if imp_y is None:
                continue

            for idx in range(cen_y.shape[0]):
                self.assertGreater(imp_y[idx], cen_y[idx])
            self.assertTrue(numpy.isfinite(imp_y).all())

    def testRealImputation(self):

        # Without instance features
        rs = numpy.random.RandomState(1)

        cs, rh = self.get_runhistory(num_success=5, num_timeout=1, num_capped=2)

        scen = self.get_scenario()
        model = self.get_model(cs)
        imputor = rfr_imputator.RFRImputator(rng=rs,
                                             cutoff=scen.cutoff,
                                             threshold=scen.cutoff * 10,
                                             change_threshold=0.01, max_iter=10,
                                             model=model)

        r2e = runhistory2epm.RunHistory2EPM4LogCost(
            scenario=scen, num_params=3,
            success_states=[StatusType.SUCCESS, ],
            impute_censored_data=True, impute_state=[StatusType.TIMEOUT],
            imputor=imputor, rng=rs,
        )

        self.assertEqual(r2e.transform(rh)[1].shape, (8, 1))
        self.assertEqual(r2e.transform(rh)[1].shape, (8, 1))

        # Now with instance features
        instance_features = {run_key.instance_id: numpy.random.rand(10) for run_key in rh.data}
        scen = self.get_scenario(instance_features)
        model = self.get_model(cs, instance_features)

        with unittest.mock.patch.object(model, attribute='train', wraps=model.train) as train_wrapper:
            imputor = rfr_imputator.RFRImputator(rng=rs,
                                                 cutoff=scen.cutoff,
                                                 threshold=scen.cutoff * 10,
                                                 change_threshold=0.01, max_iter=10,
                                                 model=model)
            r2e = runhistory2epm.RunHistory2EPM4LogCost(
                scenario=scen, num_params=3,
                success_states=[StatusType.SUCCESS, ],
                impute_censored_data=True, impute_state=[StatusType.TIMEOUT],
                imputor=imputor, rng=rs,
            )
            X, y = r2e.transform(rh)
            self.assertEqual(X.shape, (8, 13))
            self.assertEqual(y.shape, (8, 1))
            num_calls = len(train_wrapper.call_args_list)
            self.assertGreater(num_calls, 1)
            self.assertEqual(train_wrapper.call_args_list[0][0][0].shape, (5, 13))
            self.assertEqual(train_wrapper.call_args_list[1][0][0].shape, (8, 13))

            X, y = r2e.transform(rh)
            self.assertEqual(X.shape, (8, 13))
            self.assertEqual(y.shape, (8, 1))
            self.assertGreater(len(train_wrapper.call_args_list), num_calls + 1)
            self.assertEqual(train_wrapper.call_args_list[num_calls][0][0].shape, (5, 13))
            self.assertEqual(train_wrapper.call_args_list[num_calls + 1][0][0].shape, (8, 13))
