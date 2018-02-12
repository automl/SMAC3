import copy
import unittest
import logging
from nose.plugins.attrib import attr
import numpy

from ConfigSpace import Configuration, ConfigurationSpace
from ConfigSpace.hyperparameters import UniformIntegerHyperparameter, \
    CategoricalHyperparameter, UniformFloatHyperparameter
from ConfigSpace.conditions import InCondition

from pyrfr.regression import binary_rss_forest

from smac.tae.execute_ta_run import StatusType
from smac.runhistory import runhistory, runhistory2epm
from smac.scenario import scenario
from smac.epm import rfr_imputator
from smac.epm.rf_with_instances import RandomForestWithInstances
from smac.optimizer.objective import average_cost
from smac.utils.util_funcs import get_types


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
    runtime = 10**(numpy.sin(i)+f) + seed/10000 - numpy.sin(instance_id)

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
        self.cs = ConfigurationSpace()
        self.cs.add_hyperparameter(CategoricalHyperparameter(
                name="cat_a_b", choices=["a", "b"], default_value="a"))
        self.cs.add_hyperparameter(UniformFloatHyperparameter(
                name="float_0_1", lower=0, upper=1, default_value=0.5))
        self.cs.add_hyperparameter(UniformIntegerHyperparameter(
                name='integer_0_100', lower=-10, upper=10, default_value=0))

        self.rh = runhistory.RunHistory(aggregate_func=average_cost)
        rs = numpy.random.RandomState(1)
        to_count = 0
        cn_count = 0
        for i in range(500):
            config, seed, runtime, status, instance_id = \
                generate_config(cs=self.cs, rs=rs)
            if runtime == 40:
                to_count += 1
            if runtime < 40 and status == StatusType.TIMEOUT:
                cn_count += 1
            self.rh.add(config=config, cost=runtime, time=runtime,
                        status=status, instance_id=instance_id,
                        seed=seed, additional_info=None)
        print("%d TIMEOUTs, %d censored" % (to_count, cn_count))

        self.scen = Scen()
        self.scen.run_obj = "runtime"
        self.scen.overall_obj = "par10"
        self.scen.cutoff = 40

        types, bounds = get_types(self.cs, None)
        self.model = RandomForestWithInstances(
                types=types, bounds=bounds,
                instance_features=None, seed=1234567980)

    @attr('slow')
    def testRandomImputation(self):
        rs = numpy.random.RandomState(1)

        for i in range(0, 150, 15):
            # First random imputation sanity check
            num_samples = max(1, i*10)
            num_feat = max(1, i)
            num_censored = int(num_samples*0.1)
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
                cs.add_hyperparameter(UniformFloatHyperparameter(
                    name="a_%d" % i, lower=0, upper=1, default_value=0.5)
                )

            types, bounds = get_types(cs, None)
            print(types)
            print(bounds)
            print('#'*120)
            print(cen_X)
            print(uncen_X)
            print('~'*120)
            self.model = RandomForestWithInstances(types=types, bounds=bounds,
                                                   instance_features=None,
                                                   seed=1234567980)
            imputor = rfr_imputator.RFRImputator(rng=rs,
                                                 cutoff=cutoff,
                                                 threshold=cutoff*10,
                                                 change_threshold=0.01,
                                                 max_iter=5,
                                                 model=self.model)

            imp_y = imputor.impute(censored_X=cen_X, censored_y=cen_y,
                                   uncensored_X=uncen_X,
                                   uncensored_y=uncen_y)

            if imp_y is None:
                continue

            for idx in range(cen_y.shape[0]):
                self.assertGreater(imp_y[idx], cen_y[idx])
            self.assertTrue(numpy.isfinite(imp_y).all())

    def testRealImputation(self):
        rs = numpy.random.RandomState(1)
        imputor = rfr_imputator.RFRImputator(rng=rs,
                                             cutoff=self.scen.cutoff,
                                             threshold=self.scen.cutoff*10,
                                             change_threshold=0.01, max_iter=10,
                                             model=self.model)

        r2e = runhistory2epm.RunHistory2EPM4LogCost(
            scenario=self.scen, num_params=3,
            success_states=[StatusType.SUCCESS, ],
            impute_censored_data=True, impute_state=[StatusType.TIMEOUT],
            imputor=imputor, rng=rs)
        print("%s" % str(r2e.transform(self.rh)[0]))
