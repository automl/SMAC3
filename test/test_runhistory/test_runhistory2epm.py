__author__ = "Katharina Eggensperger"
__copyright__ = "Copyright 2015, ML4AAD"
__license__ = "GPLv3"
__maintainer__ = "Katharina Eggensperger"
__email__ = "eggenspk@cs.uni-freiburg.de"
__version__ = "0.0.1"

import unittest

import numpy as np

from smac.tae.execute_ta_run import StatusType
from smac.runhistory import runhistory, runhistory2epm

from ConfigSpace import Configuration, ConfigurationSpace
from ConfigSpace.hyperparameters import UniformIntegerHyperparameter
from smac.runhistory.runhistory import RunHistory
from smac.scenario.scenario import Scenario
from smac.optimizer.objective import average_cost
from smac.epm.rfr_imputator import RFRImputator
from smac.epm.rf_with_instances import RandomForestWithInstances
from smac.utils.util_funcs import get_types


def get_config_space():
    cs = ConfigurationSpace()
    cs.add_hyperparameter(UniformIntegerHyperparameter(name='a',
                                                       lower=0,
                                                       upper=100))
    cs.add_hyperparameter(UniformIntegerHyperparameter(name='b',
                                                       lower=0,
                                                       upper=100))
    return cs


class RunhistoryTest(unittest.TestCase):

    def setUp(self):
        unittest.TestCase.setUp(self)

        self.rh = runhistory.RunHistory(aggregate_func=average_cost)
        self.cs = get_config_space()
        self.config1 = Configuration(self.cs,
                                     values={'a': 0, 'b': 100})
        self.config2 = Configuration(self.cs,
                                     values={'a': 100, 'b': 0})
        self.config3 = Configuration(self.cs,
                                     values={'a': 100, 'b': 100})

        self.scen = Scenario({'run_obj': 'runtime', 'cutoff_time': 20,
                              'cs': self.cs})
        self.types, self.bounds = get_types(self.cs, None)
        self.scen = Scenario({'run_obj': 'runtime', 'cutoff_time': 20, 'cs': self.cs,
                              'output_dir': ''})

    def test_log_runtime_with_imputation(self):
        '''
            adding some rundata to RunHistory2EPM4LogCost and impute censored data
        '''
        self.imputor = RFRImputator(rng=np.random.RandomState(seed=12345),
                                    cutoff=np.log(self.scen.cutoff),
                                    threshold=np.log(
                                        self.scen.cutoff * self.scen.par_factor),
                                    model=RandomForestWithInstances(types=self.types, bounds=self.bounds,
                                                                    instance_features=None,
                                                                    seed=12345,
                                                                    ratio_features=1.0)
                                    )

        rh2epm = runhistory2epm.RunHistory2EPM4LogCost(num_params=2,
                                                       scenario=self.scen,
                                                       impute_censored_data=True,
                                                       impute_state=[
                                                           StatusType.TIMEOUT],
                                                       imputor=self.imputor)

        self.rh.add(config=self.config1, cost=1, time=1,
                    status=StatusType.SUCCESS, instance_id=23,
                    seed=None,
                    additional_info=None)

        X, y = rh2epm.transform(self.rh)
        self.assertTrue(np.allclose(X, np.array([[0.005, 0.995]]), atol=0.001))
        self.assertTrue(np.allclose(y, np.array([[0.]])))  # 10^0 = 1

        # rh2epm should use time and not cost field later
        self.rh.add(config=self.config3, cost=200, time=20,
                    status=StatusType.TIMEOUT, instance_id=1,
                    seed=45,
                    additional_info={"start_time": 20})

        X, y = rh2epm.transform(self.rh)
        self.assertTrue(
            np.allclose(X, np.array([[0.005, 0.995], [0.995, 0.995]]), atol=0.001))
        # ln(20 * 10)
        self.assertTrue(np.allclose(y, np.array([[0.], [5.2983]]), atol=0.001))

        self.rh.add(config=self.config2, cost=100, time=10,
                    status=StatusType.TIMEOUT, instance_id=1,
                    seed=12354,
                    additional_info={"start_time": 10})

        X, y = rh2epm.transform(self.rh)
        np.testing.assert_array_almost_equal(X, np.array([[0.005, 0.995],
                                                          [0.995, 0.005],
                                                          [0.995, 0.995]]),
                                             decimal=3)
        
        np.testing.assert_array_almost_equal(y, np.array([[0.], [2.727], [5.2983]]),
                                             decimal=3)

    def test_log_cost_without_imputation(self):
        '''
            adding some rundata to RunHistory2EPM4LogCost
        '''

        rh2epm = runhistory2epm.RunHistory2EPM4LogCost(num_params=2,
                                                       scenario=self.scen)

        self.rh.add(config=self.config1, cost=1, time=1,
                    status=StatusType.SUCCESS, instance_id=23,
                    seed=None,
                    additional_info=None)

        X, y = rh2epm.transform(self.rh)
        self.assertTrue(np.allclose(X, np.array([[0.005, 0.995]]), atol=0.001))
        self.assertTrue(np.allclose(y, np.array([[0.]])))  # 10^0 = 1

        # rh2epm should use time and not cost field later
        self.rh.add(config=self.config3, cost=200, time=20,
                    status=StatusType.TIMEOUT, instance_id=1,
                    seed=45,
                    additional_info={"start_time": 20})

        X, y = rh2epm.transform(self.rh)
        self.assertTrue(
            np.allclose(X, np.array([[0.005, 0.995], [0.995, 0.995]]), atol=0.001))
        # ln(20 * 10)
        self.assertTrue(np.allclose(y, np.array([[0.], [5.2983]]), atol=0.001))

        self.rh.add(config=self.config2, cost=100, time=10,
                    status=StatusType.TIMEOUT, instance_id=1,
                    seed=12354,
                    additional_info={"start_time": 10})

        X, y = rh2epm.transform(self.rh)
        # last entry gets skipped since imputation is disabled
        self.assertTrue(np.allclose(
            X, np.array([[0.005, 0.995], [0.995, 0.995]]), atol=0.001))
        self.assertTrue(
            np.allclose(y, np.array([[0.], [5.2983]]), atol=0.001))

    def test_cost_with_imputation(self):
        '''
            adding some rundata to RunHistory2EPM4Cost and impute censored data
        '''

        self.imputor = RFRImputator(rng=np.random.RandomState(seed=12345),

                                    cutoff=self.scen.cutoff,
                                    threshold=self.scen.cutoff * self.scen.par_factor,
                                    model=RandomForestWithInstances(types=self.types, bounds=self.bounds,
                                                                    instance_features=None,
                                                                    seed=12345, n_points_per_tree=90,
                                                                    ratio_features=1.0)
                                    )

        rh2epm = runhistory2epm.RunHistory2EPM4Cost(num_params=2,
                                                       scenario=self.scen,
                                                       impute_censored_data=True,
                                                       impute_state=[
                                                           StatusType.TIMEOUT],
                                                       imputor=self.imputor)

        self.rh.add(config=self.config1, cost=1, time=1,
                    status=StatusType.SUCCESS, instance_id=23,
                    seed=None,
                    additional_info=None)

        X, y = rh2epm.transform(self.rh)
        self.assertTrue(np.allclose(X, np.array([[0.005, 0.995]]), atol=0.001))
        self.assertTrue(np.allclose(y, np.array([[1.]])))

        # rh2epm should use time and not cost field later
        self.rh.add(config=self.config3, cost=200, time=20,
                    status=StatusType.TIMEOUT, instance_id=1,
                    seed=45,
                    additional_info={"start_time": 20})

        X, y = rh2epm.transform(self.rh)
        self.assertTrue(
            np.allclose(X, np.array([[0.005, 0.995], [0.995, 0.995]]), atol=0.001))
        self.assertTrue(np.allclose(y, np.array([[1.], [200.]]), atol=0.001))

        self.rh.add(config=self.config2, cost=100, time=10,
                    status=StatusType.TIMEOUT, instance_id=1,
                    seed=12354,
                    additional_info={"start_time": 10})

        X, y = rh2epm.transform(self.rh)
        np.testing.assert_array_almost_equal(X, np.array([[0.005, 0.995],
                                                          [0.995, 0.005],
                                                          [0.995, 0.995]]),
                                             decimal=3)
        np.testing.assert_array_almost_equal(y, np.array([[1.], [11.], [200.]]),
                                             decimal=1)

    def test_cost_without_imputation(self):
        '''
            adding some rundata to RunHistory2EPM4Cost without imputation
        '''

        rh2epm = runhistory2epm.RunHistory2EPM4Cost(num_params=2,
                                                       scenario=self.scen)

        self.rh.add(config=self.config1, cost=1, time=1,
                    status=StatusType.SUCCESS, instance_id=23,
                    seed=None,
                    additional_info=None)

        X, y = rh2epm.transform(self.rh)
        self.assertTrue(np.allclose(X, np.array([[0.005, 0.995]]), atol=0.001))
        self.assertTrue(np.allclose(y, np.array([[1.]])))

        # rh2epm should use time and not cost field later
        self.rh.add(config=self.config3, cost=2, time=20,
                    status=StatusType.TIMEOUT, instance_id=1,
                    seed=45,
                    additional_info={"start_time": 20})

        X, y = rh2epm.transform(self.rh)
        self.assertTrue(
            np.allclose(X, np.array([[0.005, 0.995], [0.995, 0.995]]), atol=0.001))
        # log_10(20 * 10)
        self.assertTrue(np.allclose(y, np.array([[1.], [200.]]), atol=0.001))

        self.rh.add(config=self.config2, cost=100, time=10,
                    status=StatusType.TIMEOUT, instance_id=1,
                    seed=12354,
                    additional_info={"start_time": 10})

        X, y = rh2epm.transform(self.rh)
        # last entry gets skipped since imputation is disabled
        self.assertTrue(np.allclose(
            X, np.array([[0.005, 0.995], [0.995, 0.995]]), atol=0.001))
        self.assertTrue(
            np.allclose(y, np.array([[1.], [200.]]), atol=0.001))

    def test_cost_quality(self):
        '''
            adding some rundata to RunHistory2EPM4LogCost
        '''
        self.scen = Scenario({"cutoff_time": 20, 'cs': self.cs, 'run_obj': 'quality',
                              'output_dir': ''})

        rh2epm = runhistory2epm.RunHistory2EPM4Cost(num_params=2,
                                                       scenario=self.scen)

        self.rh.add(config=self.config1, cost=1, time=10,
                    status=StatusType.SUCCESS, instance_id=23,
                    seed=None,
                    additional_info=None)

        X, y = rh2epm.transform(self.rh)
        self.assertTrue(np.allclose(X, np.array([[0.005, 0.995]]), atol=0.001))
        # should use the cost field and not runtime
        self.assertTrue(np.allclose(y, np.array([[1.]])))

        # rh2epm should use cost and not time field later
        self.rh.add(config=self.config3, cost=200, time=20,
                    status=StatusType.TIMEOUT, instance_id=1,
                    seed=45,
                    additional_info={"start_time": 20})

        X, y = rh2epm.transform(self.rh)
        self.assertTrue(
            np.allclose(X, np.array([[0.005, 0.995], [0.995, 0.995]]), atol=0.001))
        # log_10(20 * 10)
        self.assertTrue(np.allclose(y, np.array([[1.], [200.]]), atol=0.001))

        #TODO: unit test for censored data in quality scenario
        
    def test_get_X_y(self):
        '''
            add some data to RH and check returned values in X,y format
        '''

        self.scen = Scenario({'cutoff_time': 20, 'cs': self.cs,
                              'run_obj': 'runtime', 
                              'instances': [['1'],['2']],
                              'features': {
                                  '1': [1,1],
                                  '2': [2,2]
                                  },
                              'output_dir': ''})

        rh2epm = runhistory2epm.RunHistory2EPM4Cost(num_params=2,
                                                    scenario=self.scen)

        self.rh.add(config=self.config1, cost=1, time=10,
                    status=StatusType.SUCCESS, instance_id='1',
                    seed=None,
                    additional_info=None)
        
        self.rh.add(config=self.config1, cost=2, time=10,
                    status=StatusType.SUCCESS, instance_id='2',
                    seed=None,
                    additional_info=None)
        
        self.rh.add(config=self.config2, cost=1, time=10,
                    status=StatusType.TIMEOUT, instance_id='1',
                    seed=None,
                    additional_info=None)
        
        self.rh.add(config=self.config2, cost=0.1, time=10,
                    status=StatusType.CAPPED, instance_id='2',
                    seed=None,
                    additional_info=None)
        
        X,y,c = rh2epm.get_X_y(self.rh)
        
        print(X,y,c)
        
        X_sol = np.array([[0,100,1,1],
                          [0,100,2,2],
                          [100,0,1,1],
                          [100,0,2,2]])
        self.assertTrue(np.all(X==X_sol))
        
        y_sol = np.array([1,2,1,0.1])
        self.assertTrue(np.all(y==y_sol))
        
        c_sol = np.array([False, False, True, True])
        self.assertTrue(np.all(c==c_sol))
        

if __name__ == "__main__":
    unittest.main()
