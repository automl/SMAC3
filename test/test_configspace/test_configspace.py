'''
Created on Nov 19, 2015

@author: lindauer
'''
import os
import unittest

from ConfigSpace.read_and_write import pcs
from ConfigSpace.hyperparameters import CategoricalHyperparameter
from ConfigSpace.conditions import EqualsCondition
import smac.configspace


class ConfigSpaceTest(unittest.TestCase):

    def test_spear(self):
        '''
            simply getting some random configuration from spear pcs
        '''
        file_path = os.path.join(os.path.dirname(__file__), '..',
                                 'test_files', 'spear-params.pcs')

        with open(file_path) as fp:
            pcs_str = fp.readlines()
            cs = pcs.read(pcs_str)

        for i in range(100):
            config = cs.sample_configuration()
            print(config.get_dictionary())


    def test_impute_inactive_hyperparameters(self):
        cs = smac.configspace.ConfigurationSpace()
        a = cs.add_hyperparameter(CategoricalHyperparameter('a', [0, 1],
                                                            default_value=0))
        b = cs.add_hyperparameter(CategoricalHyperparameter('b', [0, 1],
                                                            default_value=1))
        cs.add_condition(EqualsCondition(b, a, 1))
        cs.seed(1)
        configs = cs.sample_configuration(size=100)
        config_array = smac.configspace.convert_configurations_to_array(configs)
        for line in config_array:
            if line[0] == 0:
                self.assertEqual(line[1], 1)



if __name__ == "__main__":
    unittest.main()
