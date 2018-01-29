'''
Created on Nov 19, 2015

@author: lindauer
'''
import os
import unittest

from ConfigSpace.read_and_write import pcs
from ConfigSpace.hyperparameters import CategoricalHyperparameter, \
    UniformFloatHyperparameter
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


    def test_impute_inactive_hyperparameters(self):
        cs = smac.configspace.ConfigurationSpace()
        a = cs.add_hyperparameter(CategoricalHyperparameter('a', [0, 1]))
        b = cs.add_hyperparameter(CategoricalHyperparameter('b', [0, 1]))
        c = cs.add_hyperparameter(UniformFloatHyperparameter('c', 0, 1))
        cs.add_condition(EqualsCondition(b, a, 1))
        cs.add_condition(EqualsCondition(c, a, 0))
        cs.seed(1)
        configs = cs.sample_configuration(size=100)
        config_array = smac.configspace.convert_configurations_to_array(configs)
        for line in config_array:
            print(line, flush=True)
            if line[0] == 0:
                self.assertEqual(line[1], 2)
            elif line[0] == 1:
                self.assertEqual(line[2], -1)

