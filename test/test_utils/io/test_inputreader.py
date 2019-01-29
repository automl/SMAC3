'''
Created on Oct 16, 2017

@author: Joshua Marben
'''
import os
import unittest
import logging

import numpy as np

from smac.configspace import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformFloatHyperparameter
from smac.configspace import pcs
from smac.utils.io.input_reader import InputReader
from smac.utils.io.output_writer import OutputWriter


class InputReaderTest(unittest.TestCase):

    def setUp(self):
        logging.basicConfig()
        self.logger = logging.getLogger(self.__module__ + "." + self.__class__.__name__)
        self.logger.setLevel(logging.DEBUG)
        self.current_dir = os.getcwd()
        base_directory = os.path.split(__file__)[0]
        base_directory = os.path.abspath(os.path.join(base_directory, '..',
                                                      '..', '..'))
        os.chdir(base_directory)

        # Files that will be created:
        self.pcs_fn = "test/test_files/configspace.pcs"
        self.json_fn = "test/test_files/configspace.json"

        self.output_files = [self.pcs_fn, self.json_fn]

    def tearDown(self):
        for output_file in self.output_files:
            if output_file:
                try:
                    os.remove(output_file)
                except FileNotFoundError:
                    pass

        os.chdir(self.current_dir)

    def test_feature_input(self):
        feature_fn = "test/test_files/features_example.csv"
        in_reader = InputReader()
        feats = in_reader.read_instance_features_file(fn=feature_fn)
        self.assertEqual(feats[0], ["feature1", "feature2", "feature3"])
        feats_original = {"inst1":[1.0, 2.0, 3.0],
                          "inst2":[1.5, 2.5, 3.5],
                          "inst3":[1.7, 1.8, 1.9]}
        for i in feats[1]:
            self.assertEqual(feats_original[i], list(feats[1][i]))

    def test_save_load_configspace(self):
        """Check if inputreader can load different config-spaces"""
        cs = ConfigurationSpace()
        hyp = UniformFloatHyperparameter('A', 0.0, 1.0, default_value=0.5)
        cs.add_hyperparameters([hyp])

        output_writer = OutputWriter()
        input_reader = InputReader()

        # pcs_new
        output_writer.save_configspace(cs, self.pcs_fn, 'pcs_new')
        restored_cs = input_reader.read_pcs_file(self.pcs_fn)
        self.assertEqual(cs, restored_cs)
        restored_cs = input_reader.read_pcs_file(self.pcs_fn, self.logger)
        self.assertEqual(cs, restored_cs)

        # json
        output_writer.save_configspace(cs, self.json_fn, 'json')
        restored_cs = input_reader.read_pcs_file(self.json_fn)
        self.assertEqual(cs, restored_cs)
        restored_cs = input_reader.read_pcs_file(self.json_fn, self.logger)
        self.assertEqual(cs, restored_cs)

        # pcs
        with open(self.pcs_fn, 'w') as fh:
            fh.write(pcs.write(cs))
        restored_cs = input_reader.read_pcs_file(self.pcs_fn)
        self.assertEqual(cs, restored_cs)
        restored_cs = input_reader.read_pcs_file(self.pcs_fn)
        self.assertEqual(cs, restored_cs)
        restored_cs = input_reader.read_pcs_file(self.pcs_fn, self.logger)
        self.assertEqual(cs, restored_cs)
