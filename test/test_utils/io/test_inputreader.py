'''
Created on Oct 16, 2017

@author: Joshua Marben
'''
import os
import unittest
import logging

import numpy as np

from smac.utils.io.input_reader import InputReader


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

    def tearDown(self):
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
