'''
Created on Apr 12, 2015

@author: Andre Biedenkapp
'''
import os
import unittest
import logging

from smac.utils.io.cmd_reader import CMDReader

class TestArgs():

    def __init__(self, sf, seed, mi, vl):
        self.scenario_file = sf
        self.seed = seed
        self.max_iterations = mi
        self.verbose_level = vl

class CMDReaderTest(unittest.TestCase):

    def setUp(self):
        logging.basicConfig()
        self.logger = logging.getLogger(self.__module__ + "." + self.__class__.__name__)
        self.logger.setLevel(logging.DEBUG)
        self.cr = CMDReader()
        self.current_dir = os.getcwd()
        base_directory = os.path.split(__file__)[0]
        base_directory = os.path.abspath(os.path.join(base_directory, '..',
                                                      '..', '..'))
        os.chdir(base_directory)

    def tearDown(self):
        os.chdir(self.current_dir)

    def test_check_args_exception(self):  # Tests if the Exception is correctly raised
        targs = TestArgs('.', 1234, 2, 'DEBUG')
        with self.assertRaises(ValueError):
            self.cr._check_args(targs)

    def test_check_args(self):  # Tests if no Exception is raised
        targs = TestArgs('test/test_files/scenario_test/scenario.txt', 1234, 2, 'DEBUG')
        self.cr._check_args(targs)

if __name__ == "__main__":
    unittest.main()
