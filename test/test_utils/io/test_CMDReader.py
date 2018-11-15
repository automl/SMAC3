'''
Created on Apr 12, 2015

@author: Andre Biedenkapp
'''
import os
import unittest
import logging

from smac.utils.io.cmd_reader import CMDReader


class TestArgs:

    def __init__(self, sf, seed, mi, vl):
        self.scenario_file = sf
        self.seed = seed
        self.max_iterations = mi
        self.verbose_level = vl

    def cmdline(self):
        return ['--scenario-file', self.scenario_file,
                '--seed', str(self.seed),
                '--runcount-limit', str(self.max_iterations),
                '--verbose', self.verbose_level]


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
        with self.assertRaises(SystemExit):
            self.cr.read_cmd(targs.cmdline())

    def test_check_args(self):  # Tests if no Exception is raised
        targs = TestArgs('test/test_files/scenario_test/scenario.txt', 1234, 2, 'DEBUG')
        self.cr.read_cmd(targs.cmdline())

    def test_doc_files(self):
        self.cr.write_main_options_to_doc(path="test.rst")
        self.cr.write_smac_options_to_doc(path="test.rst")
        self.cr.write_scenario_options_to_doc(path="test.rst")
        os.remove("./test.rst")


if __name__ == "__main__":
    unittest.main()
