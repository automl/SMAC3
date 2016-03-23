'''
Created on Nov 19, 2015

@author: lindauer
'''
import os
import unittest

from ConfigSpace.io import pcs


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


if __name__ == "__main__":
    unittest.main()
