'''
Created on Nov 19, 2015

@author: lindauer
'''
import unittest
import logging

from ConfigSpace.io import pcs


class ConfigSpaceTest(unittest.TestCase):

    def test_spear(self):
        '''
            simply getting some random configuration from spear pcs
        '''

        with open("./test_files/spear-params.pcs") as fp:
            pcs_str = fp.readlines()
            cs = pcs.read(pcs_str)

        for i in range(100):
            config = cs.sample_configuration()
            print(config.get_dictionary())


if __name__ == "__main__":
    unittest.main()
