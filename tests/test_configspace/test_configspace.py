import os
import unittest

from ConfigSpace.read_and_write import pcs

__copyright__ = "Copyright 2021, AutoML.org Freiburg-Hannover"
__license__ = "3-clause BSD"


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
            _ = cs.sample_configuration()
