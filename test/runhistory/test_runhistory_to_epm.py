__author__ = "Katharina Eggensperger"
__copyright__ = "Copyright 2015, ML4AAD"
__license__ = "BSD"
__maintainer__ = "Katharina Eggensperger"
__email__ = "eggenspk@cs.uni-freiburg.de"
__version__ = "0.0.1"

import unittest
from smac.runhistory import runhistory, runhistory_to_epm


class RunhistoryTest(unittest.TestCase):

    def test_add(self):
        '''
            simply adding some rundata to runhistory
        '''
        rh = runhistory.RunHistory()
        rh.add(config={'a': '1', 'b': '2'}, cost=10, time=20,
               status="SUCCESS", instance_id=None,
               seed=None,
               additional_info=None)
        rh.add(config={'a': '1', 'b': '2'}, cost=10, time=20,
               status="SUCCESS", instance_id=1,
               seed=12354,
               additional_info={"start_time": 10})

        rh2epm = runhistory_to_epm.RunHistoryToEPM()

        self.assertRaises(ValueError, rh2epm.transform, rh)
        print(rh2epm.fit(rh))


if __name__ == "__main__":
    unittest.main()
