from smac.optimizer.random_configuration_chooser import ChooserNoCoolDown

__author__ = "Aaron Kimmig"
__copyright__ = "Copyright 2015, ML4AAD"
__license__ = "3-clause BSD"
__maintainer__ = "Aaron Kimmig"
__email__ = "kimmiga@cs.uni-freiburg.de"
__version__ = "0.0.1"


"""
This class is used to test in test_deterministic_smac if the CLI accepts
custom Random Conf choosers.
"""
class RandomConfigurationChooserImpl(ChooserNoCoolDown):
    """
    Implementation of helper class to configure interleaving of
    random configurations in a list of challengers.
    """

    def __init__(self):
        super().__init__(modulus=3.0)
