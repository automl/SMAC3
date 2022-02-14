import unittest

from smac import extras_installed, extras_require

__copyright__ = "Copyright 2021, AutoML.org Freiburg-Hannover"
__license__ = "3-clause BSD"


def requires_extra(name):
    return unittest.skipUnless(name in extras_installed, "requires {}".format(extras_require[name]))
