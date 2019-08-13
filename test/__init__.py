import unittest

from smac import extras_require, extras_installed


def requires_extra(name):
    return unittest.skipUnless(name in extras_installed,
                               'requires {}'.format(extras_require[name]))
