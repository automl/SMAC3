import json
import os
import sys

from smac.utils import dependencies

__copyright__ = "Copyright 2021, AutoML.org Freiburg-Hannover"
__license__ = "3-clause BSD"
__version__ = '1.2'
__author__ = 'Marius Lindauer, Katharina Eggensperger, Matthias Feurer, André Biedenkapp, ' \
             'Difan Deng, Carolin Benjamins, Tim Ruhkopf, René Sass ' \
             'and Frank Hutter'


with open(os.path.join(os.path.dirname(__file__), 'requirements.txt')) as fh:
    dependencies.verify_packages(fh.read())

with open(os.path.join(os.path.dirname(__file__), 'extras_require.json')) as fh:
    extras_require = json.load(fh)

extras_installed = set()  # type: set
for name, requirements in extras_require.items():
    if name in ['documentation', 'test']:
        continue

if sys.version_info < (3, 7, 0):
    raise ValueError("SMAC requires Python 3.7.0 or newer.")


if os.name != 'posix':
    print(
        'Detected unsupported operating system: %s. Please be aware, that SMAC might not run on this system.' %
        sys.platform
    )
