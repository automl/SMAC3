import json
import os
import sys

from smac.utils import dependencies

__version__ = '0.14.0dev'
__author__ = 'Marius Lindauer, Matthias Feurer, Katharina Eggensperger, Joshua Marben, ' \
             'André Biedenkapp, Francisco Rivera, Ashwin Raaghav, Aaron Klein, Stefan Falkner ' \
             'and Frank Hutter'


with open(os.path.join(os.path.dirname(__file__), 'requirements.txt')) as fh:
    dependencies.verify_packages(fh.read())

with open(os.path.join(os.path.dirname(__file__), 'extras_require.json')) as fh:
    extras_require = json.load(fh)

extras_installed = set()
for name, requirements in extras_require.items():
    if name in ['documentation', 'test']:
        continue
    if dependencies.are_valid_packages(requirements):
        extras_installed.add(name)

if sys.version_info < (3, 7, 0):
    raise ValueError("SMAC requires Python 3.7.0 or newer.")


if os.name != 'posix':
    print(
        'Detected unsupported operating system: %s. Please be aware, that SMAC might not run on this system.' %
        sys.platform
    )
