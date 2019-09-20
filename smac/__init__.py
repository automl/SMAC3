import json
import os
import sys

import lazy_import
from smac.utils import dependencies

__version__ = '0.11.1'
__author__ = 'Marius Lindauer, Matthias Feurer, Katharina Eggensperger, Joshua Marben, André Biedenkapp, Aaron Klein, Stefan Falkner and Frank Hutter'


with open(os.path.join(os.path.dirname(__file__), 'requirements.txt')) as fh:
    dependencies.verify_packages(fh.read())

with open(os.path.join(os.path.dirname(__file__), 'extras_require.json')) as fh:
    extras_require = json.load(fh)

extras_installed = set()
for name, requirements in extras_require.items():
    if dependencies.are_valid_packages(requirements):
        extras_installed.add(name)
    for requirement in requirements:
        package_name = dependencies.RE_PATTERN.match(requirement).group('name')
        if package_name == 'scikit-optimize':
            package_name = 'skopt'
        lazy_import.lazy_module(package_name)

if sys.version_info < (3, 5, 2):
    raise ValueError("SMAC requires Python 3.5.2 or newer.")


if os.name != 'posix':
    print(
        'Detected unsupported operating system: %s. Please be aware, that SMAC might not run on this system.' %
        sys.platform
    )
