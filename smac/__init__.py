import os
import sys

from smac.utils import dependencies

__version__ = '0.10.1.dev'
__author__ = 'Marius Lindauer, Matthias Feurer, Katharina Eggensperger, Joshua Marben, André Biedenkapp, Aaron Klein, Stefan Falkner and Frank Hutter'

__MANDATORY_PACKAGES__ = """
numpy>=1.7.1
scipy>=0.18.1
six
psutil
pynisher>=0.4.1
ConfigSpace>=0.4.6,<0.5
scikit-learn>=0.18.0
pyrfr>=0.5.0
joblib
"""
dependencies.verify_packages(__MANDATORY_PACKAGES__)

if sys.version_info < (3, 5, 2):
    raise ValueError("SMAC requires Python 3.5.2 or newer.")


if os.name != 'posix':
    print(
        'Detected unsupported operating system: %s. Please be aware, that SMAC might not run on this system.' %
        sys.platform
    )
