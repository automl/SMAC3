import os
import sys

if sys.version_info < (3, 5, 2):
    raise ValueError("SMAC requires Python 3.5.2 or newer.")

from smac.__version__ import __version__
from smac.utils import dependencies

AUTHORS = "Marius Lindauer, Matthias Feurer, Katharina Eggensperger, " \
          "Aaron Klein, Stefan Falkner and Frank Hutter"

__MANDATORY_PACKAGES__ = '''
numpy>=1.7.1
scipy>=0.18.1
scikit-learn>=0.18.0
pyrfr>=0.5.0
ConfigSpace>=0.4.6,<0.5
emcee>=2.1.0
george
cython
psutil
pyDOE
statsmodels
joblib
sobol_seq
'''


dependencies.verify_packages(__MANDATORY_PACKAGES__)

if os.name != 'posix':
    print(
        'Detected unsupported operating system: %s. Please be aware, that SMAC might not run on this system.' %
        sys.platform
    )
