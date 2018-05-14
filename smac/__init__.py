import sys

if sys.version_info < (3,5):
    raise ValueError("SMAC requires Python 3.5 or newer.")

from smac.__version__ import __version__
AUTHORS = "Marius Lindauer, Matthias Feurer, Katharina Eggensperger, " \
          "Aaron Klein, Stefan Falkner and Frank Hutter"
