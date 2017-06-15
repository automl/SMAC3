import sys
import numpy as np

from branin import branin

"""
Example for the use of SMAC through the commandline. To run this example, change
to the branin-directory and type:
    python ../../scripts/smac --scenario branin_scenario.txt
We optimize the branin-function, which has two parameters: x1 and x2.
To use the cmd-line, we need two files:
    a scenario-file, located in examples/branin/branin_scenario.txt
    a pcs-file, located in examples/branin/branin_pcs.pcs
The code below will be called by SMAC with parameters as cmd-line arguments and
prints the results so SMAC can interpret them again.
"""

if __name__ == '__main__':
    seed = sys.argv[5]
    x = float(sys.argv[7])
    y = float(sys.argv[9])
    tmp = branin((x, y))
    print('Result for SMAC: SUCCESS, -1, -1, %f, %s' % (tmp, seed))
