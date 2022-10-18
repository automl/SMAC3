"""
Branin
^^^^^^

This file is a wrapper used by SMAC to optimize parameters on the branin function.
To run this example in the terminal, execute:

.. code-block:: bash

    python ./scripts/smac.py --scenario examples/commandline/branin/scenario.txt


Inside the scenario, this file and also ``configspace.pcs`` is referenced and therefore used
for the optimization. A full call by SMAC looks like this:

.. code-block:: bash

    <algo>           <instance> <instance specific> <cutoff time>  <runlength> <seed> <parameters>
    python branin.py 0          0                   9999999        0           12345  -x1 0 -x2 0


Since SMAC processes results from the commandline, print-statements are
crucial. The format of the results must be the following to ensure correct usage:

.. code-block:: bash

    Result for SMAC: <STATUS>, <runtime>, <runlength>, <quality>, <seed>, <instance-specifics>

"""

import sys

import numpy as np

__copyright__ = "Copyright 2021, AutoML.org Freiburg-Hannover"
__license__ = "3-clause BSD"


def branin(x):
    x1 = x[0]
    x2 = x[1]
    a = 1.0
    b = 5.1 / (4.0 * np.pi**2)
    c = 5.0 / np.pi
    r = 6.0
    s = 10.0
    t = 1.0 / (8.0 * np.pi)
    ret = a * (x2 - b * x1**2 + c * x1 - r) ** 2 + s * (1 - t) * np.cos(x1) + s

    return ret


if __name__ == "__main__":
    # Unused in this example:
    # instance, instance_specific, cutoff, runlength = sys.argv[1:5]
    seed = sys.argv[5]

    # sys.argv[6] and sys.argv[8] are the names of the target algorithm
    # parameters (here: "-x1", "-x2")
    x = float(sys.argv[7])
    y = float(sys.argv[9])
    result = branin((x, y))

    # This line is important so the result can be processed by SMAC:
    print("Result for SMAC: SUCCESS, -1, -1, %f, %s" % (result, seed))
