import sys

from branin import branin

"""
This is a wrapper used by SMAC to optimize parameters on the branin-function.

To run this example, execute:
    $ cd examples/branin
    $ python ../../scripts/smac --scenario scenario.txt

We optimize the branin-function (see "examples/branin/branin.py").

To use the commandline, we need two files:
- a scenario-file: located in "examples/branin/scenario.txt"
                   specifies SMAC-parameters, e.g. runtime, output, etc.
- a pcs-file:      located in "examples/branin/param_config_space.pcs"
                   specifies the parameter configuration space (here: x1, x2)

SMAC calls this wrapper during optimization, because it is specified in the
"algo"-parameter of the scenario-file.
SMAC calls this file via the commandline, passing information in additional
commandline-arguments.
The target algorithm parameters (here: x1, x2) are also passed as
commandline-arguments.
A full call by SMAC looks like this:
    <algo>           <instance> <instance specific> <cutoff time>  <runlength> <seed> <algorithm parameters>
    python branin.py 0          0                   99999999999999 0           12345  -x1 0 -x2 0

SMAC processes results from the commandline, therefore the print-statement is
crucial. The format of the results must be:
    Result for SMAC: <STATUS>, <runtime>, <runlength>, <quality>, <seed>, <instance-specifics>
"""

if __name__ == '__main__':
    # Unused in this example:
    # instance, instance_specific, cutoff, runlength = sys.argv[1:5]
    seed = sys.argv[5]
    # sys.argv[6] and sys.argv[8] are the names of the target algorithm
    # parameters (here: "-x1", "-x2")
    x = float(sys.argv[7])
    y = float(sys.argv[9])
    result = branin((x, y))
    print('Result for SMAC: SUCCESS, -1, -1, %f, %s' % (result, seed))
