#!/bin/bash

# An example showing how to use commandline to optimization with ROAR facade
echo $PATH
which python
python ../../scripts/smac.py --scenario spear_qcp/scenario.txt --verbose DEBUG --mode ROAR
