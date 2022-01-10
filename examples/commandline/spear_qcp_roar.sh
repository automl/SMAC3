#!/bin/bash

# An example showing how to use commandline to optimization with ROAR facade
echo "SPEAR_QCP_ROAR.SH"
python -c "import sys; print(sys.path)"
python ../../scripts/smac.py --scenario spear_qcp/scenario.txt --verbose DEBUG --mode ROAR
