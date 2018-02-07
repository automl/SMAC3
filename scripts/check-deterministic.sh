#!/bin/bash

cd $(dirname $0)/../examples/branin

cat << EOF > /tmp/scenario.txt
algo = python cmdline_wrapper.py
paramfile = param_config_space.pcs
run_obj = quality
runcount_limit = 100
deterministic = 1
EOF

python ../../scripts/smac --scenario /tmp/scenario.txt --seed 1 --verbose_level INFO --mode SMAC | tee /tmp/out1.log
python ../../scripts/smac --scenario /tmp/scenario.txt --seed 1 --verbose_level INFO --mode SMAC | tee /tmp/out2.log

echo "=== diff /tmp/out{1,2}.log ==="
diff /tmp/out1.log /tmp/out2.log