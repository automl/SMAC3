#!/bin/bash

cd $(dirname $0)/../examples/branin

cat << EOF > /tmp/scenario.txt
algo = python cmdline_wrapper.py
paramfile = param_config_space.pcs
run_obj = quality
runcount_limit = 100
deterministic = 1
EOF

INITIAL_GIT_REVISION=$(git branch --no-color 2> /dev/null | sed -e '/^[^*]/d' -e "s/* \(.*\)/\1/")

function test_deterministic () {
	python ../../scripts/smac --scenario /tmp/scenario.txt --seed 1 --verbose_level INFO --mode SMAC | tee "$1"
	python ../../scripts/smac --scenario /tmp/scenario.txt --seed 1 --verbose_level INFO --mode SMAC | tee "$2"
	GIT_REVISION=$(git rev-parse HEAD~1)
	echo ==============================
	echo === diff between two runs on $(git branch --no-color 2> /dev/null | sed -e '/^[^*]/d' -e "s/* \(.*\)/\1/") ===
	echo ==============================
	diff "$1" "$2"
}

test_deterministic /tmp/smac_det_1.log /tmp/smac_det_2.log

if [[ $# -ge 1 ]] ; then
	git checkout $1 || (echo "argument provided must be a git revision" ; exit 1)
	test_deterministic /tmp/smac_det_3.log /tmp/smac_det_4.log

	echo ==============================
	echo === diff between runs on $INITIAL_GIT_REVISION and $1 ===
	echo ==============================
	diff /tmp/smac_det_1.log /tmp/smac_det_3.log
fi
