#!/bin/bash

# check if test cases left any files after execution by looking for files not tracked by git
CHECK_CMD=(git ls-files '*output*' 'run_*' 'test/' --others --exclude-standard)
if [[ ! -z "$(${CHECK_CMD[@]})" ]]
then
	echo 'ERROR: tests did not clean up all files they generated:'
	"${CHECK_CMD[@]}" | sed 's/^/  /'
	exit 1
fi
