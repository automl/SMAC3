#!/bin/bash
flake8 --max-line-length=120 --show-source $options --ignore W605,E402,W503 --exclude .git,__pycache__,doc,build,dist,examples/spear_qcp/target_algorithm
