#!/bin/bash
flake8 --max-line-length=120 --show-source $options ./smac --ignore W605
flake8 --max-line-length=120 --show-source $options ./test --ignore W605
