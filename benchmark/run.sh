#!/bin/bash

# Env name
ENVNAME=smac

# Declare the old versions here
declare -a versions=("1.4.0")

# Loop through old versions
for version in "${versions[@]}"
do
    # Setup environment
    conda env remove -n SMACBench
    conda create -n SMACBench python=3.10 -y
    conda run -n SMACBench pip install smac==$version
    conda run -n SMACBench pip install -r requirements.txt

    if [ "$version" = "1.4.0" ]; then
        # Use different scikit-learn
        conda run -n SMACBench pip install scikit-learn==1.1.0
    fi

    conda run --no-capture-output -n SMACBench python src/benchmark.py
done

# Benchmark the current version
conda run -n ${ENVNAME} pip install -r requirements.txt
conda run --no-capture-output -n ${ENVNAME} python src/benchmark.py

# Clean-up
rm -rf smac3-output*
rm -rf smac3_output