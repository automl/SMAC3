
## declare an array variable
declare -a versions=("1.4.0" "2.0.0a3")

conda create -n SMACBench python=3.9 -y
conda activate SMACBench

# Loop through versions
for i in "${versions[@]}"
do
    # Setup environment
    pip install smac==$version
    pip install -r benchmark/requirements.txt

    python benchmark/benchmark.py
done

# Clean-up
rm -rf smac3-output*