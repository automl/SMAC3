# SMAC Benchmark (Beta)

Calculates multiple tasks for specified versions and compares performance.
Each version is derived from pip and installed in an empty environment (``SMACBenchmark``).


## Results

![trajectory_trials](report/trajectory_walltime.png "Trajectory (Walltime)")


## Getting Started

- Make sure you have anaconda installed.
- Execute ``bash ./run.sh`` inside the benchmark directory. You can specify the versions to run there.
- Have your current version of SMAC installed in the environment ``SMAC``. After the old versions have been finished, the script benchmarks the current version.
- Alternatively, just execute ``python src/benchmark.py`` with a SMAC environment of your   choice.


## Note

- Versions before 2.0 might not support the new sklearn (>1.2) anymore
