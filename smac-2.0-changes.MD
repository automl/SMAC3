# Unified transformation guide

- Use future annotations: `from __future__ import annotations`. For example:
    ```
    def test_func(value: int | None = None) -> int | None:
        return value
    ```

- Remove `self.logger`. Create on top a global logger:
logger = get_logger(__name__)

- Remove `rng` from init and replace it with `seed: int = 0`

- Comments should start upper-case. For example: `# This is a comment`.
If multiple sentences are used, use punctuation. For example: `# This is one sentence. This is another.`

- Docstring structure is shown in the following. Please start the general comment and the parameter description
upper-case.

    ```
    """
    First a general comment.

    Parameters
    ----------
    X : np.ndarray | None, defaults to None
        Explain what this parameter does.

    Returns
    -------
    np.ndarray(N,1)
        Acquisition function values wrt X.
    """
    ```

- Everytime you see a variable you don't know, look it up and replace it with the variable provided
in the `variable renaming`. If the variable is not listed below, add the variable and change it
accordingly. Just make sure it is persistent.

- If you find something with `run_obj == "runtime"` remove it. Since adaptive capping 
is bounded to `runtime`, you can remove that too.

- Please always check whether the docstrings are still correct. If not, correct them accordingly.

- Replace all copyrights/authors/license with the following:
    ```
    __copyright__ = "Copyright 2022, automl.org"
    __license__ = "3-clause BSD"
    ```



# Variable renaming

We removed run_obj == runtime COMPLETELY
=> no adaptive capping

cutoff/cutoff-time/cutoff_time => algorithm_walltime_limit
algo_runs_timelimit/tuner-timout/tuner_timout => cputime_limit
tae/ta/target_algorithm => target_algorithm
sls_max_steps => max_steps_local_search
rh/run_history/runhistory => runhistory
rand_prob/rand-prob => random_propability
num_trees => n_trees
output_dir => output_directory
ta_run_limit/ta-run-limit/runcount-limit/runcount_limit => n_runs
cs/configspace/config_space => configspace
max_config_fracs => max_config_ratio

AbstractTAFunc => AbstractAlgorithmExecuter
ExecuteTAFuncArray/ExecuteTAFuncDict => AlgorithmExecuter
ExecuteTARunOld => CommandlineAlgorithmExecuter
run_obj => REMOVED
cost_for_crash => crash_cost
par_factor => REMOVED (depending on run_obj == runtime)
always_race_against => race_against
runhistory2epm => runhistory_transformer
Runhistory2EPM => RunhistoryTransformer
num_obj => n_objectives (use `count_objectives` from the config object)

## Initial design
n_configs_x_params => n_configs_per_hyperparameter
max_config_fracs => 

## Algorithm Configuration
feature_array => instance_features
train_insts/train_instances => instances

## Intensification
min_chall => min_challenger
minR => min_config_calls
maxR => max_config_calls