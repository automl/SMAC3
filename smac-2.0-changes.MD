# Unified transformation guide

- Please only push new examples/tests (when converting from -old) when they pass (it gets too messy otherwise).
- Please use grammarly in docstrings/text in general (let's say if 3 sentences or more). Often,
punctuations, spelling, and so on can be easily removed.
- Please see `examples/svm_cv.py` how to write a clean example with configspace 0.6.
- Use future annotations: `from __future__ import annotations`. For example:
    ```
    def test_func(value: int | None = None) -> int | None:
        return value
    ```

    In that context, we also use the new operations:
    `list[int]`, `dict[int, int]`, etc.

    And in the parameters:
    `smbo: "SMBO"` => `smbo: SMBO`
    
    Please change if you see the old style.

- Remove `self.logger` in all classes. Create on top a global logger:
logger = get_logger(__name__) (from smac.utils.logging import get_logger)

- Remove `rng` from init and replace it with `seed: int = 0`. Create new `rng = np.random.RandomState(seed)` if needed.

- Comments should start upper-case. For example: `# This is a comment`.
If multiple sentences are used, use punctuation. For example: `# This is one sentence. This is another.`

- Docstring structure is shown in the following. Please start the general comment and the parameter description
upper-case.

    ```
    """
    First a general comment.

    Note
    ----
    This could be a note here. Can also be a warning.

    Parameters
    ----------
    X : np.ndarray | None, defaults to None
        Explain what this parameter does.

    Returns
    -------
    optional_a_name_here : np.ndarray(N,1)
        Acquisition function values wrt X.
    """
    ```

- Everytime you see a variable you don't know, look it up and replace it with the variable provided
in the `variable renaming` below. If the variable is not listed below, add the variable and change it
accordingly. Just make sure it is persistent.

- If you find something with `run_obj == "runtime"` remove it. Since adaptive capping 
is bounded to `runtime`, you can remove that too.

- Please always check whether the docstrings are still correct. If not, correct them accordingly. Sometimes the ordering has changed.

- Get rid of `# type: ignore`.


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
max_config_fracs => max_config_ratio

## Algorithm Configuration
feature_array => instance_features
train_insts/train_instances => instances

## Intensification
min_chall => min_challenger
minR => min_config_calls
maxR => max_config_calls