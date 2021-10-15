:algo_runs_timelimit: Maximum amount of CPU-time used for optimization. Default: inf.
:always_race_default: Race new incumbents always against default configuration.
:cost_for_crash: Defines the cost-value for crashed runs on scenarios with quality as run-obj. Default: 2147483647.0.
:cutoff: Maximum runtime, after which the target algorithm is cancelled. **Required if *run_obj* is runtime.**
:deterministic: If true, SMAC assumes that the target function or algorithm is deterministic (the same static seed of 0 is always passed to the function/algorithm). If false, different random seeds are passed to the target function/algorithm.
:execdir: Specifies the path to the execution-directory. Default: ..
:feature_fn: Specifies the file with the instance-features.
:initial_incumbent: DEFAULT is the default from the PCS. Default: DEFAULT.
:memory_limit: Maximum available memory the target algorithm can occupy before being cancelled in MB.
:overall_obj: PARX, where X is an integer defining the penalty imposed on timeouts (i.e. runtimes that exceed the *cutoff-time*). Default: par10.
:pcs_fn: Specifies the path to the PCS-file.
:run_obj: Defines what metric to optimize. When optimizing runtime, *cutoff_time* is required as well.
:save_instantly: If true, runhistory and stats are saved immediately on changes. Otherwise, runhistory and states are only saved once after the optimization process has finished.
:ta: Specifies the target algorithm call that *SMAC* will optimize. Interpreted as a bash-command.
:ta_run_limit: Maximum number of algorithm-calls during optimization. Default: inf.
:test_inst_fn: Specifies the file with the test-instances.
:train_inst_fn: Specifies the file with the training-instances.
:wallclock_limit: Maximum amount of wallclock-time used for optimization. Default: inf.


