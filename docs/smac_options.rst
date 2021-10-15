:abort_on_first_run_crash: If true, *SMAC* will abort if the first run of the target algorithm crashes. Default: True.
:acq_opt_challengers: Number of challengers returned by acquisition function optimization. Also influences the number of randomly sampled configurations to optimized the acquisition function Default: 5000.
:always_race_default: Race new incumbents always against default configuration.
:hydra_iterations: number of hydra iterations. Only active if mode is set to Hydra Default: 3.
:input_psmac_dirs: For parallel SMAC, multiple output-directories are used.
:intens_adaptive_capping_slackfactork: Slack factor of adpative capping (factor * adpative cutoff). Only active if obj is runtime. If set to very large number it practically deactivates adaptive capping. Default: 1.2.
:intens_min_chall: Minimal number of challengers to be considered in each intensification run (> 1). Set to 1 and in combination with very small intensification-percentage. it will deactivate randomly sampled configurations (and hence, extrapolation of random forest will be an issue.) Default: 2.
:intensification_percentage: The fraction of time to be used on intensification (versus choice of next Configurations). Default: 0.5.
:limit_resources: If true, *SMAC* will use pynisher to limit time and memory for the target algorithm. Allows SMAC to use all resources available. Applicable only to func TAEs. Set to 'True' by default. (Use with caution!) Default: True.
:maxR: Maximum number of calls per configuration. Default: 2000.
:minR: Minimum number of calls per configuration. Default: 1.
:output_dir: Specifies the output-directory for all emerging files, such as logging and results. Default: smac3-output_2021-10-15_17:08:00_490278.
:rand_prob: probablity to run a random configuration instead of configuration optimized on the acquisition function Default: 0.5.
:random_configuration_chooser: path to a python module containing a class`RandomConfigurationChooserImpl` implementingthe interface of `RandomConfigurationChooser`
:rf_do_bootstrapping: Use bootstraping in random forest. Default: True.
:rf_max_depth: Maximum depth of each tree in the random forest. Default: 20.
:rf_min_samples_leaf: Minimum required number of samples in each leaf of a tree in the random forest. Default: 3.
:rf_min_samples_split: Minimum number of samples to split for building a tree in the random forest. Default: 3.
:rf_num_trees: Number of trees in the random forest (> 1). Default: 10.
:rf_ratio_features: Ratio of sampled features in each split ([0.,1.]). Default: 0.8333333333333334.
:shared_model: Whether to run SMAC in parallel mode.
:sls_max_steps: Maximum number of local search steps in one iteration during the optimization of the acquisition function.
:sls_n_steps_plateau_walk: Maximum number of steps on plateaus during the optimization of the acquisition function. Default: 10.
:transform_y: Transform all observed cost values via log-transformations or inverse scaling. The subfix "s" indicates that SMAC scales the y-values accordingly to apply the transformation. Default: NONE.
:use_ta_time: Instead of measuring SMAC's wallclock time, only consider time reported by the target algorithm (ta).


