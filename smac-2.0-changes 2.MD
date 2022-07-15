

cutoff/cutoff-time/cutoff_time => algorithm_walltime_limit
algo_runs_timelimit/tuner-timout/tuner_timout => cputime_limit
tae/ta/target_algorithm => algorithm
sls_max_steps => max_steps_local_search
rh/run_history/runhistory => runhistory
rand_prob/rand-prob => random_propability
feature_array => instance_features
train_insts => train_instances
num_trees => n_trees
output_dir => output_directory
ta_run_limit/ta-run-limit/runcount-limit/runcount_limit => n_runs


AbstractTAFunc => AbstractAlgorithmExecuter
ExecuteTAFuncArray/ExecuteTAFuncDict => AlgorithmExecuter
ExecuteTARunOld => CommandlineAlgorithmExecuter
run_obj => REMOVED
cost_for_crash => crash_cost
par_factor => REMOVED (depending on run_obj == runtime)
always_race_against => race_against
runhistory2epm => runhistory_transformer
Runhistory2EPM => RunhistoryTransformer

# Initial design
n_configs_x_params => n_configs_per_hyperparameter
max_config_fracs => 

min_chall => min_challenger
minR => min_config_calls
maxR => max_config_calls