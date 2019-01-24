import importlib
import logging
import numpy as np

# Import ConfigSpace and different types of parameters
from smac.configspace import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformFloatHyperparameter,\
 UniformIntegerHyperparameter, CategoricalHyperparameter
 
from ConfigSpace.conditions import InCondition

# Import SMAC-utilities
from smac.scenario.scenario import Scenario
from smac.facade.smac_facade import SMAC
###
from smac.facade.borf_facade import BORF
from smac.optimizer.random_configuration_chooser import ChooserProb
###
from smac.optimizer.acquisition import LCB, EI, LogEI, PI
from smac.initial_design.latin_hypercube_design import LHDesign
from smac.initial_design.sobol_design import SobolDesign
from smac.initial_design.default_configuration_design import DefaultConfiguration
from smac.runhistory.runhistory2epm import RunHistory2EPM4InvScaledCost, \
RunHistory2EPM4LogScaledCost, RunHistory2EPM4Cost

def wrapper(x, **kwargs):
    return benchmark.objective_function(x)['function_value']

def run_smac(smac_config):

    # TODO: treat it as non-deterministic

    print(smac_config)
    # Build Configuration Space which defines all parameters and their ranges
    cs = ConfigurationSpace()
    
    x0 = UniformFloatHyperparameter("x0", -5, 5, default_value=-3)
    x1 = UniformFloatHyperparameter("x1", -5, 5, default_value=-4)
    cs.add_hyperparameters([x0,x1])
    
    
    # Scenario object
    scenario = Scenario({"run_obj": "quality",   # we optimize quality (alternatively runtime)
                         "runcount-limit": num_function_evals,  # maximum function evaluations
                         "cs": cs,               # configuration space
                         "deterministic": "true",
                         "output_dir": None
                         })
    
    # Optimize, using a SMAC-object
    print("Optimizing! Depending on your machine, this might take a few minutes.")
    smac = SMAC(scenario=scenario, 
                rng=np.random.RandomState(42),
                tae_runner=wrapper,
                initial_design=smac_config['init_design'],
                initial_design_kwargs={'n_configs_x_params':smac_config['n_configs_x_params'],
                                       'max_config_fracs':1.0},
                random_configuration_chooser_kwargs={'prob': smac_config['rand_prob']},
                runhistory2epm=smac_config['y_trans'],
                model_kwargs={'num_trees': smac_config['num_trees'],
                              'log_y': smac_config['log_y'],
                              'do_bootstrapping': smac_config['do_bootstrapping'],
                              'ratio_features': smac_config['ratio_features'],
                              'min_samples_split': smac_config['min_samples_split'],
                              'min_samples_leaf': smac_config['min_samples_leaf']},
                acquisition_function=smac_config['acq_func'],
                acquisition_function_kwargs={'par':smac_config['par'] if smac_config['par'] is not None else smac_config['lcb_par']}
            )
    
    # hard coded for artificial functions
    smac.solver.scenario.intensification_percentage = 1e-10
    smac.solver.intensifier.min_chall = 1
    smac.solver.acq_optimizer.n_sls_iterations = 10
    smac.solver.scenario.acq_opt_challengers = 1000
    smac.solver.predict_incumbent = True
    
    #import warnings
    #warnings.filterwarnings('error')
    #import traceback
    #try:
    #    incumbent = smac.optimize()
    #except Warning:
    #    traceback.print_exc()
    incumbent = smac.optimize()
         
    final_value = smac.solver.runhistory.get_cost(incumbent)
    
    traj = smac.get_trajectory()
    
    t_last = 0
    c_last = 0
    auc = 0
    for id_, t_entry in enumerate(traj):
        if id_ == 0: # skip first dummy entry
            continue
        c = np.log10(t_entry.train_perf)
        t = t_entry.ta_runs
        auc += (t - t_last) * c_last
        t_last = t
        c_last = c
    
    auc += (num_function_evals - t_last) * c_last 
    
    return auc / num_function_evals # average incumbent value 

HPOLIB_SYNTH_FUNCTIONS = {
    "Branin": "hpolib.benchmarks.synthetic_functions.branin",
    #"Bohachevsky": "hpolib.benchmarks.synthetic_functions.bohachevsky",
    "Camelback": "hpolib.benchmarks.synthetic_functions.camelback",
    #"Forrester": "hpolib.benchmarks.synthetic_functions.forrester",
    "GoldsteinPrice": "hpolib.benchmarks.synthetic_functions.goldstein_price",
    "Hartmann3": "hpolib.benchmarks.synthetic_functions.hartmann3",
    "Hartmann6": "hpolib.benchmarks.synthetic_functions.hartmann6",
    #"Levy1D": "hpolib.benchmarks.synthetic_functions.levy",
    "Levy2D": "hpolib.benchmarks.synthetic_functions.levy",
    "Rosenbrock2D": "hpolib.benchmarks.synthetic_functions.rosenbrock",
    "Rosenbrock5D": "hpolib.benchmarks.synthetic_functions.rosenbrock",
    #"Rosenbrock10D": "hpolib.benchmarks.synthetic_functions.rosenbrock",
    "SinOne": "hpolib.benchmarks.synthetic_functions.sin_one",
    "SinTwo": "hpolib.benchmarks.synthetic_functions.sin_two"
}

benchmark_name = "Branin"
seed = 42
module = importlib.import_module(HPOLIB_SYNTH_FUNCTIONS[benchmark_name])
benchmark = getattr(module, benchmark_name)(rng=1)
print("Run SMAC on %s with seed %d" %(benchmark.get_meta_information()["name"], seed))

cs = benchmark.get_configuration_space()
cs.random = np.random.RandomState(seed=seed)

num_function_evals = benchmark.get_meta_information()['num_function_evals']

logging.basicConfig(level=logging.INFO)  # logging.DEBUG for debug output

cs = ConfigurationSpace()

# Random Forest hyperparameters
num_trees = UniformIntegerHyperparameter("num_trees", 2, 100, default_value=10, log=True) 
log_y = CategoricalHyperparameter("log_y", [True, False], default_value=False)
do_bootstrapping = CategoricalHyperparameter("do_bootstrapping", [True, False], default_value=True)
ratio_features = UniformFloatHyperparameter('ratio_features', 0.5, 1.0, default_value=1.0)
min_samples_split = UniformIntegerHyperparameter("min_samples_split", 1, 100, default_value=2, log=True) 
min_samples_leaf = UniformIntegerHyperparameter("min_samples_leaf", 1, 100, default_value=1, log=True)

cs.add_hyperparameters([num_trees, log_y, do_bootstrapping, ratio_features, min_samples_split, min_samples_leaf])

# acquisition function
acq_func = CategoricalHyperparameter("acq_func", [LCB, EI, PI], default_value=EI)
lcb_par = UniformFloatHyperparameter("lcb_par", 10**-5, 10**5, default_value=0.1, log=True)
par = UniformFloatHyperparameter("par", 0, 10, default_value=0)
cs.add_hyperparameters([acq_func, lcb_par, par])

use_lcb_par = InCondition(child=lcb_par, parent=acq_func, values=[LCB])
use_par = InCondition(child=par, parent=acq_func, values=[EI, PI])
cs.add_conditions([use_lcb_par, use_par])

# y transformation
y_trans = CategoricalHyperparameter("y_trans", 
                                    [RunHistory2EPM4InvScaledCost, RunHistory2EPM4LogScaledCost, RunHistory2EPM4Cost], 
                                    default_value=RunHistory2EPM4LogScaledCost)
cs.add_hyperparameters([y_trans])

# initial design
init_design = CategoricalHyperparameter("init_design", [LHDesign, SobolDesign, DefaultConfiguration], default_value=SobolDesign)
n_configs_x_params = UniformIntegerHyperparameter("n_configs_x_params", 0, 20, default_value=10)
cs.add_hyperparameters([init_design, n_configs_x_params])

# randomly sampled configuration interleaved
rand_prob = UniformFloatHyperparameter("rand_prob", 0.0, 0.5, default_value=0.0)
cs.add_hyperparameters([rand_prob])

#def_value = run_smac(cs.get_default_configuration())

# Scenario object
scenario = Scenario({"run_obj": "quality",   # we optimize quality (alternatively runtime)
                     "runcount-limit": 100,  # maximum function evaluations
                     "cs": cs,               # configuration space
                     "deterministic": "true",
                     "output_dir": None
                     })

# Optimize, using a SMAC-object
smac = SMAC(scenario=scenario, 
        rng=np.random.RandomState(42),
        tae_runner=run_smac,
        )
smac.output_dir = None

meta_inc  = smac.optimize()
run_smac(meta_inc)

