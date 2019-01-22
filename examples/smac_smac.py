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
from smac.optimizer.acquisition import LCB, EI, LogEI, PI
from smac.initial_design.latin_hypercube_design import LHDesign
from smac.initial_design.sobol_design import SobolDesign
from smac.initial_design.default_configuration_design import DefaultConfiguration
from smac.runhistory.runhistory2epm import RunHistory2EPM4InvScaledCost, \
RunHistory2EPM4LogCost, RunHistory2EPM4Cost

def rosenbrock_2d(x):
    """ The 2 dimensional Rosenbrock function as a toy model
    The Rosenbrock function is well know in the optimization community and
    often serves as a toy problem. It can be defined for arbitrary
    dimensions. The minimium is always at x_i = 1 with a function value of
    zero. All input parameters are continuous. The search domain for
    all x's is the interval [-5, 5].
    """
    x1 = x["x0"]
    x2 = x["x1"]

    val = 100. * (x2 - x1 ** 2.) ** 2. + (1 - x1) ** 2.
    return val


def run_smac(smac_config):

    # TODO: treat it as non-deterministic

    print(smac_config)
    # Build Configuration Space which defines all parameters and their ranges
    cs = ConfigurationSpace()
    
    x0 = UniformFloatHyperparameter("x0", -5, 5, default_value=-3)
    x1 = UniformFloatHyperparameter("x1", -5, 5, default_value=-4)
    cs.add_hyperparameters([x0,x1])
    
    
    max_runs = 200
    # Scenario object
    scenario = Scenario({"run_obj": "quality",   # we optimize quality (alternatively runtime)
                         "runcount-limit": max_runs,  # maximum function evaluations
                         "cs": cs,               # configuration space
                         "deterministic": "true"
                         })
    
    # Optimize, using a SMAC-object
    print("Optimizing! Depending on your machine, this might take a few minutes.")
    smac = SMAC(scenario=scenario, 
                rng=np.random.RandomState(42),
                tae_runner=rosenbrock_2d,
                initial_design=smac_config['init_design'],
                initial_design_kwargs={'n_configs_x_params':smac_config['n_configs_x_params'],
                                       'max_config_fracs':1.0},
                runhistory2epm=RunHistory2EPM4InvScaledCost,
                model_kwargs={'num_trees': smac_config['num_trees'],
                              'log_y': smac_config['log_y'],
                              'do_bootstrapping': smac_config['do_bootstrapping'],
                              'ratio_features': smac_config['ratio_features'],
                              'min_samples_split': smac_config['min_samples_split'],
                              'min_samples_leaf': smac_config['min_samples_leaf']},
                acquisition_function_optimizer_kwargs={'max_steps':100},
                acquisition_function=smac_config['acq_func'],
                acquisition_function_kwargs={'par':smac_config['par'] if smac_config['par'] is not None else smac_config['lcb_par']}
            )
    
    smac.solver.scenario.intensification_percentage = 1e-10
    smac.solver.intensifier.min_chall = 1
    
    incumbent = smac.optimize()
    final_value = smac.solver.runhistory.get_cost(incumbent)
    
    traj = smac.get_trajectory()
    
    t_last = 0
    c_last = 0
    auc = 0
    for id_, t_entry in enumerate(traj):
        if id_ == 0: # skip first dummy entry
            continue
        c = t_entry.train_perf
        t = t_entry.ta_runs
        auc += (t - t_last) * c_last
        t_last = t
        c_last = c
    
    auc += (max_runs - t_last) * c_last 
    
    return auc / max_runs # average incumbent value 


logging.basicConfig(level=logging.INFO)  # logging.DEBUG for debug output

cs = ConfigurationSpace()

# Random Forest hyperparameters
num_trees = UniformIntegerHyperparameter("num_trees", 2, 100, default_value=10, log=True) 
log_y = CategoricalHyperparameter("log_y", [True, False], default_value=False)
do_bootstrapping = CategoricalHyperparameter("do_bootstrapping", [True, False], default_value=False)
ratio_features = UniformFloatHyperparameter('ratio_features', 0.5, 1.0, default_value=1.0)
min_samples_split = UniformIntegerHyperparameter("min_samples_split", 2, 100, default_value=2, log=True) 
min_samples_leaf = UniformIntegerHyperparameter("min_samples_leaf", 2, 100, default_value=2, log=True)

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
y_trans = CategoricalHyperparameter("y_trans", [RunHistory2EPM4InvScaledCost, RunHistory2EPM4LogCost, RunHistory2EPM4Cost], default_value=RunHistory2EPM4Cost)
cs.add_hyperparameters([y_trans])

# initial design
init_design = CategoricalHyperparameter("init_design", [LHDesign, SobolDesign, DefaultConfiguration], default_value=LHDesign)
n_configs_x_params = UniformIntegerHyperparameter("n_configs_x_params", 0, 20, default_value=2)
cs.add_hyperparameters([init_design, n_configs_x_params])


#def_value = run_smac(cs.get_default_configuration())

# Scenario object
scenario = Scenario({"run_obj": "quality",   # we optimize quality (alternatively runtime)
                     "runcount-limit": 10,  # maximum function evaluations
                     "cs": cs,               # configuration space
                     "deterministic": "true",
                     "output_dir": None
                     })

# Optimize, using a SMAC-object
smac = SMAC(scenario=scenario, 
        rng=np.random.RandomState(42),
        tae_runner=run_smac,
        )

meta_inc  = smac.optimize()
run_smac(meta_inc)

