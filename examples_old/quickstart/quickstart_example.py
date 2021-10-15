"""
==========
Quickstart
==========
"""

###############################################################################
# If you have not installed *SMAC* yet take a look at the
# :ref:`installation instructions <installation>` and make sure that all the requirements are fulfilled.
# Examples to illustrate the usage of *SMAC* - either by reading in a scenario file,
# or by directly using *SMAC* in Python - are provided in the examples-folder.
#
# To get started, we will walk you through a few examples.
#
# * First, we explain the basic usage of *SMAC* by optimizing the
# :ref:`Branin <branin-example>`-function as a toy example.
# * Second, we explain the usage of *SMAC* within Python by optimizing a :ref:`Support Vector Machine <svm-example>`.
# * Third, we show a real-world example, using an algorithm-wrapper to optimize the
# :ref:`SPEAR SAT-solver <spear-example>`.
#
# .. _TAE: ../../tae.html


###############################################################################
# .. _branin-example:
#
# Branin
# ======
# First of, we'll demonstrate the usage of *SMAC* on the minimization of
# `the standard 2-dimensional continuous test function Branin <https://www.sfu.ca/~ssurjano/branin.html>`_.
#

import numpy as np


def branin(x):
    x1 = x[0]
    x2 = x[1]
    a = 1.
    b = 5.1 / (4. * np.pi ** 2)
    c = 5. / np.pi
    r = 6.
    s = 10.
    t = 1. / (8. * np.pi)
    ret = a * (x2 - b * x1 ** 2 + c * x1 - r) ** 2 + s * (1 - t) * np.cos(x1) + s
    return ret

###############################################################################
# This example aims to explain the basic usage of *SMAC*. There are different ways to use *SMAC*:
#
# f_min-wrapper
# ~~~~~~~~~~~~~
# The easiest way to use *SMAC* is to use the :func:`f_min SMAC wrapper
# <smac.facade.func_facade.fmin_smac>`.
# We import the fmin-function and wrap it around are simple branin function.
#


from smac.facade.func_facade import fmin_smac

x, cost, _ = fmin_smac(func=branin,  # function
                       x0=[0, 0],  # default configuration
                       bounds=[(-5, 10), (0, 15)],  # limits
                       maxfun=10,  # maximum number of evaluations
                       rng=3)  # random seed
print("Optimum at {} with cost of {}".format(x, cost))

###############################################################################
# This way, you can optimize a blackbox-function with minimal effort. However, to
# use all of *SMAC's* capability, a :ref:`scenario` object should be used.
#
# Command line
# ~~~~~~~~~~~~
# A more evolved example can be found in ``examples/quickstart/branin/``.
# In this directory you can find:
#
# * ``branin.py``
# * ``cmdline_wrapper.py``
# * ``scenario.txt``
# * ``param_config_space.pcs``
#
# To run the example scenario, change into the root directory of *SMAC* and type the following commands:
#
#
# .. code-block:: bash
#
#     cd examples/quickstart/branin
#     python ../../scripts/smac --scenario scenario.txt
#
# The Python command runs *SMAC* with the specified scenario. The scenario file consists of the following lines:
#
# .. literalinclude:: ../../../examples/quickstart/branin/scenario.txt
#
# The **algo** parameter specifies how *SMAC* calls the target algorithm to be optimized.
# This is further explained in the chapter about the Target Algorithm Evaluator (:ref:`TAE <tae>`).
# An algorithm call by *SMAC* is of the following format:
#
#     .. code-block:: bash
#
#         <algo> <instance> <instance specific> <cutoff time> <runlength> <seed> <algorithm parameters>
#         python branin.py 0 0 999999999.0 0 1148756733 -x1 -1.1338595629 -x2 13.8770222718
#
# The **paramfile** parameter tells *SMAC* which Parameter Configuration Space
# :ref:`PCS <paramcs>`-file to use. This file contains a list of the algorithm's parameters,
# their domains and default values:
#
#     .. literalinclude:: ../../../examples/quickstart/branin/param_config_space.pcs
#
# ``x1`` and ``x2`` are both continuous parameters. ``x1`` can take any real value
# in the range ``[-5, 10]``, ``x2`` in the range ``[0, 15]`` and both have
# the default value ``0``.
#
# The **run_obj** parameter specifies what *SMAC* is supposed to **optimize**. Here we optimize solution quality.
# The **runcount_limit** specifies the maximum number of algorithm calls.
#
# *SMAC* reads the results from the command line output. The wrapper returns the
# results of the algorithm in a specific format:
#
#     .. code-block:: bash
#
#         Result for SMAC: <STATUS>, <running time>, <runlength>, <quality>, <seed>, <instance-specifics>
#         Result for SMAC: SUCCESS, 0, 0, 48.948190, 1148756733
#
#     | The second line is the result of the above listed algorithm call.
#     | *STATUS:* can be either *SUCCESS*, *CRASHED*, *SAT*, *UNSAT*, *TIMEOUT*
#     | *running time:* is the measured running time for an algorithm call
#     | *runlength:* is the number of steps needed to find a solution
#     | *quality:* the solution quality
#     | *seed:* the seed that was used with the algorithm call
#     | *instance-specifics:* additional information
#
# *SMAC* will terminate with the following output:
#
#     .. code-block:: bash
#
#         INFO:intensifier:Updated estimated error of incumbent on 122 runs: 0.5063
#         DEBUG:root:Remaining budget: inf (wallclock), inf (ta costs), -6.000000 (target runs)
#         INFO:Stats:##########################################################
#         INFO:Stats:Statistics:
#         INFO:Stats:#Target algorithm runs: 506
#         INFO:Stats:Used wallclock time: 44.00 sec
#         INFO:Stats:Used target algorithm runtime: 0.00 sec
#         INFO:Stats:##########################################################
#         INFO:SMAC:Final Incumbent: Configuration:
#           x1, Value: 9.556406137303922
#           x2, Value: 2.429138598022513
#
# Furthermore, *SMACs* trajectory and runhistory will be stored in ``branin/``.
#

###############################################################################
# .. _svm-example:
#
# Using *SMAC* in Python: SVM
# ---------------------------
# To explain the use of *SMAC* within Python, let's look at a real-world example,
# optimizing the hyperparameters of a Support Vector Machine (SVM) trained on the widely known `IRIS-dataset
# <https://en.wikipedia.org/wiki/Iris_flower_data_set>`_.
# This example is located in :code:`examples/general/svm.py`.
#
# To use *SMAC* directly with Python, we first import the necessary modules
#

from ConfigSpace.conditions import InCondition
from ConfigSpace.hyperparameters import CategoricalHyperparameter, \
    UniformFloatHyperparameter, UniformIntegerHyperparameter
from sklearn import svm, datasets
from sklearn.model_selection import cross_val_score

# Import ConfigSpace and different types of parameters
from smac.configspace import ConfigurationSpace
from smac.facade.smac_hpo_facade import SMAC4HPO
# Import SMAC-utilities
from smac.scenario.scenario import Scenario

# We load the iris-dataset (a widely used benchmark)
iris = datasets.load_iris()

###############################################################################
# Next we need a configuration space from which to sample from

# Build Configuration Space which defines all parameters and their ranges
cs = ConfigurationSpace()

# We define a few possible types of SVM-kernels and add them as "kernel" to our cs
kernel = CategoricalHyperparameter("kernel", ["linear", "rbf", "poly", "sigmoid"], default_value="poly")
cs.add_hyperparameter(kernel)

# There are some hyperparameters shared by all kernels
C = UniformFloatHyperparameter("C", 0.001, 1000.0, default_value=1.0)
shrinking = CategoricalHyperparameter("shrinking", ["true", "false"], default_value="true")
cs.add_hyperparameters([C, shrinking])

# Others are kernel-specific, so we can add conditions to limit the searchspace
degree = UniformIntegerHyperparameter("degree", 1, 5, default_value=3)  # Only used by kernel poly
coef0 = UniformFloatHyperparameter("coef0", 0.0, 10.0, default_value=0.0)  # poly, sigmoid
cs.add_hyperparameters([degree, coef0])
use_degree = InCondition(child=degree, parent=kernel, values=["poly"])
use_coef0 = InCondition(child=coef0, parent=kernel, values=["poly", "sigmoid"])
cs.add_conditions([use_degree, use_coef0])

# This also works for parameters that are a mix of categorical and values from a range of numbers
# For example, gamma can be either "auto" or a fixed float
gamma = CategoricalHyperparameter("gamma", ["auto", "value"], default_value="auto")  # only rbf, poly, sigmoid
gamma_value = UniformFloatHyperparameter("gamma_value", 0.0001, 8, default_value=1)
cs.add_hyperparameters([gamma, gamma_value])
# We only activate gamma_value if gamma is set to "value"
cs.add_condition(InCondition(child=gamma_value, parent=gamma, values=["value"]))
# And again we can restrict the use of gamma in general to the choice of the kernel
cs.add_condition(InCondition(child=gamma, parent=kernel, values=["rbf", "poly", "sigmoid"]))

print(cs)

###############################################################################
# Of course we also define a function to evaluate the configured SVM on the IRIS-dataset.
# Some options, such as the *kernel* or *C*, can be passed directly.
# Others, such as *gamma*, need to be translated before the call to the SVM.


def svm_from_cfg(cfg):
    """ Creates a SVM based on a configuration and evaluates it on the
    iris-dataset using cross-validation.

    Parameters:
    -----------
    cfg: Configuration (ConfigSpace.ConfigurationSpace.Configuration)
        Configuration containing the parameters.
        Configurations are indexable!

    Returns:
    --------
    A crossvalidated mean score for the svm on the loaded data-set.
    """
    # For deactivated parameters, the configuration stores None-values.
    # This is not accepted by the SVM, so we remove them.
    cfg = {k: cfg[k] for k in cfg if cfg[k]}
    # We translate boolean values:
    cfg["shrinking"] = True if cfg["shrinking"] == "true" else False
    # And for gamma, we set it to a fixed value or to "auto" (if used)
    if "gamma" in cfg:
        cfg["gamma"] = cfg["gamma_value"] if cfg["gamma"] == "value" else "auto"
        cfg.pop("gamma_value", None)  # Remove "gamma_value"

    clf = svm.SVC(**cfg, random_state=42)
    scores = cross_val_score(clf, iris.data, iris.target, cv=5)
    return 1 - np.mean(scores)  # Minimize!


def_value = svm_from_cfg(cs.get_default_configuration())
print("Default Value: %.2f" % (def_value))

###############################################################################
# We need a Scenario-object to configure the optimization process.
# We provide a :ref:`list of possible options <scenario>` in the scenario.
#
# The initialization of a scenario in the code uses the same keywords as a
# scenario-file, which we used in the Branin example.
#

scenario = Scenario({"run_obj": "quality",  # we optimize quality (alternatively runtime)
                     "runcount-limit": 5,  # max. number of function evaluations; for this example set to a low number
                     "cs": cs,  # configuration space
                     "deterministic": "true"
                     })

###############################################################################
# Now we're ready to create a *SMAC*-instance, which handles the Bayesian
# Optimization-loop and calculates the incumbent.
# To automatically handle the exploration of the search space
# and evaluation of the function, SMAC needs as inputs the scenario object
# as well as the function.
#
smac = SMAC4HPO(scenario=scenario, rng=np.random.RandomState(42),
                tae_runner=svm_from_cfg)

incumbent = smac.optimize()

inc_value = svm_from_cfg(incumbent)
print("Optimized Value: %.2f" % (inc_value))

###############################################################################
# Internally SMAC keeps track of the number of algorithm calls and the remaining time budget via a Stats object.
#
# After successful execution of the optimization loop the Stats object outputs the result of the loop.
#
# .. code-block:: bash
#
#     INFO:smac.stats.stats.Stats:##########################################################
#     INFO:smac.stats.stats.Stats:Statistics:
#     INFO:smac.stats.stats.Stats:#Incumbent changed: 88
#     INFO:smac.stats.stats.Stats:#Target algorithm runs: 200 / 200.0
#     INFO:smac.stats.stats.Stats:Used wallclock time: 41.73 / inf sec
#     INFO:smac.stats.stats.Stats:Used target algorithm runtime: 17.44 / inf sec
#     INFO:smac.stats.stats.Stats:##########################################################
#     INFO:smac.facade.smac_facade.SMAC:Final Incumbent: Configuration:
#       C, Value: 357.4171743725004
#       coef0, Value: 9.593372746957046
#       degree, Value: 1
#       gamma, Value: 'value'
#       gamma_value, Value: 0.0029046235175726105
#       kernel, Value: 'poly'
#       shrinking, Value: 'false'
#
# We further query the target function at the incumbent, using the function evaluator
# so that as final output we can see the error value of the incumbent.
#
# .. code-block:: bash
#
#    Optimized Value: 0.02
#
# As a bonus, we can validate our results. This is more useful when optimizing on
# instances, but we include the code so it is easily applicable for any usecase.
#
# We can also validate our results (though this makes a lot more sense with instances)

smac.validate(config_mode='inc',  # We can choose which configurations to evaluate
              # instance_mode='train+test',  # Defines what instances to validate
              repetitions=100,  # Ignored, unless you set "deterministic" to "false" in line 95
              n_jobs=1)  # How many cores to use in parallel for optimization

###############################################################################
# .. _spear-example:
#
# Spear-QCP
# ---------
# For this example we use *SMAC* to optimize the runtime required by the SAT solver
# `Spear <http://www.domagoj-babic.com/index.php/ResearchProjects/Spear>`_
# to solve a small subset of the QCP-dataset.
#
# In *SMACs* root-directory type:
#
# .. code-block:: bash
#
#     cd examples/quickstart/spear_qcp && ls -l
#
# In this folder you see the following files and directories:
#
# * **features.txt**:
#
#     The :ref:`feature file <feature>` contains the features for each instance in a csv-format.
#
#     +--------------------+--------------------+--------------------+-----+
#     |      instance      | name of feature 1  | name of feature 2  | ... |
#     +====================+====================+====================+=====+
#     | name of instance 1 | value of feature 1 | value of feature 2 | ... |
#     +--------------------+--------------------+--------------------+-----+
#     |         ...        |          ...       |          ...       | ... |
#     +--------------------+--------------------+--------------------+-----+
#
# * **instances.txt**
#   The ref:`instance` contains the names of all
#   instances one might want to consider during the optimization process.
#
# * **scenario.txt**
#   The :ref:`scenario <scenario>` file contains all the necessary information about the configuration scenario at hand.
#
#   .. literalinclude:: ../../../examples/quickstart/spear_qcp/scenario.txt
#
#   For this example the following options are used:
#
#   * *algo:*
#
#     .. code-block:: bash
#
#         python -u ./target_algorithm/scripts/SATCSSCWrapper.py \
#             --mem-limit 1024 \
#             --script ./target_algorithm/spear-python/spearCSSCWrapper.py
#
#     This specifies the wrapper that *SMAC* executes with a pre-specified
#     syntax in order to evaluate the algorithm to be optimized.
#     This wrapper script takes an instantiation of the parameters as input,
#     runs the algorithm with these parameters, and returns
#     the cost of running the algorithm; since every algorithm has a different
#     input and output format, this wrapper acts as an interface between the
#     algorithm and *SMAC*, which executes the wrapper through a command line call.
#
#     An example call would look something like this:
#
#     .. code-block:: bash
#
#         <algo> <instance> <instance specifics> <cutoff time> <runlength> <seed> <algorithm parameters>
#
#     For *SMAC* to be able to interpret the results of the algorithm run,
#     the wrapper returns the results of the algorithm run as follows:
#
#     .. code-block:: bash
#
#         STATUS, running time, runlength, quality, seed, instance-specifics
#
#   * *paramfile:*
#
#     This parameter specifies which pcs-file to use and where it is located.
#
#     The PCS-file specifies the Parameter Configuration Space file, which
#     lists the algorithm's parameters, their domains, and default values (one per line)
#
#     In this example we are dealing with 26 parameters of which 12 are categorical
#     and 14 are continuous. Out of these 26
#     parameters, 9 parameters are conditionals (they are only active if
#     their parent parameter takes on a certain value).
#
#   * *execdir:* Specifies the directory in which the target algorithm will be run.
#
#   * *deterministic:* Specifies if the target algorithm is deterministic.
#
#   * *run_obj:* This parameter tells *SMAC* what is to be optimized, i.e. running time or (solution) quality.
#
#   * *overall_obj:* Specifies how to evaluate the error values, e.g as mean or PARX.
#
#   * *cutoff_time:* The target algorithms cutoff time.
#
#   * *wallclock-limit:* This parameter is used to give the time budget for the configuration task in seconds.
#
#   * *instance_file:* See instances.txt above.
#
#   * *feature_file:* See features.txt above.
#
# * **run.sh**
#   A shell script calling *SMAC* with the following command:
#
#   .. code-block:: bash
#
#       python ../../scripts/smac --scenario scenario.txt --verbose DEBUG``
#
#   This runs *SMAC* with the scenario options specified in the scenario.txt file.
#
# * **target_algorithms** contains the wrapper and the executable for Spear.
#
# * **instances** folder contains the instances on which *SMAC* will configure Spear.
#
# To run the example type one of the two commands below into a terminal:
#
# .. code-block:: bash
#
#     bash run.sh
#     python ../../scripts/smac --scenario scenario.txt
#
# *SMAC* will run for a few seconds and generate a lot of logging output.
# After *SMAC* finished the configuration process you'll get some final statistics about the configuration process:
#
# .. code-block:: bash
#
#    INFO:    ##########################################################
#    INFO:    Statistics:
#    INFO:    #Incumbent changed: 2
#    INFO:    #Target algorithm runs: 17 / inf
#    INFO:    Used wallclock time: 35.38 / 30.00 sec
#    INFO:    Used target algorithm runtime: 20.10 / inf sec
#    INFO:    ##########################################################
#    INFO:    Final Incumbent: Configuration:
#      sp-clause-activity-inc, Value: 0.9846527087294622
#      sp-clause-decay, Value: 1.1630090545101102
#      sp-clause-del-heur, Value: '0'
#      sp-first-restart, Value: 30
#      sp-learned-clause-sort-heur, Value: '10'
#      sp-learned-clauses-inc, Value: 1.2739749314202675
#      sp-learned-size-factor, Value: 0.8355014264152971
#      sp-orig-clause-sort-heur, Value: '1'
#      sp-phase-dec-heur, Value: '0'
#      sp-rand-phase-dec-freq, Value: '0.01'
#      sp-rand-phase-scaling, Value: 0.3488055907688382
#      sp-rand-var-dec-freq, Value: '0.05'
#      sp-rand-var-dec-scaling, Value: 0.46427056372562864
#      sp-resolution, Value: '0'
#      sp-restart-inc, Value: 1.7510945705535836
#      sp-update-dec-queue, Value: '1'
#      sp-use-pure-literal-rule, Value: '0'
#      sp-var-activity-inc, Value: 0.9000377944957962
#      sp-var-dec-heur, Value: '14'
#      sp-variable-decay, Value: 1.8292433459523076
#
#
#
# The statistics further show the used wallclock time, target algorithm running
# time and the number of executed target algorithm runs,
# and the corresponding budgets---here we exhausted the wallclock time budget.
#
# The directory in which you invoked *SMAC* now contains a new folder called **SMAC3-output_YYYY-MM-DD_HH:MM:SS**.
# The .json file contains the information about the target algorithms *SMAC* just
# executed. In this file you can see the *status* of the algorithm run, *misc*,
# the *instance* on which the algorithm was evaluated, which *seed* was used,
# how much *time* the algorithm needed and with which *configuration* the algorithm
# was run.
# In the folder *SMAC* generates a file for the runhistory, and two files for the trajectory.
#
#
# .. _hydra-example:
#
# Hydra on Spear-QCP
# ------------------
#
# For this example we use *Hydra* to build a portfolio on the same example data
# presented in :ref:`Spear-QCP <spear-example>`
# Hydra is a portfolio builder that aims to build a portfolio by iteratively adding complementary configurations to the
# already existing portfolio. To select these complementary configurations *Hydra* compares new configurations to the
# portfolio and only considers configurations that improve the portfolio performance.
# In the first iteration Hydra runs standard *SMAC* to determine a well performing configuration
# across all instances as a starting point for the portfolio. In following iterations
# *Hydra* adds one configuration that
# improves the portfolio performance.
#
# To run Hydra for three iterations you can run the following code in the spear-qcp example folder.
#
# .. code-block:: bash
#
#       python ../../scripts/smac --scenario scenario.txt --verbose DEBUG --mode Hydra --hydra_iterations 3
#
# As the individual SMAC scenario takes 30 seconds to run Hydra will run for ~90 seconds on this example.
# You will see the same output to the terminal as with standard SMAC. In the folder
# where you executed the above command,
# you will find a *hydra-output-yyy-mm-dd_hh:mm:ss_xyz* folder. This folder contains the results of all three performed
# SMAC runs, as well as the resulting portfolio (as pkl file).
#
# The resulting portfolio can be used with any algorithm selector such as
# `AutoFolio <https://github.com/mlindauer/AutoFolio>`_
