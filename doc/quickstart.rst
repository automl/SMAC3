.. _scenario: options.html#scenario
.. _PCS: options.html#paramcs
.. _TAE: tae.html

Quick Start
===========
If you have not installed *SMAC* yet take a look at the `installation instructions <installation.html>`_ and make sure that all the requirements are fulfilled.
Examples to illustrate the usage of *SMAC* - either by reading in a scenario file, or by directly using *SMAC* in Python - are provided in the examples-folder.

To get started, we will walk you through a few examples.

* First, we explain the basic usage of *SMAC* by optimizing the `Branin`__-function as a toy example.
* Second, we explain the usage of *SMAC* within Python by optimizing a `Support Vector Machine`__.
* Third, we show a real-world example, using an algorithm-wrapper to optimize the `SPEAR SAT-solver`__.

__ branin-example_
__ svm-example_
__ spear-example_

.. _branin-example:

Branin
------
First of, we'll demonstrate the usage of *SMAC* on the minimization of a standard 2-dimensional continuous test function (`Branin <https://www.sfu.ca/~ssurjano/branin.html>`_).
This example aims to explain the basic usage of *SMAC*. There are different ways to use *SMAC*:

f_min-wrapper
~~~~~~~~~~~~~
The easiest way to use *SMAC* is to use the `f_min SMAC wrapper
<apidoc/smac.facade.func_facade.html#smac.facade.func_facade.fmin_smac>`_. It is
implemented in `examples/branin/branin_fmin.py` and requires no extra files. We
import the fmin-function and the Branin-function:

.. literalinclude:: ../examples/branin/branin_fmin.py
   :lines: 3-4 
   :lineno-match:

And run the f_min-function:

.. literalinclude:: ../examples/branin/branin_fmin.py
   :start-after: logging.basicConfig 
   :lineno-match:

This way, you can optimize a blackbox-function with minimal effort. However, to
use all of *SMAC's* capability, a scenario_ object should be used.

Command line
~~~~~~~~~~~~
A more evolved example can be found in ``examples/branin/``. In this directory you can find
a wrapper ``cmdline_wrapper.py``, a scenario-file ``scenario.txt``, and a
*PCS*-file ``param_config_space.pcs``.
To run the example scenario, change into the root directory of *SMAC* and type the following commands:

.. code-block:: bash

    cd examples/branin
    python ../../scripts/smac --scenario scenario.txt

The Python command runs *SMAC* with the specified scenario. The scenario file consists of the following lines:

    .. literalinclude:: ../examples/branin/scenario.txt

The **algo** parameter specifies how *SMAC* calls the target algorithm to be optimized.
This is further explained in the chapter about the `Target Algorithm Evaluator (TAE) <tae.html>`_.
An algorithm call by *SMAC* is of the following format:

    .. code-block:: bash

        <algo> <instance> <instance specific> <cutoff time> <runlength> <seed> <algorithm parameters>
        python branin.py 0 0 999999999.0 0 1148756733 -x1 -1.1338595629 -x2 13.8770222718
    
The **paramfile** parameter tells *SMAC* which Parameter Configuration Space (PCS_)-file to use. This file contains a list of the algorithm's parameters, their domains and default values:

    .. literalinclude:: ../examples/branin/param_config_space.pcs

    x1 and x2 are both continuous parameters. x1 can take any real value in the range [-5, 10], x2 in the range [0, 15] and both have the default value 0.

The **run_obj** parameter specifies what *SMAC* is supposed to **optimize**. Here we optimize solution quality.

The **runcount_limit** specifies the maximum number of algorithm calls.

*SMAC* reads the results from the command line output. The wrapper returns the
results of the algorithm in a specific format:

    .. code-block:: bash

        Result for SMAC: <STATUS>, <running time>, <runlength>, <quality>, <seed>, <instance-specifics>
        Result for SMAC: SUCCESS, 0, 0, 48.948190, 1148756733

    | The second line is the result of the above listed algorithm call.
    | *STATUS:* can be either *SUCCESS*, *CRASHED*, *SAT*, *UNSAT*, *TIMEOUT*
    | *running time:* is the measured running time for an algorithm call
    | *runlength:* is the number of steps needed to find a solution
    | *quality:* the solution quality
    | *seed:* the seed that was used with the algorithm call
    | *instance-specifics:* additional information

*SMAC* will terminate with the following output:

    .. code-block:: bash

        INFO:intensifier:Updated estimated error of incumbent on 122 runs: 0.5063
        DEBUG:root:Remaining budget: inf (wallclock), inf (ta costs), -6.000000 (target runs)
        INFO:Stats:##########################################################
        INFO:Stats:Statistics:
        INFO:Stats:#Target algorithm runs: 506
        INFO:Stats:Used wallclock time: 44.00 sec
        INFO:Stats:Used target algorithm runtime: 0.00 sec
        INFO:Stats:##########################################################
        INFO:SMAC:Final Incumbent: Configuration:
          x1, Value: 9.556406137303922
          x2, Value: 2.429138598022513

Furthermore, *SMACs* trajectory and runhistory will be stored in ``branin/``.


.. _svm-example:

Using *SMAC* in Python: SVM
---------------------------
To explain the use of *SMAC* within Python, let's look at a real-world example,
optimizing the hyperparameters of a Support Vector Machine (SVM) trained on the widely known `IRIS-dataset
<https://en.wikipedia.org/wiki/Iris_flower_data_set>`_.
This example is located in :code:`examples/svm.py`.

To use *SMAC* directly with Python, we first import the necessary modules

.. literalinclude:: ../examples/SMAC4HPO_svm.py
   :lines: 9-22
   :lineno-match:
   
We import the `SVM from Scikit-Learn <http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html>`_, 
 different hyperparameter types, the ConfigurationSpace, and the objects we need from *SMAC*.
The ConfigurationSpace is used to define the hyperparameters we want to optimize as well as
their domains. Possible hyperparameter types are floats, integers and categorical parameters.

We optimize a SVM with the hyperparameters *kernel*, *C*, *gamma*, *coef0*, *degree* and *shrinking*, which
are further explained in the documentation of sklearn. Note that modifying *C* can
quickly lead to overfitting, which is why we use cross-validation to evaluate
the configuration.

Let's start by creating a ConfigSpace-object and adding the first hyperparameter: the choice of
the kernel.

.. literalinclude:: ../examples/SMAC4HPO_svm.py
   :lines: 60-65
   :lineno-match:

We can add Integers, Floats or Categoricals to the ConfigSpace-object all at
once, by passing them in a list. 

.. literalinclude:: ../examples/SMAC4HPO_svm.py
   :lines: 67-70
   :lineno-match:

Not every kernel uses all the parameters. The sklearn-implementation of the SVM accepts all hyperparameters we want to optimize, but ignores all those incompatible with the chosen kernel.
We can reflect this in optimization using **conditions** to deactivate hyperparameters that are irrelevant to the current kernel.
Deactivated hyperparameters are not considered during optimization, limiting the search-space to reasonable configurations.
This way human knowledge about the problem is introduced.

.. literalinclude:: ../examples/SMAC4HPO_svm.py
   :lines: 72-78
   :lineno-match:

Conditions can be used for various reasons. The `gamma`-hyperparameter for
example can be set to "auto" or to a fixed float-value. We introduce a hyperparameters
that is only activated if `gamma` is not set to "auto".

.. literalinclude:: ../examples/SMAC4HPO_svm.py
   :lines: 80-88
   :lineno-match:

Of course we also define a function to evaluate the configured SVM on the IRIS-dataset.
Some options, such as the *kernel* or *C*, can be passed directly.
Others, such as *gamma*, need to be translated before the call to the SVM.

.. literalinclude:: ../examples/SMAC4HPO_svm.py
   :pyobject: svm_from_cfg
   :lineno-match:

We need a Scenario-object to configure the optimization process.
We provide a `list of possible options`__ in the scenario.

__ scenario_

The initialization of a scenario in the code uses the same keywords as a
scenario-file, which we used in the Branin example.

.. literalinclude:: ../examples/SMAC4HPO_svm.py
   :lines: 91-96
   :lineno-match:

Now we're ready to create a *SMAC*-instance, which handles the Bayesian
Optimization-loop and calculates the incumbent. 
To automatically handle the exploration of the search space 
and evaluation of the function, SMAC needs as inputs the scenario object 
as well as the function.

.. literalinclude:: ../examples/SMAC4HPO_svm.py
   :lines: 103-112
   :lineno-match:

We start the optimization loop.

Internally SMAC keeps track of the number of algorithm calls and the remaining time budget via a Stats object.

After successful execution of the optimization loop the Stats object outputs the result of the loop.

.. code-block:: bash

    INFO:smac.stats.stats.Stats:##########################################################
    INFO:smac.stats.stats.Stats:Statistics:
    INFO:smac.stats.stats.Stats:#Incumbent changed: 88
    INFO:smac.stats.stats.Stats:#Target algorithm runs: 200 / 200.0
    INFO:smac.stats.stats.Stats:Used wallclock time: 41.73 / inf sec 
    INFO:smac.stats.stats.Stats:Used target algorithm runtime: 17.44 / inf sec
    INFO:smac.stats.stats.Stats:##########################################################
    INFO:smac.facade.smac_facade.SMAC:Final Incumbent: Configuration:
      C, Value: 357.4171743725004
      coef0, Value: 9.593372746957046
      degree, Value: 1
      gamma, Value: 'value'
      gamma_value, Value: 0.0029046235175726105
      kernel, Value: 'poly'
      shrinking, Value: 'false'

We further query the target function at the incumbent, using the function evaluator
so that as final output we can see the error value of the incumbent.

.. code-block:: bash

   Optimized Value: 0.02

As a bonus, we can validate our results. This is more useful when optimizing on
instances, but we include the code so it is easily applicable for any usecase.

.. literalinclude:: ../examples/SMAC4HPO_svm.py
   :lines: 115-
   :lineno-match:

.. _spear-example:

Spear-QCP
---------
For this example we use *SMAC* to optimize the runtime required by the SAT solver `Spear <http://www.domagoj-babic.com/index.php/ResearchProjects/Spear>`_
to solve a small subset of the QCP-dataset.

In *SMACs* root-directory type:

.. code-block:: bash

    cd examples/spear_qcp && ls -l

In this folder you see the following files and directories:

* **features.txt**:

    The `feature file <options.html#feature>`_ contains the features for each instance in a csv-format.

    +--------------------+--------------------+--------------------+-----+
    |      instance      | name of feature 1  | name of feature 2  | ... |
    +====================+====================+====================+=====+
    | name of instance 1 | value of feature 1 | value of feature 2 | ... |
    +--------------------+--------------------+--------------------+-----+
    |         ...        |          ...       |          ...       | ... |
    +--------------------+--------------------+--------------------+-----+

* **instances.txt**
    The `instance file <options.html#instance>`_ contains the names of all instances one might want to consider during the optimization process.

* **scenario.txt**
    The scenario_ file contains all the necessary information about the configuration scenario at hand.
    
    .. literalinclude:: ../examples/spear_qcp/scenario.txt
    
    For this example the following options are used:

    * *algo:*

        .. code-block:: bash

            python -u ./target_algorithm/scripts/SATCSSCWrapper.py --mem-limit 1024 --script ./target_algorithm/spear-python/spearCSSCWrapper.py

        This specifies the wrapper that *SMAC* executes with a pre-specified syntax in order to evaluate the algorithm to be optimized.
        This wrapper script takes an instantiation of the parameters as input, runs the algorithm with these parameters, and returns
        the cost of running the algorithm; since every algorithm has a different input and output format, this wrapper acts as an interface between the
        algorithm and *SMAC*, which executes the wrapper through a command line call.

        An example call would look something like this:

        .. code-block:: bash

            <algo> <instance> <instance specifics> <cutoff time> <runlength> <seed> <algorithm parameters>

        For *SMAC* to be able to interpret the results of the algorithm run, the wrapper returns the results of the algorithm run as follows:

        .. code-block:: bash

            STATUS, running time, runlength, quality, seed, instance-specifics

    * *paramfile:*

        This parameter specifies which pcs-file to use and where it is located.

        The PCS-file specifies the Parameter Configuration Space file, which lists the algorithm's parameters, their domains, and default values (one per line)

        In this example we are dealing with 26 parameters of which 12 are categorical and 14 are continuous. Out of these 26
        parameters, 9 parameters are conditionals (they are only active if their parent parameter takes on a certain value).

    * *execdir:* Specifies the directory in which the target algorithm will be run.

    * *deterministic:* Specifies if the target algorithm is deterministic.

    * *run_obj:* This parameter tells *SMAC* what is to be optimized, i.e. running time or (solution) quality.

    * *overall_obj:* Specifies how to evaluate the error values, e.g as mean or PARX.

    * *cutoff_time:* The target algorithms cutoff time.

    * *wallclock-limit:* This parameter is used to give the time budget for the configuration task in seconds.

    * *instance_file:* See instances.txt above.

    * *feature_file:* See features.txt above.

* **run.sh**
    A shell script calling *SMAC* with the following command:

    ``python ../../scripts/smac --scenario scenario.txt --verbose DEBUG``

    This runs *SMAC* with the scenario options specified in the scenario.txt file.

* **target_algorithms** contains the wrapper and the executable for Spear.
* **instances** folder contains the instances on which *SMAC* will configure Spear.

To run the example type one of the two commands below into a terminal:

.. code-block:: bash

    bash run.sh
    python ../../scripts/smac --scenario scenario.txt

*SMAC* will run for a few seconds and generate a lot of logging output.
After *SMAC* finished the configuration process you'll get some final statistics about the configuration process:

.. code-block:: bash

   INFO:    ##########################################################
   INFO:    Statistics:
   INFO:    #Incumbent changed: 2
   INFO:    #Target algorithm runs: 17 / inf
   INFO:    Used wallclock time: 35.38 / 30.00 sec
   INFO:    Used target algorithm runtime: 20.10 / inf sec
   INFO:    ##########################################################
   INFO:    Final Incumbent: Configuration:
     sp-clause-activity-inc, Value: 0.9846527087294622
     sp-clause-decay, Value: 1.1630090545101102
     sp-clause-del-heur, Value: '0'
     sp-first-restart, Value: 30
     sp-learned-clause-sort-heur, Value: '10'
     sp-learned-clauses-inc, Value: 1.2739749314202675
     sp-learned-size-factor, Value: 0.8355014264152971
     sp-orig-clause-sort-heur, Value: '1'
     sp-phase-dec-heur, Value: '0'
     sp-rand-phase-dec-freq, Value: '0.01'
     sp-rand-phase-scaling, Value: 0.3488055907688382
     sp-rand-var-dec-freq, Value: '0.05'
     sp-rand-var-dec-scaling, Value: 0.46427056372562864
     sp-resolution, Value: '0'
     sp-restart-inc, Value: 1.7510945705535836
     sp-update-dec-queue, Value: '1'
     sp-use-pure-literal-rule, Value: '0'
     sp-var-activity-inc, Value: 0.9000377944957962
     sp-var-dec-heur, Value: '14'
     sp-variable-decay, Value: 1.8292433459523076



The statistics further show the used wallclock time, target algorithm running time and the number of executed target algorithm runs,
and the corresponding budgets---here we exhausted the wallclock time budget.

The directory in which you invoked *SMAC* now contains a new folder called **SMAC3-output_YYYY-MM-DD_HH:MM:SS**.
The .json file contains the information about the target algorithms *SMAC* just executed. In this file you can see the *status* of the algorithm run, *misc*, the *instance* on which the algorithm was evaluated, which *seed* was used, how much *time* the algorithm needed and with which *configuration* the algorithm was run.
In the folder *SMAC* generates a file for the runhistory, and two files for the trajectory.


.. _hydra-example:

Hydra on Spear-QCP
------------------

For this example we use *Hydra* to build a portfolio on the same example data presented in `Spear-QCP`__.
Hydra is a portfolio builder that aims to build a portfolio by iteratively adding complementary configurations to the
already existing portfolio. To select these complementary configurations *Hydra* compares new configurations to the
portfolio and only considers configurations that improve the portfolio performance.
In the first iteration Hydra runs standard *SMAC* to determine a well performing configuration
across all instances as a starting point for the portfolio. In following iterations *Hydra* adds one configuration that
improves the portfolio performance.

__ spear-example_

To run Hydra for three iterations you can run the following code in the spear-qcp example folder.

 ``python ../../scripts/smac --scenario scenario.txt --verbose DEBUG --mode Hydra --hydra_iterations 3``

As the individual SMAC scenario takes 30 seconds to run Hydra will run for ~90 seconds on this example.
You will see the same output to the terminal as with standard SMAC. In the folder where you executed the above command,
you will find a *hydra-output-yyy-mm-dd_hh:mm:ss_xyz* folder. This folder contains the results of all three performed
SMAC runs, as well as the resulting portfolio (as pkl file).

The resulting portfolio can be used with any algorithm selector such as `AutoFolio <https://github.com/mlindauer/AutoFolio>`_