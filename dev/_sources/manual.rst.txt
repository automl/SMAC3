Manual
======
.. role:: bash(code)
    :language: bash


In the following we will show how to use **SMAC3**.

.. _quick:

Quick Start
-----------
| If you have not installed *SMAC* yet take a look at the `installation instructions <installation.html>`_ and make sure that all the requirements are fulfilled.
| In the examples folder, you can find examples that illustrate how to reads scenario files that allow you to automatically configure an algorithm, as well as examples that show how to directly use *SMAC* in Python.

We'll demonstrate the usage of *SMAC* on the most basic scenario, the optimization of a continuous blackbox function. The first simple example is the minimization of a standard 2-dimensional continuous test function (`branin <https://www.sfu.ca/~ssurjano/branin.html>`_).

To run the example scenario, change into the root-directory of *SMAC* and type the following commands:

.. code-block:: bash

    cd examples/branin
    python ../../scripts/smac --scenario branin_scenario.txt

The python command runs *SMAC* with the specified scenario. The scenario file contains the following four lines:

.. code-block:: bash

    algo = python branin.py
    paramfile = branin_pcs.pcs
    run_obj = quality
    runcount_limit = 500

The **algo** parameter specifies how *SMAC* can call the function or evaluate an algorithm that *SMAC* is optimizing. An algorithm call by *SMAC* will look like this:

    .. code-block:: bash

        python branin.py 0 0 999999999.0 0 1148756733 -x1 -1.1338595629 -x2 13.8770222718

    The first two parameter after the branin.py do not matter for this example since no instances are needed for the function optimization. For algorithm optimization, the first parameter holds the instance name on which the algorithm is evaluated and the second can provide extra information about the instance (rarely used).

    The third parameter gives the runtime cutoff (maximal runtime) an algorithm is allowed to run and the fourth the runlength (maximal number of steps).

    The fifth parameter is the random seed which is followed by the algorithm/function parameters.
    
    The general syntax of algorithm calls is the following

    .. code-block:: bash

        <algo> <instance> <instance specific> <running time cutoff> <run length> <seed> <algorithm parameters>

The **paramfile** parameter tells *SMAC* which Parameter Configuration File to use. This file contains a list of the algorithm's parameters, their domains and default values as follows:

    .. code-block:: bash

        x1 [-5,10] [0]
        x2 [0,15]  [0]

    x1 and x2 are both continuous parameters. x1 can take any real value in the range [-5, 10], x2 in the range [0, 15] and both have the default value 0.

The **run_obj** parameter specifies what *SMAC* is supposed to optimize. For this example we are optimizing the solution quality.
For *SMAC* to be able to interpret the results of an algorithm run, a wrapper should return the results of the algorithm run as follows:

    .. code-block:: bash

        Result for SMAC: <STATUS>, <running time>, <run length>, <quality>, <seed>, <instance-specifics>
        Result for SMAC: SUCCESS, 0, 0, 48.948190, 1148756733

    | The second line is the result of the above listed algorithm call.
    | *STATUS:* can be either *SUCCESS*, *CRASHED*, *SAT*, *UNSAT*, *TIMEOUT*
    | *running time:* is the measured running time for an algorithm call
    | *run length:* are the amount of steps needed to find a solution
    | *quality:* the solution quality
    | *seed:* the seed that was used with the algorithm call
    | *instance-specifics:* additional information

The **runcount_limit** specifies the maximum number of algorithm calls.

*SMAC* will terminate with the following output:

    .. code-block:: bash

        INFO:intensifier:Updated estimated performance of incumbent on 122 runs: 0.5063
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

    Further a folder containing *SMACs* trajectory and the runhistory will be created in the branin folder.

Using *SMAC* directly in python
-------------------------------
| For demonstration purposes we are going to look at the example :bash:`leadingones.py`
|
| In this example we are going to optimize the following function with 16 categorical parameters.
| For a given sequence of 0,1, we count how many leading 1s we have at the beginning of the sequence. 

    .. code-block:: python

        def leading_ones(cfg, seed):
            """ Leading ones
            score is the number of 1 starting from the first parameter
            e.g., 111001 -> 3; 0110111 -> 0
            """

            arr_ = [0] * len(cfg.keys())
            for p in cfg:
                arr_[int(p)] = cfg[p]

            count = 0
            for v in arr_:
                if v == 1:
                    count += 1
                else:
                    break

            return -count

| Thus the optimum is -16 and the optimal configuration is x_1 = 1, ..., x_16 = 1
|
| To use *SMAC* directly with Python, we first have to import the necessary modules

    .. code-block:: python
        :lineno-start: 3

        import numpy as np

        from smac.configspace import ConfigurationSpace
        from ConfigSpace.hyperparameters import CategoricalHyperparameter, \
            UniformFloatHyperparameter, UniformIntegerHyperparameter
        from ConfigSpace.conditions import InCondition

        from smac.tae.execute_func import ExecuteTAFunc
        from smac.scenario.scenario import Scenario
        from smac.facade.smac_facade import SMAC

First, we import the ConfigurationSpace and Parametertypes in order to later declare different parameters.

Now, we build the Configuration Space:

    .. code-block:: python
        :lineno-start: 38

        # build Configuration Space which defines all parameters and their ranges
        n_params = 16
        use_conditionals = True # using conditionals should help a lot in this example

        cs = ConfigurationSpace()
        previous_param = None
        for n in range(n_params):
            p = CategoricalHyperparameter("%d" % (n), [0, 1], default=0)
            cs.add_hyperparameter(p)

            if n > 0 and use_conditionals:
                cond = InCondition(
                    child=p, parent=previous_param, values=[1])
                cs.add_condition(cond)

            previous_param = p

cs is the Configuration space Object. 
We declare each of the 16 parameters to be Categorical parameters 
that can take the values 0 or 1 and are set by default to 0. 
They are also given the names '1' to '16'.
Further we illustrate how to setup conditionals in this example.

Parameter 'i+1' is conditioned on parameter 'i' 
and thus only activated if parameter 'i' is set to 1. 
For example parameter '1' is only active once parameter '0' is set to 1. 
Using conditionals in such a way restricts the search space quite a bit. 
This way *SMAC* won't have to query regions in the search space that are non-improving, 
like '0100000000000000' or '0100000000000001'. Both return the same value as the default, i.e. 0.

After the configuration space was setup we can create a scenario object.

    .. code-block:: python
        :lineno-start: 53

        # SMAC scenario object
        scenario = Scenario({"run_obj": "quality",  # we optimize quality (alternative runtime)
                             "runcount-limit": n_params*2,  # at most 200 function evaluations
                             "cs": cs,  # configuration space
                             "deterministic": "true"
                             })

The Scenario object contains information about the optimization scenario, such as the runcount-limit or what metric to optimize.
It uses the same keywords as a scenario files, we showed in the branin example.

To evaluate the "leading ones" function, we register it with the TargetAlgorithmFunction evaluator.

    .. code-block:: python
        :lineno-start: 60

        # register function to be optimize
        taf = ExecuteTAFunc(leading_ones)

        # example call of the function
        # it returns: Status, Cost, Runtime, Additional Infos
        def_value = taf.run(cs.get_default_configuration())[1]
        print("Default Value: %.2f" % (def_value))

We register the function to optimize together with the evaluator that handles calling the function with a specified configuration.

Afterwards,
the default value is queried by calling the run method of the evaluator with the default configuration of the configuration space.

To handle the Bayesian optimization loop we can create a SMAC object.
To automatically handle the exploration of the search space 
and querying of the function SMAC needs as inputs the scenario object as well as the function evaluator.

    .. code-block:: python
        :lineno-start: 68

        # Optimize
        smac = SMAC(scenario=scenario, rng=np.random.RandomState(42),
                    tae_runner=taf)
        try:
            incumbent = smac.optimize()
        finally:
            smac.stats.print_stats()
            incumbent = smac.solver.incumbent

        inc_value = taf.run(incumbent)[1]
        print("Optimized Value: %.2f" % (inc_value))

We start the optimization loop and set the maximum number of iterations to 999.

Initernally SMAC keeps track of the number of algorithm calls and the remaining time budget via a stats object.

After successful execution of the optimization loop the stats opject outputs the result of the loop.

We can directly access the incumbent configuration which is stored in the SMAC object and print it to the terminal (line 75).

We further query the target function at the incumbent, using the function evaluator so that as final output we can see performance value of the incumbent.



Spear-QCP
---------
| For this example we use *SMAC* to optimize `Spear <http://www.domagoj-babic.com/index.php/ResearchProjects/Spear>`_ on a small subset of the QCP-dataset.
| In *SMACs* root-directory type:

.. code-block:: bash

    cd examples/spear_qcp && ls -l

In this folder you see the following files and directories:
    * **features.txt**:

     The feature file is contains the features for each instance in a csv-format.

     +--------------------+--------------------+--------------------+-----+
     |      instance      | name of feature 1  | name of feature 2  | ... |
     +====================+====================+====================+=====+
     | name of instance 1 | value of feature 1 | value of feature 2 | ... |
     +--------------------+--------------------+--------------------+-----+
     |         ...        |          ...       |          ...       | ... |
     +--------------------+--------------------+--------------------+-----+

    * **instances.txt**
        The instance file contains the names of all instances one might want to consider during the optimization process.

    * **scenario.txt**
        The scenario file contains all the necessary information about the configuration scenario at hand.
        For this example the following options were used:

        * *algo:*

            .. code-block:: bash

                python -u ./target_algorithm/scripts/SATCSSCWrapper.py --mem-limit 1024 --script ./target_algorithm/spear-python/spearCSSCWrapper.py

            This specifies the wrapper that *SMAC* executes with a pre-specified syntax in order to evaluate the algorithm to be optimized.
            This wrapper script takes an instantiation of the parameters as input, runs the algorithm with these parameters, and returns
            the performance of the algorithm; since every algorithm has a different input and output format, this wrapper acts as a interface between the
            algorithm and *SMAC*, which executes the wrapper through a command line call.

            An example call would look something like this:

            .. code-block:: bash

                <algo> <instance> <instance_specifics> <running time cutoff> <run length> <seed> <algorithm parameters>

            For *SMAC* to be able to interpret the results of the algorithm run, the wrapper returns the results of the algorithm run as follows:
            :bash:`STATUS, runtime, runlength, quality, seed, instance-specifics`

        * *paramfile:*

            This parameter specifies which pcs-file to use and where it is located.

            The pcs-file specifies the Parameter Configuration Space file, which lists the algorithm's parameters, their domains, and default values (one per line)

            In this example we are dealing with 26 parameters of which 12 are categorical and 14 are continuous. Out of these 26
            parameters, 9 parameters are conditionals (they are only active if their parent parameter takes on a certain value).

        * *execdir:* Specifies the directory in which the target algorithm will be run.

        * *deterministic:* Specifies if the configuration scenario is deterministic.

        * *run_obj:* This parameter tells *SMAC* what is to be optimized, i.e. runtime or (solution) quality.

        * *overall_obj:* Specifies how to evaluat the performance values, e.g as mean or PARX.

        * *cutoff_time:* Gives the target algorithms cuttof time.

        * *wallclock-limit:* This parameter is used to give the time budget for the configuration task in seconds.

        * *instance_file:* See instances.txt above.

        * *feature_file:* See features.txt above.

    * **run.sh**
        A shell script calling *SMAC* with the following command:
        :bash:`python ../../scripts/smac --scenario scenario.txt --verbose DEBUG`
        This runs *SMAC* with the scenario options specified in the scenario.txt file.

    * **target_algorithms** contains the wrapper and the executable for Spear.
    * **instances** folder contains the instances on which *SMAC* will configure Spear.

To run the example type one of the two commands below into a terminal:

.. code-block:: bash

    bash run.sh
    python ../../scripts/smac --scenario scenario.txt --verbose DEBUG

| *SMAC* will run for a few seconds and generate a lot of logging output.
| After *SMAC* finished the configuration process you'll get some final statistics about the configuration process:

.. code-block:: bash

    DEBUG:root:Remaining budget: -11.897580 (wallclock), inf (ta costs), inf (target runs)
    INFO:Stats:##########################################################
    INFO:Stats:Statistics:
    INFO:Stats:#Target algorithm runs: 28
    INFO:Stats:Used wallclock time: 21.90 sec
    INFO:Stats:Used target algorithm runtime: 15.72 sec
    INFO:Stats:##########################################################
    INFO:SMAC:Final Incumbent: Configuration:
      sp-clause-activity-inc, Value: 0.956325431976
      sp-clause-decay, Value: 1.77371504106
      sp-clause-del-heur, Value: 2
      sp-first-restart, Value: 52
      sp-learned-clause-sort-heur, Value: 13
      sp-learned-clauses-inc, Value: 1.12196861555
      sp-learned-size-factor, Value: 0.760013050806
      sp-max-res-lit-inc, Value: 0.909236510144
      sp-max-res-runs, Value: 3
      sp-orig-clause-sort-heur, Value: 1
      sp-phase-dec-heur, Value: 6
      sp-rand-phase-dec-freq, Value: 0.0001
      sp-rand-phase-scaling, Value: 0.825118640774
      sp-rand-var-dec-freq, Value: 0.05
      sp-rand-var-dec-scaling, Value: 1.05290899107
      sp-res-cutoff-cls, Value: 5
      sp-res-cutoff-lits, Value: 1378
      sp-res-order-heur, Value: 6
      sp-resolution, Value: 1
      sp-restart-inc, Value: 1.84809841772
      sp-update-dec-queue, Value: 1
      sp-use-pure-literal-rule, Value: 0
      sp-var-activity-inc, Value: 1.00507435273
      sp-var-dec-heur, Value: 4
      sp-variable-decay, Value: 1.91690063007


The first line shows why *SMAC* terminated. The wallclock time-budget is exhausted. The target algorithm runtime (ta cost) and target algorithm runs were not exhausted since the budget for these were not specified and thus set to the default, i.e., infinity.

The statistics further show the used wallclock time, target algorithm runtime and the number of executed target algorithm runs.

| The directory in which you invoked *SMAC* now contain a new folder called **SMAC3-output_YYYY-MM-DD_HH:MM:SS**.
| The .json file contains the information about the target algorithms *SMAC* just executed. In this file you can see the *status* of the algorithm run, *misc*, the *instance* on which the algorithm was evaluated, which *seed* was used, how much *time* the algorithm needed and with which *configuration* the algorithm was run.
| In the folder *SMAC* generates a file for the runhistory, and two files for the trajectory.
