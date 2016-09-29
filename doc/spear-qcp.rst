
Spear-QCP
_________
| For this example we use *SMAC* to optimize `Spear <http://www.domagoj-babic.com/index.php/ResearchProjects/Spear>`_ on a small subset of the QCP-dataset.
| In *SMACs* root-directory type:

.. code-block:: bash

    cd examples/spear_qcp && ls -l

In this folder you'll see the following files and directories:
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

        * *algo*:

            .. code-block:: bash

                python -u ./target_algorithm/scripts/SATCSSCWrapper.py --mem-limit 1024 --script ./target_algorithm/spear-python/spearCSSCWrapper.py

            This specifies the wrapper that *SMAC* executes with a prespecified syntax in order to evaluate the algorithm to be optimized.
            This wrapper script takes an instantiation of the parameters as input, runs the algorithm with these parameters, and returns
            how well it did; since every algorithm has a different input and output format, this wrapper acts as a mediator between the
            algorithm and *SMAC*, which executes the wrapper through a command line call.

            An example call would look something like this:

            .. code-block:: bash

                <algo> <instance> <instance_specifics> <runtime cutoff> <runlength> <seed> <solver parameters>

            For *SMAC* to be able to interpret the results of the algorithm run, the wrapper returns the results of the algorithm run as follows:
            :bash:`STATUS, runtime, runlength, quality, seed, instance-specifics`

        * *paramfile*:
            This parameter specifies which pcs-file to use and where it is located.

            The pcs-file specifies the Parameter Configuration Space file, which lists the algorithm's parameters, their domains, and default values (one per line)

            In this example we are dealing with 26 parameters of which 12 are categorical and 14 are continuous. Out of these 26
            parameters, 9 parameters are conditionals (they are only active if their parent parameter takes on a certain value).

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


The first line shows why *SMAC* terminated. The wallclock time-budget is exhausted. The target algorithm runtime (ta cost) and target algorithm runs were not exhausted since the budget for these were not specified and thus defaulted to infinity.

The statistics further show the used wallclock time, target algorithm runtime and the number of executed target algorithm runs.

| In directory in which you invoked *SMAC* now contain a new folder called **SMAC3-output_YYYY-MM-DD_HH:MM:SS** as well as a file called **target_algo_run.json**.
| The .json file contains the information about the target algorithms *SMAC* just executed. In this file you can see the *status* of the algorithm run, *misc*, the *instance* on which the algorithm was evaluated, which *seed* was used, how much *time* the algorithm needed and with which *configuration* the algorithm was run.
| In the folder *SMAC* generates a file for the runhistory, and two files for the trajectory.