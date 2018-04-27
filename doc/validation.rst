Validation
----------

During optimization, *SMAC* does not automatically evaluate all configurations on all instances.
The `validate-module <apidoc/smac.utils.validate.html>`_ provides a convenient
way to do this after a configuration run is finished. You can specify a set of configurations and a set of instances,
whereupon the validator will run every configuration on every instance. This is
either done by calling the target algorithm - or it is estimated using an EPM.
This functionality can be important when comparing default and incumbent, giving
a best estimate of the actual quality of the incumbent or investigating the
quality of the incumbents founds over the time of the optimization.
Validation supports parallel computing, non-deterministic target algorithms and
reusing results from runhistories.

.. note::

        When evaluating the cost via target-algorithm runs (no EPM), runs should be
        reused from a given runhistory only on comparable hardware!

Commandline 
~~~~~~~~~~~

.. code-block:: bash

        python smac-validate.py --scenario SCENARIO --trajectory TRAJECTORY --output OUTPUT [--configs CONFIG_MODE] [--instances INSTANCE_MODE] [--[no-]epm] [--runhistory RUNHISTORY] [--seed SEED] [--repetitions REPETITIONS] [--n_jobs N_JOBS] [--tae TAE]


Required:
     * *scenario*: Path to the file that specifies the `scenario <options.html#scenario>`_ THAT IS USED FOR THE VALIDATION.
     * *trajectory*: Path to the trajectory of the *SMAC*-run.
     * *output*: Path to save output-runhistory to.
Optional:
     * *instances*: Which instances to validate on, train or test are defined in scenario. From: [train, test, train+test] **Default**: test
     * *configs*: What configurations to validate. From: [def, inc, def+inc, wallclock_time, cpu_time, all]
       def and inc validate on default and/or incumbent. all validates on all
       configurations in the specified trajectory
       wallclock_time/cpu_time validate at cpu- or wallclock-timesteps of 
       [max_time/2^0, max_time/2^1, max_time/2^3, ..., default] **Default**: def+inc
     * *[no-]epm]*: epm uses an EPM which is built upon the given runhistory to estimate the costs of the config/instance-pairs;
       no-epm evaluates the config/instance-pairs by actually running the target algorithm **Default**: no-epm
     * *runhistory*: path to a runhistory. If specified, runs from the runhistory will not be re-evaluated. IMPORTANT: THIS MAY BE DIFFICULT WHEN USING DIFFERENT HARDWARE **Default**: None
     * *repetitions*: for non-deterministic target algorithms, this option
       specifies the number of different seeds that are evaluated per
       config/instance-pair **Default**: 1
     * *n_jobs*: if no-epm is used, this is the number of cores to use for
       evaluation in parallel **Default**: 1 
     * *tae* from [old, aclib], if no-epm is used, this specifies the format of
       the target algorithm (see `tae <tae.html>`_) **Default**: old
     * *verbose_level*: in [INFO, DEBUG], specifies the logging-verbosity. **Default**: INFO
     * *seed*: seed to be used for validation


Usage in Python
~~~~~~~~~~~~~~~

To validate directly in Python (e.g. to perform a validation immediately after an
optimization), the `Validator <apidoc/smac.utils.validate.html#Validator>`_ can be used. It provides two different
methods, `validate <apidoc/smac.utils.validate.html#smac.utils.validate.Validator.validate>`_ and
`validate_epm <apidoc/smac.utils.validate.html#smac.utils.validate.Validator.validate_epm>`_. Both return runhistories
containing results for all desired config/instance-pairs.
To validate an actual `*SMAC*-object <apidoc/smac.facade.smac_facade.html>`_,
there is also a `method <apidoc/smac.facade.smac_facade.html#smac.facade.smac_facade.SMAC.validate>`_ within *SMAC*
(see also: `SVM-example <quickstart.html#using-smac-in-python-svm>`_).
