Validation
==========

During optimization, SMAC does not automatically evaluate all configurations on all instances.
The :mod:`validate-module <smac.utils.validate>` provides a convenient
way to do this after a configuration run is finished. You can specify a set of configurations and a set of instances,
whereupon the validator will run every configuration on every instance. This is
either done by calling the target algorithm or it is estimated using an EPM.
This functionality can be important when comparing default and incumbent, giving
a best estimate of the actual quality of the incumbent or investigating the
quality of the incumbents founds over the time of the optimization.
Validation supports parallel computing, non-deterministic target algorithms and
reusing results from runhistories.

.. warning::

      When evaluating the cost via target-algorithm runs (no EPM), runs should be
      reused from a given runhistory only on comparable hardware!


Python
------

To validate directly in Python (e.g. to perform a validation immediately after an
optimization), the :class:`Validator <smac.utils.validate.Validator>` can be used. It provides two different
methods, :meth:`validate <smac.utils.validate.Validator.validate>` and
:meth:`validate_epm <smac.utils.validate.Validator.validate_epm>`. Both return runhistories
containing results for all desired config/instance-pairs.
To validate an actual :class:`SMAC-object <smac.facade.smac_ac_facade.SMAC4AC>`,
it has its own method :meth:`validate <smac.facade.smac_ac_facade.SMAC4AC.validate>` within SMAC.


Commandline
-----------

.. code-block:: bash

      python3 smac-validate.py --scenario SCENARIO --trajectory TRAJECTORY --output OUTPUT [--configs CONFIG_MODE] [--instances INSTANCE_MODE] [--[no-]epm] [--runhistory RUNHISTORY] [--seed SEED] [--repetitions REPETITIONS] [--n_jobs N_JOBS] [--tae TAE]


Required:
  :scenario: Path to the file that specifies the :ref:`scenario <Scenario>` that is used for the validation.
  :trajectory: Path to the trajectory of the SMAC run.
  :output: Path to save output-runhistory to.

Optional:
  :instances: Which instances to validate on, train or test are defined in scenario. From: [train, test, train+test]. **Default**: test

  :configs: What configurations to validate. From: [def, inc, def+inc, wallclock_time, cpu_time, all].
        `def` and `inc` validate on default and/or incumbent.
        `all` validates on all configurations in the specified trajectory.
        `wallclock_time`/`cpu_time` validate at cpu- or wallclock-time steps of [max_time/2^0, max_time/2^1, max_time/2^3, ..., default].
        **Default**: def+inc

  :[no-]epm: `epm` uses an EPM which is built upon the given runhistory to estimate the costs of the config/instance-pairs.
        `no-epm` evaluates the config/instance-pairs by actually running the target algorithm. *Default*: no-epm
  :runhistory: Path to a runhistory. If specified, runs from the runhistory will not be re-evaluated. *Default*: None

  :repetitions: For non-deterministic target algorithms, this option specifies the number of different seeds that are evaluated per config/instance-pair. *Default*: 1.

  :n_jobs: if `no-epm` is used, this is the number of cores to use for evaluation in parallel. *Default*: 1.

  :tae: From [old, aclib]. If `no-epm` is used, this specifies the format of the (see :ref:`TAE<Target Algorithm Evaluator>`). *Default*: old.

  :verbose_level: In [INFO, DEBUG]. Specifies the logging-verbosity. *Default*: INFO.

  :seed: Seed to be used for validation



