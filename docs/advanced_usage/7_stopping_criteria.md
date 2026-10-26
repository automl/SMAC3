# Stopping Criteria

In addition to the standard stopping criteria like number of trials or wallclock time, SMAC also provides 
more advanced criteria.


## Termination Cost Threshold

SMAC can stop the optimization process after a user-defined cost was reached. In each iteration, the average cost 
(using ``average_cost`` from the run history) from the incumbent is compared to the termination cost threshold. If one
of the objective costs is below its associated termination cost threshold, the optimization process is stopped.
Note, since the ``average_cost`` method is used, all instance-seed-budget trials of the incumbent are considered so far.
In other words, the process can be stopped even if the incumbent has not been evaluated on all instances, on the 
highest fidelity, or on all seeds.


```python
scenario = Scenario(
    ...
    objectives=["accuracy", "runtime"],
    termination_cost_threshold=[0.1, np.inf]
    ...
)
```

In the code above, the optimization process is stopped if the average accuracy of the incumbent is below 0.1. The 
runtime is ignored completely as it is set to infinity. Note here again that SMAC minimizes the objective values.


## Initial Design Diagnostics

Once every configuration of the initial design has been evaluated, SMAC checks whether the collected
evaluations carry any feedback signal for the surrogate model. Two degenerate outcomes are possible:

* **All configurations failed.** The target function or the pipeline around it is probably broken.
  Every failed trial is assigned the same penalty cost (``crash_cost``), so there is nothing to build
  the surrogate model on.
* **All successful configurations have the same cost.** The metric or the search space is likely
  misconfigured. This may also be a huge plateau — in either case the surrogate model has no signal
  to learn from.

In both cases continuing the run cannot produce anything useful, so SMAC can stop early instead of
spending the remaining budget. A third, purely informational check reports how many configurations
failed when only *some* of them did; partial failures are a normal part of hyperparameter
optimization and never stop the run.

The behaviour is controlled via the scenario:

```python
scenario = Scenario(
    ...
    initial_design_diagnostics="warn",  # "off" | "warn" (default) | "abort"
    ...
)
```

* ``"off"``: No diagnostics are performed.
* ``"warn"`` (default): Findings are logged as warnings, the optimization continues. Nothing but the
  log output changes.
* ``"abort"``: Findings are logged, and the optimization is stopped gracefully if the initial design
  carries no feedback signal at all.

A configuration counts as failed only if none of its evaluations succeeded. A configuration that
fails on one instance but succeeds on another has not failed. Costs are compared with a small
tolerance, so objectives that are constant except for floating point noise are detected as well.

    At least three configurations have to be evaluated before any conclusion is drawn. Some facades
    (for example [RandomFacade][smac.facade.random_facade] or
    [MultiObjectiveFacade][smac.facade.multi_objective_facade]) use an initial design that proposes
    only the default configuration; in that case the diagnostics wait for the configurations
    evaluated next. A default configuration that happens to be invalid for the given dataset is a
    common and recoverable situation and must not stop a run on its own.

    With more than one worker, trials that were already submitted when the diagnosis fires still run
    to completion. The abort is therefore effective but not exact — expect one more trial than the
    diagnosis point suggests.

