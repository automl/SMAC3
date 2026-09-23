# Ask-and-Tell Interface

SMAC provides an ask-and-tell interface in v2.0, giving the user the opportunity to ask for the next trial 
and report the results of the trial. 

!!! warning

    When specifying ``n_trials`` in the scenario and trials have been registered by the user, SMAC will 
    count the users trials as well. However, the wallclock time will first start when calling ``optimize``.

!!! warning

    It might be the case that not all user-provided trials can be considered. Take Successive Halving, for example, 
    when specifying the min and max budget, intermediate budgets are calculated. If the user provided trials with
    different budgets, they, obviously, can not be considered. However, all user-provided configurations will flow 
    into the intensification process.

!!! warning 

    In pure ask-and-tell usage, SMAC does not hard-stop `ask()` when `n_trials` is depleted. (This is true for any kind of budget exhaustion and not only `n_trials` eg. walltime, cputime. warning logs all the budget variables in the case of exhaustion).
    This means `ask()` can still return additional trials after budget exhaustion.
    SMAC now emits a runtime warning in this case and keeps this behaviour for backward compatibility.
    If you want strict stopping in your loop, stop calling `ask()` when the optimizer reports no remaining budget (for example, `smac.optimizer.budget_exhausted` or `smac.optimizer.remaining_trials <= 0`)

Notice: if you are exclusively using the ask-and-tell interface and do not use `smac.optimize()`, then smac no longer
is responsible for the evaluation of the trials and therefore the Facade no longer will require a specified `target_algorithm` argument.

Please have a look at our [ask-and-tell example](../examples/1%20Basics/3_ask_and_tell.md).

You can configure post-budget `ask()` behavior with `warn_mode` in the facade:

- `warn_once`: warn only on the first `ask()` call after budget exhaustion.
- `warn_never`: never warn.
- `warn_always`: warn on every `ask()` call after budget exhaustion.
- `exception`: raise `AskAndTellBudgetExhaustedError` instead of returning another trial.
