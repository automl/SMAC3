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

Notice: if you are exclusively using the ask-and-tell interface and do not use `smac.optimize()`, then smac no longer
is responsible for the evaluation of the trials and therefore the Facade no longer will require a specified `target_algorithm` argument.

Please have a look at our [ask-and-tell example](../examples/1%20Basics/3_ask_and_tell.md).
