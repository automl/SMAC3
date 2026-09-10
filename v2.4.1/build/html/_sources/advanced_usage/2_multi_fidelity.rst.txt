Multi-Fidelity Optimization
===========================

Multi-fidelity refers to running an algorithm on multiple budgets (such as number of epochs or
subsets of data) and thereby evaluating the performance prematurely. You can run a multi-fidelity optimization
when using :ref:`Successive Halving<smac.intensifier.successive\\_halving>` or 
:ref:`Hyperband<smac.intensifier.hyperband>`. `Hyperband` is the default intensifier in the 
:ref:`multi-fidelity facade<smac.facade.multi\\_fidelity\\_facade>` and requires the arguments 
``min_budget`` and ``max_budget`` in the scenario if no instances are used.

In general, multi-fidelity works for both real-valued and instance budgets. In the real-valued case,
the budget is directly passed to the target function. In the instance case, the budget is not passed to the 
target function but ``min_budget`` and ``max_budget`` are used internally to determine the number of instances of 
each stage. That's also the reason why ``min_budget`` and ``max_budget`` are *not required* when using instances: 
The ``max_budget`` is simply the max number of instances, whereas the ``min_budget`` is simply 1.

Please have a look into our :ref:`multi-fidelity examples<Multi-Fidelity and Multi-Instances>` to see how to use
multi-fidelity optimization in real-world applications.