Multi-Objective Optimization
============================

Often we do not only want to optimize just a single objective, but multiple instead. SMAC offers a multi-objective 
optimization interface to do exactly that. Right now, the algorithm used for this is a mean aggregation strategy or 
ParEGO [Know06]_. In both cases, multiple objectives are aggregated into a single scalar objective, which is then 
optimized by SMAC. However, the run history still keeps the original objectives.


The basic recipe is as follows:

- Specify the objectives in the scenario object as list. For example, ``Scenario(objectives=["obj1", "obj2"])``.
- Make sure that your target function returns a cost *dictionary* containing the objective names as keys
  and the objective values as values, e.g. ``{'obj1': 0.3, 'obj2': 200}``. Alternatively, you can simply
  return a list, e.g., ``[0.3, 200]``.
- Now you can optionally pass a custom multi-objective algorithm class to the SMAC
  facade (via ``multi_objective_algorithm``). In all facades, a mean aggregation strategy is used as the 
  multi-objective algorithm default.


.. warning ::

   The multi-objective algorithm influences which configurations are sampled next. More specifically, 
   since only one surrogate model is trained, multiple objectives have to be scalarized into a single objective.
   This scalarized value is used to train the surrogate model, which is used by the acquisition function/maximizer
   to sample the next configurations.  


You receive the incumbents (points on the Pareto front) after the optimization process directly. Alternatively, you can 
use the method ``get_incumbents`` in the intensifier.

.. code-block:: python

   smac = ...
   incumbents = smac.optimize()

   # Or you use the intensifier
   incumbents = smac.intensifier.get_incumbents()


We show an example of how to use multi-objective with plots in our :ref:`examples<Multi-Objective>`.
