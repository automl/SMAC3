Multi-Objective Optimization
============================

Often we do not only want to optimize just a single objective, but multiple instead. SMAC offers a multi-objective 
optimization interface to do exactly that. Right now, the algorithm used for this is a mean aggregation strategy or 
ParEGO [Know06]_. In both cases, multiple objectives are aggregated into a single scalar objective, which is then 
optimized by SMAC. However, the run history still keeps the original objectives.


The basic recipe is as follows:

#. Make sure that your target function returns a cost *dictionary* containing the objective names as keys
   and the objective values as values, e.g. ``{'myobj1': 0.3, 'myobj2': 200}``. Alternatively, you can simply
   return a list, e.g., ``[0.3, 200]``.
#. When instantiating SMAC pass the names of your objectives to the scenario object via the ``objectives``
   argument, e.g., ``multi_objectives: ["myobj1", "myobj2"]``.
#. Now you can optionally pass a custom multi-objective algorithm class to the SMAC
   facade (via ``multi_objective_algorithm``). In all facades, a mean aggregation strategy is used as the 
   multi-objective algorithm default.


.. warning ::

    Depending on the multi-objective algorithm, the incumbent might be ambiguous because there might be multiple 
    incumbents on the Pareto front. Let's take ParEGO for example:
    Everytime a new configuration is sampled, the weights are updated (see runhistory encoder). Therefore, calling
    the ``get_incumbent`` method in the runhistory might return a different configuration based on the internal state 
    of the multi-objective algorithm. 


You can use ``get_pareto_front`` in the run history to get the configurations on the Pareto front.


.. code-block:: python

    smac = ...
    smac.optimize()

    for incumbent, costs in smac.get_pareto_front():
        print(incumbent, costs)




We show an example of how to use multi-objective with plots in our :ref:`examples<Multi-Objective>`.
