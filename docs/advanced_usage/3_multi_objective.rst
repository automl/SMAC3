Multi-Objective Optimization
============================

Often we do not only want to optimize just a single objective, but multiple instead. SMAC offers a multi-objective 
optimization interface to do exactly that. Right now, the algorithm used for this is a mean aggregation strategy or 
ParEGO [Know06]_. In both cases, multiple objectives are aggregated into a single scalar objective, which is then 
optimized by SMAC. However, the run history still keeps the original objectives.


The basic recipe is as follows:

#. Make sure that your target function returns a cost *dictionary* containing the objective names as keys
   and the objective values as values, e.g. ``{'myobj1': 0.3, 'myobj2': 200}``. Alternatively, you can simply
   return a list, e.g, ``[0.3, 200]``.
#. When instantiating SMAC pass the names of your objectives to the scenario object via the ``objectives``
   argument, e.g., ``multi_objectives: ["myobj1", "myobj2"]``.
#. Now you can optionally pass a custom multi-objective algorithm class to the SMAC
   facade (via ``multi_objective_algorithm``). In all facades, a mean aggregation strategy is used as the 
   multi-objective algorithm default.


.. warning ::

    To judge whether a configuration is better than another (comparison happens in the intensifier), we need to 
    scalarize the multi-objective values. This, however, is *not* done by the multi-objective algorithm but directly in 
    the runhistory object using the method ``average_cost``. The values are always normalized (based on all costs so 
    far) and weighted averaged (based on the objective weights provided by the user).


We show an example of how to use multi-objective with plots in our :ref:`examples<Multi-Objective>`.
