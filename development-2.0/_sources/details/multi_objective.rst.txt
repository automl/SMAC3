Multi-Objective Optimization
============================

Often we do not only want to optimize just cost or runtime, but both or other objectives instead.
SMAC offers a multi-objective optimization interface to do exactly that.
Right now, the algorithm used for this is `ParEgo`_ [Christescu & Knowles, 2015].
`ParEgo`_ weights and sums the individual objectives so that we can optimize a single scalar.

The costs returned by your target algorithm are stored as usual in the runhistory object, such that
you can recover the Pareto front later on.


The basic recipe is as follows:

#. Make sure that your target algorithm returns a cost *dictionary* containing the objective names as keys
   and the objective values as values, e.g. ``{'myobj1': 0.3, 'myobj2': 200}``. Alternatively, you can simply
   return a list, e.g ``[0.3, 200]``.
#. When instantiating SMAC pass the names of your objectives to the scenario object via the ``multi_objectives``
   argument, e.g. ``multi_objectives = "myobj1, myobj2"`` or ``multi_objectives = ["myobj1", "myobj2"]``.
   Please set ``run_obj = 'quality'``.
#. Now you can optionally pass a custom multi-objective algorithm class or further kwargs to the SMAC
   facade (via ``multi_objective_algorithm`` and/or ``multi_objective_kwargs``).
   Per default, a mean aggregation strategy is used as the multi-objective algorithm.


We show an example of how to use multi-objective with a nice Pareto front plot in our examples:
:ref:`Scalarized Multi-Objective Using ParEGO`.


.. _ParEgo: https://www.cs.bham.ac.uk/~jdk/UKCI-2015.pdf
.. _example: https://github.com/automl/SMAC3/blob/master/examples/python/scalarized_multi_objective.py
