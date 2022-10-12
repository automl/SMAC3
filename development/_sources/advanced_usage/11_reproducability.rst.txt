Reproducability
===============


The intensifier ``Intensifier`` does not ensure reproducability because it is depending on computation time.
The intensification of configurations are depending on the ``intensifier_percentage`` and how long the optimization 
run took so far. Based on that, a ``time_bound`` for the intensification procedure is computed (in ``base_smbo``),
which is further used to determine whether the next iteration should start (``intensifier.process_results``).

.. warning ::
 
    The intensifier ``Intensifier`` (used by HyperparameterOptimizationFacade) is *not* reproducable!


However, since ``time_bound`` is only used in ``Intensifier``, all other intensifiers guarantee reproducability when 
using one worker.