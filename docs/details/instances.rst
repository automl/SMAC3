Instances and Features
======================

Often you want to optimize the cost across different datasets, subsets, or even different
transformations. For this purpose, you can use instances. A randomly selected instance is passed to
the target algorithm evaluator, in which you can access it.

To work with instances, you need to add your predefined instances (list) to the scenario object. The
items of the instances can be chosen individually. In the following example,
I want to use five different subsets, identified by its id:

.. code-block:: bash

    instances = [0, 1, 2, 3, 4]
    scenario = Scenario({
      ...
      'instances': instances,
      ...
    })


Alternatively, you can also pass ``instance_file`` to the scenario object.
Additionally to the instances, there is the option to define ``features``. Those instance features are
used to expand the internal X matrix and thus play a role in training the underlying optimizer.
See `here <https://github.com/automl/SMAC3/blob/master/smac/runhistory/runhistory2epm.py#L423>`_ for
the exact implementation.

For example, if I want to add the number of samples and the mean of each subset, I can do as
follows:

.. code-block:: bash

    instance_features = {
      0: [121, 0.6],
      1: [140, 0.65],
      2: [99, 0.45],
      3: [102, 0.59],
      4: [132, 0.48],
    }
    scenario = Scenario({
      ...
      'instances': instances,
      'features': instance_features
      ...
    })


