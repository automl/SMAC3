Optimization across Instances
=============================

Often you want to optimize the cost across different datasets, subsets, or even different
augmentations. For this purpose, you can use instances.

To work with instances, you need to add your pre-defined instance names to the scenario object.
In the following example, we want to use five different subsets, identified by its id:

.. code-block:: python

    instances = ["d0", "d1", "d2", "d3", "d4"]
    scenario = Scenario(
      ...
      "instances": instances,
      ...
    )


Additionally to the instances, there is the option to define ``instance_features``. Those instance features are
used to expand the internal X matrix and thus play a role in training the underlying surrogate model. For example, if I 
want to add the number of samples and the mean of each subset, I can do as follows:

.. code-block:: bash

    instance_features = {
      "d0": [121, 0.6],
      "d1": [140, 0.65],
      "d2": [99, 0.45],
      "d3": [102, 0.59],
      "d4": [132, 0.48],
    }

    scenario = Scenario(
      ...
      "instances": instances,
      "instance_features": instance_features
      ...
    )
