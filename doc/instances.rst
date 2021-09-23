Instances
---------

Often you want to optimize the cost across different datasets, subsets, or even different
transformations. For this purpose, you can use instances. A randomly selected instance is passed to
the target algorithm evaluator, in which you can access it.

To work with instances, you need to add your predefined instances (list) to the scenario object. The
items of the instances can be chosen individually. In the following example, I want to use five different subsets, identified by its id:

.. code-block:: bash
    instances = [0, 1, 2, 3, 4]
    scenario = Scenario({
      ...
      'instances': instances,
      ...
    })


Alternatively, you can also pass `instance_file` to the scenario object.
Additionally to the instances, there is the option to define `features`. Those instance features are
used to expand the internal X matrix and thus play a role in training the underlying optimizer. See `here <https://github.com/automl/SMAC3/blob/master/smac/runhistory/runhistory2epm.py#L423>`_ for
the exact implementation.

For example, if I want to add the number of samples and the mean for each subset, I can do as
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

Lastly, here are some more examples to incorporate instances in your code:
- `SMAC4MF SGD <https://github.com/automl/SMAC3/blob/master/examples/SMAC4MF/SMAC4MF_sgd_example.py>`_
- `SMAC4MF MLP <https://automl.github.io/SMAC3/master/examples/SMAC4MF/SMAC4MF_mlp_example.html#sphx-glr-examples-smac4mf-smac4mf-mlp-example-py`_
- `Spear <https://github.com/automl/SMAC3/tree/master/examples/quickstart/spear_qcp`_


F.A.Q.
~~~~~~

.. rubric:: When using subsets, does it make sense to use instances over having cross-validation directly in my TAE?

It is recommended if the sum of the performance values across the subsets makes sense, and dividing the sum is a certain amount of information about the total sum. Ultimately, SMAC optimizes for the total and can take advantage of the fact that only parts are evaluated if there are bad configurations.
