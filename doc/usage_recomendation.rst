.. _scenario: options.html#scenario
.. _PCS: options.html#paramcs
.. _TAE: tae.html

Usage Recommendation
===========
*SMAC* of course itself offers a lot of design choices, some of which are crucial to achieve peak performance. Luckily, often it is enough to distinguish between a few problem classes.
To make the usage of *SMAC* as easy as possible, we provide several facades designed for these different use cases. Here we give some general recommendations on
when to use which facade. This recommendation is based on our experience and technical limitations and is by far not intended to be complete:

+-----------------------+----------+-----+-----+-----+-----+-----+-----+-----+-------+
|                       | SMAC4BO  | SMAC4HPO              | SMAC4AC                 |
+=======================+==========+=====+=====+=====+=====+=====+=====+=====+=======+
| # parameter           | low      | low/medium/high       | low/medium/high         |
+-----------------------+----------+-----+-----+-----+-----+-----+-----+-----+-------+
| categorical parameter | no       | yes | yes | no  | no  | yes | yes | no  | no    |
+-----------------------+----------+-----+-----+-----+-----+-----+-----+-----+-------+
| conditional parameter | no       | no  | yes | no  | yes | no  | yes | no  | yes   |
+-----------------------+----------+-----+-----+-----+-----+-----+-----+-----+-------+
| instances             | no       | None or CV-folds      | yes                     |
+-----------------------+----------+-----+-----+-----+-----+-----+-----+-----+-------+
| deterministic         | yes      | can handle both       | can handle both         |
+-----------------------+----------+-----+-----+-----+-----+-----+-----+-----+-------+
| objective             | any      | e.g. validation loss  | e.g. runtime or quality |
+-----------------------+----------+-----+-----+-----+-----+-----+-----+-----+-------+

Some examples one typical use cases:

*SMAC4BO*: Vanilla Bayesian Optimization using a *Gaussian Process* and *Expected Improvement*
   - Optimizing the objective value of Branin and other low dimensional artificial testfunctions
   - Find the best learning rate for training a neuronal network wrt. to RMSE on a validation dataset

*SMAC4HPO*: Bayesian optimization using a *Random Forest*
  - Find the optimal choice of machine learning algorithm and its hyperparameters wrt. validation error
  - Tune the architecture and training parameters of a neural network wrt. to classification error on a validation dataset
  - Optimize hyperparameters of a SVM wrt. the CV-fold error
  - Minimize objective values of problems that are noisy and/or yield crashed runs (e.g. due to mem-outs)
  - Find the best setting of an RL-agent to minimize training time for a RL problem (or a set of RL problems)

*SMAC4AC*: Algorithm configuration using a *Random Forest*
  - Minimizing the average time it takes for a SAT-solver to solve a set of SAT instances
  - Configure the CPLEX solver to solve a set of mixed-integer-problems as fast as possible
  - Optimize the average quality of solutions returned by a configurable TSP solver

**Important:** If your problem is not covered in this table, this doesn't mean you can't benefit from using our tool. In case of doubt, please create an issue on Github.

