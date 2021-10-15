.. _SMAC4HPO_examples:

========
SMAC4HPO
========
SMAC4HPO is designed for hyperparameter optimization (HPO) problems.

SMAC4HPO uses an RF as its surrogate model. It is able to scale to higher evaluation budgets (more than 1000)
and higher number of dimensions. It is recommended to apply SMAC4HPO to mixed data types as well as conditional
hyperparameters.

SMAC4HPO by default only contains single fidelity approach. Typical functions optimized by SMAC4HPO
requires the following inputs:
 - *cfg*, the input configuration, example can be found [here](./SMAC4HPO_scm_example.py).
 - *seed*, random seed, applied when the function to be evaluated is not deterministic. (Optional)
 - *instance*, the instance to be evaluated, instances could be, for example, each fold of a k-folder cross validation. (Optional).








