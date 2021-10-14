Package Overview 
================

SMAC supports you in determining well-performing hyperparameter configurations for your algorithms.
By being a robust and flexible framework for `Bayesian Optimization (BO) TODO <linktoBO>`_ SMAC can improve performance whithin few function evaluations.
It offers several `facadesTODO <linktofacades>`_ and pre-sets for typical use cases, such as optimizing hyperparameters, solving low dimensional continuous (artificial) global optimization problems and configuring algorithms to perform well across multiple problem `instancesTODO <linktoinstances>`_.


.. note::
    Attention: This package is a reimplementation of the original SMAC tool
    (see reference below).
    However, the reimplementation slightly differs from the original SMAC.
    For comparisons against the original SMAC, we refer to a stable release of SMAC (v2) in Java
    which can be found `here <http://www.cs.ubc.ca/labs/beta/Projects/SMAC/>`_.


SMAC has following characteristics and capabilities:

- global optimizer
    - Bayesian Optimization → sample-efficient, no gradients required.
- optimize following types of functions:
    - black-box (BB): A function where we can only observe input and output behaviour. Can be undifferentiable.
    - grey-box: We have access to intermediate results/performances.
- optimize few up to many hyperparameters
- optimize categorical, continuous and conditional (hierarchical) hyperparameters
- different/any objectives possible, e.g., quality or runtime
- multi-fidelity
    - If you want to optimize a grey-box function and you can specify the budget with which your algorithm can run (e.g., certain number of epochs,
      iterations or steps or total runtime), SMAC can take intermediate performance into account and already discard
      unsuitable hyperparameter configurations early on → speed-up.
- instances
    - Find well-performing hyperparameter configurations not only for one instance of an algorithm, but for many.