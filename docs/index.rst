Home
====

.. toctree::
   :hidden:
   :maxdepth: 2

   pages/getting_started/index
   pages/details/index
   pages/api/index
   pages/glossary
   pages/FAQ
   pages/contact_citation_license


SMAC is a tool for algorithm configuration to optimize the parameters of
arbitrary algorithms, including hyperparameter optimization of Machine Learning algorithms. The main core consists of
Bayesian Optimization in combination with an aggressive racing mechanism to
efficiently decide which of two configurations performs better.

For a detailed description of the main ideas, we refer to:

```
Hutter, F. and Hoos, H. H. and Leyton-Brown, K.
Sequential Model-Based Optimization for General Algorithm Configuration
In: Proceedings of the conference on Learning and Intelligent OptimizatioN (LION 5)
```

```
Marius Lindauer and Katharina Eggensperger and Matthias Feurer and André Biedenkapp and Difan Deng and Carolin Benjamins and René Sass and Frank Hutter
SMAC3: A Versatile Bayesian Optimization Package for Hyperparameter Optimization
```

SMAC3 is written in Python3 and continuously tested with Python 3.7, 3.8 and 3.9. Its Random
Forest is written in C++. In further texts, SMAC is representatively mentioned for SMAC3.