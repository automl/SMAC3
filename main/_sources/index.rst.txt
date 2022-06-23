Home
====

.. toctree::
   :hidden:
   :maxdepth: 2

   getting_started/index
   examples/index
   details/index
   api
   glossary
   faq
   license


SMAC is a tool for algorithm configuration to optimize the parameters of
arbitrary algorithms, including hyperparameter optimization of Machine Learning algorithms. The main core consists of
Bayesian Optimization in combination with an aggressive racing mechanism to
efficiently decide which of two configurations performs better.

SMAC3 is written in Python3 and continuously tested with Python 3.7, 3.8 and 3.9. Its Random
Forest is written in C++. In further texts, SMAC is representatively mentioned for SMAC3.

If you use SMAC, please cite our `JMLR paper <https://jmlr.org/papers/v23/21-0888.html>`_:

.. code-block:: text

    @inproceedings {lindauer-arxiv21a,
      author = {Marius Lindauer and Katharina Eggensperger and Matthias Feurer and André Biedenkapp and Difan Deng and Carolin Benjamins and Tim Ruhkopf and René Sass and Frank Hutter},
      title = {SMAC3: A Versatile Bayesian Optimization Package for Hyperparameter Optimization},
      booktitle = {ArXiv: 2109.09831},
      year = {2021},
      url = {https://arxiv.org/abs/2109.09831}
    }

For the original idea, we refer to:

.. code-block:: text

   Hutter, F. and Hoos, H. H. and Leyton-Brown, K.
   Sequential Model-Based Optimization for General Algorithm Configuration
   In: Proceedings of the conference on Learning and Intelligent OptimizatioN (LION 5)


Contact
-------

SMAC3 is developed by `<https://automl.org>`_.
If you want to contribute or found an issue please visit our github page `<https://github.com/automl/SMAC3>`_.
Our guidelines for contributing to this package can be found `here <https://github.com/automl/SMAC3/blob/main/CONTRIBUTING.md>`_.
