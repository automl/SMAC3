.. SMAC3 documentation master file, created by
   sphinx-quickstart on Mon Sep 14 12:36:21 2015.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

SMAC3 documentation!
=================================
SMAC is a tool for algorithm configuration.
It optimizes configurations of arbitrary algorithms across a set of instances.
This includes, but is not limited to, optimization of parameters of hard combinatorial problem solvers and
hyperparameter optimization of various machine learning algorithms.
The main core consists of Bayesian Optimization in combination with a simple racing mechanism on
the instances to efficiently decide which of two configurations performs better.


Contents:
---------
.. toctree::
   :titlesonly:
   :maxdepth: 1

   installation
   quickstart
   usage_recomendation
   manual
   api
   examples/index
   faq
   contact
   license


.. note::

   For a detailed description of its main idea, we refer to:

      Hutter, F. and Hoos, H. H. and Leyton-Brown, K.
      Sequential Model-Based Optimization for General Algorithm Configuration
      In: Proceedings of the conference on Learning and Intelligent OptimizatioN (LION 5)

   The SMAC3 package is not meant to be a reference implementation for SMAC according the above cited LION paper.
   Please use SMAC v2 (Java) for comparisons against SMAC.

.. note::

   If you used SMAC in one of your research projects,
   please cite us:


     | @misc{smac-2017,
     |    title={SMAC v3: Algorithm Configuration in Python},
     |    author={Marius Lindauer and Katharina Eggensperger and Matthias Feurer and Stefan Falkner and André Biedenkapp and Frank Hutter},
     |    year={2017},
     |    publisher={GitHub},
     |    howpublished={\\url{https://github.com/automl/SMAC3}}
     | }

SMAC3 is mainly written in Python 3 and continuously tested with Python 3.6-3.8.
Its `Random Forest <https://github.com/automl/random_forest_run>`_ is written in
C++11.


.. Indices and tables
.. ------------------
..
.. * :ref:`genindex`
.. * :ref:`modindex`
.. * :ref:`search`

