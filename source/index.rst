.. SMAC3 documentation master file, created by
   sphinx-quickstart on Mon Sep 14 12:36:21 2015.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to SMAC3's documentation!
=================================
SMAC is a tool for algorithm configuration
to optimize the parameters of arbitrary algorithms across a set of instances.
This also includes hyperparameter optimization of ML algorithms.
The main core consists of Bayesian Optimization in combination with a simple racing mechanism to
efficiently decide which of two configuration performs better.

.. note::

   For a detailed description of its main idea,
   we refer to

      Hutter, F. and Hoos, H. H. and Leyton-Brown, K.
      Sequential Model-Based Optimization for General Algorithm Configuration
      In: Proceedings of the conference on Learning and Intelligent OptimizatioN (LION 5)


SMAC v3 is mainly written in Python 3.4. We expect that it should be also compatible with Python 2.7.
Its `Random Forest <https://bitbucket.org/aadfreiburg/random_forest_run>`_ is written in C++.

Contents:
---------
.. toctree::
   :maxdepth: 2

   installation
   manual
   contact
   license



Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

