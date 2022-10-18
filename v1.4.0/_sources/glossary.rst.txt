Glossary
========

.. glossary::

    BO
        Bayesian Optimization. A Black-Box optimization algorithm weighing exploration & exploitation
        to find the minimum of its objective.

    HB
        `Hyperband <https://arxiv.org/abs/1603.06560>`_. A novel bandit-based algorithm for hyperparameter
        optimization. Hyperband is an extension of successive halving and therefore works with
        multi-fidelities.

    BOHB
        `Bayesian optimization and Hyperband <https://arxiv.org/abs/1807.01774>`_.

    SMAC
        Sequential Model-Based Algorithm Configuration.

    ROAR
        Random Online Adaptive Racing. A simple model-free instantiation of the general SMBO framework.
        It selects configurations uniformly random and iteratively compares them against the current incumbent
        using the intensification mechanism. See `SMAC extended <https://ai.dmi.unibas.ch/research/reading_group/hutter-et-al-tr2010.pdf>`_
        chapter 3.2 for details.

    BB
        Black-Box. Refers to an algorithm being optimized, where only input and output are observable.

    MF
        Multi-Fidelity. Refers to running an algorithm on multiple budgets (such as number of epochs or
        subsets of data) and thereby evaluating the performance prematurely.

    TAE
        Target Algorithm Evaluator. Your model, which returns a cost based on the given config,
        budget and instance.

    RF
        Random Forest.

    GP
        Gaussian Process.

    GP-MCMC
        Gaussian Process with Markov-Chain Monte-Carlo.

    Objective
        An objective is a metric to evaluate the quality or performance of an Algorithm.

    Budget
        Budget is another word for fidelity. Examples are the number of training epochs or the size of
        the data subset the algorithm is trained on.

    PCS
        `ConfigurationSpace <https://automl.github.io/ConfigSpace/master/API-Doc.html>`_ can be written/read from a PCS file.

    EPM
        Empirical Performance Models. Empirical performance models are regression models that characterize a given
        algorithm’s performance across problem instances and/or parameter settings. These models can predict the
        performance of algorithms on previously unseen input, including previously unseen problem instances and or
        previously untested parameter settings and are useful for analyzing of how an algorithm performs under different
        conditions, select promising configurations for a new problem instance, or surrogate benchmarks.

    Intensification
        A mechanism, that governs how many evaluations to perform with each configuration and when to trust a configuration
        enough to make it the new current best known configuration (the incumbent).

    CV
        Cross-Validation. 

    CLI
        Command-Line Interface.