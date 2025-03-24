Glossary
========

.. glossary::

    BB
        See :term:`Black-Box`.

    BO
        See :term:`Bayesian Optimization`.

    BOHB
        `Bayesian optimization and Hyperband <https://arxiv.org/abs/1807.01774>`_.

    CLI
        Command-Line Interface.

    CV
        Cross-Validation.

    GP
        Gaussian Process.

    GP-MCMC
        Gaussian Process with Markov-Chain Monte-Carlo.

    HB
        See :term:`Hyperband`.

    HP
        Hyperparameter.

    MF
        See :term:`Multi-Fidelity`.

    RF
        Random Forest.

    ROAR
        See :term:`Random Online Adaptive Racing`.

    SMAC
        Sequential Model-Based Algorithm Configuration.

    SMBO
        Sequential Mode-Based Optimization.

    Bayesian Optimization
        Bayesian optimization is a sequential design strategy for global optimization of black-box functions that does 
        not assume any functional forms. It is usually employed to optimize expensive-to-evaluate functions.
        A Bayesian optimization weights exploration and exploitation to find the minimum of its objective.

    Black-Box
        Refers to an algorithm being optimized, where only input and output are observable.

    Budget
        Budget is another word for fidelity. Examples are the number of training epochs or the size of
        the data subset the algorithm is trained on. However, budget can also be used in the context of
        instances. For example, if you have 100 instances (let's say we optimize across datasets) and you want to run
        your algorithm on 10 of them, then the budget is 10.

    Hyperband
        `Hyperband <https://arxiv.org/abs/1603.06560>`_. A novel bandit-based algorithm for hyperparameter
        optimization. Hyperband is an extension of successive halving and therefore works with
        multi-fidelities.

    Incumbent
        The incumbent is the current best known configuration.

    Instances
        Often you want to optimize across different datasets, subsets, or even different transformations (e.g.
        augmentation). In general, each of these is called an instance. Configurations are evaluated on multiple
        instances so that a configuration is found which performs superior on all instances instead of only
        a few.

    Intensification
        A mechanism that governs how many evaluations to perform with each configuration and when to trust a
        configuration enough to make it the new current best known configuration (the incumbent).

    Multi-Fidelity
        Multi-fidelity refers to running an algorithm on multiple budgets (such as number of epochs or
        subsets of data) and thereby evaluating the performance prematurely.

    Multi-Objective
        A multi-objective optimization problem is a problem with more than one objective.
        The goal is to find a solution that is optimal or at least a good compromise in all objectives.

    Objective
        An objective is a metric to evaluate the quality or performance of an algorithm.

    Random Online Adaptive Racing
        Random Online Adaptive Racing. A simple model-free instantiation of the general :term:`SMBO` framework.
        It selects configurations uniformly at random and iteratively compares them against the current incumbent
        using the intensification mechanism. See `SMAC extended <https://ai.dmi.unibas.ch/research/reading_group/hutter-et-al-tr2010.pdf>`_
        chapter 3.2 for details.

    Target Function
        Your model, which returns a cost based on the given config, seed, budget, and/or instance.

    Trial
        Trial is a single run of a target function on a combination of configuration, seed, budget and/or instance.
