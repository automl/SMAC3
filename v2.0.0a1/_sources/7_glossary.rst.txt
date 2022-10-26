Glossary
========

.. glossary::

    SMAC
        Sequential Model-Based Algorithm Configuration.

    BO
        See :term:`Bayesian Optimization`.

    HB
        See :term:`Hyperband`.

    BOHB
        `Bayesian optimization and Hyperband <https://arxiv.org/abs/1807.01774>`_.

    ROAR
        See :term:`Random Online Adaptive Racing`.

    BB
        See :term:`Black-Box`.
    
    MF
        See :term:`Multi-Fidelity`.

    RF
        Random Forest.

    GP
        Gaussian Process.

    GP-MCMC
        Gaussian Process with Markov-Chain Monte-Carlo.

    CV
        Cross-Validation. 

    CLI
        Command-Line Interface.

    HP
        Hyperparameter.

    Bayesian Optimization
        Bayesian optimization is a sequential design strategy for global optimization of black-box functions that does 
        not assume any functional forms. It is usually employed to optimize expensive-to-evaluate functions.
        A Bayesian optimization weights exploration and exploitation to find the minimum of its objective.

    Hyperband
        `Hyperband <https://arxiv.org/abs/1603.06560>`_. A novel bandit-based algorithm for hyperparameter
        optimization. Hyperband is an extension of successive halving and therefore works with
        multi-fidelities.

    Random Online Adaptive Racing
        Random Online Adaptive Racing. A simple model-free instantiation of the general SMBO framework.
        It selects configurations uniformly random and iteratively compares them against the current incumbent
        using the intensification mechanism. See `SMAC extended <https://ai.dmi.unibas.ch/research/reading_group/hutter-et-al-tr2010.pdf>`_
        chapter 3.2 for details.

    Black-Box
        Refers to an algorithm being optimized, where only input and output are observable.

    Target Function
        Your model, which returns a cost based on the given config, seed, budget, and/or instance.

    Trial
        Trial is a single run of a target function on a combination of configuration, seed, budget and/or instance.

    Objective
        An objective is a metric to evaluate the quality or performance of an algorithm.

    Multi-Objective
        A multi-objective optimization problem is a problem with more than one objective. 
        The goal is to find a solution that is optimal or at least a good compromise in all objectives.

    Budget
        Budget is another word for fidelity. Examples are the number of training epochs or the size of
        the data subset the algorithm is trained on. However, budget can also be used in the context of 
        instances. For example, if you have 100 instances (let's say we optimize across datasets) and you want to run 
        your algorithm on 10 of them, then the budget is 10.

    Multi-Fidelity
        Multi-fidelity refers to running an algorithm on multiple budgets (such as number of epochs or
        subsets of data) and thereby evaluating the performance prematurely.

    Instances
        Often you want to optimize across different datasets, subsets, or even different transformations (e.g. 
        augmentation). In general, each of these is called an instance. Configurations are evaluated on multiple
        instances so that a configuration found which performs superior on all instances instead of only 
        a few.

    Intensification
        A mechanism, that governs how many evaluations to perform with each configuration and when to trust a 
        configuration enough to make it the new current best known configuration (the incumbent).

    Incumbent
        The incumbent is the current best known configuration.
