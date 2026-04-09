Home
# SMAC3 Documentation

<img src="images/smac_icon.png" alt="SMAC3 Logo" width="150"/>

## Introduction

SMAC is a tool for algorithm configuration to optimize the parameters of arbitrary algorithms, including hyperparameter optimization of Machine Learning algorithms. The main core consists of Bayesian Optimization in combination with an aggressive racing mechanism to efficiently decide which of two configurations performs better.

SMAC3 is written in Python3 and continuously tested with Python 3.8, 3.9, and 3.10. Its Random Forest is written in C++. In the following, SMAC is representatively mentioned for SMAC3.

## Ecosystem

SMAC3 integrates with several tools in the AutoML ecosystem to enhance hyperparameter optimization workflows:

### DeepCAVE
[DeepCAVE](https://github.com/automl/DeepCAVE) is an interactive visualization tool for optimization. It provides advanced plotting and analysis of SMAC runs, enabling users to efficiently generate insights for AutoML problems. DeepCAVE brings the human back into the loop with its intuitive graphical user interface.

### CARPS
[CARPS](https://github.com/carps-ai/carps) (Configuration And Running Parallel Systems) is a framework for benchmarking and parallelizing optimization methods. It offers a lightweight interface between optimizers and benchmarks, supports native SMAC3 integration, and includes many HPO tasks from various domains such as black-box, multi-fidelity, multi-objective, and multi-objective multi-fidelity optimization.

### HyperSweeper
[HyperSweeper](https://github.com/automl/HyperSweeper) is designed for efficient hyperparameter optimization of large models, especially when objective functions are expensive to evaluate. It supports distributed computation on clusters (using Slurm, Joblib, or Ray) and evaluates functions as separate jobs for scalability.

### Optuna Integration
SMAC3 is available as a sampler in [Optuna](https://optuna.org/), allowing users to leverage SMAC's optimization strategies within Optuna's flexible framework for hyperparameter optimization.


## Features

* Open source + active maintenance
* Rich search space with floats, ordinals, categoricals and conditions
* Ask-and-Tell Interface
* Continue and Warmstart Optimization
* Intensification mechanism to efficiently compare configurations
* User priors
* Parallelization, local and on a cluster with Dask
* Multi-fidelity optimization, e.g. when we can evaluate our function with different resolutions
* Multi-objective optimization with ParEGO
* Optimization across many tasks (aka algorithm configuration)
* Function to optimize can either be pythonic or called via a script
* Easily extensible with callbacks

## Cite Us
If you use SMAC, please cite our [JMLR paper](https://jmlr.org/papers/v23/21-0888.html):

```bibtex
@article{lindauer-jmlr22a,
       author  = {Marius Lindauer and Katharina Eggensperger and Matthias Feurer and André Biedenkapp and Difan Deng and Carolin Benjamins and Tim Ruhkopf and René Sass and Frank Hutter},
       title   = {SMAC3: A Versatile Bayesian Optimization Package for Hyperparameter Optimization},
       journal = {Journal of Machine Learning Research},
       year    = {2022},
       volume  = {23},
       number  = {54},
       pages   = {1--9},
       url     = {http://jmlr.org/papers/v23/21-0888.html}
}
```

For the original idea, we refer to:

```text
Hutter, F. and Hoos, H. H. and Leyton-Brown, K.
Sequential Model-Based Optimization for General Algorithm Configuration
In: Proceedings of the conference on Learning and Intelligent Optimization (LION 5)
```

## Contact

SMAC3 is developed by [AutoML.org](https://www.automl.org). If you want to contribute or found an issue, please visit our [GitHub page](https://github.com/automl/SMAC3). Our guidelines for contributing to this package can be found [here](https://github.com/automl/SMAC3/blob/main/CONTRIBUTING.md).
