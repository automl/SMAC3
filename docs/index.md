Home
# SMAC3 Documentation

<img src="images/smac_icon.png" alt="SMAC3 Logo" width="150"/>

## Introduction

SMAC is a tool for algorithm configuration to optimize the parameters of arbitrary algorithms, including hyperparameter optimization of Machine Learning algorithms. The main core consists of Bayesian Optimization in combination with an aggressive racing mechanism to efficiently decide which of two configurations performs better.

SMAC3 is written in Python3 and continuously tested with Python 3.8, 3.9, and 3.10. Its Random Forest is written in C++. In the following, SMAC is representatively mentioned for SMAC3.


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
