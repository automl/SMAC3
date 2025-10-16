# 🌍 SMAC Ecosystem

SMAC (Sequential Model-based Algorithm Configuration) is a black-box optimization framework
used for hyperparameter tuning.
SMAC3 is not only used by itself but also as a backend for other HPO tools:
Auto-WEKA [Thornton et al., 2013, Kotthoff et al., 2017]: A tool for Algorithm Selection and HPO
Auto-Sklearn [Feurer et al., 2015, Feurer et al., 2022]: An automated machine learning toolkit and a drop-in replacement for a scikit-learn estimator
Auto-Pytorch [Mendoza et al., 2019, Zimmer et al., 2021, Deng et al., 2022]: A tool for joint NAS and HPO for Deep Learning
SMAC is extended for multi-objective algorithm configuration by MO-SMAC [Rook et al., 2025]
SMAC is supported by Optuna as a sampler


---

## 🔗 Key Ecosystem Tools

- [**DeepCAVE**](https://github.com/automl/DeepCAVE) — Visualization and analysis of SMAC runs  
- [**CARPS**](https://github.com/automl/CARPS) — Experiment and configuration management  
- [**HyperSweeper**](https://github.com/automl/hypersweeper) — Distributed hyperparameter optimization  
- [**Optuna Integration**](https://optuna.org) — SMAC can act as an Optuna sampler  

---

## DeepCave

- Visualization and analysis tool for AutoML, especially for the sub-problem hyperparameter optimization
- Allows to efficiently generate insights for AutoML problems and brings the human back in the loop
- Interactive GUI

## CARPS

- Framework for benchmarking N optimization methods on M benchmarks
- Lightweight interface between optimizer and benchmark
- Many included HPO tasks from different task types BB, MF, MO, MOMF
- Subselections for task types
- Tutorials available for easy integration of your own optimizer or tasks

## HyperSweeper

- For expensive objective functions
- On a cluster (slurm, joblib, ray)
- Evaluates functions as separate jobs
- Custom hydra sweeper



## 🧩 Integration Examples

SMAC powers:
- [Auto-Sklearn](https://github.com/automl/auto-sklearn)
- OpenML AutoML Benchmarks

---

## 📚 References

- [SMAC3: A Versatile Bayesian Optimization Package (arXiv 2021)](https://arxiv.org/abs/2109.06716)  
- [Sequential Model-Based Optimization for General Algorithm Configuration (Hutter et al., 2011)](https://www.cs.ubc.ca/~hutter/papers/10-LION5-SMAC.pdf)

---

