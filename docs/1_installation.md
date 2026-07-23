# Installation
## Requirements

SMAC is written in python3 and therefore requires an environment with python>=3.9.

!!! info 

    SMAC is tested on Linux and Mac machines with python >=3.9.


## SetUp

We recommend using Anaconda to create and activate an environment:

```bash
conda create -n SMAC python=3.10
conda activate SMAC
```

## Install SMAC
You can install SMAC either using PyPI or Conda-forge.

### PYPI
To install SMAC with PyPI call:

```bash
pip install smac
```

Or alternatively, clone the environment from GitHub directly:

```bash
git clone https://github.com/automl/SMAC3.git && cd SMAC3
pip install -e ".[dev]"
```

### Conda-forge

Installing SMAC from the `conda-forge` channel can be achieved by adding `conda-forge` to your channels with:

```bash
conda config --add channels conda-forge
conda config --set channel_priority strict
```

You must have `conda >= 4.9` installed. To update conda or check your current conda version, please follow the instructions from [the official anaconda documentation](https://docs.anaconda.com/anaconda/install/update-version/). Once the `conda-forge` channel has been enabled, SMAC can be installed with:

```bash
conda install smac
```

Read [SMAC feedstock](https://github.com/conda-forge/smac-feedstock) for more details.

## Running SMAC with pyrfr
starting from 2.4.0, SMAC uses random forest from [sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html)
instead of random forest from [pyrfr](https://pypi.org/project/pyrfr/) as the default surrogate model for HPO tasks.
However, you could still use the old pyrfr surrogate model by calling `smac.facade.old.HyperparameterOptimizationRFRFacade`
and `smac.facade.old.MultiFidelityRFRFacade`.

The Random Forest used in SMAC requires SWIG as a build dependency. 
You could install swig either on the system level e.g. using the following command for Linux:
```bash
apt-get install swig
```

Or install swig inside of an already created conda environment using:

```bash
conda install gxx_linux-64 gcc_linux-64 swig
```

And then install smac with the pyrfr option:
```
pip install smac[pyrfr]
```


### pyrfr on Windows (native or via WSL, experimental)

SMAC can be installed under Windows in a WSL (Windows Subsystem for Linux). 
You can find an instruction on how to do this here: [Experimental](./10_experimental.md).
However, this is experimental and might not work in each case. 
If you would like to suggest any changes, please let us know. 
