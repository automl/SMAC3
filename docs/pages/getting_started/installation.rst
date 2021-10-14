Installation
============

Requirements
~~~~~~~~~~~~

SMAC is written in python3 and therefore requires an environment with python>=3.7.
Furthermore, the random forest used in SMAC requires SWIG as a build dependency. Install it either in your
environment or on your system directly. The command to install swig on linux machines is the following:

.. code-block::

    apt-get install swig


Anaconda
~~~~~~~~

Create and activate environment:

.. code-block::

    conda create -n SMAC python=3.9
    conda activate SMAC


If you haven't installed swig yet, you can install it directly inside the environment:

.. code-block::

    conda install gxx_linux-64 gcc_linux-64 swig


Now install SMAC via PyPI:

.. code-block::

    pip install smac


Or alternatively, clone the environment from GitHub directly:

.. code-block::

    git clone https://github.com/automl/SMAC3.git && cd SMAC3
    pip install -r requirements.txt
    pip install .


Optional Dependencies
~~~~~~~~~~~~~~~~~~~~~

SMAC comes with a set of optional dependencies that can be installed using `setuptools
extras <https://setuptools.pypa.io/en/latest/setuptools.html#declaring-extras-optional-features-with-their-own-dependencies>`_:

- `lhd`: Latin Hypercube Design
- `gp`: Gaussian Process Models

These can be installed from PyPI or manually:

.. code-block::

    pip install smac[gp,lhd]

.. code-block::

    pip install .[gp,lhd]

For convenience, there is also an all meta-dependency that installs all optional dependencies:

.. code-block::

    pip install smac[all]

    