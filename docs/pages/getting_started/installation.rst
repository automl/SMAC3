Installation
============

Requirements
~~~~~~~~~~~~

SMAC is written in python3 and therefore requires an environment with python>=3.7.
Furthermore, the random forest used in SMAC requires SWIG as a build dependency. Install it either in your
environment or on your system directly. The command to install swig on linux machines is the following:

.. code-block::

    apt-get install swig

SMAC is tested on Linux and Mac (Intel) machines with python 3.7, 3.8, and 3.9.

.. warning::
    When using Mac, make sure ``smac.optimize`` is
    wrapped inside ``if __name__ == "__main__"``.

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


.. warning::

    Please note that :term:`CLI` is only available when installing from GitHub.


Conda-forge
^^^^^^^^^^^^

Installing `auto-sklearn` from the `conda-forge` channel can be achieved by adding `conda-forge` to your channels with:

.. code:: bash

    conda config --add channels conda-forge
    conda config --set channel_priority strict


You must have `conda >=4.9`. To update conda or check your current conda version, please follow the instructions from `the official anaconda documentation <https://docs.anaconda.com/anaconda/install/update-version/>`_ . Once the `conda-forge` channel has been enabled, `auto-sklearn` can be installed with:

.. code:: bash

    conda install smac


It is possible to list all of the versions of `smac` available on your platform with:

.. code:: bash

    conda search smac --channel conda-forge

to read in more details check
`smac feedstock <https://github.com/conda-forge/smac-feedstock>`_.

for more information about Conda forge check
`conda-forge documentations <https://conda-forge.org/docs/>`_.


Optional Dependencies
~~~~~~~~~~~~~~~~~~~~~

SMAC comes with a set of optional dependencies that can be installed using `setuptools
extras <https://setuptools.pypa.io/en/latest/setuptools.html#declaring-extras-optional-features-with-their-own-dependencies>`_:


For convenience, there is an all meta-dependency that installs ``all`` optional dependencies:

.. code-block::

    pip install smac[all]

    