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

    Please note that calling SMAC via :term:`CLI` is only available when installing from GitHub. We
    refer to :ref:`Branin` for more details.



Conda-forge
~~~~~~~~~~~

Installing SMAC from the `conda-forge` channel can be achieved by adding `conda-forge` to your channels with:

.. code:: bash

    conda config --add channels conda-forge
    conda config --set channel_priority strict


You must have `conda >= 4.9` installed. To update conda or check your current conda version, please follow the instructions from `the official anaconda documentation <https://docs.anaconda.com/anaconda/install/update-version/>`_ . Once the `conda-forge` channel has been enabled, SMAC can be installed with:

.. code:: bash

    conda install smac
    

Read `SMAC feedstock <https://github.com/conda-forge/smac-feedstock>`_ for more details.
    