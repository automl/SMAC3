Installation
============

.. _requirements:

Requirements
------------
The random forest used in SMAC depends on a C++11 compatible compiler
and on SWIG (>3.0).

To install both requirements system-wide on a linux system with apt, 
please call:

.. code-block:: bash

    sudo apt-get install build-essential swig

If you use Anaconda, you have to install both gcc and SWIG from Anaconda to
prevent compilation errors:

.. code-block:: bash

    conda install gxx_linux-64 gcc_linux-64 swig

.. _installation_pypi:

Installation from pypi
----------------------
To install SMAC3 from pypi, please use the following command on the command
line:

.. code-block:: bash

    pip install smac
    
If you want to install it in the user space (e.g., because of missing
permissions), please add the option :code:`--user` or create a virtualenv.

.. _manual_installation:

Manual Installation
-------------------
To install SMAC3 from command line, please type the following commands on the
command line:

.. code-block:: bash

    git clone https://github.com/automl/SMAC3
    cd SMAC3
    cat requirements.txt | xargs -n 1 -L 1 pip install
    python setup.py install
