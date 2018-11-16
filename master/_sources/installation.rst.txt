Installation
============

.. _requirements:

System Requirements
------------

SMAC has the following system requirements:

  * Linux operating system (for example Ubuntu),
  * Python (>=3.5.2).
  * C++ compiler (with C++11 supports) and SWIG (version 3.0 or later)

To install the C++ compiler and SWIG system-wide on a linux system with apt,
please call:

.. code-block:: bash

    sudo apt-get install build-essential swig

If you use Anaconda, you have to install both, gcc and SWIG, from Anaconda to
prevent compilation errors:

.. code-block:: bash

    conda install gxx_linux-64 gcc_linux-64 swig

.. _installation_pypi:

Installation from pypi
----------------------
To install SMAC3 from pypi, please use the following command on the command
line:

.. code-block:: bash

    # First install all requirements
    curl https://raw.githubusercontent.com/automl/smac3/master/requirements.txt | xargs -n 1 -L 1 pip install
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
