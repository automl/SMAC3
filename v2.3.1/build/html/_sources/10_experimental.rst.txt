Experimental
============

.. warning::
    This part is experimental and might not work in each case. If you would like to suggest any changes, please let us know. 


Installation in Windows via WSL
------------------------------

SMAC can be installed in a WSL (Windows Subsystem for Linux) under Windows.

**1) Install WSL under Windows**

Install WSL under Windows. This SMAC installation workflow was tested with Ubuntu 18.04. For Ubuntu 20.04, 
it has been observed that the SMAC installation results in a segmentation fault (core dumped).

**2) Get Anaconda**

Download an Anaconda Linux version to drive D under Windows, e.g. D:\\Anaconda3-2023.03-1-Linux-x86_64
    
In the WSL, Windows resources are mounted under /mnt:

.. code:: bash

    cd /mnt/d
    bash Anaconda3-2023.03-1-Linux-x86_64

Enter this command to create the environment variable:

.. code:: bash

    export PATH="$PATH:/home/${USER}/anaconda3/bin

Input 'python' to check if the installation was successful.

**3) Install SMAC**

Change to your home folder and install the general software there:

.. code:: bash

    cd /home/${USER}
    sudo apt-get install software-properties-common
    sudo apt-get update
    sudo apt-get install build-essential swig
    conda install gxx_linux-64 gcc_linux-64 swig
    curl https://raw.githubusercontent.com/automl/smac3/master/requirements.txt | xargs -n 1 -L 1 pip install
