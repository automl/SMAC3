Experimental
============

.. warning::
    This part is experimental and might not work in each case. In case you would like to suggest any changes, please let us know. 


Installation in Windows via WSL
------------------------------

SMAC can be installed in a WSL (Windows-Subsystem für Linux) under windows.

**1) Install WSL under Windows**

This workflow was tested with Ubuntu 18.04. For the 20.04 version, it is possible that the installation will result in a segmentation fault (core dumped).

**2) Get Anaconda**

Download an Anaconda Linux version to drive D under Windows, e.g. \\Anaconda\\Anaconda2-4.2.0-linux-x86_64.sh
    
Windows resources are mounted under /mnt in the WSL:

.. code:: bash

    cd /mnt/d
    bash Anaconda 2-4.2.0-linux-x86_ 64.sh

    
Enter this command to create the environment variable:

.. code:: bash

    export PATH="$PATH:/home/USERNAME/anaconda3/bin

Input 'python' to check if the installation was successful.

**3) Install SMAC**

Change to your home folder and install the general software there:

.. code:: bash

    cd /home/USERNAME
    sudo apt-get install software-properties-common
    sudo apt-get update
    sudo apt-get install build-essential swig
    conda install gxx_linux-64 gcc_linux-64 swig
    curl https://raw.githubusercontent.com/automl/smac3/master/requirements.txt | xargs -n 1 -L 1 pip install
