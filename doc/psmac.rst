Parallel SMAC (pSMAC)
---------------------

*SMAC* also provides a parallel mode to use several parallel computational resources (such as CPU cores).
This variant of SMAC is called *pSMAC* (parallel SMAC).
The general idea is that all target algorithm run evaluations are shared between the individual *SMAC* runs
such that all *SMAC* are better informed and can work together.

.. note::

	To use *pSMAC*, please note that it communicates via the file space,
	i.e., all *pSMAC* runs write from time to time its runhistory (all target algorithm evaluations)
	to disk and read the runhistories of all other *pSMAC* runs.
	So, a requirement for *pSMAC* is that it can write to a shared file space.

.. _psmaccommandline:

Commandline 
~~~~~~~~~~~
To use *pSMAC* via the commandline interface,
please specify the following two arguments:

     * **--shared_model True**: this will activate the information sharing between the *pSMAC* runs
     * **--input_psmac_dirs <output_path>**: list of *pSMAC*'s output directories (you can use shell-extension with asterix-notation)
     
.. note::

	*pSMAC* has no option to specify the number of parallel runs. You have to start as many *pSMAC* runs as you want to run.

On the command line an exemplary call could be:

.. code-block:: bash

        python smac --scenario SCENARIO --seed INT --shared_model True --input_psmac_dirs smac3-output*

We recommend that each *pSMAC* uses a different random **seed**.

If you want to verify that all arguments are correct and *pSMAC* finds all file on the file space,
please set the **verbose** level to **DEBUG** and grep in the following way

.. code-block:: bash
  
		python smac --verbose DEBUG [...] | grep -E "Loaded [0-9]+ new runs"

.. _psmacinpython:

Usage in Python
~~~~~~~~~~~~~~~

The same arguments used on the commandline can also be passed to the *Scenario* constructor.
See above for a detailed description.

.. code-block:: python

        scenario = Scenario({...
			"shared_model": True,
			"input_psmac_dirs": <output_path>
			})
				
        

