Parallelism
===========

SMAC supports multiple workers natively via Dask. Just specify ``n_workers`` in the scenario and you are ready to go. 


.. note :: 
    
    Please keep in mind that additional workers are only used to evaluate trials. The main thread still orchestrates the
    optimization process, including training the surrogate model.


.. warning ::

    Using high number of workers when the target function evaluation is fast might be counterproductive due to the 
    overhead of communcation. Consider using only one worker in this case.


.. warning ::

    When using multiple workers, SMAC is not reproducible anymore.


.. warning ::

    You cannot use resource limitation (pynisher, via the `scenario` arguments `trail_walltime_limit` and `trial_memory_limit`).
    This is because pynisher works by running your function inside of a subprocess.
    Once in the subprocess, the resources will be limited for that process before running your function. 
    This does not work together with pickling - which is required by dask to schedule jobs on the cluster, even on a local one.


.. warning ::

    Start/run SMAC inside ``if __name__ == "__main__"`` in your script otherwise Dask is not able to correctly
    spawn jobs and probably this runtime error will be raised:

    .. code-block ::

        RuntimeError: 
            An attempt has been made to start a new process before the
            current process has finished its bootstrapping phase.

            This probably means that you are not using fork to start your
            child processes and you have forgotten to use the proper idiom
            in the main module:

                if __name__ == '__main__':
                    freeze_support()
                    ...

            The "freeze_support()" line can be omitted if the program
            is not going to be frozen to produce an executable.




Running on a Cluster
--------------------
You can also pass a custom dask client, e.g. to run on a slurm cluster.
See our :ref:`parallelism example<Parallelization-on-Cluster>`.

.. warning ::

    On some clusters you cannot spawn new jobs when running a SLURMCluster inside a
    job instead of on the login node. No obvious errors might be raised but it can hang silently.

.. warning ::

    Sometimes you need to modify your launch command which can be done with
    ``SLURMCluster.job_class.submit_command``. 

.. code-block:: python

    cluster.job_cls.submit_command = submit_command
    cluster.job_cls.cancel_command = cancel_command
