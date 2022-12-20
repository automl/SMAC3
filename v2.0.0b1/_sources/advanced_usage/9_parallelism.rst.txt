Parallelism
===========

SMAC supports multiple workers natively. Just specify ``n_workers`` in the scenario and you are ready to go. 


.. note :: 
    
    Please keep in mind that additional workers are only used to evaluate trials. The main thread still orchestrates the
    optimization process, including training the surrogate model.