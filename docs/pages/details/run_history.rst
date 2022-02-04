Run-History
===========


Iterating over Run-History
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code::

   smac = SMAC4AC(...)
   smac.optimize(...)
   rh = smac.get_runhistory()
   for (config_id, instance_id, seed, budget), (cost, time, status, starttime, endtime, additional_info) in rh.data.items():
      config = rh.ids_config[config_id]
      ...
   
