Callbacks
=========

Callbacks allow customizing the behavior of SMAC to ones needs. Currently, the list of
implemented callbacks is very limited, but they can easily be added.


How to add a new callback
^^^^^^^^^^^^^^^^^^^^^^^^^

* Implement a callback class in ``smac/callbacks.py``. There are no restrictions on how such a
  callback must look like, but it is recommended to implement the main logic inside the `__call__`
  function, such as for example in ``IncorporateRunResultCallback``.

* Add your callback to ``smac.smbo.optimizer.SMBO._callbacks``, using the name of your callback
  as the key, and an empty list as the value.

* Add your callback to ``smac.smbo.optimizer.SMBO._callback_to_key``, using the callback class as
  the key, and the name as value (the name used in 2.).

* Implement calling all registered callbacks at the correct place. This is as simple as 
  ``for callback in self._callbacks['your_callback']: callback(*args, **kwargs)``, where you
  obviously need to change the callback name and signature.