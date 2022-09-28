Components
==========

Additionally to the basic components mentioned in GETTING STARTED, all other components are
explained in the following to get a better picture. These components are all used to guide the optimization process and 
simple changes can influence the results drastically.


Model
-----


Acquisition Function
--------------------


Acquisition Maximier
--------------------


Initial Design 
--------------


Random Design
-------------


Intensifier
-----------


Multi Objective Algorithm
-------------------------

Please refer to multi objective for more information about multi objective optimization.


Runhistory
----------

how to iterate

.. code::

    ...
    smac = BlackboxFacade(...)
    for trial_info, trial_value in smac.runhistory.items():
        ...



Runhistory Encoder
------------------


Callbacks
---------

Callbacks provide the ability to easily execute code before, inside, and after the Bayesian Optimization loop.
To add a callback, you have to inherit from ``smac.callback.Callback`` and overwrite the abstract methods.
Afterwards, you can pass the callbacks to any facade. 

.. code-block:: python

    from smac import MultiFidelityFacade
    from smac.callback import Callback


    class CustomCallback(Callback):
        def on_start(self, smbo: SMBO) -> None:
            pass

        @abstractmethod
        def on_end(self, smbo: SMBO) -> None:
            pass

        @abstractmethod
        def on_iteration_start(self, smbo: SMBO) -> None:
            pass

        @abstractmethod
        def on_iteration_end(self, smbo: SMBO, info: RunInfo, value: RunValue) -> bool | None:
            # We just do a simple printing here
            print(info, value)


    smac = MultiFidelityFacade(
        ...
        callbacks=[CustomCallback()]
    )
    smac.optimize()