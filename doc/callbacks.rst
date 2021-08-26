Callbacks
---------

Callbacks allow customizing the behavior of SMAC to ones needs. Currently, the list of implemented callbacks is
very limited, but they can easily be added.

If you want to create a new callback, please check `smac.callbacks` and create a new pull request.


IncorporateRunResultCallback 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Callback to react on a new run result.

Called after the finished run is added to the runhistory.
Optionally return `False` to (gracefully) stop the optimization.



