Ask-and-Tell Interface
======================


.. note ::

    The initial design is already part of the ask and tell.


.. warning ::

    Using ask+tell instead of optimize can result in different results. Because
    - callbacks are ignored
    - skipped trials are handled differently
    - 

.. warning :: 

    Calling multiple asks in a row is not yet supported.


Calling Tell without Ask
------------------------

Does only work with the intensifier.
It does not make sense to tell SMAC trials in advance when using SH. Reason: It's heavily depending on a budget+instance combination and even if the user provides it, SMAC have to wait till the other trials have been finished too.


