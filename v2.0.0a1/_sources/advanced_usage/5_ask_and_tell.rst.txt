Ask-and-Tell Interface
======================

SMAC provides an ask-and-tell interface in v2.0, giving the user the opportunity to ask for the next trial 
and report the results of the trial. However, the ask-and-tell interfaces comes in v2.0 with some 
limitations which you should be aware of.

.. note ::

    The initial design is already part of the ask and tell. Therefore, if you ask for the first trials, 
    you will receive configurations from the initial design.


.. warning ::

    Using ask-and-tell instead of the optimize method might result in different results because 
    some callbacks are ignored and skipped trials are handled differently. In fact, skipped trials
    are ignored completely and can results in being stuck when only skipped trials are found.

Please have a look at our :ref:`ask-and-tell example<Ask-and-Tell>`.


Calling Tell without Ask
------------------------

Sometimes you want to report pre-evaluated trials to the optimization. You can realize this by calling the
``tell`` method without calling ``ask`` before. But be aware that this *only* works with the ``Intensifier`` and not
with ``Successive Halving`` or ``Hyperband``. The ``Intensifier`` checks the run history and detects pre-evaluated 
trials and incorporates it into the optimization. Since ``Successive Halving`` and ``Hyperband`` are more 
complicated (needs specific budgets and pre-defined number of configurations in each stage), it is not supported yet.

.. warning ::

    Calling ``tell`` without ``ask`` does only work for specific intensifiers.


Calling Multiple Tells
----------------------

Calling multiple times ``tell`` before ``ask`` is not supported yet.  

