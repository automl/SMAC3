Installation
============

.. role:: bash(code)
    :language: bash

Requirements
------------
SMAC3 requires:

* `NumPy <https://pypi.python.org/pypi/numpy/1.6.1>`_ (Version >= 1.6.1)
* `SciPy <https://pypi.python.org/pypi/scipy/0.15.1>`_ (Version >= 0.13.1)
* `pynisher <https://pypi.python.org/pypi/pynisher/0.4.1>`_ (Version >= 0.4.1)
* `ConfigSpace <https://pypi.python.org/pypi/ConfigSpace/0.2.0>`_ (Version >= 0.2.0)
* `setuptools <https://pypi.python.org/pypi/setuptools>`_
* `six <https://pypi.python.org/pypi/six>`_
* `Cython <https://pypi.python.org/pypi/Cython/>`_
* `pyrfr <https://pypi.python.org/pypi/pyrfr/0.2.0>`_

.. _installation:

Manual Installation
-------------------
| To install EPM from command line type the following commands in the terminal:
.. code-block:: bash

    cat requirements.txt | xargs -n 1 -L 1 pip install
    python setup.py install