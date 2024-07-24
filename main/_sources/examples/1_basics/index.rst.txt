

.. _sphx_glr_examples_1_basics:

Basics
------


.. raw:: html

    <div class="sphx-glr-thumbnails">

.. thumbnail-parent-div-open

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="An example of applying SMAC to optimize a quadratic function.">

.. only:: html

  .. image:: /examples/1_basics/images/thumb/sphx_glr_1_quadratic_function_thumb.png
    :alt:

  :ref:`sphx_glr_examples_1_basics_1_quadratic_function.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Quadratic Function</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="An example of optimizing a simple support vector machine on the IRIS dataset. We use the hyperparameter optimization facade, which uses a random forest as its surrogate model. It is able to scale to higher evaluation budgets and a higher number of dimensions. Also, you can use mixed data types as well as conditional hyperparameters.">

.. only:: html

  .. image:: /examples/1_basics/images/thumb/sphx_glr_2_svm_cv_thumb.png
    :alt:

  :ref:`sphx_glr_examples_1_basics_2_svm_cv.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Support Vector Machine with Cross-Validation</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="This examples show how to use the Ask-and-Tell interface.">

.. only:: html

  .. image:: /examples/1_basics/images/thumb/sphx_glr_3_ask_and_tell_thumb.png
    :alt:

  :ref:`sphx_glr_examples_1_basics_3_ask_and_tell.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Ask-and-Tell</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Using callbacks is the easieast way to integrate custom code inside the Bayesian optimization loop. In this example, we disable SMAC&#x27;s default logging option and use the custom callback to log the evaluated trials. Furthermore, we print some stages of the optimization process.">

.. only:: html

  .. image:: /examples/1_basics/images/thumb/sphx_glr_4_callback_thumb.png
    :alt:

  :ref:`sphx_glr_examples_1_basics_4_callback.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Custom Callback</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="SMAC can also be continued from a previous run. To do so, it reads in old files (derived from scenario&#x27;s name, output_directory and seed) and sets the corresponding components. In this example, an optimization of a simple quadratic function is continued.">

.. only:: html

  .. image:: /examples/1_basics/images/thumb/sphx_glr_5_continue_thumb.png
    :alt:

  :ref:`sphx_glr_examples_1_basics_5_continue.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Continue an Optimization</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Example for optimizing a Multi-Layer Perceptron (MLP) setting priors over the optimum on the hyperparameters. These priors are derived from user knowledge (from previous runs on similar tasks, common knowledge or intuition gained from manual tuning). To create the priors, we make use of the Normal and Beta Hyperparameters, as well as the &quot;weights&quot; property of the CategoricalHyperparameter. This can be integrated into the optimiztion for any SMAC facade, but we stick with the hyperparameter optimization facade here. To incorporate user priors into the optimization, you have to change the acquisition function to PriorAcquisitionFunction.">

.. only:: html

  .. image:: /examples/1_basics/images/thumb/sphx_glr_6_priors_thumb.png
    :alt:

  :ref:`sphx_glr_examples_1_basics_6_priors.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">User Priors over the Optimum</div>
    </div>


.. thumbnail-parent-div-close

.. raw:: html

    </div>


.. toctree::
   :hidden:

   /examples/1_basics/1_quadratic_function
   /examples/1_basics/2_svm_cv
   /examples/1_basics/3_ask_and_tell
   /examples/1_basics/4_callback
   /examples/1_basics/5_continue
   /examples/1_basics/6_priors

