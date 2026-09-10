:orphan:

Examples
========

We provide several examples of how to use SMAC with Python. Practical use-cases were chosen to show the
variety of SMAC.


.. raw:: html

    <div class="sphx-glr-thumbnails">

.. thumbnail-parent-div-open

.. thumbnail-parent-div-close

.. raw:: html

    </div>

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

Multi-Fidelity and Multi-Instances
----------------------------------




.. raw:: html

    <div class="sphx-glr-thumbnails">

.. thumbnail-parent-div-open

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Example for optimizing a Multi-Layer Perceptron (MLP) using multiple budgets. Since we want to take advantage of multi-fidelity, the MultiFidelityFacade is a good choice. By default, MultiFidelityFacade internally runs with hyperband as intensification, which is a combination of an aggressive racing mechanism and Successive Halving. Crucially, the target  function must accept a budget variable, detailing how much fidelity smac wants to allocate to this configuration. In this example, we use both SuccessiveHalving and Hyperband to compare the results.">

.. only:: html

  .. image:: /examples/2_multi_fidelity/images/thumb/sphx_glr_1_mlp_epochs_thumb.png
    :alt:

  :ref:`sphx_glr_examples_2_multi_fidelity_1_mlp_epochs.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Multi-Layer Perceptron Using Multiple Epochs</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Example for optimizing a Multi-Layer Perceptron (MLP) across multiple (dataset) instances.">

.. only:: html

  .. image:: /examples/2_multi_fidelity/images/thumb/sphx_glr_2_sgd_datasets_thumb.png
    :alt:

  :ref:`sphx_glr_examples_2_multi_fidelity_2_sgd_datasets.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Stochastic Gradient Descent On Multiple Datasets</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="In Hyperband, normally SMAC calculates a typical Hyperband round. If the number of trials is not used up by one single round, the next round is started. Instead of specifying the number of trial beforehand, specify the total budget in terms of the fidelity units and let SMAC calculate how many trials that would be.">

.. only:: html

  .. image:: /examples/2_multi_fidelity/images/thumb/sphx_glr_3_specify_HB_via_total_budget_thumb.png
    :alt:

  :ref:`sphx_glr_examples_2_multi_fidelity_3_specify_HB_via_total_budget.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Specify Number of Trials via a Total Budget in Hyperband</div>
    </div>


.. thumbnail-parent-div-close

.. raw:: html

    </div>

Multi-Objective
---------------




.. raw:: html

    <div class="sphx-glr-thumbnails">

.. thumbnail-parent-div-open

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="A simple example on how to use multi-objective optimization is shown. The 2D Schaffer function is used. In the plot you can see that all points are on the Pareto front. However, since we set the objective weights, you can notice that SMAC prioritizes the second objective over the first one.">

.. only:: html

  .. image:: /examples/3_multi_objective/images/thumb/sphx_glr_1_schaffer_thumb.png
    :alt:

  :ref:`sphx_glr_examples_3_multi_objective_1_schaffer.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">2D Schaffer Function with Objective Weights</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="An example of how to use multi-objective optimization with ParEGO. Both accuracy and run-time are going to be optimized on the digits dataset using an MLP, and the configurations are shown in a plot, highlighting the best ones in  a Pareto front. The red cross indicates the best configuration selected by SMAC.">

.. only:: html

  .. image:: /examples/3_multi_objective/images/thumb/sphx_glr_2_parego_thumb.png
    :alt:

  :ref:`sphx_glr_examples_3_multi_objective_2_parego.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">ParEGO</div>
    </div>


.. thumbnail-parent-div-close

.. raw:: html

    </div>

Command-Line Interface
----------------------

SMAC can call a target function from a script. This is useful if you want to optimize non-python code.


.. raw:: html

    <div class="sphx-glr-thumbnails">

.. thumbnail-parent-div-open

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="This simple example shows how to call a script with the following content:">

.. only:: html

  .. image:: /examples/5_commandline/images/thumb/sphx_glr_1_call_target_function_script_thumb.png
    :alt:

  :ref:`sphx_glr_examples_5_commandline_1_call_target_function_script.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Call Target Function From Script</div>
    </div>


.. thumbnail-parent-div-close

.. raw:: html

    </div>


.. toctree::
   :hidden:
   :includehidden:


   /examples/1_basics/index.rst
   /examples/2_multi_fidelity/index.rst
   /examples/3_multi_objective/index.rst
   /examples/5_commandline/index.rst


.. only:: html

  .. container:: sphx-glr-footer sphx-glr-footer-gallery

    .. container:: sphx-glr-download sphx-glr-download-python

      :download:`Download all examples in Python source code: examples_python.zip </examples/examples_python.zip>`

    .. container:: sphx-glr-download sphx-glr-download-jupyter

      :download:`Download all examples in Jupyter notebooks: examples_jupyter.zip </examples/examples_jupyter.zip>`
