

.. _sphx_glr_examples_2_multi_fidelity:

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


.. toctree::
   :hidden:

   /examples/2_multi_fidelity/1_mlp_epochs
   /examples/2_multi_fidelity/2_sgd_datasets
   /examples/2_multi_fidelity/3_specify_HB_via_total_budget

