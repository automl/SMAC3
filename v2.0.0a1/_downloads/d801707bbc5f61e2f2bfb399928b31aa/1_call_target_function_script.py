"""
Call Target Function From Script
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This simple example shows how to call a script with the following content:

.. code-block:: bash

    #!/bin/bash

    # Set arguments first
    for argument in "$@"
    do
        key=$(echo $argument | cut -f1 -d=)
        value=$(echo $argument | cut -f2 -d=)  

        if [[ $key == *"--"* ]]; then
            v="${key/--/}"
            declare $v="${value}"
        fi
    done

    # We simply set the cost to our parameter
    cost=$x0

    # Return everything
    echo "cost=$cost"

"""

from ConfigSpace import ConfigurationSpace

from smac import BlackBoxFacade, Scenario

__copyright__ = "Copyright 2021, AutoML.org Freiburg-Hannover"
__license__ = "3-clause BSD"


if __name__ == "__main__":
    cs = ConfigurationSpace({"x0": (0, 1000)}, seed=0)

    # Scenario object specifying the optimization "environment"
    scenario = Scenario(cs, deterministic=True, n_trials=30)

    # Now we use SMAC to find the best hyperparameters
    smac = BlackBoxFacade(
        scenario,
        "./script.sh",  # We pass the filename of our script here
        overwrite=True,  # Overrides any previous results that are found that are inconsistent with the meta-data
    )
    incumbent = smac.optimize()

    # Get cost of default configuration
    default_cost = smac.validate(cs.get_default_configuration())
    print(f"Default cost: {default_cost}")

    # Let's calculate the cost of the incumbent
    incumbent_cost = smac.validate(incumbent)
    print(f"Incumbent cost: {incumbent_cost}")
