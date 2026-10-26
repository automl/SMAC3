import traceback

name = "SMAC3"
package_name = "smac"
author = (
    "\tMarius Lindauer, Katharina Eggensperger, Matthias Feurer, André Biedenkapp, "
    "Difan Deng,\n\tCarolin Benjamins, Tim Ruhkopf, René Sass and Frank Hutter"
)

author_email = "smac@ai.uni-hannover.de"
maintainer = (
    "Marcel Wever, Anthony Angrimson, Lukas Fehring, Michael Gratius, Leona Hennig, "
    "Daphne Theodorakopoulos, René Traoré, Eva Vormschlag and Andreas Bentzen Winje"
)
maintainer_email = "smac@ai.uni-hannover.de"
description = "SMAC3, a Python implementation of 'Sequential Model-based Algorithm Configuration'."
url = "https://www.ai.uni-hannover.de/"
project_urls = {
    "Documentation": "https://automl.github.io/SMAC3/latest",
    "Source Code": "https://github.com/automl/SMAC3",
}
# Single source of truth for the copyright notice. Run `python scripts/update_files.py` after changing it
# to propagate it to all file headers, LICENSE.txt and README.md.
copyright_year = "2026"
copyright_holder = "Leibniz University Hanover, Institute of AI"
copyright = f"Copyright {copyright_year}, {copyright_holder}"
version = "2.4.1"


try:
    from smac.callback.callback import Callback
    from smac.facade import (
        AlgorithmConfigurationFacade,
        BlackBoxFacade,
        HPIFacade,
        HyperbandFacade,
        HyperparameterOptimizationFacade,
        MultiFidelityFacade,
        MultiObjectiveFacade,
        RandomFacade,
    )
    from smac.runhistory.runhistory import RunHistory
    from smac.scenario import Scenario

    __all__ = [
        "Scenario",
        "RunHistory",
        "BlackBoxFacade",
        "HyperparameterOptimizationFacade",
        "MultiFidelityFacade",
        "AlgorithmConfigurationFacade",
        "RandomFacade",
        "HPIFacade",
        "HyperbandFacade",
        "MultiObjectiveFacade",
        "Callback",
    ]
except ModuleNotFoundError as e:
    print(e)
    traceback.print_exc()
