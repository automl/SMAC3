import datetime
import traceback

name = "SMAC3"
package_name = "smac"
author = (
    "\tMarius Lindauer, Katharina Eggensperger, Matthias Feurer, André Biedenkapp, "
    "Difan Deng,\n\tCarolin Benjamins, Tim Ruhkopf, René Sass and Frank Hutter"
)

author_email = "fh@cs.uni-freiburg.de"
description = "SMAC3, a Python implementation of 'Sequential Model-based Algorithm Configuration'."
url = "https://www.automl.org/"
project_urls = {
    "Documentation": "https://https://github.com/automl.github.io/SMAC3/main",
    "Source Code": "https://github.com/https://github.com/automl/smac",
}
copyright = f"""
    Copyright {datetime.date.today().strftime('%Y')}, Marius Lindauer, Katharina Eggensperger,
    Matthias Feurer, André Biedenkapp, Difan Deng, Carolin Benjamins, Tim Ruhkopf, René Sass
    and Frank Hutter"""
version = "2.0.0a2"


try:
    from smac.callback import Callback
    from smac.facade import (
        AlgorithmConfigurationFacade,
        BlackBoxFacade,
        HyperbandFacade,
        HyperparameterOptimizationFacade,
        MultiFidelityFacade,
        RandomFacade,
    )
    from smac.runhistory.runhistory import RunHistory
    from smac.scenario import Scenario
    from smac.utils.logging import setup_logging

    setup_logging(0)

    __all__ = [
        "Scenario",
        "RunHistory",
        "BlackBoxFacade",
        "HyperparameterOptimizationFacade",
        "MultiFidelityFacade",
        "AlgorithmConfigurationFacade",
        "RandomFacade",
        "HyperbandFacade",
        "Callback",
    ]
except ModuleNotFoundError as e:
    print(e)
    traceback.print_exc()
