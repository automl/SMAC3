import os
import sys
import datetime


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
    and Frank Hutter
"""
version = "1.3.2"


if "setup.py" not in sys.argv[0]:
    import smac
    from smac.utils import dependencies

    with open(os.path.join(smac.__path__[0], "requirements.txt")) as fh:
        dependencies.verify_packages(fh.read())

    if sys.version_info < (3, 7, 0):
        raise ValueError("SMAC requires Python 3.7.0 or newer.")

    if os.name != "posix":
        print(
            f"Detected unsupported operating system: {sys.platform}."
            "Please be aware, that SMAC might not run on this system."
        )
