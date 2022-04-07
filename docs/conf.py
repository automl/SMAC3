import automl_sphinx_theme

from smac import copyright, author, version, name
from smac.utils.io.cmd_reader import CMDReader


options = {
    "copyright": copyright,
    "author": author,
    "version": version,
    "name": name,
    "html_theme_options": {
        "github_url": "https://github.com/https://github.com/automl/SMAC3",
        "twitter_url": "https://twitter.com/automl_org?lang=de",
    },
    "ignore_pattern": ".*pcs$|.*scenario.txt$|.*spear_qcp$",
    "sphinx_gallery_conf": {
        "plot_gallery": True,
    },
}

automl_sphinx_theme.set_options(globals(), options)

# Write outputs
cmd_reader = CMDReader()
cmd_reader.write_main_options_to_doc()
cmd_reader.write_smac_options_to_doc()
cmd_reader.write_scenario_options_to_doc()
