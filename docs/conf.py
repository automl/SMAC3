import automl_sphinx_theme

from smac import copyright, author, version, name
from sphinx_gallery.sorting import FileNameSortKey

# from smac.cli.cmd_reader import CMDReader


options = {
    "copyright": copyright,
    "author": author,
    "version": version,
    "versions": {
        f"v{version}": "#",
        "v2.0.0b1": "https://automl.github.io/SMAC3/v2.0.0b1/",
        "v2.0.0a2": "https://automl.github.io/SMAC3/v2.0.0a2/",
        "v2.0.0a1": "https://automl.github.io/SMAC3/v2.0.0a1/",
        "v1.4.0": "https://automl.github.io/SMAC3/v1.4.0/",
    },
    "name": name,
    "html_theme_options": {
        "github_url": "https://github.com/automl/SMAC3",
        "twitter_url": "https://twitter.com/automl_org?lang=de",
    },
    # "ignore_pattern": ".*pcs$|.*scenario.txt$|.*spear_qcp$",
    "sphinx_gallery_conf": {
        "plot_gallery": True,
        "within_subsection_order": FileNameSortKey,
        "filename_pattern": "/",  # We want to execute all files in `examples`
    },
}

automl_sphinx_theme.set_options(globals(), options)

# Write outputs
# cmd_reader = CMDReader()
# cmd_reader.write_main_options_to_doc()
# cmd_reader.write_smac_options_to_doc()
# cmd_reader.write_scenario_options_to_doc()
