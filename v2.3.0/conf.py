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
        "v2.3.0": "https://automl.github.io/SMAC3/v2.3.0/",
        "v2.2.0": "https://automl.github.io/SMAC3/v2.2.0/",
        "v2.1.0": "https://automl.github.io/SMAC3/v2.1.0/",
        "v2.0.1": "https://automl.github.io/SMAC3/v2.0.1/",
        "v2.0.0": "https://automl.github.io/SMAC3/v2.0.0/",
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
        "binder": {
            # Required keys
            "org": "automl",
            "repo": "SMAC3",
            "branch": "main",
            "binderhub_url": "https://mybinder.org",
            "dependencies": ["../.binder/apt.txt", "../.binder/requirements.txt"],
            # "filepath_prefix": "<prefix>" # A prefix to prepend to any filepaths in Binder links.
            # Jupyter notebooks for Binder will be copied to this directory (relative to built documentation root).
            "notebooks_dir": "notebooks/",
            "use_jupyter_lab": True,
            # Whether Binder links should start Jupyter Lab instead of the Jupyter Notebook interface.
        },
        "ignore_pattern": ".*7_parallelization_cluster.py$",
    },
}

automl_sphinx_theme.set_options(globals(), options)

# Write outputs
# cmd_reader = CMDReader()
# cmd_reader.write_main_options_to_doc()
# cmd_reader.write_smac_options_to_doc()
# cmd_reader.write_scenario_options_to_doc()
