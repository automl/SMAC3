"""A custom Sphinx HTML Translator for Bootstrap layout
"""
from distutils.version import LooseVersion
from docutils import nodes

import sphinx
from sphinx.writers.html5 import HTML5Translator
from sphinx.util import logging
from sphinx.ext.autosummary import autosummary_table

logger = logging.getLogger(__name__)
