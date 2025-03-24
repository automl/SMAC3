"""This module is a hook that when any code is being rendered, it will
print the path to the file being rendered.

This makes it easier to identify which file is being rendered when an error happens.
"""
from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Any

import mkdocs
import mkdocs.plugins

if TYPE_CHECKING:
    import mkdocs.structure.pages

log = logging.getLogger("mkdocs")

RENDER_EXAMPLES_ENV_VAR = "SMAC_DOC_RENDER_EXAMPLES"
EXEC_DOCS_ENV_VAR = "SMAC_EXEC_DOCS"

truthy_values = {"yes", "on", "true", "1", "all"}


def on_pre_page(
    page: mkdocs.structure.pages.Page,
    config: Any,
    files: Any,
) -> mkdocs.structure.pages.Page | None:
    render_examples = os.environ.get(RENDER_EXAMPLES_ENV_VAR, "true")
    render_code = os.environ.get(EXEC_DOCS_ENV_VAR, "true")
    if render_examples.lower() in truthy_values or render_code.lower() in truthy_values:
        log.info(f"{page.file.src_path}")
