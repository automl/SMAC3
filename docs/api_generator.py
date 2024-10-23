"""Generate the code reference pages and navigation.

# https://mkdocstrings.github.io/recipes/
"""
from __future__ import annotations

import logging
from pathlib import Path

import mkdocs_gen_files

logger = logging.getLogger(__name__)

source_path = "smac"

# Modules whose members should not include inherited attributes or methods
# NOTE: Given the current setup, we can only operate at a module level.
# Ideally we specify options (at least at a module level) and we render
# them into strings using a yaml parser. For now this is fine though
NO_INHERITS = ("sklearn.evaluation",)
TAB = "    "

for path in sorted(Path(source_path).rglob("*.py")):
    module_path = path.relative_to(source_path).with_suffix("")
    doc_path = path.relative_to(source_path).with_suffix(".md")
    full_doc_path = Path("api", doc_path)

    parts = tuple(module_path.parts)

    if parts[-1] in ("__main__", "__version__", "__init__"):
        continue

    if any(part.startswith("_") for part in parts):
        continue

    with mkdocs_gen_files.open(full_doc_path, "w") as fd:
        ident = ".".join(parts)
        fd.write(f"::: {ident}")

        if ident.endswith(NO_INHERITS):
            fd.write(f"\n{TAB}options:")
            fd.write(f"\n{TAB}{TAB}inherited_members: false")

    mkdocs_gen_files.set_edit_path(full_doc_path, path)