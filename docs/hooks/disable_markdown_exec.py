"""This disable markdown_exec based on an environment variable.
This speeds up the build of the docs for faster iteration.

This is done by overwriting the module responsible for compiling and executing the code
by overriding the `exec(...)` global variable that is used to run the code.
We hijack it and print a helpful message about how to run the code cell instead.

https://github.com/pawamoy/markdown-exec/blob/adff40b2928dbb2d22f27684e085f02d39a07291/src/markdown_exec/formatters/python.py#L42-L70
"""
from __future__ import annotations

import logging
import os
from typing import Any

import mkdocs
import mkdocs.plugins
import mkdocs.structure.pages

RUN_CODE_BLOCKS_ENV_VAR = "SMAC_EXEC_DOCS"

logger = logging.getLogger("mkdocs")


def _print_msg(compiled_code: Any, code_block_id: int, exec_globals: dict) -> None:
    _print = exec_globals["print"]
    _print(
        f"Env variable {RUN_CODE_BLOCKS_ENV_VAR}=0 - No code to display."
        "\nUse `just docs-code` (or `just docs-full` for examples) to run"
        " the code block and display output."
    )

truthy_values = {"yes", "on", "true", "1"}

@mkdocs.plugins.event_priority(100)
def on_startup(**kwargs: Any):
    run_code_blocks = os.environ.get(RUN_CODE_BLOCKS_ENV_VAR, "true")
    if run_code_blocks.lower() not in truthy_values:
        logger.warning(
            f"Disabling markdown-exec due to {RUN_CODE_BLOCKS_ENV_VAR}={run_code_blocks}"
            "\n.Use `just docs-full` to run and render examples.",
        )
        from markdown_exec.formatters import python

        setattr(python, "exec_python", _print_msg)
