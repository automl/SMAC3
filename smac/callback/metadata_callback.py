from __future__ import annotations

import json
import platform
from datetime import datetime

import smac
from smac.callback.callback import Callback
from smac.main.smbo import SMBO

__copyright__ = "Copyright 2023, AutoML.org Freiburg-Hannover"
__license__ = "3-clause BSD"


class MetadataCallback(Callback):
    def __init__(self, **kwargs: str | int | float | dict | list) -> None:
        # Arguments must be json serializable
        self.kwargs = kwargs

    def on_start(self, smbo: SMBO) -> None:
        """Called before the optimization starts."""
        path = smbo._scenario.output_directory
        meta_dict = {
            "utc_time": str(datetime.utcnow()),
            "os": platform.platform(),
            "smac_version": getattr(smac, "version"),
        }
        for key, value in self.kwargs.items():
            meta_dict[key] = value

        path.mkdir(parents=True, exist_ok=True)

        with open(path / "metadata.json", "w") as fp:
            json.dump(meta_dict, fp, indent=2)
