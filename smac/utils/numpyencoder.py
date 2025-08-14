from __future__ import annotations

from typing import Any

import json

import numpy as np


class NumpyEncoder(json.JSONEncoder):
    """Custom encoder for numpy data types

    From https://stackoverflow.com/a/61903895
    """

    def default(self, obj: Any) -> Any:
        """Handle numpy datatypes if present by converting to native python

        Parameters
        ----------
        obj : Any
            Object to serialize

        Returns
        -------
        Any
            Object in native python
        """
        if isinstance(
            obj,
            (
                np.int_,
                np.intc,
                np.intp,
                np.int8,
                np.int16,
                np.int32,
                np.int64,
                np.uint8,
                np.uint16,
                np.uint32,
                np.uint64,
            ),
        ):
            return int(obj)

        elif isinstance(obj, (np.float16, np.float32, np.float64)):
            return float(obj)

        elif isinstance(obj, (np.complex64, np.complex128)):
            return {"real": obj.real, "imag": obj.imag}

        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()

        elif isinstance(obj, (np.bool_)):
            return bool(obj)

        elif isinstance(obj, (np.void)):
            return None

        return json.JSONEncoder.default(self, obj)
