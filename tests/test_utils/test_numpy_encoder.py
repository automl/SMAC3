from __future__ import annotations

import json

import numpy as np
import pytest

from smac.utils.numpyencoder import NumpyEncoder


# Test cases for NumpyEncoder
def test_numpy_encoder():
    data = {
        "int": np.int32(1),
        "float": np.float32(1.23),
        "complex": np.complex64(1 + 2j),
        "array": np.array([1, 2, 3]),
        "bool": np.bool_(True),
        "void": np.void(b"void"),
    }

    expected_output = {
        "int": 1,
        "float": 1.23,
        "complex": {"real": 1.0, "imag": 2.0},
        "array": [1, 2, 3],
        "bool": True,
        "void": None,
    }

    encoded_data = json.dumps(data, cls=NumpyEncoder)
    decoded_data = json.loads(encoded_data)

    assert np.isclose(decoded_data["float"], expected_output["float"])  # float ist not exactly the same
    del decoded_data["float"]
    del expected_output["float"]
    assert decoded_data == expected_output


# Test if default method raises TypeError for unsupported types
def test_numpy_encoder_unsupported_type():
    with pytest.raises(TypeError):
        json.dumps(set([1, 2, 3]), cls=NumpyEncoder)
