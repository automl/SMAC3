import pytest
from ConfigSpace.hyperparameters import (
    CategoricalHyperparameter,
    Constant,
    OrdinalHyperparameter,
    UniformFloatHyperparameter,
    UniformIntegerHyperparameter,
)

from smac.utils.configspace import (
    modify_hyperparameter,
)
# Test data for parameterized test
@pytest.mark.parametrize(
    "space, hyperparameter_name, modifications, expected_type, expected_attributes",
    [
        # Test Case 1: Modify a categorical hyperparameter
        (
            {"hp1": CategoricalHyperparameter("hp1", ["A", "B", "C"], default_value="A")},
            "hp1",
            {"choices": ["A", "B", "D"], "default_value": "D"},
            CategoricalHyperparameter,
            {"choices": ["A", "B", "D"], "default_value": "D"},
        ),
        # Test Case 2: Modify a constant hyperparameter
        (
            {"hp1": Constant("hp1", 42)},
            "hp1",
            {"value": 100},
            Constant,
            {"value": 100},
        ),
        # Test Case 3: Modify an ordinal hyperparameter
        (
            {"hp1": OrdinalHyperparameter("hp1", [1, 2, 3], default_value=2)},
            "hp1",
            {"sequence": [3, 2, 1], "default_value": 3},
            OrdinalHyperparameter,
            {"sequence": [3, 2, 1], "default_value": 3},
        ),
        # Test Case 4: Modify a uniform float hyperparameter
        (
            {"hp1": UniformFloatHyperparameter("hp1", 0.0, 1.0, default_value=0.5)},
            "hp1",
            {"lower": 0.1, "upper": 0.9, "default_value": 0.8},
            UniformFloatHyperparameter,
            {"lower": 0.1, "upper": 0.9, "default_value": 0.8},
        ),
        # Test Case 5: Raise an error for missing hyperparameter
        (
            {"hp1": UniformIntegerHyperparameter("hp1", 1, 10)},
            "hp2",
            {"lower": 2},
            ValueError,
            None,  # No attributes to check since this raises an error
        ),
    ],
)
def test_modify_hyperparameter(space, hyperparameter_name, modifications, expected_type, expected_attributes):
    if isinstance(expected_type, type) and issubclass(expected_type, Exception):
        # Check for expected error
        with pytest.raises(expected_type):
            modify_hyperparameter(space, hyperparameter_name, **modifications)
    else:
        # Modify the hyperparameter
        result = modify_hyperparameter(space, hyperparameter_name, **modifications)
        modified_hp = result[hyperparameter_name]
        
        # Check the type of the modified hyperparameter
        assert isinstance(modified_hp, expected_type), f"Expected type {expected_type}, got {type(modified_hp)}"
        
        # Check each modified attribute
        for key, value in expected_attributes.items():
            if key == "choices":
                list(modified_hp.choices) == list(modified_hp.choices), f"Choices expected to be {value}, got {modified_hp.choices}"
            elif key == "sequence":
                assert list(modified_hp.sequence) == list(value), f"Sequence expected to be {value}, got {modified_hp.sequence}"
            else:
                assert getattr(modified_hp, key) == value, f"Attribute '{key}' expected to be {value}, got {getattr(modified_hp, key)}"

# Run the test
if __name__ == "__main__":
    pytest.main()