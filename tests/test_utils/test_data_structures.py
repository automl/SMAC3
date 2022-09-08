from smac.utils.data_structures import recursively_compare_dicts


def test_compare_dicts():
    A = {
        "a": 1,
        "b": {
            "c": 2,
        },
    }

    B = {
        "a": 2,
        "b": {
            "c": 2,
        },
    }

    C = {
        "a": 1,
        "b": {
            "c": 5,
        },
    }

    diff = recursively_compare_dicts(A, A)
    assert len(diff) == 0

    diff = recursively_compare_dicts(A, C)
    assert diff == ["root.b.c: 2 != 5"]

    diff = recursively_compare_dicts(A, B)
    assert diff == ["root.a: 1 != 2"]

    diff = recursively_compare_dicts(B, C)
    assert "root.b.c: 2 != 5" in diff and "root.a: 2 != 1" in diff


def test_different_keys():
    A = {"a": 1}
    B = {"b": 2}

    diff = recursively_compare_dicts(A, B)
    assert diff == ["root + {'a'} - {'b'}"]


def test_different_lists():
    A = {"a": [1, 2, 3]}
    B = {"a": [1, 2, 4]}

    diff = recursively_compare_dicts(A, B)
    assert diff == ["root.a[2]: 3 != 4"]
