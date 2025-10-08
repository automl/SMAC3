from __future__ import annotations


def normalize_costs(
    values: list[float],
    bounds: list[tuple[float, float]] | None = None,
) -> list[float]:
    """
    Normalizes a list of floats with corresponding bounds.

    Parameters
    ----------
    values : list[float]
        List of costs to be normalized.
    bounds : list[tuple[float, float]] | None, optional, defaults to None
        List of tuple of bounds. If no bounds are passed, the input is returned.

    Returns
    -------
    normalized_costs : list[float]
        Normalized costs based on the bounds. If no bounds are given, the original values are returned.
        Also, if min and max bounds are the same, the value of the corresponding objective is set to 1.
    """
    if bounds is None:
        return values

    if len(values) != len(bounds):
        raise ValueError("Number of values and bounds must be equal.")

    costs = []
    for v, b in zip(values, bounds):
        assert not isinstance(v, list)
        p = v - b[0]
        q = b[1] - b[0]

        if q < 1e-10:
            cost = 1.0
        else:
            cost = p / q
        costs.append(cost)

    return costs
