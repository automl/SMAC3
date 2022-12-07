from __future__ import annotations

from typing import Iterable

from smac.utils.logging import get_logger

logger = get_logger(__name__)


def recursively_compare_dicts(
    d1: dict,
    d2: dict,
    *,
    level: str = "root",
    diff: list[str] | None = None,
) -> list[str]:
    """Compares dictionaries recursively. Returns a list of differences in string format.

    Parameters
    ----------
    d1 : dict
        First dictionary.
    d2 : dict
        Second dictionary.
    level : str, defaults to "root"
        How the first level is called.
    diff : list[str] | None, defaults to None
        Used for recursion.

    Returns
    -------
    list[str]
        List of differences in string format.
    """
    if diff is None:
        diff = []

    if isinstance(d1, dict) and isinstance(d2, dict):
        if d1.keys() != d2.keys():
            s1 = set(d1.keys())
            s2 = set(d2.keys())
            # logger.info("{:<20} + {} - {}".format(level, s1 - s2, s2 - s1))
            # logger.info("{} - {}".format(s1 - s2, s2 - s1))
            diff += [f"{level} + {s1 - s2} - {s2 - s1}"]
            common_keys = s1 & s2
        else:
            common_keys = set(d1.keys())

        for k in common_keys:
            recursively_compare_dicts(d1[k], d2[k], level="{}.{}".format(level, k), diff=diff)

    elif isinstance(d1, list) and isinstance(d2, list):
        if len(d1) != len(d2):
            diff += [f"{level}: len1={len(d1)}; len2={len(d2)}"]
            # logger.info("{:<20} len1={}; len2={}".format(level, len(d1), len(d2)))
            # logger.info("len1={}; len2={}".format(len(d1), len(d2)))
        common_len = min(len(d1), len(d2))

        for i in range(common_len):
            recursively_compare_dicts(d1[i], d2[i], level="{}[{}]".format(level, i), diff=diff)

    else:
        if d1 != d2:
            diff += [f"{level}: {d1} != {d2}"]
            # logger.info("{:<20} {} != {}".format(level, d1, d2))
            # logger.info("len1={}; len2={}".format(len(d1), len(d2)))

    return diff


def batch(iterable: list, n: int = 1) -> Iterable[list]:
    """Batches an iterable into chunks of size n."""
    length = len(iterable)
    for ndx in range(0, length, n):
        yield iterable[ndx : min(ndx + n, length)]
