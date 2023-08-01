from __future__ import annotations

import hashlib
import logging
from functools import partial

import numpy as np
from ConfigSpace import Configuration, ConfigurationSpace
from ConfigSpace.hyperparameters import (
    BetaFloatHyperparameter,
    BetaIntegerHyperparameter,
    CategoricalHyperparameter,
    Constant,
    NormalFloatHyperparameter,
    NormalIntegerHyperparameter,
    OrdinalHyperparameter,
    UniformFloatHyperparameter,
    UniformIntegerHyperparameter,
)
from ConfigSpace.util import get_one_exchange_neighbourhood

__copyright__ = "Copyright 2022, automl.org"
__license__ = "3-clause BSD"


get_one_exchange_neighbourhood = partial(get_one_exchange_neighbourhood, stdev=0.05, num_neighbors=8)


def convert_configurations_to_array(configs: list[Configuration]) -> np.ndarray:
    """Impute inactive hyperparameters in configurations with their default.

    Parameters
    ----------
    configs : List[Configuration]
        List of configuration objects.

    Returns
    -------
    np.ndarray
    """
    return np.array([config.get_array() for config in configs], dtype=np.float64)


def get_types(
    configspace: ConfigurationSpace,
    instance_features: dict[str, list[float]] | None = None,
) -> tuple[list[int], list[tuple[float, float]]]:
    """Return the types of the hyperparameters and the bounds of the
    hyperparameters and instance features.

    Warning
    -------
    The bounds for the instance features are *not* added in this function.
    """
    # Extract types vector for rf from config space and the bounds
    types = [0] * len(configspace.get_hyperparameters())
    bounds = [(np.nan, np.nan)] * len(types)

    for i, param in enumerate(configspace.get_hyperparameters()):
        parents = configspace.get_parents_of(param.name)
        if len(parents) == 0:
            can_be_inactive = False
        else:
            can_be_inactive = True

        if isinstance(param, (CategoricalHyperparameter)):
            n_cats = len(param.choices)
            if can_be_inactive:
                n_cats = len(param.choices) + 1
            types[i] = n_cats
            bounds[i] = (int(n_cats), np.nan)
        elif isinstance(param, (OrdinalHyperparameter)):
            n_cats = len(param.sequence)
            types[i] = 0
            if can_be_inactive:
                bounds[i] = (0, int(n_cats))
            else:
                bounds[i] = (0, int(n_cats) - 1)
        elif isinstance(param, Constant):
            # For constants we simply set types to 0 which makes it a numerical parameter
            if can_be_inactive:
                bounds[i] = (2, np.nan)
                types[i] = 2
            else:
                bounds[i] = (0, np.nan)
                types[i] = 0
            # and we leave the bounds to be 0 for now
        elif isinstance(param, UniformFloatHyperparameter):
            # Are sampled on the unit hypercube thus the bounds
            # are always 0.0, 1.0
            if can_be_inactive:
                bounds[i] = (-1.0, 1.0)
            else:
                bounds[i] = (0, 1.0)
        elif isinstance(param, UniformIntegerHyperparameter):
            if can_be_inactive:
                bounds[i] = (-1.0, 1.0)
            else:
                bounds[i] = (0, 1.0)
        elif isinstance(param, NormalFloatHyperparameter):
            if can_be_inactive:
                raise ValueError("Inactive parameters not supported for Beta and Normal Hyperparameters")

            bounds[i] = (param._lower, param._upper)
        elif isinstance(param, NormalIntegerHyperparameter):
            if can_be_inactive:
                raise ValueError("Inactive parameters not supported for Beta and Normal Hyperparameters")

            bounds[i] = (param.nfhp._lower, param.nfhp._upper)
        elif isinstance(param, BetaFloatHyperparameter):
            if can_be_inactive:
                raise ValueError("Inactive parameters not supported for Beta and Normal Hyperparameters")

            bounds[i] = (param._lower, param._upper)
        elif isinstance(param, BetaIntegerHyperparameter):
            if can_be_inactive:
                raise ValueError("Inactive parameters not supported for Beta and Normal Hyperparameters")

            bounds[i] = (param.bfhp._lower, param.bfhp._upper)
        elif not isinstance(
            param,
            (
                UniformFloatHyperparameter,
                UniformIntegerHyperparameter,
                OrdinalHyperparameter,
                CategoricalHyperparameter,
                NormalFloatHyperparameter,
                NormalIntegerHyperparameter,
                BetaFloatHyperparameter,
                BetaIntegerHyperparameter,
            ),
        ):
            raise TypeError("Unknown hyperparameter type %s" % type(param))

    if instance_features is not None:
        n_features = len(list(instance_features.values())[0])
        types = types + [0] * n_features

    return types, bounds


def get_conditional_hyperparameters(X: np.ndarray, Y: np.ndarray | None = None) -> np.ndarray:
    """Returns conditional hyperparameters if values with -1 or smaller are observed. X is used
    if Y is not specified.
    """
    # Taking care of conditional hyperparameters according to Levesque et al.
    X_cond = X <= -1

    if Y is not None:
        Y_cond = Y <= -1
    else:
        Y_cond = X <= -1

    active = ~((np.expand_dims(X_cond, axis=1) != Y_cond).any(axis=2))
    return active


def get_config_hash(config: Configuration, chars: int = 6) -> str:
    """Returns a hash of the configuration."""
    return hashlib.sha1(str(config).encode("utf-8")).hexdigest()[:chars]


def print_config_changes(
    incumbent: Configuration | None,
    challenger: Configuration | None,
    logger: logging.Logger,
) -> None:
    """Compares two configurations and prints the differences."""
    if incumbent is None or challenger is None:
        return

    inc_keys = set(incumbent.keys())
    all_keys = inc_keys.union(challenger.keys())

    lines = []
    for k in sorted(all_keys):
        inc_k = incumbent.get(k, "-inactive-")
        cha_k = challenger.get(k, "-inactive-")
        lines.append(f"--- {k}: {inc_k} -> {cha_k}" + " (unchanged)" if inc_k == cha_k else "")

    msg = "\n".join(lines)
    logger.debug(msg)


# def check_subspace_points(
#     X: np.ndarray,
#     cont_dims: np.ndarray | list = [],
#     cat_dims: np.ndarray | list = [],
#     bounds_cont: np.ndarray | None = None,
#     bounds_cat: list[tuple] | None = None,
#     expand_bound: bool = False,
# ) -> np.ndarray:
#     """Check which points are place inside a given subspace.

#     Parameters
#     ----------
#     X: Optional[np.ndarray(N,D)],
#         points to be checked, where D = D_cont + D_cat
#     cont_dims: Union[np.ndarray(D_cont), List]
#         which dimensions represent continuous hyperparameters
#     cat_dims: Union[np.ndarray(D_cat), List]
#         which dimensions represent categorical hyperparameters
#     bounds_cont: optional[List[Tuple]]
#         subspaces bounds of categorical hyperparameters, its length is the number of continuous hyperparameters
#     bounds_cat: Optional[List[Tuple]]
#         subspaces bounds of continuous hyperparameters, its length is the number of categorical hyperparameters
#     expand_bound: bool
#         if the bound needs to be expanded to contain more points rather than the points inside the subregion
#     Return
#     ----------
#     indices_in_ss:np.ndarray(N)
#         indices of data that included in subspaces
#     """
#     if len(X.shape) == 1:
#         X = X[np.newaxis, :]
#     if len(cont_dims) == 0 and len(cat_dims) == 0:
#         return np.ones(X.shape[0], dtype=bool)

#     if len(cont_dims) > 0:
#         if bounds_cont is None:
#             raise ValueError("bounds_cont must be given if cont_dims provided")

#         if len(bounds_cont.shape) != 2 or bounds_cont.shape[1] != 2 or bounds_cont.shape[0] != len(cont_dims):
#             raise ValueError(
#                 f"bounds_cont (with shape  {bounds_cont.shape}) should be an array with shape of"
#                 f"({len(cont_dims)}, 2)"
#             )

#         data_in_ss = np.all(X[:, cont_dims] <= bounds_cont[:, 1], axis=1) & np.all(
#             X[:, cont_dims] >= bounds_cont[:, 0], axis=1
#         )

#         if expand_bound:
#             bound_left = bounds_cont[:, 0] - np.min(X[data_in_ss][:, cont_dims] - bounds_cont[:, 0], axis=0)
#             bound_right = bounds_cont[:, 1] + np.min(bounds_cont[:, 1] - X[data_in_ss][:, cont_dims], axis=0)
#             data_in_ss = np.all(X[:, cont_dims] <= bound_right, axis=1) & np.all(X[:, cont_dims] >= bound_left,
# axis=1)
#     else:
#         data_in_ss = np.ones(X.shape[0], dtype=bool)

#     if len(cat_dims) == 0:
#         return data_in_ss
#     if bounds_cat is None:
#         raise ValueError("bounds_cat must be given if cat_dims provided")

#     if len(bounds_cat) != len(cat_dims):
#         raise ValueError(
#             f"bounds_cat ({len(bounds_cat)}) and cat_dims ({len(cat_dims)}) must have " f"the same number of elements"
#         )

#     for bound_cat, cat_dim in zip(bounds_cat, cat_dims):
#         data_in_ss &= np.in1d(X[:, cat_dim], bound_cat)

#     return data_in_ss
