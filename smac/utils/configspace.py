from __future__ import annotations

from typing import Iterator

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
    Hyperparameter,
    IntegerHyperparameter,
    NormalFloatHyperparameter,
    NormalIntegerHyperparameter,
    NumericalHyperparameter,
    OrdinalHyperparameter,
    UniformFloatHyperparameter,
    UniformIntegerHyperparameter,
)
from ConfigSpace.types import f64
from ConfigSpace.util import (
    ForbiddenValueError,
    deactivate_inactive_hyperparameters,
    get_one_exchange_neighbourhood,
)

__copyright__ = "Copyright 2025, Leibniz University Hanover, Institute of AI"
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
    types = [0] * len(list(configspace.values()))
    bounds = [(np.nan, np.nan)] * len(types)

    for i, param in enumerate(list(configspace.values())):
        parents = configspace.parents_of[param.name]
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

            bounds[i] = (param.lower_vectorized, param.upper_vectorized)
        elif isinstance(param, NormalIntegerHyperparameter):
            if can_be_inactive:
                raise ValueError("Inactive parameters not supported for Beta and Normal Hyperparameters")

            bounds[i] = (param.lower_vectorized, param.upper_vectorized)
        elif isinstance(param, BetaFloatHyperparameter):
            if can_be_inactive:
                raise ValueError("Inactive parameters not supported for Beta and Normal Hyperparameters")

            bounds[i] = (param.lower_vectorized, param.upper_vectorized)
        elif isinstance(param, BetaIntegerHyperparameter):
            if can_be_inactive:
                raise ValueError("Inactive parameters not supported for Beta and Normal Hyperparameters")

            bounds[i] = (param.lower_vectorized, param.upper_vectorized)
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


def transform_continuous_designs(
    design: np.ndarray, origin: str, configspace: ConfigurationSpace
) -> list[Configuration]:
    """Transforms the continuous designs into a discrete list of configurations.

    Parameters
    ----------
    design : np.ndarray
        Array of hyperparameters originating from the initial design strategy.
    origin : str | None, defaults to None
        Label for a configuration where it originated from.
    configspace : ConfigurationSpace

    Returns
    -------
    configs : list[Configuration]
        Continuous transformed configs.
    """
    params = configspace.get_hyperparameters()
    for idx, param in enumerate(params):
        if isinstance(param, IntegerHyperparameter):
            design[:, idx] = param._inverse_transform(param._transform(design[:, idx]))
        elif isinstance(param, NumericalHyperparameter):
            continue
        elif isinstance(param, Constant):
            design_ = np.zeros(np.array(design.shape) + np.array((0, 1)))
            design_[:, :idx] = design[:, :idx]
            design_[:, idx + 1 :] = design[:, idx:]
            design = design_
        elif isinstance(param, CategoricalHyperparameter):
            v_design = design[:, idx]
            v_design[v_design == 1] = 1 - 10**-10
            design[:, idx] = np.array(v_design * len(param.choices), dtype=int)
        elif isinstance(param, OrdinalHyperparameter):
            v_design = design[:, idx]
            v_design[v_design == 1] = 1 - 10**-10
            design[:, idx] = np.array(v_design * len(param.sequence), dtype=int)
        else:
            raise ValueError("Hyperparameter not supported when transforming a continuous design.")

    configs = []
    for vector in design:
        try:
            conf = deactivate_inactive_hyperparameters(
                configuration=None, configuration_space=configspace, vector=vector
            )
        except ForbiddenValueError:
            continue

        conf.origin = origin
        configs.append(conf)

    return configs


def get_k_exchange_neighbourhood(
    configuration: Configuration,
    seed: int | np.random.RandomState,
    num_neighbors: int = 4,
    stdev: float = 0.2,
    exchange_size: int = 1,
) -> Iterator[Configuration]:
    """Generate Configurations in the k-exchange neighborhood of a given configuration.

    Each neighbor is obtained by randomly selectin 'exchange_size' hyperparameters
    from the original configuration and modifying them:
    - Continous/integer hyperparameters are sampled around the current value using a Gaussian distribution
    - Categorical/ordinal hyperparameters are sampled from their discrete neighbors.

    Parameters
    ----------
    configuration: Configuration
        Configuration for which neighbors are computed.
    seed: int | np.random.RandomState
        Sets the random seed to a fixed value.
    num_neighbors: int
        Number of neighbors to attempt generating.
    stdev: float
        Standard deviation used for sampling continous/integer hyperparameters.
    exchange_size: int
        Number of hyperparameters to modify in each neighbor.

    Returns
    -------
    Iterator[Configuration]
        Iterator over neighbor configurations
    """
    OVER_SAMPLE_CONTINUOUS_MULT = 5
    space = configuration.config_space
    config = configuration
    arr = configuration._vector
    dag = space._dag

    # neighbor_sample_size: How many neighbors we should sample for a given
    #   hyperparameter at once.
    # max_iter_per_selection: How many times we loop trying to generate a valid
    #   configuration with a given hyperparameter, every time it gets sampled. If
    #   not a single valid configuration is generated in this many iterations, it's
    #   marked as failed.
    # std: The standard deviation to use for the neighborhood of a hyperparameter when
    #   sampling neighbors.
    # should_shuffle: Whether or not we should shuffle the neighbors of a hyperparameter
    #   once generated
    # generated: Whether or not we have already generated the neighbors for this
    #   hyperparameter, set to false until sampled.
    # should_regen: Whether or not we should regenerate more neighbors for this
    #   hyperparameter at all.
    # -> dict[HP, (neighbor_sample_size, std, should_shuffle, generated, should_regen)]
    sample_strategy: dict[str, tuple[int, int, float | None, bool, bool, bool]] = {}

    # n_to_gen: Per hyperparameter, how many configurations we should generate with this
    #   hyperparameter as the one where the values change.
    # neighbors_generated_for_hp: The neighbors that were generated for this hp that can
    #   be retrieved.
    # -> tuple[HP, hp_idx, n_to_gen, neighbors_generated_for_hp]
    neighbors_to_generate: list[tuple[Hyperparameter, int, int, list[f64]]] = []

    nan_hps = np.isnan(arr)
    UFH = UniformFloatHyperparameter
    UIH = UniformIntegerHyperparameter
    for hp_name, node in dag.nodes.items():
        hp = node.hp
        hp_idx = node.idx

        # Skip inactive or fixed hyperparameters
        if hp.size == 1 or nan_hps[hp_idx]:
            continue

        # Determine neighbor sampling strategy per hyperparameter type
        if isinstance(hp, CategoricalHyperparameter):
            neighbor_sample_size = hp.size - 1
            n_to_gen = neighbor_sample_size
            max_iter_per_selection = neighbor_sample_size
            _std = None
            should_shuffle = True
            should_regen = False
        elif isinstance(hp, OrdinalHyperparameter):
            neighbor_sample_size = int(hp.get_num_neighbors(config[hp_name]))
            _std = None
            n_to_gen = neighbor_sample_size
            max_iter_per_selection = neighbor_sample_size
            should_shuffle = True
            should_regen = False
        elif np.isinf(hp.size):  # Continous hyperparameters
            neighbor_sample_size = num_neighbors * OVER_SAMPLE_CONTINUOUS_MULT
            n_to_gen = num_neighbors
            max_iter_per_selection = max(neighbor_sample_size, 100)
            _std = stdev if isinstance(hp, UFH) else None
            should_shuffle = False
            should_regen = True
        else:  # Discrete integer hyperparameters
            _possible_neighbors = int(hp.size - 1)
            neighbor_sample_size = int(min(num_neighbors, _possible_neighbors))
            n_to_gen = num_neighbors
            max_iter_per_selection = neighbor_sample_size
            _std = stdev if isinstance(hp, UIH) else None
            should_shuffle = True
            should_regen = _possible_neighbors >= num_neighbors

        sample_strategy[hp_name] = (
            neighbor_sample_size,
            max_iter_per_selection,
            _std,
            should_shuffle,
            False,
            should_regen,
        )
        neighbors_to_generate.append((hp, hp_idx, n_to_gen, []))

    random = np.random.RandomState(seed) if isinstance(seed, int) else seed
    arr = config.get_array()

    if len(neighbors_to_generate) == 0:
        return

    assert not any(n_to_gen == 0 for _, _, n_to_gen, _ in neighbors_to_generate)

    # Compose a finite set of hyperparameter index combinations
    n_hps = len(neighbors_to_generate)
    k = min(exchange_size, n_hps)

    # Generate neighbors until we reach the target number
    while True:

        # Randomly pick 'exchange_size' hyperparameters to modify
        # Only from HP's that were not exhausted before
        available_indices = [i for i, (_, _, n_left, _) in enumerate(neighbors_to_generate) if n_left > 0]
        if len(available_indices) == 0:
            break
        chosen_indices = random.choice(available_indices, size=min(k, len(available_indices)), replace=False)
        new_arr = arr.copy()
        valid = True

        # Modify each chosen hyperparameter
        for idx in chosen_indices:
            hp, hp_idx, n_left, pool = neighbors_to_generate[idx]
            hp_name = hp.name
            neighbor_sample_size, max_iter, _std, shuffle, generated, regen = sample_strategy[hp_name]

            # Generate new neighbors if pool is empty
            if len(pool) == 0:
                if generated and not regen:
                    neighbors_to_generate[idx] = (hp, hp_idx, 0, pool)
                    continue
                elif not generated or regen:
                    vec = arr[hp_idx]
                    _neighbors = hp._neighborhood(vec, n=neighbor_sample_size, seed=random, std=_std)
                    if shuffle:
                        random.shuffle(_neighbors)
                    pool = _neighbors.tolist()

                    if len(pool) == 0:
                        valid = False
                        break

                    sample_strategy[hp_name] = (neighbor_sample_size, max_iter, _std, shuffle, True, regen)
                    neighbors_to_generate[idx] = (hp, hp_idx, n_left, pool)
                else:
                    valid = False
                    break

            # pop one neighbor value for this hyperparameter
            val = pool.pop()
            neighbors_to_generate[idx] = (hp, hp_idx, n_left - 1, pool)
            new_arr[hp_idx] = val

        if not valid:
            continue

        # Check for forbidden constraints
        for forbidden_list in dag.forbidden_lookup.values():
            if any(f.is_forbidden_vector(new_arr) for f in forbidden_list):
                valid = False
                break
        if not valid:
            continue

        yield Configuration(space, vector=new_arr)


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
