from __future__ import annotations

import hashlib
import logging
from functools import partial
from typing import Any, Mapping, Sequence, Dict

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
    Hyperparameter,
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

### Below functions are needed to adapt the Configuration Space during the optimization process. Should this functionality be added to ConfigurationSpace, we can remove these functions.

def modify_hyperparameter(space:Dict[str,tuple[int, int]
                | tuple[float, float]
                | Sequence[Any]
                | int
                | float
                | str
                | Hyperparameter,
            ], hyperparameter_name:str, **modifications):
    """
    Modifies a specific hyperparameter in the space dictionary of a ConfigurationSpace by applying given modifications.

    Parameters:
        space (Dict): The `space` dictionary extracted from a ConfigurationSpace.
        hyperparameter_name (str): The name of the hyperparameter to modify.
        **modifications: The keyword arguments specifying the modifications (e.g., `lower`, `upper`, etc.).

    Returns:
        dict: The modified space dictionary.
    """
    # Check if the hyperparameter exists in the space dictionary
    if hyperparameter_name not in space.keys():
        raise ValueError(f"Hyperparameter '{hyperparameter_name}' not found in the space.")

    # Get the original hyperparameter
    original_hp = space[hyperparameter_name]
    hp_class = type(original_hp)

    # Prepare updated attributes
    new_args = original_hp.__dict__.copy()
    new_args.update(modifications)

    # Recreate the hyperparameter with updated values
    if hp_class is CategoricalHyperparameter:
        updated_hp = CategoricalHyperparameter(
            name=new_args["name"],
            choices=new_args["choices"],
            default_value=new_args.get("default_value", None)
        )
    elif hp_class is Constant:
        updated_hp = Constant(
            name=new_args["name"],
            value=new_args["value"]
        )
    elif hp_class is OrdinalHyperparameter:
        updated_hp = OrdinalHyperparameter(
            name=new_args["name"],
            sequence=new_args["sequence"],
            default_value=new_args.get("default_value", None)
        )
    elif hp_class is UniformFloatHyperparameter:
        updated_hp = UniformFloatHyperparameter(
            name=new_args["name"],
            lower=new_args["lower"],
            upper=new_args["upper"],
            default_value=new_args.get("default_value", None),
            log=new_args.get("log", False)
        )
    elif hp_class is UniformIntegerHyperparameter:
        updated_hp = UniformIntegerHyperparameter(
            name=new_args["name"],
            lower=new_args["lower"],
            upper=new_args["upper"],
            default_value=new_args.get("default_value", None),
            log=new_args.get("log", False)
        )
    elif hp_class is NormalFloatHyperparameter:
        updated_hp = NormalFloatHyperparameter(
            name=new_args["name"],
            mu=new_args["mu"],
            sigma=new_args["sigma"],
            lower=new_args.get("lower", None),
            upper=new_args.get("upper", None),
            default_value=new_args.get("default_value", None),
            log=new_args.get("log", False)
        )
    elif hp_class is NormalIntegerHyperparameter:
        updated_hp = NormalIntegerHyperparameter(
            name=new_args["name"],
            mu=new_args["mu"],
            sigma=new_args["sigma"],
            lower=new_args.get("lower", None),
            upper=new_args.get("upper", None),
            default_value=new_args.get("default_value", None),
            log=new_args.get("log", False)
        )
    elif hp_class is BetaFloatHyperparameter:
        updated_hp = BetaFloatHyperparameter(
            name=new_args["name"],
            alpha=new_args["alpha"],
            beta=new_args["beta"],
            lower=new_args.get("lower", None),
            upper=new_args.get("upper", None),
            default_value=new_args.get("default_value", None),
            log=new_args.get("log", False)
        )
    elif hp_class is BetaIntegerHyperparameter:
        updated_hp = BetaIntegerHyperparameter(
            name=new_args["name"],
            alpha=new_args["alpha"],
            beta=new_args["beta"],
            lower=new_args.get("lower", None),
            upper=new_args.get("upper", None),
            default_value=new_args.get("default_value", None),
            log=new_args.get("log", False)
        )
    else:
        raise ValueError(f"Unsupported hyperparameter type: {hp_class}")

    # Replace the hyperparameter in the space dictionary
    del space[hyperparameter_name]
    space[hyperparameter_name] = updated_hp 

    return space

def recreate_configspace(
    name: str,
    space: dict,
    conditions: list = None,
    forbidden_clauses: list = None,
    default_configuration: dict = None,
) -> ConfigurationSpace:
    """
    Recreates a ConfigurationSpace from the modified hyperparameter space and other attributes.
    Needed upon modifying the Configuration Space due to caching issues.

    Parameters:
        space (dict): The modified space dictionary containing hyperparameters.
        name (str): The name for the new ConfigurationSpace.
        conditions (list): List of conditions to apply to the ConfigurationSpace.
        forbidden_clauses (list): List of forbidden clauses for the ConfigurationSpace.
        default_configuration (dict): Default configuration for the ConfigurationSpace.

    Returns:
        ConfigurationSpace: The reconstructed ConfigurationSpace object.
    """
    # Initialize a new ConfigurationSpace with the specified name
    cs = ConfigurationSpace(name=name)

    # Add hyperparameters to the ConfigurationSpace
    for hp_name, hp_obj in space.items():
        cs.add(hp_obj)

    # Add conditions, if any
    if conditions:
        for condition in conditions:
            cs.add(condition)

    # Add forbidden clauses, if any
    if forbidden_clauses:
        for clause in forbidden_clauses:
            cs.add(clause)

    # Set the default configuration, if provided
    if default_configuration:
        cs.default_configuration = default_configuration

    return cs

def update_configspace(smac: "AbstractFacade", new_configspace: ConfigurationSpace): # Typehints lead to circular imports
    """
    Provided a new Configuration Space, updates the SMAC object and its components.

    Parameters:
        smac (AbstractFacade): The SMAC object to update.
        new_configspace (ConfigurationSpace): The new ConfigurationSpace object to set.
    """
    # Update the Configuration Space in SMAC
    smac.scenario.configspace = new_configspace
    smac._model._configspace = new_configspace
    smac._optimizer._configspace = new_configspace
    smac._intensifier._scenario.configspace = new_configspace
    smac._acquisition_maximizer._configspace = new_configspace

    from smac.acquisition.maximizer.local_and_random_search import LocalAndSortedRandomSearch
    if isinstance(smac._acquisition_maximizer, LocalAndSortedRandomSearch):
        smac._acquisition_maximizer._local_search._configspace = new_configspace
        smac._acquisition_maximizer._random_search._configspace = new_configspace


### Above functions are needed to adapt the Configuration Space during the optimization process. Should this functionality be added to ConfigurationSpace, we can remove these functions.

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
