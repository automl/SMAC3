import typing

import logging

import numpy as np
from ConfigSpace import ConfigurationSpace
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

from smac.utils.constants import MAXINT

__copyright__ = "Copyright 2021, AutoML.org Freiburg-Hannover"
__license__ = "3-clause BSD"


def get_types(
    config_space: ConfigurationSpace,
    instance_features: typing.Optional[np.ndarray] = None,
) -> typing.Tuple[typing.List[int], typing.List[typing.Tuple[float, float]]]:
    """Return the types of the hyperparameters and the bounds of the
    hyperparameters and instance features.
    """
    # Extract types vector for rf from config space and the bounds
    types = [0] * len(config_space.get_hyperparameters())
    bounds = [(np.nan, np.nan)] * len(types)

    for i, param in enumerate(config_space.get_hyperparameters()):
        parents = config_space.get_parents_of(param.name)
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
            # for constants we simply set types to 0 which makes it a numerical
            # parameter
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
        types = types + [0] * instance_features.shape[1]

    return types, bounds


def get_rng(
    rng: typing.Optional[typing.Union[int, np.random.RandomState]] = None,
    run_id: typing.Optional[int] = None,
    logger: typing.Optional[logging.Logger] = None,
) -> typing.Tuple[int, np.random.RandomState]:
    """Initialize random number generator and set run_id.

    * If rng and run_id are None, initialize a new generator and sample a run_id
    * If rng is None and a run_id is given, use the run_id to initialize the rng
    * If rng is an int, a RandomState object is created from that.
    * If rng is RandomState, return it
    * If only run_id is None, a run_id is sampled from the random state.

    Parameters
    ----------
    rng : np.random.RandomState|int|None
    run_id : int, optional
    logger: logging.Logger, optional

    Returns
    -------
    int
    np.random.RandomState
    """
    if logger is None:
        logger = logging.getLogger("GetRNG")
    # initialize random number generator
    if rng is not None and not isinstance(rng, (int, np.random.RandomState)):
        raise TypeError(
            "Argument rng accepts only arguments of type None, int or np.random.RandomState, "
            "you provided %s." % str(type(rng))
        )
    if run_id is not None and not isinstance(run_id, int):
        raise TypeError(
            "Argument run_id accepts only arguments of type None, int, " "you provided %s." % str(type(run_id))
        )

    if rng is None and run_id is None:
        # Case that both are None
        logger.debug("No rng and no run_id given: using a random value to initialize run_id.")
        rng_return = np.random.RandomState()
        run_id_return = rng_return.randint(MAXINT)
    elif rng is None and isinstance(run_id, int):
        logger.debug("No rng and no run_id given: using run_id %d as seed.", run_id)
        rng_return = np.random.RandomState(seed=run_id)
        run_id_return = run_id
    elif isinstance(rng, int) and run_id is None:
        run_id_return = rng
        rng_return = np.random.RandomState(seed=rng)
    elif isinstance(rng, int) and isinstance(run_id, int):
        run_id_return = run_id
        rng_return = np.random.RandomState(seed=rng)
    elif isinstance(rng, np.random.RandomState) and run_id is None:
        rng_return = rng
        run_id_return = rng.randint(MAXINT)
    elif isinstance(rng, np.random.RandomState) and isinstance(run_id, int):
        rng_return = rng
        run_id_return = run_id
    else:
        raise ValueError(
            "This should not happen! Please contact the developers! Arguments: rng=%s of type %s and "
            "run_id=%s of type %s" % (rng, type(rng), str(run_id), type(run_id))
        )
    return run_id_return, rng_return
