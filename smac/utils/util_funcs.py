import typing
import logging
import numpy as np

from ConfigSpace.hyperparameters import CategoricalHyperparameter, \
    UniformFloatHyperparameter, UniformIntegerHyperparameter, Constant, \
    OrdinalHyperparameter
from smac.utils.constants import MAXINT


def get_types(config_space, instance_features=None):
    """TODO"""
    # Extract types vector for rf from config space and the bounds
    types = np.zeros(len(config_space.get_hyperparameters()),
                     dtype=np.uint)
    bounds = [(np.nan, np.nan)]*types.shape[0]

    for i, param in enumerate(config_space.get_hyperparameters()):
        if isinstance(param, (CategoricalHyperparameter)):
            n_cats = len(param.choices)
            types[i] = n_cats
            bounds[i] = (int(n_cats), np.nan)

        elif isinstance(param, (OrdinalHyperparameter)):
            n_cats = len(param.sequence)
            types[i] = 0
            bounds[i] = (0, int(n_cats) - 1)

        elif isinstance(param, Constant):
            # for constants we simply set types to 0
            # which makes it a numerical parameter
            types[i] = 0
            bounds[i] = (0, np.nan)
            # and we leave the bounds to be 0 for now
        elif isinstance(param, UniformFloatHyperparameter):         # Are sampled on the unit hypercube thus the bounds
            # bounds[i] = (float(param.lower), float(param.upper))  # are always 0.0, 1.0
            bounds[i] = (0.0, 1.0)
        elif isinstance(param, UniformIntegerHyperparameter):
            # bounds[i] = (int(param.lower), int(param.upper))
            bounds[i] = (0.0, 1.0)
        elif not isinstance(param, (UniformFloatHyperparameter,
                                    UniformIntegerHyperparameter,
                                    OrdinalHyperparameter)):
            raise TypeError("Unknown hyperparameter type %s" % type(param))

    if instance_features is not None:
        types = np.hstack(
            (types, np.zeros((instance_features.shape[1]))))

    types = np.array(types, dtype=np.uint)
    bounds = np.array(bounds, dtype=object)
    return types, bounds


def get_rng(
        rng: typing.Optional[typing.Union[int, np.random.RandomState]] = None,
        run_id: typing.Optional[int] = None,
        logger: typing.Optional[logging.Logger] = None,
) -> typing.Tuple[int, np.random.RandomState]:
    """
    Initialize random number generator and set run_id

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
        logger = logging.getLogger('GetRNG')
    # initialize random number generator
    if rng is not None and not isinstance(rng, (int, np.random.RandomState)):
        raise TypeError('Argument rng accepts only arguments of type None, int or np.random.RandomState, '
                        'you provided %s.' % str(type(rng)))
    if run_id is not None and not isinstance(run_id, int):
        raise TypeError('Argument run_id accepts only arguments of type None, int or np.random.RandomState, '
                        'you provided %s.' % str(type(run_id)))

    if rng is None and run_id is None:
        # Case that both are None
        logger.debug('No rng and no run_id given: using a random value to initialize run_id.')
        rng = np.random.RandomState()
        run_id = rng.randint(MAXINT)
    elif rng is None and isinstance(run_id, int):
        logger.debug('No rng and no run_id given: using run_id %d as seed.', run_id)
        rng = np.random.RandomState(seed=run_id)
    elif isinstance(rng, int):
        if run_id is None:
            run_id = rng
        else:
            pass
        rng = np.random.RandomState(seed=rng)
    elif isinstance(rng, np.random.RandomState):
        if run_id is None:
            run_id = rng.randint(MAXINT)
        else:
            pass
    else:
        raise ValueError('This should not happen! Please contact the developers! Arguments: rng=%s of type %s and '
                         'run_id=% of type %s' % (rng, type(rng), run_id, type(run_id)))
    return run_id, rng
