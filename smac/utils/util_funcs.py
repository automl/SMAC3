import numpy as np

from ConfigSpace.hyperparameters import CategoricalHyperparameter, \
    UniformFloatHyperparameter, UniformIntegerHyperparameter, Constant, \
    OrdinalHyperparameter


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
