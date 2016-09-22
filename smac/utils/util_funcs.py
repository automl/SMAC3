import numpy as np

from ConfigSpace.hyperparameters import CategoricalHyperparameter, \
    UniformFloatHyperparameter, UniformIntegerHyperparameter, Constant

def get_types(config_space, instance_features=None):
    # Extract types vector for rf from config space
    types = np.zeros(len(config_space.get_hyperparameters()),
                     dtype=np.uint)

    for i, param in enumerate(config_space.get_hyperparameters()):
        if isinstance(param, (CategoricalHyperparameter)):
            n_cats = len(param.choices)
            types[i] = n_cats

        elif isinstance(param, Constant):
            # for constants we simply set types to 0
            # which makes it a numerical parameter
            types[i] = 0
            # and we leave the bounds to be 0 for now
        elif not isinstance(param, (UniformFloatHyperparameter,
                                    UniformIntegerHyperparameter)):
            raise TypeError("Unknown hyperparameter type %s" % type(param))

    if instance_features is not None:
        types = np.hstack(
            (types, np.zeros((instance_features.shape[1]))))

    types = np.array(types, dtype=np.uint)
    return types