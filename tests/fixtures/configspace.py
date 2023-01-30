import pytest
from ConfigSpace import (
    Categorical,
    ConfigurationSpace,
    EqualsCondition,
    Float,
    InCondition,
    Integer,
)


@pytest.fixture
def configspace_small() -> ConfigurationSpace:
    cs = ConfigurationSpace(seed=0)

    a = Integer("a", (1, 10000), default=1)
    b = Float("b", (1e-5, 1e-1), log=True, default=1e-1)
    c = Categorical("c", ["cat", "dog", "mouse"], default="cat")

    # Add all hyperparameters at once:
    cs.add_hyperparameters([a, b, c])

    return cs


@pytest.fixture
def configspace_large() -> ConfigurationSpace:
    cs = ConfigurationSpace(seed=0)

    n_layer = Integer("n_layer", (1, 5), default=1)
    n_neurons = Integer("n_neurons", (8, 256), log=True, default=10)
    activation = Categorical("activation", ["logistic", "tanh", "relu"], default="tanh")
    solver = Categorical("solver", ["lbfgs", "sgd", "adam"], default="adam")
    batch_size = Integer("batch_size", (30, 300), default=200)
    learning_rate = Categorical("learning_rate", ["constant", "invscaling", "adaptive"], default="constant")
    learning_rate_init = Float("learning_rate_init", (0.0001, 1.0), default=0.001, log=True)

    # Add all hyperparameters at once:
    cs.add_hyperparameters(
        [
            n_layer,
            n_neurons,
            activation,
            solver,
            batch_size,
            learning_rate,
            learning_rate_init,
        ]
    )

    # Adding conditions to restrict the hyperparameter space...
    # ... since learning rate is used when solver is 'sgd'.
    use_lr = EqualsCondition(child=learning_rate, parent=solver, value="sgd")
    # ... since learning rate initialization will only be accounted for when using 'sgd' or 'adam'.
    use_lr_init = InCondition(child=learning_rate_init, parent=solver, values=["sgd", "adam"])
    # ... since batch size will not be considered when optimizer is 'lbfgs'.
    use_batch_size = InCondition(child=batch_size, parent=solver, values=["sgd", "adam"])

    # We can also add multiple conditions on hyperparameters at once:
    cs.add_conditions([use_lr, use_batch_size, use_lr_init])

    return cs
