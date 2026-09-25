"""A prior given by its values rather than by a named distribution.

`ConfigSpacePrior` can say what ConfigSpace can express. This one says what an
interface can draw: a region ruled out, two peaks, a shape with no name. The
positions are in the vectorized representation deliberately, so that what was
drawn is what is read with nothing in between to disagree about.
"""

import numpy as np
import pytest
from ConfigSpace import Categorical, ConfigurationSpace, Float

from smac.acquisition.weight import TabulatedPrior
from smac.utils.configspace import convert_configurations_to_array


@pytest.fixture
def space():
    cs = ConfigurationSpace(seed=0)
    cs.add([Float("lr", (1e-4, 1e-1), log=True), Float("mom", (0.0, 1.0)),
            Categorical("kern", ["rbf", "linear", "poly"])])
    return cs


@pytest.fixture
def columns(space):
    return list(space.keys())


def _peak(at=0.25, width=0.06, n=41):
    xs = np.linspace(0, 1, n)
    return [[float(x), float(np.exp(-0.5 * ((x - at) / width) ** 2))] for x in xs]


def _row(columns, **values):
    row = np.zeros((1, len(columns)))
    for name, value in values.items():
        row[0, columns.index(name)] = value
    return row


def test_it_names_the_whole_space_not_only_what_it_tabulates(space):
    """`validate_against` evaluates column by column, so a prior which named
    only its own hyperparameter would apply each belief to the wrong one."""
    prior = TabulatedPrior(space, {"lr": _peak()})

    prior.validate_against(space)
    assert prior.hyperparameter_names == list(space.keys())


def test_the_density_follows_the_table(space, columns):
    prior = TabulatedPrior(space, {"lr": _peak(at=0.25)})

    at_peak = prior.pdf(_row(columns, lr=0.25)).item()
    off_peak = prior.pdf(_row(columns, lr=0.75)).item()

    assert at_peak > off_peak
    assert off_peak == pytest.approx(0.0, abs=1e-6)


def test_an_untabulated_hyperparameter_contributes_a_constant(space, columns):
    """Which is how a belief about part of the space is written: name the whole
    space, tabulate the part you have an opinion about."""
    prior = TabulatedPrior(space, {"lr": _peak()})

    low = prior.pdf(_row(columns, lr=0.25, mom=0.1)).item()
    high = prior.pdf(_row(columns, lr=0.25, mom=0.9)).item()

    assert low == pytest.approx(high)


def test_the_density_is_flat_beyond_the_outermost_knots(space, columns):
    prior = TabulatedPrior(space, {"mom": [[0.3, 2.0], [0.7, 5.0]]})

    assert prior.pdf(_row(columns, mom=0.0)).item() == pytest.approx(2.0)
    assert prior.pdf(_row(columns, mom=1.0)).item() == pytest.approx(5.0)


def test_it_can_be_coarsened_for_a_forest(space, columns):
    """A random forest predicts a piecewise constant function, and multiplying
    one by a smoothly varying density removes the step structure it relies on —
    the reason `discretize_pdf` exists."""
    prior = TabulatedPrior(space, {"lr": _peak()})
    grid = np.zeros((41, len(columns)))
    grid[:, columns.index("lr")] = np.linspace(0, 1, 41)

    exact = prior.pdf(grid)
    coarse = prior.pdf(grid, resolution=4)

    assert len(np.unique(np.round(coarse, 9))) <= 4
    assert len(np.unique(np.round(exact, 9))) > 4


def test_sampling_follows_a_continuous_table(space, columns):
    prior = TabulatedPrior(space, {"lr": _peak(at=0.8, width=0.03)})

    drawn = convert_configurations_to_array(prior.sample(400, np.random.RandomState(0)))

    assert drawn[:, columns.index("lr")].mean() == pytest.approx(0.8, abs=0.05)


def test_sampling_a_categorical_names_a_choice_that_exists(space, columns):
    """Its vectorized value is a choice index, so interpolating between two of
    them would name a choice that is not there."""
    prior = TabulatedPrior(space, {"kern": [[0, 0.0], [1, 0.0], [2, 1.0]]})

    drawn = convert_configurations_to_array(prior.sample(60, np.random.RandomState(0)))
    indices = drawn[:, columns.index("kern")]

    assert set(np.unique(indices)) == {2.0}


def test_an_untabulated_hyperparameter_is_sampled_as_the_space_would(space, columns):
    prior = TabulatedPrior(space, {"lr": _peak()})

    drawn = convert_configurations_to_array(prior.sample(400, np.random.RandomState(0)))

    assert drawn[:, columns.index("mom")].mean() == pytest.approx(0.5, abs=0.06)


def test_a_table_of_zeros_falls_back_to_uniform(space, columns):
    """It carries no information about where to look, which is not the same as
    dividing by nothing."""
    prior = TabulatedPrior(space, {"mom": [[0.0, 0.0], [1.0, 0.0]]})

    drawn = convert_configurations_to_array(prior.sample(300, np.random.RandomState(0)))

    assert drawn[:, columns.index("mom")].mean() == pytest.approx(0.5, abs=0.08)


def test_a_negative_density_is_refused(space):
    with pytest.raises(ValueError, match="not a density"):
        TabulatedPrior(space, {"mom": [[0.0, 1.0], [1.0, -1.0]]})


def test_a_table_for_an_unknown_hyperparameter_is_refused(space):
    with pytest.raises(ValueError, match="does not have"):
        TabulatedPrior(space, {"nonesuch": _peak()})


def test_a_single_knot_is_refused(space):
    with pytest.raises(ValueError, match="at least two"):
        TabulatedPrior(space, {"mom": [[0.5, 1.0]]})
