from __future__ import annotations

from abc import abstractmethod
from typing import Any, Mapping

import numpy as np
from ConfigSpace import Configuration, ConfigurationSpace
from ConfigSpace.hyperparameters import FloatHyperparameter, Hyperparameter

from smac.acquisition.weight.abstract_weight import AbstractAcquisitionWeight
from smac.acquisition.weight.decay import DecaySchedule, PolynomialDecay
from smac.model.abstract_model import AbstractModel
from smac.model.random_forest.abstract_random_forest import AbstractRandomForest
from smac.utils.logging import get_logger

__copyright__ = "Copyright 2025, Leibniz University Hanover, Institute of AI"
__license__ = "3-clause BSD"

logger = get_logger(__name__)


class AbstractInputPrior:
    """A belief about where in the configuration space the optimum lies.

    A prior is a density over the search space, evaluated on the vectorized representation of a configuration - the
    same array `convert_configurations_to_array` produces, with one column per hyperparameter in the order the
    configuration space lists them. It is not required to be normalized: only the relative values matter, because
    the prior is multiplied into an acquisition function which is itself only ranked.

    Sampling is optional. A prior which can produce configurations lets the acquisition function maximizer draw
    candidates from it, which is what makes a sharply peaked prior usable at all - the maximizer would otherwise
    have to stumble into the peak by chance. A prior which cannot returns `None` and simply does not contribute
    candidates.
    """

    @property
    @abstractmethod
    def hyperparameter_names(self) -> list[str]:
        """The hyperparameters the prior is defined over, in column order."""
        raise NotImplementedError

    @property
    def meta(self) -> dict[str, Any]:
        """Returns the meta data of the created object."""
        return {"name": self.__class__.__name__, "hyperparameters": self.hyperparameter_names}

    @abstractmethod
    def pdf(self, X: np.ndarray, resolution: int | None = None) -> np.ndarray:
        """Evaluates the density.

        Parameters
        ----------
        X : np.ndarray [N, D]
            Points in the vectorized representation of the configuration space.
        resolution : int | None, defaults to None
            Number of distinct levels to coarsen the density to, or `None` for the exact density. Random forest
            surrogates need a piecewise constant acquisition function to be well behaved, so a prior evaluated
            alongside one is coarsened. A prior with no continuous part may ignore this.

        Returns
        -------
        np.ndarray [N, 1]
            Non-negative density at X.
        """
        raise NotImplementedError

    def sample(self, n: int, rng: np.random.RandomState | None = None) -> list[Configuration] | None:
        """Draws configurations from the prior, or returns `None` if it cannot be sampled from.

        Parameters
        ----------
        n : int
            Number of configurations to draw.
        rng : np.random.RandomState | None, defaults to None
            Random state to draw with, where the prior supports one.
        """
        return None

    def validate_against(self, configspace: ConfigurationSpace) -> None:
        """Checks that the prior lines up with the search space it will be evaluated against.

        The density is evaluated column by column against the vectorized representation of a configuration, so a
        prior whose hyperparameters are named differently or ordered differently would silently apply each belief
        to the wrong hyperparameter.
        """
        expected = list(_hyperparameters_of(configspace))
        actual = self.hyperparameter_names

        if actual != expected:
            raise ValueError(
                f"The prior is defined over {actual}, but the search space has {expected}. A prior is evaluated "
                "column by column against the vectorized configurations, so the hyperparameters have to match in "
                "name and order. Hyperparameters the prior has no belief about are given a uniform distribution "
                "rather than left out."
            )


def _hyperparameters_of(configspace: Any) -> dict[str, Hyperparameter]:
    """Returns the hyperparameters of a configuration space, by name, in column order."""
    if isinstance(configspace, Mapping):
        return dict(configspace)

    try:
        return dict(configspace)
    except TypeError:
        # Objects which only offer the older accessor, such as test doubles.
        return dict(configspace.get_hyperparameters_dict())


class ConfigSpacePrior(AbstractInputPrior):
    """A prior expressed as a configuration space whose hyperparameters carry distributions.

    This is how priors are written down in ConfigSpace: a normal or beta distribution on a numerical
    hyperparameter, weights on a categorical one. The density is the product of the per-hyperparameter densities,
    so beliefs about different hyperparameters are independent - a belief about a *combination* of hyperparameters
    cannot be expressed this way and needs a prior of its own.

    A hyperparameter the user has no belief about keeps its uniform distribution and contributes a constant factor,
    which is how a prior over only part of the search space is written: name the whole space, distribute only the
    part you have an opinion about.

    Parameters
    ----------
    configspace : ConfigurationSpace
        A copy of the search space with distributions placed on it. `create_prior_configspace_copy` builds one
        from point estimates.
    """

    def __init__(self, configspace: ConfigurationSpace) -> None:
        self._configspace = configspace
        self._hyperparameters = _hyperparameters_of(configspace)

    @property
    def configspace(self) -> ConfigurationSpace:
        """The configuration space carrying the distributions."""
        return self._configspace

    @property
    def hyperparameter_names(self) -> list[str]:  # noqa: D102
        return list(self._hyperparameters)

    def pdf(self, X: np.ndarray, resolution: int | None = None) -> np.ndarray:  # noqa: D102
        if len(X.shape) == 1:
            X = X[:, np.newaxis]

        values = np.ones((len(X), 1))

        # The hyperparameters are listed in the same order as the columns of the vectorized configurations.
        for hyperparameter, column in zip(self._hyperparameters.values(), X.T):
            if resolution is not None and isinstance(hyperparameter, FloatHyperparameter):
                values = values * discretize_pdf(hyperparameter, column, resolution)
            else:
                values = values * density_of(hyperparameter, column)

        return values

    def sample(self, n: int, rng: np.random.RandomState | None = None) -> list[Configuration] | None:  # noqa: D102
        if not hasattr(self._configspace, "sample_configuration"):
            return None

        if n == 1:
            return [self._configspace.sample_configuration()]

        return list(self._configspace.sample_configuration(size=n))


class TabulatedPrior(AbstractInputPrior):
    """A prior given by its values rather than by a named distribution.

    Where `ConfigSpacePrior` writes a belief as a distribution ConfigSpace can express, this writes it as a table:
    positions along one hyperparameter and a density at each. Anything an interface can draw can be stated this
    way - a region ruled out, two peaks, a shape with no name - which is the point of it and is what a named
    distribution cannot do.

    Positions are in the **vectorized** representation, the one `pdf` is evaluated against: the unit interval for
    a numerical hyperparameter, and the choice index for a categorical. That is deliberate rather than a
    convenience, because it is the representation the acquisition function already works in, so what was drawn is
    what is read with no conversion in between to disagree about.

    A hyperparameter with no table contributes a constant factor, which is how a belief about part of the search
    space is written: name the whole space, tabulate the part you have an opinion about. Between knots the density
    is linear, and beyond the outermost ones it is flat - a table dense enough to have been drawn is dense enough
    to be read back.

    Parameters
    ----------
    configspace : ConfigurationSpace
        The search space the prior is evaluated against. Its hyperparameters, in order, are the columns.
    tables : Mapping[str, Any]
        Knots per hyperparameter, as `(position, density)` pairs. Densities must be non-negative and need not be
        normalized, since only ratios matter to a function which is itself only ranked.
    """

    def __init__(self, configspace: ConfigurationSpace, tables: Mapping[str, Any]) -> None:
        self._configspace = configspace
        self._hyperparameters = _hyperparameters_of(configspace)
        self._tables: dict[str, tuple[np.ndarray, np.ndarray]] = {}

        for name, knots in tables.items():
            if name not in self._hyperparameters:
                raise ValueError(
                    f"The prior tabulates {name!r}, which the search space does not have. Its hyperparameters "
                    f"are {list(self._hyperparameters)}."
                )
            points = np.asarray(knots, dtype=float)
            if points.ndim != 2 or points.shape[1] != 2 or len(points) < 2:
                raise ValueError(
                    f"The table for {name!r} must be at least two (position, density) pairs, got shape "
                    f"{points.shape}."
                )
            order = np.argsort(points[:, 0])
            positions, densities = points[order, 0], points[order, 1]
            if np.any(densities < 0):
                raise ValueError(f"The table for {name!r} has a negative density, which is not a density.")
            self._tables[name] = (positions, densities)

    @property
    def tables(self) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        """The knots, by hyperparameter, as `(positions, densities)`."""
        return dict(self._tables)

    @property
    def hyperparameter_names(self) -> list[str]:  # noqa: D102
        return list(self._hyperparameters)

    @property
    def meta(self) -> dict[str, Any]:  # noqa: D102
        data = super().meta
        data["tabulated"] = {name: int(len(x)) for name, (x, _) in self._tables.items()}
        return data

    def pdf(self, X: np.ndarray, resolution: int | None = None) -> np.ndarray:  # noqa: D102
        if len(X.shape) == 1:
            X = X[:, np.newaxis]

        values = np.ones((len(X), 1))

        for name, column in zip(self._hyperparameters, X.T):
            table = self._tables.get(name)
            if table is None:
                continue
            positions, densities = table
            density = np.interp(np.asarray(column, dtype=float), positions, densities)
            if resolution is not None:
                density = _coarsen(density, float(densities.max()), resolution)
            values = values * density.reshape((-1, 1))

        return values

    def sample(self, n: int, rng: np.random.RandomState | None = None) -> list[Configuration] | None:  # noqa: D102
        if not hasattr(self._configspace, "sample_configuration"):
            return None

        drawn = self._configspace.sample_configuration(size=n) if n > 1 else [self._configspace.sample_configuration()]
        if not self._tables:
            return list(drawn)

        # Every untabulated hyperparameter keeps the draw the configuration space made for it, so the parts of the
        # space nobody has an opinion about are sampled exactly as they would have been.
        random = rng if rng is not None else np.random.RandomState()
        names = list(self._hyperparameters)
        out = []
        for configuration in drawn:
            vector = np.array(configuration.get_array(), dtype=float)
            for name, (positions, densities) in self._tables.items():
                column = names.index(name)
                continuous = isinstance(self._hyperparameters[name], FloatHyperparameter)
                vector[column] = (
                    _sample_continuous(positions, densities, float(random.random_sample()))
                    if continuous
                    else _sample_discrete(positions, densities, random)
                )
            out.append(Configuration(self._configspace, vector=vector))
        return out


def _coarsen(values: np.ndarray, upper: float, number_of_bins: int) -> np.ndarray:
    """Coarsens a tabulated density to a fixed number of levels, for the reason `discretize_pdf` explains."""
    if number_of_bins < 1:
        raise ValueError(f"The number of bins must be at least one, got {number_of_bins}.")
    if upper <= 0:
        return values

    bin_values = np.linspace(0.0, upper, number_of_bins)
    bin_indices = np.clip(np.round(values * number_of_bins / upper), 0, number_of_bins - 1).astype(int)
    return bin_values[bin_indices]


def _sample_continuous(positions: np.ndarray, densities: np.ndarray, u: float) -> float:
    """Draws one value by inverting the cumulative mass of a piecewise linear density.

    A table which is zero everywhere carries no information about where to look, so it falls back to uniform over
    its own range rather than dividing by nothing.
    """
    mass = np.concatenate([[0.0], np.cumsum(0.5 * (densities[1:] + densities[:-1]) * np.diff(positions))])
    if mass[-1] <= 0:
        return float(positions[0] + u * (positions[-1] - positions[0]))
    return float(np.interp(u * mass[-1], mass, positions))


def _sample_discrete(positions: np.ndarray, densities: np.ndarray, rng: np.random.RandomState) -> float:
    """Draws one of the tabulated positions, weighted by its density.

    A categorical's vectorized value is a choice index, so interpolating between two of them would name a choice
    that does not exist.
    """
    total = float(densities.sum())
    if total <= 0:
        return float(positions[int(rng.randint(len(positions)))])
    return float(rng.choice(positions, p=densities / total))


def density_of(hyperparameter: Hyperparameter, X_col: np.ndarray) -> np.ndarray:
    """Evaluates the density of one hyperparameter on the vectorized representation.

    Returns
    -------
    np.ndarray [N, 1]
    """
    # `_pdf` still works but is deprecated. It is kept as a fallback for objects which only offer it.
    density = getattr(hyperparameter, "pdf_vector", None)

    if density is not None:
        return np.asarray(density(X_col), dtype=float).reshape((-1, 1))

    return np.asarray(hyperparameter._pdf(X_col[:, np.newaxis]), dtype=float).reshape((-1, 1))


def discretize_pdf(hyperparameter: FloatHyperparameter, X_col: np.ndarray, number_of_bins: int) -> np.ndarray:
    """Coarsens the density of a continuous hyperparameter to a fixed number of levels.

    Random forest surrogates predict a piecewise constant function, and an acquisition function multiplied by a
    smoothly varying density is no longer piecewise constant. Every candidate inside one leaf then gets a slightly
    different value and the local search follows the prior's gradient rather than the model, which removes the
    randomness the forest relies on. Coarsening the density restores the step structure.

    Parameters
    ----------
    hyperparameter : FloatHyperparameter
        The hyperparameter whose density is coarsened.
    X_col : np.ndarray [N, ]
        The values of that hyperparameter, in the vectorized representation.
    number_of_bins : int
        The number of distinct density values allowed.

    Returns
    -------
    np.ndarray [N, 1]
        The coarsened density.
    """
    if number_of_bins < 1:
        raise ValueError(f"The number of bins must be at least one, got {number_of_bins}.")

    pdf_values = density_of(hyperparameter, X_col)

    lower, upper = (0.0, hyperparameter.get_max_density())
    bin_values = np.linspace(lower, upper, number_of_bins)
    bin_indices = np.clip(
        np.round((pdf_values - lower) * number_of_bins / (upper - lower)), 0, number_of_bins - 1
    ).astype(int)

    return bin_values[bin_indices]


class PriorWeight(AbstractAcquisitionWeight):
    r"""A user belief about where the optimum lies, fading as the surrogate learns.

    $$w(\mathbf{X}) = \left( \pi(\mathbf{X}) + \text{floor} \right)^{\beta / (t - t_0 + 1)}$$

    See "piBO: Augmenting Acquisition Functions with User Beliefs for Bayesian Optimization" by Carl Hvarfner et
    al. [[HSSL22][HSSL22]] for the mechanism, and "Dynamic Priors in Bayesian Optimization for Hyperparameter
    Optimization" by Lukas Fehring et al. [[FWS+25][FWS+25]] for what supplying one during a run means.

    The decay is anchored at $t_0$, the trial count when the belief was supplied. A belief stated before the run
    begins is anchored at the end of the initial design, and a belief stated at trial 120 is anchored there - so it
    arrives at full strength however late it is, rather than inheriting the near-flat exponent an older belief has
    already decayed to. Several priors with different anchors are what `PriorEnsemble` exists to hold.

    Parameters
    ----------
    prior : AbstractInputPrior | None, defaults to None
        The belief. `None` reads it off the configuration space the surrogate model was built over, which is how a
        prior declared on `Scenario.configspace` reaches the acquisition function.
    decay : DecaySchedule | None, defaults to None
        How the belief fades. `None` is the piBO schedule with a decay factor of one; a sensible factor is
        ``scenario.n_trials`` / 10.
    floor : float, defaults to 1e-12
        Lowest possible value of the prior, so that a configuration the user considers impossible is merely
        unattractive rather than unreachable. Without it the search could never recover from a mistaken belief.
    t0 : int | None, defaults to None
        Trial count to anchor the decay at. `None` anchors at the first update, which for a prior present from the
        start is the end of the initial design.
    discretize : bool | None, defaults to None
        Whether to coarsen the density of continuous hyperparameters. `None` decides by the surrogate model:
        random forests need it. See `discretize_pdf`.
    discrete_bins_factor : float, defaults to 10.0
        Multiple on the number of allowed levels when coarsening. The count shrinks with the decay exponent, so a
        belief that has faded is also flattened.
    name : str | None, defaults to None
        A label for the belief, for logging and for telling several of them apart.
    """

    def __init__(
        self,
        prior: AbstractInputPrior | None = None,
        decay: DecaySchedule | None = None,
        *,
        floor: float = 1e-12,
        t0: int | None = None,
        discretize: bool | None = None,
        discrete_bins_factor: float = 10.0,
        name: str | None = None,
    ) -> None:
        super().__init__(floor=floor, decay=decay if decay is not None else PolynomialDecay(beta=1.0))

        self._prior = prior
        self._t0 = t0
        self._discretize = discretize
        self._discrete_bins_factor = discrete_bins_factor
        self._name = name

    @property
    def name(self) -> str:  # noqa: D102
        return self._name if self._name is not None else self.__class__.__name__

    @property
    def meta(self) -> dict[str, Any]:  # noqa: D102
        meta = super().meta
        meta.update(
            {
                "prior": self._prior.meta if self._prior is not None else None,
                "t0": self._t0,
                "discretize": self._discretize,
                "discrete_bins_factor": self._discrete_bins_factor,
            }
        )

        return meta

    @property
    def prior(self) -> AbstractInputPrior | None:
        """The belief, once it is known."""
        return self._prior

    @property
    def t0(self) -> int | None:
        """The trial count the decay is measured from, once the weight is anchored."""
        return self._t0

    @property
    def model(self) -> AbstractModel | None:  # noqa: D102
        return self._model

    @model.setter
    def model(self, model: AbstractModel) -> None:
        self._model = model

        if self._prior is None:
            # A prior declared on the search space itself: the surrogate was built over a copy carrying the
            # distributions, so the belief can be read straight off it.
            self._prior = ConfigSpacePrior(model._configspace)

        if isinstance(model, AbstractRandomForest) and self._discretize is None:
            logger.info("Coarsening the prior density, which a random forest surrogate needs.")
            self._discretize = True

    def anchor(self, t0: int) -> None:
        """Fixes the trial count the decay is measured from.

        Refuses to move an anchor which is already set: re-anchoring a belief silently restores its full strength,
        which would keep it from ever fading.
        """
        if self._t0 is not None:
            raise ValueError(f"The prior is already anchored at trial {self._t0} and cannot be re-anchored.")

        self._t0 = t0

    def _compute_steps(self, **kwargs: Any) -> int:  # noqa: D102
        assert "num_data" in kwargs, "A prior weight needs to know how many trials have finished."
        num_data = int(kwargs["num_data"])

        if self._t0 is None:
            self._t0 = num_data

        return num_data - self._t0

    def _compute(self, X: np.ndarray) -> np.ndarray:  # noqa: D102
        assert self._prior is not None, "The prior weight has no belief and should not have been active."

        resolution: int | None = None
        if self._discretize:
            resolution = max(1, int(np.ceil(self._discrete_bins_factor * self._decay(self._steps))))

        return self._prior.pdf(X, resolution)

    def is_active(self) -> bool:
        """Whether the belief is known yet. It is not, until the surrogate model has been handed over."""
        return self._prior is not None
