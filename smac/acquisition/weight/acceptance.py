from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING, Any

import copy

import numpy as np
from ConfigSpace import Configuration, ConfigurationSpace

from smac.acquisition.function.abstract_acquisition_function import (
    AbstractAcquisitionFunction,
)
from smac.acquisition.function.confidence_bound import LCB, AbstractConfidenceBound
from smac.acquisition.weight.prior import (
    AbstractInputPrior,
    ConfigSpacePrior,
    PriorWeight,
    TabulatedPrior,
)
from smac.model.abstract_model import AbstractModel
from smac.runhistory.runhistory import RunHistory
from smac.utils.configspace import (
    convert_configurations_to_array,
    create_prior_configspace_copy,
)
from smac.utils.logging import get_logger

if TYPE_CHECKING:
    from smac.runhistory.encoder.abstract_encoder import AbstractRunHistoryEncoder

__copyright__ = "Copyright 2025, Leibniz University Hanover, Institute of AI"
__license__ = "3-clause BSD"

logger = get_logger(__name__)


class AbstractPriorAcceptancePolicy:
    """Decides whether a user belief about the optimum is plausible enough to act on.

    A belief that points somewhere bad costs trials, and the surrogate model usually has an opinion about the
    region the belief names. Checking that opinion before acting is what keeps a mistaken belief from steering
    the search, without taking the decision away from the user: the default policy accepts everything.

    A policy must accept when there is nothing to judge on - no model, or no finished trials. Rejecting a belief
    because nothing is known yet would reject exactly the beliefs stated at the start of a run, which are the
    ones with the most to contribute.
    """

    @property
    def meta(self) -> dict[str, Any]:
        """Returns the meta data of the created object."""
        return {"name": self.__class__.__name__}

    @property
    def requires_model(self) -> bool:
        """Whether the policy judges against the surrogate model.

        A policy that does is handed the model fitted to everything observed so far, together with `eta`,
        `num_data` and the runhistory encoder; one that does not is handed the model as it last was, and the model
        is not trained for it.
        """
        return False

    @abstractmethod
    def accept(
        self,
        prior: PriorWeight,
        *,
        configspace: ConfigurationSpace,
        model: AbstractModel | None,
        runhistory: RunHistory | None,
        incumbent: Configuration | None,
        rng: np.random.RandomState | None = None,
        eta: float | None = None,
        num_data: int | None = None,
        runhistory_encoder: AbstractRunHistoryEncoder | None = None,
    ) -> bool:
        """Whether the belief should be acted on.

        Parameters
        ----------
        prior : PriorWeight
            The belief, not yet anchored.
        configspace : ConfigurationSpace
            The search space.
        model : AbstractModel | None
            The surrogate model of the objective, or `None` if it has not been fitted yet.
        runhistory : RunHistory | None
            What has been observed so far.
        incumbent : Configuration | None
            The best configuration so far, or `None` if there is none yet.
        rng : np.random.RandomState | None, defaults to None
            Random state to draw samples with.
        eta : float | None, defaults to None
            The incumbent's cost as the model sees it - what an acquisition function improves on. Passed to a
            policy that `requires_model`.
        num_data : int | None, defaults to None
            The number of finished trials, as the acquisition function counts them. Passed to a policy that
            `requires_model`.
        runhistory_encoder : AbstractRunHistoryEncoder | None, defaults to None
            How costs were transformed for the model, so that what the model says can be read in the objective's
            own units. Passed to a policy that `requires_model`.
        """
        raise NotImplementedError


class AcceptAllPriors(AbstractPriorAcceptancePolicy):
    """Acts on every belief. The default: the user is in charge."""

    def accept(self, prior: PriorWeight, **kwargs: Any) -> bool:  # noqa: D102
        return True


class IncumbentComparisonPolicy(AbstractPriorAcceptancePolicy):
    r"""Rejects a belief whose region the surrogate model thinks is worse than the incumbent's neighbourhood.

    Follows the safeguard of "Dynamic Priors in Bayesian Optimization for Hyperparameter Optimization" by Lukas
    Fehring et al. [[FWS+25][FWS+25]]:

    $$\mathbb{E}_{\lambda \sim \pi}[\xi(\lambda)] - \mathbb{E}_{\lambda \sim N_{\hat{\lambda}}}[\xi(\lambda)] > \tau$$

    Configurations are drawn from the belief and from a belief-shaped neighbourhood of the current incumbent,
    both are scored with $\xi$ under the current surrogate, and the belief is accepted when its mean score beats
    the incumbent neighbourhood's by more than $\tau$.

    Note the reference distribution is the incumbent's neighbourhood, not the search space as a whole. Comparing
    against the whole space would accept almost any belief once the search has narrowed, since most of the space
    is worse than anywhere the search is still looking.

    Note also that the threshold is in the objective's units, so it has to be set for the scale of the objective
    at hand; the default comes from a benchmark whose objective lives in [0, 1]. A confidence bound is read back
    through the runhistory encoder to get there, since the model may be fitted to transformed costs - logarithms,
    under `HyperparameterOptimizationFacade`. With an encoder that cannot be inverted, or another acquisition
    function, the threshold is in the units the model works in.

    Parameters
    ----------
    n_samples : int, defaults to 100
        Number of configurations drawn from each distribution.
    n_samples_per_hyperparameter : int | None, defaults to None
        Draw this many configurations per hyperparameter of the search space instead of `n_samples`.
    threshold : float, defaults to -0.15
        How much worse than the incumbent's neighbourhood the belief's region may look and still be accepted.
        Negative, so a belief is rejected only when the model is fairly confident it is bad.
    acquisition_function : AbstractAcquisitionFunction | None, defaults to None
        Used to score the samples. `None` is a lower confidence bound, which judges a region by what it might
        deliver rather than only by its mean, so an unexplored region is not rejected for being unexplored.
    std_denominator : float, defaults to 4.0
        Width of the incumbent's neighbourhood, as a divisor of the range of each hyperparameter.
    """

    def __init__(
        self,
        *,
        n_samples: int = 100,
        n_samples_per_hyperparameter: int | None = None,
        threshold: float = -0.15,
        acquisition_function: AbstractAcquisitionFunction | None = None,
        std_denominator: float = 4.0,
    ) -> None:
        _check_sample_counts(n_samples, n_samples_per_hyperparameter)

        self._n_samples = n_samples
        self._n_samples_per_hyperparameter = n_samples_per_hyperparameter
        self._threshold = threshold
        self._acquisition_function = acquisition_function if acquisition_function is not None else LCB()
        self._std_denominator = std_denominator

    @property
    def meta(self) -> dict[str, Any]:  # noqa: D102
        meta = super().meta
        meta.update(
            {
                "n_samples": self._n_samples,
                "n_samples_per_hyperparameter": self._n_samples_per_hyperparameter,
                "threshold": self._threshold,
                "acquisition_function": self._acquisition_function.meta,
                "std_denominator": self._std_denominator,
            }
        )

        return meta

    @property
    def requires_model(self) -> bool:  # noqa: D102
        return True

    def accept(
        self,
        prior: PriorWeight,
        *,
        configspace: ConfigurationSpace,
        model: AbstractModel | None,
        runhistory: RunHistory | None,
        incumbent: Configuration | None,
        rng: np.random.RandomState | None = None,
        eta: float | None = None,
        num_data: int | None = None,
        runhistory_encoder: AbstractRunHistoryEncoder | None = None,
    ) -> bool:
        """Whether the surrogate model thinks the belief's region is worth looking at."""
        if model is None or incumbent is None or runhistory is None or len(runhistory) == 0:
            logger.debug("Nothing has been observed yet, so there is no ground to reject a prior on.")

            return True

        if prior.prior is None:
            return True

        n_samples = _sample_count(configspace, self._n_samples, self._n_samples_per_hyperparameter)
        prior_samples = prior.prior.sample(n_samples, rng)
        if prior_samples is None:
            logger.debug("The prior cannot be sampled from, so its region cannot be judged.")

            return True

        neighbourhood = create_prior_configspace_copy(
            configspace,
            dict(incumbent),
            std_denominator=self._std_denominator,
        )
        incumbent_samples = list(neighbourhood.sample_configuration(size=n_samples))

        self._acquisition_function.update(model=model, num_data=len(runhistory), eta=0.0)

        prior_scores = self._acquisition_function(prior_samples)
        incumbent_scores = self._acquisition_function(incumbent_samples)
        if isinstance(self._acquisition_function, AbstractConfidenceBound):
            prior_scores = _in_objective_units(prior_scores, runhistory_encoder)
            incumbent_scores = _in_objective_units(incumbent_scores, runhistory_encoder)

        prior_value = float(np.mean(prior_scores))
        incumbent_value = float(np.mean(incumbent_scores))
        difference = prior_value - incumbent_value

        if difference > self._threshold:
            return True

        logger.info(
            f"Rejecting a user prior: the surrogate scores its region {-difference:.4f} below the incumbent's "  # noqa: E231,E501
            f"neighbourhood, past the threshold of {-self._threshold:.4f}."  # noqa: E231
        )

        return False


class ClimbingComparisonPolicy(AbstractPriorAcceptancePolicy):
    r"""Rejects a belief when the search it leads to looks worse to the surrogate than the incumbent's region.

    `IncumbentComparisonPolicy` compares the mean score of configurations drawn from the belief with that of
    configurations drawn around the incumbent, which is fair for a belief about the whole configuration - the kind
    [[FWS+25][FWS+25]] state. A belief about only some hyperparameters leaves the others to be drawn at random, and
    their random values pull its mean down by more the more of them there are: a correct belief about one
    hyperparameter is refused for what it does not say. This policy judges a belief by where it leads instead:

    1. Draw `n_samples` configurations from the belief and as many from a normal around the incumbent. In the
       belief's draws, the hyperparameters it says nothing about come from the search space, except for a
       `neighbourhood_share` of the draws, where they come from the incumbent's neighbourhood.
    2. Rank the belief's draws by a confidence bound weighted by the belief, the neighbourhood's by the confidence
       bound alone.
    3. From the best `top_k` of each, climb with local search on the same functions, free to change every
       hyperparameter.
    4. Score where the climbs ended with the confidence bound alone, and accept the belief unless the mean of its
       side falls short of the neighbourhood's by more than `threshold`.

    Climbing optimizes the hyperparameters the belief says nothing about rather than leaving them to chance, so the
    verdict does not drift with how many of them there are. Starting some of the belief's climbs from the
    incumbent's neighbourhood gives them the informed starts the neighbourhood's own climbs have; starting the rest
    from the search space lets them find that the belief pairs with different values than the incumbent's.

    Both sides are scored by the same unweighted bound, so how the belief happens to be scaled cannot matter: it
    only decides which configurations are tried. While climbing, the belief weights the bound without the clipping
    of `WeightedAcquisitionFunction`, under which every configuration that does not improve on the incumbent would
    score zero and a climb starting among them could not move. An improvement $d$ on the incumbent is weighted as
    $d \cdot w$ and a shortfall as $d / w$, which favours the belief on both sides of zero.

    The belief is judged at the strength it would be added with. Its draws are made from copies, with a copy of
    `rng`, so judging changes neither the belief, the search space nor the search's own random state.

    Parameters
    ----------
    n_samples : int, defaults to 5000
        Number of configurations drawn for each side.
    n_samples_per_hyperparameter : int | None, defaults to None
        Draw this many configurations per hyperparameter of the search space instead of `n_samples`.
    top_k : int, defaults to 10
        Number of draws to climb from on each side, and to average over.
    neighbourhood_share : float, defaults to 0.5
        Share of the belief's draws whose other hyperparameters come from the incumbent's neighbourhood.
    threshold : float, defaults to -0.15
        How much worse than the incumbent's side the belief's side may score and still be accepted, in the
        objective's units: the scores are read back through the runhistory encoder, as `IncumbentComparisonPolicy`
        reads them. With an encoder that cannot be inverted, in the units the model works in.
    std_denominator : float, defaults to 4.0
        Width of the incumbent's neighbourhood, as a divisor of the range of each hyperparameter.
    max_steps : int | None, defaults to None
        Maximum number of steps of each climb. `None` leaves it to the local search.
    n_steps_plateau_walk : int, defaults to 10
        Number of steps a climb walks along a plateau before it stops.
    """

    def __init__(
        self,
        *,
        n_samples: int = 5000,
        n_samples_per_hyperparameter: int | None = None,
        top_k: int = 10,
        neighbourhood_share: float = 0.5,
        threshold: float = -0.15,
        std_denominator: float = 4.0,
        max_steps: int | None = None,
        n_steps_plateau_walk: int = 10,
    ) -> None:
        _check_sample_counts(n_samples, n_samples_per_hyperparameter)
        if top_k < 1:
            raise ValueError(f"At least one climb is needed, got top_k={top_k}.")
        if not 0.0 <= neighbourhood_share <= 1.0:
            raise ValueError(f"The neighbourhood share must be in [0, 1], got {neighbourhood_share}.")

        self._n_samples = n_samples
        self._n_samples_per_hyperparameter = n_samples_per_hyperparameter
        self._top_k = top_k
        self._neighbourhood_share = neighbourhood_share
        self._threshold = threshold
        self._std_denominator = std_denominator
        self._max_steps = max_steps
        self._n_steps_plateau_walk = n_steps_plateau_walk

    @property
    def meta(self) -> dict[str, Any]:  # noqa: D102
        meta = super().meta
        meta.update(
            {
                "n_samples": self._n_samples,
                "n_samples_per_hyperparameter": self._n_samples_per_hyperparameter,
                "top_k": self._top_k,
                "neighbourhood_share": self._neighbourhood_share,
                "threshold": self._threshold,
                "std_denominator": self._std_denominator,
                "max_steps": self._max_steps,
                "n_steps_plateau_walk": self._n_steps_plateau_walk,
            }
        )

        return meta

    @property
    def requires_model(self) -> bool:  # noqa: D102
        return True

    def accept(
        self,
        prior: PriorWeight,
        *,
        configspace: ConfigurationSpace,
        model: AbstractModel | None,
        runhistory: RunHistory | None,
        incumbent: Configuration | None,
        rng: np.random.RandomState | None = None,
        eta: float | None = None,
        num_data: int | None = None,
        runhistory_encoder: AbstractRunHistoryEncoder | None = None,
    ) -> bool:
        """Whether the search the belief leads to looks good enough to the surrogate."""
        scores = self.compare(
            prior,
            configspace=configspace,
            model=model,
            runhistory=runhistory,
            incumbent=incumbent,
            rng=rng,
            eta=eta,
            num_data=num_data,
            runhistory_encoder=runhistory_encoder,
        )
        if scores is None:
            return True

        prior_value, incumbent_value = scores
        difference = prior_value - incumbent_value

        if difference > self._threshold:
            return True

        logger.info(
            f"Rejecting a user prior: the surrogate scores where it leads {-difference:.4f} below where the "  # noqa: E231,E501
            f"incumbent's neighbourhood leads, past the threshold of {-self._threshold:.4f}."  # noqa: E231
        )

        return False

    def compare(
        self,
        prior: PriorWeight,
        *,
        configspace: ConfigurationSpace,
        model: AbstractModel | None,
        runhistory: RunHistory | None,
        incumbent: Configuration | None,
        rng: np.random.RandomState | None = None,
        eta: float | None = None,
        num_data: int | None = None,
        runhistory_encoder: AbstractRunHistoryEncoder | None = None,
    ) -> tuple[float, float] | None:
        """Scores both sides: the mean confidence bound where the belief's climbs ended, and where the
        neighbourhood's did, in the objective's units where the encoder allows. Higher is better.

        Takes the arguments of `accept`. Without `eta`, the incumbent's cost is read off the model's prediction;
        without `num_data`, the number of entries in the runhistory is used.

        Returns
        -------
        tuple[float, float] | None
            The belief's score and the neighbourhood's, or `None` when there is nothing to judge on: no model, no
            observations, or a belief that cannot be sampled from.
        """
        from smac.acquisition.maximizer.local_search import LocalSearch

        if model is None or incumbent is None or runhistory is None or len(runhistory) == 0:
            logger.debug("Nothing has been observed yet, so there is no ground to reject a prior on.")

            return None

        if prior.prior is None:
            return None

        random = copy.deepcopy(rng) if rng is not None else np.random.RandomState()
        n_samples = _sample_count(configspace, self._n_samples, self._n_samples_per_hyperparameter)

        believed = _draw(prior.prior, n_samples, random)
        if believed is None:
            logger.debug("The prior cannot be sampled from, so where it leads cannot be judged.")

            return None

        neighbourhood = create_prior_configspace_copy(
            configspace,
            dict(incumbent),
            std_denominator=self._std_denominator,
        )
        neighbourhood.seed(_seed(random))
        nearby = _in_space(configspace, _sample(neighbourhood, n_samples))
        believed = self._pool(
            _in_space(configspace, believed),
            _unstated(prior.prior, configspace),
            configspace,
            neighbourhood,
        )
        if len(believed) == 0 or len(nearby) == 0:
            return None

        bound = LCB()
        bound.update(
            model=model,
            num_data=num_data if num_data is not None else len(runhistory),
            eta=0.0,
        )
        if eta is None:
            means, _ = model.predict_marginalized(convert_configurations_to_array([incumbent]))
            eta = float(means[0, 0])

        weighted = _BeliefWeightedBound(bound, prior, eta)

        def climb(draws: list[Configuration], function: AbstractAcquisitionFunction) -> list[Configuration]:
            values = np.asarray(function(draws), dtype=float).reshape(-1)
            best = np.lexsort((random.random_sample(len(values)), -values))[: self._top_k]
            search = LocalSearch(
                configspace,
                function,
                max_steps=self._max_steps,
                n_steps_plateau_walk=self._n_steps_plateau_walk,
                seed=_seed(random),
            )

            return [config for _, config in search.climb([draws[i] for i in best])]

        prior_value = float(np.mean(_in_objective_units(bound(climb(believed, weighted)), runhistory_encoder)))
        incumbent_value = float(np.mean(_in_objective_units(bound(climb(nearby, bound)), runhistory_encoder)))

        return prior_value, incumbent_value

    def _pool(
        self,
        believed: list[Configuration],
        unstated: set[str],
        configspace: ConfigurationSpace,
        neighbourhood: ConfigurationSpace,
    ) -> list[Configuration]:
        """The belief's draws, a `neighbourhood_share` of them with their unstated hyperparameters replaced by
        values drawn around the incumbent. A replacement the search space does not admit - a condition the mixed
        values do not satisfy - keeps the draw as it was.
        """
        n_pooled = int(round(self._neighbourhood_share * len(believed)))
        if n_pooled == 0 or len(unstated) == 0:
            return believed

        replacements = _sample(neighbourhood, n_pooled)
        pooled = list(believed)
        for i, replacement in enumerate(replacements):
            values = dict(believed[i])
            for name in unstated:
                if name in replacement:
                    values[name] = replacement[name]
                else:
                    values.pop(name, None)
            try:
                pooled[i] = Configuration(configspace, values=values)
            except Exception:  # noqa: BLE001
                continue

        return pooled


class _BeliefWeightedBound(AbstractAcquisitionFunction):
    r"""A confidence bound weighted by a belief, for `ClimbingComparisonPolicy` to climb on.

    Order-equivalent to $d \cdot w$ where the bound improves on the incumbent by $d \geq 0$, and to $d / w$ where it
    falls short. Computed from the logarithms of $|d|$ and $w$, mapped monotonically onto $(-1, 1)$, so that a sharp
    belief at full strength neither underflows nor overflows; only the order of the values is used.
    """

    def __init__(self, bound: AbstractAcquisitionFunction, weight: PriorWeight, eta: float) -> None:
        super().__init__()
        self._bound = bound
        self._weight = weight
        self._eta = eta
        self._model = bound.model

    @property
    def name(self) -> str:  # noqa: D102
        return f"Belief-weighted {self._bound.name}"

    def _compute(self, X: np.ndarray) -> np.ndarray:
        if len(X.shape) == 1:
            X = X[:, np.newaxis]

        improvement = np.asarray(self._bound._compute(X), dtype=float).reshape((-1, 1)) + self._eta
        log_weight = np.asarray(self._weight(X, log=True), dtype=float).reshape((-1, 1))

        with np.errstate(divide="ignore"):
            log_magnitude = np.log(np.abs(improvement))

        level = np.where(improvement >= 0, log_magnitude + log_weight, log_magnitude - log_weight)
        ranked = np.arctan(level) / np.pi + 0.5

        return np.where(improvement > 0, ranked, np.where(improvement < 0, -ranked, 0.0))


def _check_sample_counts(n_samples: int, n_samples_per_hyperparameter: int | None) -> None:
    if n_samples < 1:
        raise ValueError(f"At least one sample is needed, got {n_samples}.")

    if n_samples_per_hyperparameter is not None and n_samples_per_hyperparameter < 1:
        raise ValueError(f"At least one sample per hyperparameter is needed, got {n_samples_per_hyperparameter}.")


def _sample_count(configspace: ConfigurationSpace, n_samples: int, n_samples_per_hyperparameter: int | None) -> int:
    """`n_samples`, or `n_samples_per_hyperparameter` times the number of hyperparameters when that is given."""
    if n_samples_per_hyperparameter is None:
        return n_samples

    return n_samples_per_hyperparameter * max(1, len(list(configspace.keys())))


def _seed(random: np.random.RandomState) -> int:
    return int(random.randint(0, 2**31 - 1))


def _sample(configspace: ConfigurationSpace, n: int) -> list[Configuration]:
    drawn = configspace.sample_configuration(size=n)

    return [drawn] if isinstance(drawn, Configuration) else list(drawn)


def _draw(belief: AbstractInputPrior, n: int, random: np.random.RandomState) -> list[Configuration] | None:
    """Draws from a copy of the belief, its search space seeded from `random`, so that the draws are
    reproducible and the belief's own search space - usually the scenario's - keeps its random state.
    """
    belief = copy.deepcopy(belief)
    space = getattr(belief, "_configspace", None)
    if space is not None and hasattr(space, "seed"):
        space.seed(_seed(random))

    return belief.sample(n, random)


def _in_space(configspace: ConfigurationSpace, configurations: list[Configuration]) -> list[Configuration]:
    """The configurations as members of `configspace`, which neighbours are generated from and the model
    reads; one it does not admit is dropped.
    """
    out = []
    for configuration in configurations:
        try:
            out.append(Configuration(configspace, values=dict(configuration)))
        except Exception:  # noqa: BLE001
            continue

    return out


def _unstated(belief: AbstractInputPrior, configspace: ConfigurationSpace) -> set[str]:
    """The hyperparameters the belief says nothing about.

    A tabulated belief says something about exactly the hyperparameters it tabulates; a configuration space
    belief about those whose distribution differs from the search space's. Any other belief is taken to speak
    for every hyperparameter.
    """
    names = set(configspace.keys())
    if isinstance(belief, TabulatedPrior):
        return names - set(belief.tables)

    if isinstance(belief, ConfigSpacePrior):
        space = belief.configspace

        return {name for name in names if name in space and space[name] == configspace[name]}

    return set()


def _in_objective_units(scores: np.ndarray, encoder: AbstractRunHistoryEncoder | None) -> np.ndarray:
    """Confidence bound scores in the objective's units, still higher-is-better.

    A confidence bound scores a configuration by the negated bound on its cost as the model sees it, so the bound is
    mapped back through the encoder and negated again. Returned as given when there is no encoder or it cannot be
    inverted, with a note in the log, since the comparison is then in the model's units.
    """
    scores = np.asarray(scores, dtype=float).reshape((-1, 1))
    if encoder is None:
        return scores

    costs = encoder.inverse_transform_response_values(-scores)
    if costs is None:
        logger.debug(
            f"{encoder.__class__.__name__} cannot be inverted, so the prior is judged in the units the model works in."
        )

        return scores

    return -np.asarray(costs, dtype=float).reshape((-1, 1))
