import collections
from enum import Enum
import json
from typing import List, Dict, Union, Optional, Any, Type, Iterable, cast, Tuple

import numpy as np

from smac.configspace import Configuration, ConfigurationSpace
from smac.tae import StatusType
from smac.utils.logging import PickableLoggerAdapter
from smac.utils.multi_objective import normalize_costs


__author__ = "Marius Lindauer"
__copyright__ = "Copyright 2015, ML4AAD"
__license__ = "3-clause BSD"
__maintainer__ = "Marius Lindauer"
__email__ = "lindauer@cs.uni-freiburg.de"
__version__ = "0.0.1"


# NOTE class instead of collection to have a default value for budget in RunKey
class RunKey(
    collections.namedtuple("RunKey", ["config_id", "instance_id", "seed", "budget"])
):
    __slots__ = ()

    def __new__(
        cls,  # No type annotation because the 1st argument for a namedtuble is always the class type,
        # see https://docs.python.org/3/reference/datamodel.html#object.__new__
        config_id: int,
        instance_id: Optional[str],
        seed: Optional[int],
        budget: float = 0.0,
    ) -> "RunKey":
        return super().__new__(cls, config_id, instance_id, seed, budget)


# NOTE class instead of collection to have a default value for budget/source_id in RunInfo
class RunInfo(
    collections.namedtuple(
        "RunInfo",
        [
            "config",
            "instance",
            "instance_specific",
            "seed",
            "cutoff",
            "capped",
            "budget",
            "source_id",
        ],
    )
):
    __slots__ = ()

    def __new__(
        cls,  # No type annotation because the 1st argument for a namedtuble is always the class type,
        # see https://docs.python.org/3/reference/datamodel.html#object.__new__
        config: Configuration,
        instance: Optional[str],
        instance_specific: str,
        seed: int,
        cutoff: Optional[float],
        capped: bool,
        budget: float = 0.0,
        # In the context of parallel runs, one will have multiple suppliers of
        # configurations. source_id is a new mechanism to track what entity launched
        # this configuration
        source_id: int = 0,
    ) -> "RunInfo":
        return super().__new__(
            cls,
            config,
            instance,
            instance_specific,
            seed,
            cutoff,
            capped,
            budget,
            source_id,
        )


InstSeedKey = collections.namedtuple("InstSeedKey", ["instance", "seed"])

InstSeedBudgetKey = collections.namedtuple(
    "InstSeedBudgetKey", ["instance", "seed", "budget"]
)

RunValue = collections.namedtuple(
    "RunValue", ["cost", "time", "status", "starttime", "endtime", "additional_info"]
)


class EnumEncoder(json.JSONEncoder):
    """Custom encoder for enum-serialization
    (implemented for StatusType from tae).
    Using encoder implied using object_hook as defined in StatusType
    to deserialize from json.
    """

    def default(self, obj: object) -> Any:
        if isinstance(obj, StatusType):
            return {"__enum__": str(obj)}
        return json.JSONEncoder.default(self, obj)


class DataOrigin(Enum):
    """
    Definition of how data in the runhistory is used.

    * ``INTERNAL``: internal data which was gathered during the current
      optimization run. It will be saved to disk, used for building EPMs and
      during intensify.
    * ``EXTERNAL_SAME_INSTANCES``: external data, which was gathered by running
       another program on the same instances as the current optimization run
       runs on (for example pSMAC). It will not be saved to disk, but used both
       for EPM building and during intensify.
    * ``EXTERNAL_DIFFERENT_INSTANCES``: external data, which was gathered on a
       different instance set as the one currently used, but due to having the
       same instance features can still provide useful information. Will not be
       saved to disk and only used for EPM building.
    """

    INTERNAL = 1
    EXTERNAL_SAME_INSTANCES = 2
    EXTERNAL_DIFFERENT_INSTANCES = 3


class RunHistory(object):
    """Container for target algorithm run information.

    Most importantly, the runhistory contains an efficient mapping from each evaluated configuration to the
    empirical cost observed on either the full instance set or a subset. The cost is the average over all
    observed costs for one configuration:

    * If using budgets for a single instance, only the cost on the highest observed budget is returned.
    * If using instances as the budget, the average cost over all evaluated instances is returned.
    * Theoretically, the runhistory object can handle instances and budgets at the same time. This is
      neither used nor tested.
    * Capped runs are not included in this cost.

    Note
    ----
    Guaranteed to be picklable.

    Parameters
    ----------
    overwrite_existing_runs : bool (default=False)
        If set to ``True`` and a run of a configuration on an instance-budget-seed-pair already exists,
        it is overwritten. Allows to overwrites old results if pairs of algorithm-instance-seed were measured
        multiple times

    Attributes
    ----------
    data : collections.OrderedDict()
        Internal data representation
    config_ids : dict
        Maps config -> id
    ids_config : dict
        Maps id -> config
    num_runs_per_config : dict
        Maps config_id -> number of runs

    """

    def __init__(
        self,
        overwrite_existing_runs: bool = False,
    ) -> None:
        self.logger = PickableLoggerAdapter(
            self.__module__ + "." + self.__class__.__name__
        )

        # By having the data in a deterministic order we can do useful tests
        # when we serialize the data and can assume it's still in the same
        # order as it was added.
        self.data = collections.OrderedDict()  # type: Dict[RunKey, RunValue]

        # for fast access, we have also an unordered data structure
        # to get all instance seed pairs of a configuration.
        # This does not include capped runs.
        self._configid_to_inst_seed_budget = (
            {}
        )  # type: Dict[int, Dict[InstSeedKey, List[float]]]

        self.config_ids = {}  # type: Dict[Configuration, int]
        self.ids_config = {}  # type: Dict[int, Configuration]
        self._n_id = 0

        # Stores cost for each configuration ID
        self._cost_per_config = {}  # type: Dict[int, np.ndarray]
        # Stores min cost across all budgets for each configuration ID
        self._min_cost_per_config = {}  # type: Dict[int, np.ndarray]
        # runs_per_config maps the configuration ID to the number of runs for that configuration
        # and is necessary for computing the moving average
        self.num_runs_per_config = {}  # type: Dict[int, int]

        # Store whether a datapoint is "external", which means it was read from
        # a JSON file. Can be chosen to not be written to disk
        self.external = {}  # type: Dict[RunKey, DataOrigin]

        self.overwrite_existing_runs = overwrite_existing_runs
        self.num_obj = -1  # type: int
        self.objective_bounds = []  # type: List[Tuple[float, float]]

    def add(
        self,
        config: Configuration,
        cost: Union[int, float, list, np.ndarray],
        time: float,
        status: StatusType,
        instance_id: Optional[str] = None,
        seed: Optional[int] = None,
        budget: float = 0.0,
        starttime: float = 0.0,
        endtime: float = 0.0,
        additional_info: Optional[Dict] = None,
        origin: DataOrigin = DataOrigin.INTERNAL,
        force_update: bool = False,
    ) -> None:
        """Adds a data of a new target algorithm (TA) run;
        it will update data if the same key values are used
        (config, instance_id, seed)

        Parameters
        ----------
            config : dict (or other type -- depending on config space module)
                Parameter configuration
            cost: Union[int, float, list, np.ndarray]
                Cost of TA run (will be minimized)
            time: float
                Runtime of TA run
            status: str
                Status in {SUCCESS, TIMEOUT, CRASHED, ABORT, MEMOUT}
            instance_id: str
                String representing an instance (default: None)
            seed: int
                Random seed used by TA (default: None)
            budget: float
                budget (cutoff) used in intensifier to limit TA (default: 0)
            starttime: float
                starting timestamp of TA evaluation
            endtime: float
                ending timestamp of TA evaluation
            additional_info: dict
                Additional run infos (could include further returned
                information from TA or fields such as start time and host_id)
            origin: DataOrigin
                Defines how data will be used.
            force_update: bool (default: False)
                Forces the addition of a config to the history
        """

        if config is None:
            raise TypeError("Configuration to add to the runhistory must not be None")
        elif not isinstance(config, Configuration):
            raise TypeError(
                "Configuration to add to the runhistory is not of type Configuration, but %s"
                % type(config)
            )

        # Squeeze is important to reduce arrays with one element
        # to scalars.
        cost = np.asarray(cost).squeeze()

        # Get the config id
        config_id_tmp = self.config_ids.get(config)
        if config_id_tmp is None:
            self._n_id += 1
            self.config_ids[config] = self._n_id
            config_id = cast(int, self.config_ids.get(config))
            self.ids_config[self._n_id] = config
        else:
            config_id = cast(int, config_id_tmp)

        if self.num_obj == -1:
            self.num_obj = np.size(cost)
        else:
            if np.size(cost) != self.num_obj:
                raise ValueError(
                    f"Cost is not of the same length ({np.size(cost)}) as the number "
                    f"of objectives ({self.num_obj})"
                )

        k = RunKey(config_id, instance_id, seed, budget)
        v = RunValue(cost.tolist(), time, status, starttime, endtime, additional_info)

        # Construct keys and values for the data dictionary
        for key, value in (
            ("config", config.get_dictionary()),
            ("config_id", config_id),
            ("instance_id", instance_id),
            ("seed", seed),
            ("budget", budget),
            ("cost", cost.tolist()),
            ("time", time),
            ("status", status),
            ("starttime", starttime),
            ("endtime", endtime),
            ("additional_info", additional_info),
            ("origin", config.origin),
        ):
            self._check_json_serializable(key, value, EnumEncoder, k, v)

        # Each runkey is supposed to be used only once. Repeated tries to add
        # the same runkey will be ignored silently if not capped.
        if self.overwrite_existing_runs or force_update or self.data.get(k) is None:
            self._add(k, v, status, origin)
        elif status != StatusType.CAPPED and self.data[k].status == StatusType.CAPPED:
            # overwrite capped runs with uncapped runs
            self._add(k, v, status, origin)
        elif (
            status == StatusType.CAPPED
            and self.data[k].status == StatusType.CAPPED
            and cost > self.data[k].cost
        ):
            # overwrite if censored with a larger cutoff
            self._add(k, v, status, origin)

    def _check_json_serializable(
        self,
        key: str,
        obj: Any,
        encoder: Type[json.JSONEncoder],
        runkey: RunKey,
        runvalue: RunValue,
    ) -> None:
        try:
            json.dumps(obj, cls=encoder)
        except Exception as e:
            raise ValueError(
                "Cannot add %s: %s of type %s to runhistory because it raises an error during JSON encoding, "
                "please see the error above.\nRunKey: %s\nRunValue %s"
                % (key, str(obj), type(obj), runkey, runvalue)
            ) from e

    def _update_objective_bounds(self) -> None:
        """Update the objective bounds based on the data in the runhistory."""

        all_costs = []
        for (costs, _, status, _, _, _) in self.data.values():
            if status == StatusType.SUCCESS:
                if not isinstance(costs, Iterable):
                    costs = [costs]

                assert len(costs) == self.num_obj
                all_costs.append(costs)

        all_costs = np.array(all_costs, dtype=float)

        if len(all_costs) == 0:
            self.objective_bounds = [(np.inf, -np.inf)] * self.num_obj
            return

        min_values = np.min(all_costs, axis=0)
        max_values = np.max(all_costs, axis=0)

        self.objective_bounds = []
        for min_v, max_v in zip(min_values, max_values):
            self.objective_bounds += [(min_v, max_v)]

    def _add(
        self, k: RunKey, v: RunValue, status: StatusType, origin: DataOrigin
    ) -> None:
        """
        Actual function to add new entry to data structures.
        """
        self.data[k] = v
        self.external[k] = origin

        # Update objective bounds
        self._update_objective_bounds()

        # Capped data is added above
        # Do not register the cost until the run has completed
        if origin in (
            DataOrigin.INTERNAL,
            DataOrigin.EXTERNAL_SAME_INSTANCES,
        ) and status not in [StatusType.CAPPED, StatusType.RUNNING]:
            # also add to fast data structure
            is_k = InstSeedKey(k.instance_id, k.seed)
            self._configid_to_inst_seed_budget[
                k.config_id
            ] = self._configid_to_inst_seed_budget.get(k.config_id, {})
            if is_k not in self._configid_to_inst_seed_budget[k.config_id].keys():
                # add new inst-seed-key with budget to main dict
                self._configid_to_inst_seed_budget[k.config_id][is_k] = [k.budget]
            elif k.budget not in is_k:
                # append new budget to existing inst-seed-key dict
                self._configid_to_inst_seed_budget[k.config_id][is_k].append(k.budget)

            # if budget is used, then update cost instead of incremental updates
            if not self.overwrite_existing_runs and k.budget == 0:
                # assumes an average across runs as cost function aggregation, this is used for algorithm configuration
                # (incremental updates are used to save time as getting the cost for > 100 instances is high)
                self.incremental_update_cost(self.ids_config[k.config_id], v.cost)
            else:
                # this is when budget > 0 (only successive halving and hyperband so far)
                self.update_cost(config=self.ids_config[k.config_id])
                if k.budget > 0:
                    if (
                        self.num_runs_per_config[k.config_id] != 1
                    ):  # This is updated in update_cost
                        raise ValueError("This should not happen!")

    def update_cost(self, config: Configuration) -> None:
        """Store the performance of a configuration across the instances in
        self.cost_per_config and also updates self.runs_per_config;

        Note
        ----
        This method ignores capped runs.

        Parameters
        ----------
        config: Configuration
            configuration to update cost based on all runs in runhistory
        """
        config_id = self.config_ids[config]
        # removing duplicates while keeping the order
        inst_seed_budgets = list(
            dict.fromkeys(
                self.get_runs_for_config(config, only_max_observed_budget=True)
            )
        )
        self._cost_per_config[config_id] = self.average_cost(config, inst_seed_budgets)
        self.num_runs_per_config[config_id] = len(inst_seed_budgets)

        all_inst_seed_budgets = list(
            dict.fromkeys(
                self.get_runs_for_config(config, only_max_observed_budget=False)
            )
        )
        self._min_cost_per_config[config_id] = self.min_cost(
            config, all_inst_seed_budgets
        )

    def incremental_update_cost(
        self, config: Configuration, cost: Union[np.ndarray, list, float, int]
    ) -> None:
        """Incrementally updates the performance of a configuration by using a
        moving average;

        Parameters
        ----------
        config: Configuration
            configuration to update cost based on all runs in runhistory
        cost: float
            cost of new run of config
        """

        config_id = self.config_ids[config]
        n_runs = self.num_runs_per_config.get(config_id, 0)
        old_cost = self._cost_per_config.get(config_id, 0.0)

        if self.num_obj > 1:
            cost = self.average_cost(config)

        self._cost_per_config[config_id] = ((old_cost * n_runs) + cost) / (n_runs + 1)
        self.num_runs_per_config[config_id] = n_runs + 1

    def get_cost(self, config: Configuration) -> float:
        """Returns empirical cost for a configuration.

        See the class docstring for how the costs are computed. The costs are not re-computed, but are read from cache.

        Parameters
        ----------
        config: Configuration

        Returns
        -------
        cost: float
            Computed cost for configuration
        """
        config_id = self.config_ids.get(config)
        return self._cost_per_config.get(config_id, np.nan)  # type: ignore[arg-type] # noqa F821

    def get_runs_for_config(
        self, config: Configuration, only_max_observed_budget: bool
    ) -> List[InstSeedBudgetKey]:
        """Return all runs (instance seed pairs) for a configuration.

        Note
        ----
        This method ignores capped runs.

        Parameters
        ----------
        config : Configuration from ConfigSpace
            Parameter configuration
        only_max_observed_budget : bool
            Select only the maximally observed budget run for this configuration
        Returns
        -------
        instance_seed_budget_pairs : list<tuples of instance, seed, budget>
        """
        config_id = self.config_ids.get(config)
        runs = self._configid_to_inst_seed_budget.get(config_id, {}).copy()  # type: ignore[arg-type] # noqa F821

        # select only the max budget run if specified
        if only_max_observed_budget:
            for k, v in runs.items():
                runs[k] = [max(v)]

        # convert to inst-seed-budget key
        rval = [
            InstSeedBudgetKey(k.instance, k.seed, budget)
            for k, v in runs.items()
            for budget in v
        ]
        return rval

    def get_all_configs(self) -> List[Configuration]:
        """Return all configurations in this RunHistory object

        Returns
        -------
            parameter configurations: list
        """
        return list(self.config_ids.keys())

    def get_all_configs_per_budget(
        self,
        budget_subset: Optional[List] = None,
    ) -> List[Configuration]:
        """
        Return all configs in this RunHistory object that have been run on one of these budgets

        Parameters
        ----------
            budget_subset: list

        Returns
        -------
            parameter configurations: list
        """
        if budget_subset is None:
            return self.get_all_configs()
        configs = []
        for c, i, s, b in self.data.keys():
            if b in budget_subset:
                configs.append(self.ids_config[c])
        return configs

    def get_min_cost(self, config: Configuration) -> float:
        """Returns the lowest empirical cost for a configuration, across all runs (budgets)

        See the class docstring for how the costs are computed. The costs are not re-computed, but are read from cache.

        Parameters
        ----------
        config: Configuration

        Returns
        -------
        min_cost: float
            Computed cost for configuration
        """
        config_id = self.config_ids.get(config)
        return self._min_cost_per_config.get(config_id, np.nan)  # type: ignore[arg-type] # noqa F821

    def empty(self) -> bool:
        """Check whether or not the RunHistory is empty.

        Returns
        -------
        emptiness: bool
            True if runs have been added to the RunHistory,
            False otherwise
        """
        return len(self.data) == 0

    def save_json(
        self, fn: str = "runhistory.json", save_external: bool = False
    ) -> None:
        """
        saves runhistory on disk

        Parameters
        ----------
        fn : str
            file name
        save_external : bool
            Whether to save external data in the runhistory file.
        """

        data = [
            (
                [
                    int(k.config_id),
                    str(k.instance_id) if k.instance_id is not None else None,
                    int(k.seed),
                    float(k.budget) if k[3] is not None else 0,
                ],
                [v.cost, v.time, v.status, v.starttime, v.endtime, v.additional_info],
            )
            for k, v in self.data.items()
            if save_external or self.external[k] == DataOrigin.INTERNAL
        ]
        config_ids_to_serialize = set([entry[0][0] for entry in data])
        configs = {
            id_: conf.get_dictionary()
            for id_, conf in self.ids_config.items()
            if id_ in config_ids_to_serialize
        }
        config_origins = {
            id_: conf.origin
            for id_, conf in self.ids_config.items()
            if (id_ in config_ids_to_serialize and conf.origin is not None)
        }

        with open(fn, "w") as fp:
            json.dump(
                {"data": data, "config_origins": config_origins, "configs": configs},
                fp,
                cls=EnumEncoder,
                indent=2,
            )

    def load_json(self, fn: str, cs: ConfigurationSpace) -> None:
        """Load and runhistory in json representation from disk.

        Overwrites current runhistory!

        Parameters
        ----------
        fn : str
            file name to load from
        cs : ConfigSpace
            instance of configuration space
        """
        try:
            with open(fn) as fp:
                all_data = json.load(fp, object_hook=StatusType.enum_hook)
        except Exception as e:
            self.logger.warning(
                "Encountered exception %s while reading runhistory from %s. "
                "Not adding any runs!",
                e,
                fn,
            )
            return

        config_origins = all_data.get("config_origins", {})

        self.ids_config = {
            int(id_): Configuration(
                cs, values=values, origin=config_origins.get(id_, None)
            )
            for id_, values in all_data["configs"].items()
        }

        self.config_ids = {config: id_ for id_, config in self.ids_config.items()}
        self._n_id = len(self.config_ids)

        # important to use add method to use all data structure correctly
        for k, v in all_data["data"]:
            # Set num_obj first
            if self.num_obj == -1:
                if isinstance(v[0], float) or isinstance(v[0], int):
                    self.num_obj = 1
                else:
                    self.num_obj = len(np.asarray(list(map(float, v[0]))))

            if self.num_obj == 1:
                cost = float(v[0])
            else:
                cost = np.asarray(list(map(float, v[0])))

            self.add(
                config=self.ids_config[int(k[0])],
                cost=cost,
                time=float(v[1]),
                status=StatusType(v[2]),
                instance_id=k[1],
                seed=int(k[2]),
                budget=float(k[3]) if len(k) == 4 else 0,
                starttime=v[3],
                endtime=v[4],
                additional_info=v[5],
            )

    def update_from_json(
        self,
        fn: str,
        cs: ConfigurationSpace,
        origin: DataOrigin = DataOrigin.EXTERNAL_SAME_INSTANCES,
    ) -> None:
        """Update the current runhistory by adding new runs from a json file.

        Parameters
        ----------
        fn : str
            File name to load from.
        cs : ConfigSpace
            Instance of configuration space.
        origin : DataOrigin
            What to store as data origin.
        """
        new_runhistory = RunHistory()
        new_runhistory.load_json(fn, cs)
        self.update(runhistory=new_runhistory, origin=origin)

    def update(
        self,
        runhistory: "RunHistory",
        origin: DataOrigin = DataOrigin.EXTERNAL_SAME_INSTANCES,
    ) -> None:
        """Update the current runhistory by adding new runs from a RunHistory.

        Parameters
        ----------
        runhistory: RunHistory
            Runhistory with additional data to be added to self
        origin: DataOrigin
            If set to ``INTERNAL`` or ``EXTERNAL_FULL`` the data will be
            added to the internal data structure self._configid_to_inst_seed_budget
            and be available :meth:`through get_runs_for_config`.
        """

        # Configurations might be already known, but by a different ID. This
        # does not matter here because the add() method handles this
        # correctly by assigning an ID to unknown configurations and re-using
        #  the ID
        for key, value in runhistory.data.items():
            config_id, instance_id, seed, budget = key
            cost, time, status, start, end, additional_info = value
            config = runhistory.ids_config[config_id]
            self.add(
                config=config,
                cost=cost,
                time=time,
                status=status,
                instance_id=instance_id,
                starttime=start,
                endtime=end,
                seed=seed,
                budget=budget,
                additional_info=additional_info,
                origin=origin,
            )

    def _cost(
        self,
        config: Configuration,
        instance_seed_budget_keys: Optional[Iterable[InstSeedBudgetKey]] = None,
    ) -> List[np.ndarray]:
        """Return array of all costs for the given config for further calculations.

        Parameters
        ----------
        config : Configuration
            Configuration to calculate objective for
        instance_seed_budget_keys : list, optional (default=None)
            List of tuples of instance-seeds-budget keys. If None, the run_history is
            queried for all runs of the given configuration.

        Returns
        -------
        Costs: list
            Array of all costs
        """
        try:
            id_ = self.config_ids[config]
        except KeyError:  # challenger was not running so far
            return []

        if instance_seed_budget_keys is None:
            instance_seed_budget_keys = self.get_runs_for_config(
                config, only_max_observed_budget=True
            )

        costs = []
        for i, r, b in instance_seed_budget_keys:
            k = RunKey(id_, i, r, b)
            costs.append(self.data[k].cost)

        return costs

    def average_cost(
        self,
        config: Configuration,
        instance_seed_budget_keys: Optional[Iterable[InstSeedBudgetKey]] = None,
    ) -> float:
        """Return the average cost of a configuration.

        This is the mean of costs of all instance-seed pairs.

        Parameters
        ----------
        config : Configuration
            Configuration to calculate objective for
        instance_seed_budget_keys : list, optional (default=None)
            List of tuples of instance-seeds-budget keys. If None, the run_history is
            queried for all runs of the given configuration.

        Returns
        ----------
        Cost: float
            Average cost
        """

        costs = self._cost(config, instance_seed_budget_keys)
        if costs:
            if self.num_obj > 1:
                # Normalize costs
                costs = normalize_costs(costs, self.objective_bounds)

            return float(np.mean(costs))

        return np.nan

    def sum_cost(
        self,
        config: Configuration,
        instance_seed_budget_keys: Optional[Iterable[InstSeedBudgetKey]] = None,
    ) -> float:
        """Return the sum of costs of a configuration.

        This is the sum of costs of all instance-seed pairs.

        Parameters
        ----------
        config : Configuration
            Configuration to calculate objective for
        instance_seed_budget_keys : list, optional (default=None)
            List of tuples of instance-seeds-budget keys. If None, the run_history is
            queried for all runs of the given configuration.

        Returns
        ----------
        sum_cost: float
            Sum of costs of config
        """
        costs = self._cost(config, instance_seed_budget_keys)
        if costs:
            if self.num_obj > 1:
                # Normalize costs
                costs = normalize_costs(costs, self.objective_bounds)
                costs = np.mean(costs, axis=1)

        return float(np.sum(costs))

    def min_cost(
        self,
        config: Configuration,
        instance_seed_budget_keys: Optional[Iterable[InstSeedBudgetKey]] = None,
    ) -> float:
        """Return the minimum cost of a configuration

        This is the minimum cost of all instance-seed pairs.
        Warning: In the case of multi-fidelity, the minimum cost per objectives is returned.

        Parameters
        ----------
        config : Configuration
            Configuration to calculate objective for
        instance_seed_budget_keys : list, optional (default=None)
            List of tuples of instance-seeds-budget keys. If None, the run_history is
            queried for all runs of the given configuration.

        Returns
        ----------
        min_cost: float
            minimum cost of config
        """
        costs = self._cost(config, instance_seed_budget_keys)
        if costs:
            if self.num_obj > 1:
                # Normalize costs
                costs = normalize_costs(costs, self.objective_bounds)
                costs = np.mean(costs, axis=1)

            return float(np.min(costs))

        return np.nan

    def compute_all_costs(self, instances: Optional[List[str]] = None) -> None:
        """Computes the cost of all configurations from scratch and overwrites
        self.cost_perf_config and self.runs_per_config accordingly;

        Note
        ----
        This method is only used for ``merge_foreign_data`` and should be removed.

        Parameters
        ----------
        instances: List[str]
            list of instances; if given, cost is only computed wrt to this instance set
        """
        self._cost_per_config = {}
        self.num_runs_per_config = {}
        for config, config_id in self.config_ids.items():
            # removing duplicates while keeping the order
            inst_seed_budgets = list(
                dict.fromkeys(
                    self.get_runs_for_config(config, only_max_observed_budget=True)
                )
            )
            if instances is not None:
                inst_seed_budgets = list(
                    filter(
                        lambda x: x.instance in cast(List, instances), inst_seed_budgets
                    )
                )

            if inst_seed_budgets:  # can be empty if never saw any runs on <instances>
                self._cost_per_config[config_id] = self.average_cost(
                    config, inst_seed_budgets
                )
                self._min_cost_per_config[config_id] = self.min_cost(
                    config, inst_seed_budgets
                )
                self.num_runs_per_config[config_id] = len(inst_seed_budgets)

    def get_instance_costs_for_config(
        self, config: Configuration
    ) -> Dict[str, List[float]]:
        """Returns the average cost per instance (across seeds) for a configuration

        If the runhistory contains budgets, only the highest budget for a configuration is returned.

        Note
        ----
        This is used by the pSMAC facade to determine the incumbent after the evaluation.

        Parameters
        ----------
        config : Configuration from ConfigSpace
            Parameter configuration

        Returns
        -------
        cost_per_inst: dict<instance name<str>, cost<float>>
        """
        runs_ = self.get_runs_for_config(config, only_max_observed_budget=True)
        cost_per_inst = {}  # type: Dict[str, List[float]]
        for inst, seed, budget in runs_:
            cost_per_inst[inst] = cost_per_inst.get(inst, [])
            rkey = RunKey(self.config_ids[config], inst, seed, budget)
            vkey = self.data[rkey]
            cost_per_inst[inst].append(vkey.cost)
        cost_per_inst = dict(
            [(inst, np.mean(costs)) for inst, costs in cost_per_inst.items()]
        )
        return cost_per_inst
