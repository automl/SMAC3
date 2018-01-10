import collections
from enum import Enum
import json
import numpy as np
import typing

from smac.configspace import Configuration, ConfigurationSpace
from smac.tae.execute_ta_run import StatusType

__author__ = "Marius Lindauer"
__copyright__ = "Copyright 2015, ML4AAD"
__license__ = "3-clause BSD"
__maintainer__ = "Marius Lindauer"
__email__ = "lindauer@cs.uni-freiburg.de"
__version__ = "0.0.1"


RunKey = collections.namedtuple(
    'RunKey', ['config_id', 'instance_id', 'seed'])

InstSeedKey = collections.namedtuple(
    'InstSeedKey', ['instance', 'seed'])

RunValue = collections.namedtuple(
    'RunValue', ['cost', 'time', 'status', 'additional_info'])


class EnumEncoder(json.JSONEncoder):
    """Custom encoder for enum-serialization
    (implemented for StatusType from tae/execute_ta_run).
    Using encoder implied using object_hook as defined in StatusType
    to deserialize from json.
    """

    def default(self, obj):
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

    **Note:** Guaranteed to be picklable.

    Attributes
    ----------
    data : collections.OrderedDict()
        TODO
    config_ids : dict
        Maps config -> id
    ids_config : dict
        Maps id -> config
    cost_per_config : dict
        Maps config_id -> cost
    runs_per_config : dict
        Maps config_id -> number of runs

    aggregate_func
    overwrite_existing_runs
    """

    def __init__(self, 
                 aggregate_func: typing.Callable,
                 overwrite_existing_runs: bool=False
                 ):
        """Constructor

        Parameters
        ----------
        aggregate_func: callable
            function to aggregate perf across instances
        overwrite_existing_runs: bool
            allows to overwrites old results if pairs of
            algorithm-instance-seed were measured
            multiple times
        """
        # By having the data in a deterministic order we can do useful tests
        # when we serialize the data and can assume it's still in the same
        # order as it was added.
        self.data = collections.OrderedDict()

        # for fast access, we have also an unordered data structure
        # to get all instance seed pairs of a configuration
        self._configid_to_inst_seed = {}

        self.config_ids = {}  # config -> id
        self.ids_config = {}  # id -> config
        self._n_id = 0

        self.cost_per_config = {}  # config_id -> cost
        # runs_per_config is necessary for computing the moving average
        self.runs_per_config = {}  # config_id -> number of runs

        # Store whether a datapoint is "external", which means it was read from
        # a JSON file. Can be chosen to not be written to disk
        self.external = {}  # RunKey -> DataOrigin

        self.aggregate_func = aggregate_func
        self.overwrite_existing_runs = overwrite_existing_runs

    def add(self, config: Configuration, cost: float, time: float,
            status: StatusType, instance_id: str=None,
            seed: int=None,
            additional_info: dict=None,
            origin: DataOrigin=DataOrigin.INTERNAL):
        """Adds a data of a new target algorithm (TA) run;
        it will update data if the same key values are used
        (config, instance_id, seed)

        Parameters
        ----------
            config : dict (or other type -- depending on config space module)
                Parameter configuration
            cost: float
                Cost of TA run (will be minimized)
            time: float
                Runtime of TA run
            status: str
                Status in {SUCCESS, TIMEOUT, CRASHED, ABORT, MEMOUT}
            instance_id: str
                String representing an instance (default: None)
            seed: int
                Random seed used by TA (default: None)
            additional_info: dict
                Additional run infos (could include further returned
                information from TA or fields such as start time and host_id)
            origin: DataOrigin
                Defines how data will be used.
        """

        config_id = self.config_ids.get(config)
        if config_id is None:
            self._n_id += 1
            self.config_ids[config] = self._n_id
            config_id = self.config_ids.get(config)
            self.ids_config[self._n_id] = config

        k = RunKey(config_id, instance_id, seed)
        v = RunValue(cost, time, status, additional_info)

        # Each runkey is supposed to be used only once. Repeated tries to add
        # the same runkey will be ignored silently if not capped.
        if self.overwrite_existing_runs or self.data.get(k) is None:
            self._add(k, v, status, origin)
        elif status != StatusType.CAPPED and self.data[k].status == StatusType.CAPPED:
            # overwrite capped runs with uncapped runs
            self._add(k, v, status, origin)
        elif status == StatusType.CAPPED and self.data[k].status == StatusType.CAPPED and cost > self.data[k].cost:
            # overwrite if censored with a larger cutoff
            self._add(k, v, status, origin)

    def _add(self, k: RunKey, v: RunValue, status: StatusType,
             origin: DataOrigin):
        """Actual function to add new entry to data structures

        TODO

        """
        self.data[k] = v
        self.external[k] = origin

        if origin in (DataOrigin.INTERNAL, DataOrigin.EXTERNAL_SAME_INSTANCES) \
                and status != StatusType.CAPPED:
            # also add to fast data structure
            is_k = InstSeedKey(k.instance_id, k.seed)
            self._configid_to_inst_seed[
                k.config_id] = self._configid_to_inst_seed.get(k.config_id, [])
            if is_k not in self._configid_to_inst_seed[k.config_id]:
                self._configid_to_inst_seed[k.config_id].append(is_k)

            if not self.overwrite_existing_runs:
                # assumes an average across runs as cost function aggregation
                self.incremental_update_cost(self.ids_config[k.config_id], v.cost)
            else:
                self.update_cost(config=self.ids_config[k.config_id])

    def update_cost(self, config: Configuration):
        """Store the performance of a configuration across the instances in
        self.cost_perf_config and also updates self.runs_per_config;
        uses self.aggregate_func

        Parameters
        ----------
        config: Configuration
            configuration to update cost based on all runs in runhistory
        """
        inst_seeds = set(self.get_runs_for_config(config))
        perf = self.aggregate_func(config, self, inst_seeds)
        config_id = self.config_ids[config]
        self.cost_per_config[config_id] = perf
        self.runs_per_config[config_id] = len(inst_seeds)

    def compute_all_costs(self, instances: typing.List[str]=None):
        """Computes the cost of all configurations from scratch and overwrites
        self.cost_perf_config and self.runs_per_config accordingly;

        Parameters
        ----------
        instances: typing.List[str]
            list of instances; if given, cost is only computed wrt to this instance set
        """

        self.cost_per_config = {}
        self.runs_per_config = {}
        for config, config_id in self.config_ids.items():
            inst_seeds = set(self.get_runs_for_config(config))
            if instances is not None:
                inst_seeds = list(
                    filter(lambda x: x.instance in instances, inst_seeds))

            if inst_seeds:  # can be empty if never saw any runs on <instances>
                perf = self.aggregate_func(config, self, inst_seeds)
                self.cost_per_config[config_id] = perf
                self.runs_per_config[config_id] = len(inst_seeds)

    def incremental_update_cost(self, config: Configuration, cost: float):
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
        n_runs = self.runs_per_config.get(config_id, 0)
        old_cost = self.cost_per_config.get(config_id, 0.)
        self.cost_per_config[config_id] = (
            (old_cost * n_runs) + cost) / (n_runs + 1)
        self.runs_per_config[config_id] = n_runs + 1

    def get_cost(self, config: Configuration):
        """Returns empirical cost for a configuration; uses  self.cost_per_config

        Parameters
        ----------
        config: Configuration

        Returns
        -------
        cost: float
            Computed cost for configuration
        """
        config_id = self.config_ids[config]
        return self.cost_per_config.get(config_id, np.nan)

    def get_runs_for_config(self, config: Configuration):
        """Return all runs (instance seed pairs) for a configuration.

        Parameters
        ----------
        config : Configuration from ConfigSpace
            Parameter configuration

        Returns
        -------
        instance_seed_pairs : list<tuples of instance, seed>
        """
        config_id = self.config_ids.get(config)
        return self._configid_to_inst_seed.get(config_id, [])

    def get_all_configs(self):
        """Return all configurations in this RunHistory object

        Returns
        -------
            parameter configurations: list
        """
        return list(self.config_ids.keys())

    def empty(self):
        """Check whether or not the RunHistory is empty.

        Returns
        -------
        emptiness: bool
            True if runs have been added to the RunHistory,
            False otherwise
        """
        return len(self.data) == 0

    def save_json(self, fn: str="runhistory.json", save_external: bool=False):
        """
        saves runhistory on disk

        Parameters
        ----------
        fn : str
            file name
        save_external : bool
            Whether to save external data in the runhistory file.
        """

        data = [([int(k.config_id),
                  str(k.instance_id) if k.instance_id is not None else None,
                  int(k.seed)], list(v))
                for k, v in self.data.items()
                if save_external or self.external[k] == DataOrigin.INTERNAL]
        config_ids_to_serialize = set([entry[0][0] for entry in data])
        configs = {id_: conf.get_dictionary()
                   for id_, conf in self.ids_config.items()
                   if id_ in config_ids_to_serialize}

        with open(fn, "w") as fp:
            json.dump({"data": data,
                       "configs": configs}, fp, cls=EnumEncoder)

    def load_json(self, fn: str, cs: ConfigurationSpace):
        """Load and runhistory in json representation from disk.

        Overwrites current runhistory!

        Parameters
        ----------
        fn : str
            file name to load from
        cs : ConfigSpace
            instance of configuration space
        """
        with open(fn) as fp:
            all_data = json.load(fp, object_hook=StatusType.enum_hook)

        self.ids_config = {int(id_): Configuration(cs, values=values)
                           for id_, values in all_data["configs"].items()}

        self.config_ids = {config: id_ for id_, config in self.ids_config.items()}

        self._n_id = len(self.config_ids)

        # important to use add method to use all data structure correctly
        for k, v in all_data["data"]:
            self.add(config=self.ids_config[int(k[0])],
                     cost=float(v[0]),
                     time=float(v[1]),
                     status=StatusType(v[2]),
                     instance_id=k[1],
                     seed=int(k[2]),
                     additional_info=v[3])

    def update_from_json(self, fn: str, cs: ConfigurationSpace,
                         origin: DataOrigin=DataOrigin.EXTERNAL_SAME_INSTANCES):
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
        new_runhistory = RunHistory(self.aggregate_func)
        new_runhistory.load_json(fn, cs)
        self.update(runhistory=new_runhistory, origin=origin)

    def update(self, runhistory: 'RunHistory',
               origin: DataOrigin=DataOrigin.EXTERNAL_SAME_INSTANCES):
        """Update the current runhistory by adding new runs from a RunHistory.

        Parameters
        ----------
        runhistory: RunHistory
            Runhistory with additional data to be added to self
        origin: DataOrigin
            If set to ``INTERNAL`` or ``EXTERNAL_FULL`` the data will be
            added to the internal data structure self._configid_to_inst_seed
            and be available :meth:`through get_runs_for_config`.
        """

        # Configurations might be already known, but by a different ID. This
        # does not matter here because the add() method handles this
        # correctly by assigning an ID to unknown configurations and re-using
        #  the ID
        for key, value in runhistory.data.items():
            config_id, instance_id, seed = key
            cost, time, status, additional_info = value
            config = runhistory.ids_config[config_id]
            self.add(config=config, cost=cost, time=time,
                     status=status, instance_id=instance_id,
                     seed=seed, additional_info=additional_info,
                     origin=origin)
