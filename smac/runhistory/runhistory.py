import collections
import json
import numpy as np
import typing

from smac.configspace import Configuration
from smac.tae.execute_ta_run import StatusType
from smac.utils.constants import MAXINT

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


class RunHistory(object):

    '''Container for target algorithm run information.

    Guaranteed to be picklable.

    Arguments
    ---------
    aggregate_func: callable
        function to aggregate perf across instances
    '''

    def __init__(self, aggregate_func):

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

        self.aggregate_func = aggregate_func

    def add(self, config, cost, time,
            status, instance_id=None,
            seed=None,
            additional_info=None,
            external_data:bool=False):
        '''
        adds a data of a new target algorithm (TA) run;
        it will update data if the same key values are used
        (config, instance_id, seed)

        Attributes
        ----------
            config : dict (or other type -- depending on config space module)
                parameter configuration
            cost: float
                cost of TA run (will be minimized)
            time: float
                runtime of TA run
            status: str
                status in {SUCCESS, TIMEOUT, CRASHED, ABORT, MEMOUT}
            instance_id: str
                str representing an instance (default: None)
            seed: int
                random seed used by TA (default: None)
            additional_info: dict
                additional run infos (could include further returned
                information from TA or fields such as start time and host_id)
            external_data: bool
                if True, run will not be added to self._configid_to_inst_seed
                and not available through get_runs_for_config()
        '''

        config_id = self.config_ids.get(config)
        if config_id is None:
            self._n_id += 1
            self.config_ids[config] = self._n_id
            config_id = self.config_ids.get(config)
            self.ids_config[self._n_id] = config

        k = RunKey(config_id, instance_id, seed)
        v = RunValue(cost, time, status, additional_info)

        # Each runkey is supposed to be used only once. Repeated tries to add
        # the same runkey will be ignored silently.
        if self.data.get(k) is None:
            self.data[k] = v

            if not external_data:
                # also add to fast data structure
                is_k = InstSeedKey(instance_id, seed)
                self._configid_to_inst_seed[
                    config_id] = self._configid_to_inst_seed.get(config_id, [])
                self._configid_to_inst_seed[config_id].append(is_k)

            # assumes an average across runs as cost function
            self.incremental_update_cost(config, cost)

    def update_cost(self, config):
        '''
            store the performance of a configuration across the instances in self.cost_perf_config
            and also updates self.runs_per_config;
            uses self.aggregate_func

            Arguments
            --------
            config: Configuration
                configuration to update cost based on all runs in runhistory
        '''
        inst_seeds = set(self.get_runs_for_config(config))
        perf = self.aggregate_func(config, self, inst_seeds)
        config_id = self.config_ids[config]
        self.cost_per_config[config_id] = perf
        self.runs_per_config[config_id] = len(inst_seeds)

    def compute_all_costs(self, instances: typing.List[str]=None):
        '''
            computes the cost of all configurations from scratch
            and overwrites self.cost_perf_config and self.runs_per_config accordingly;

            Arguments
            ---------
            instances: typing.List[str]
                list of instances; if given, cost is only computed wrt to this instance set
        '''

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
        '''
            incrementally updates the performance of a configuration by using a moving average; 

            Arguments
            --------
            config: Configuration
                configuration to update cost based on all runs in runhistory
            cost: float
                cost of new run of config
        '''

        config_id = self.config_ids[config]
        n_runs = self.runs_per_config.get(config_id, 0)
        old_cost = self.cost_per_config.get(config_id, 0.)
        self.cost_per_config[config_id] = (
            (old_cost * n_runs) + cost) / (n_runs + 1)
        self.runs_per_config[config_id] = n_runs + 1

    def get_cost(self, config):
        '''
            returns empirical cost for a configuration;
            uses  self.cost_per_config
        '''
        config_id = self.config_ids[config]
        return self.cost_per_config.get(config_id, np.nan)

    def get_runs_for_config(self, config):
        """Return all runs (instance seed pairs) for a configuration.

        Parameters
        ----------
        config : Configuration from ConfigSpace
            parameter configuration
        Returns
        ----------
            list: tuples of instance, seed
        """
        config_id = self.config_ids.get(config)
        return self._configid_to_inst_seed.get(config_id, [])

    def get_all_configs(self):
        """ Return all configurations in this RunHistory object

        Returns
        -------
            list: parameter configurations

        """
        return list(self.config_ids.keys())

    def empty(self):
        """
        Check whether or not the RunHistory is empty.

        Returns
        ----------
            bool: True if runs have been added to the RunHistory, 
                  False otherwise
        """
        return len(self.data) == 0

    def save_json(self, fn="runhistory.json"):
        '''
        saves runhistory on disk

        Parameters
        ----------
        fn : str
            file name
        '''

        class EnumEncoder(json.JSONEncoder):
            """
            custom encoder for enum-serialization
            (implemented for StatusType from tae/execute_ta_run)
            locally defined because only ever needed here.
            using encoder implied using object_hook defined in StatusType
            to deserialize from json.
            """
            def default(self, obj):
                if isinstance(obj, StatusType):
                    return {"__enum__": str(obj)}
                return json.JSONEncoder.default(self, obj)

        configs = {id_: conf.get_dictionary()
                   for id_, conf in self.ids_config.items()}

        data = [([int(k.config_id),
                  str(k.instance_id) if k.instance_id is not None else None,
                  int(k.seed)], list(v))
                for k, v in self.data.items()]

        with open(fn, "w") as fp:
            json.dump({"data": data,
                       "configs": configs}, fp, cls=EnumEncoder)

    def load_json(self, fn, cs):
        """Load and runhistory in json representation from disk.

        Overwrites current runthistory!

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

        self.config_ids = {Configuration(cs, values=values): int(id_)
                           for id_, values in all_data["configs"].items()}

        self._n_id = len(self.config_ids)

        # important to use add method to use all data structure correctly
        for k, v in all_data["data"]:
            self.add(config=self.ids_config[int(k[0])],
                     cost=float(v[0]),
                     time=float(v[1]),
                     status=v[2],
                     instance_id=k[1],
                     seed=int(k[2]),
                     additional_info=v[3])

    def update_from_json(self, fn, cs):
        """Update the current runhistory by adding new runs from a json file.

        Parameters
        ----------
        fn : str
            file name to load from
        cs : ConfigSpace
            instance of configuration space
        """
        new_runhistory = RunHistory(self.aggregate_func)
        new_runhistory.load_json(fn, cs)
        self.update(runhistory=new_runhistory)

    def update(self, runhistory, external_data:bool=False):
        """Update the current runhistory by adding new runs from a json file.

        Parameters
        ----------
        runhistory: RunHistory
            runhistory with additional data to be added to self
        external_data: bool
            if True, run will not be added to self._configid_to_inst_seed 
            and not available through get_runs_for_config()
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
                     external_data=external_data)
