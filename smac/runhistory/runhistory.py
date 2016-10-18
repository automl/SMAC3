import collections
import json
import numpy

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

RunValue = collections.namedtuple(
    'RunValue', ['cost', 'time', 'status', 'additional_info'])


class RunHistory(object):

    '''
         saves all run informations from target algorithm runs

        Attributes
        ----------
    '''

    def __init__(self, aggregate_func):
        '''
        Constructor
        
        Arguments
        ---------
        aggregate_func: callable
            function to aggregate perf across instances
        
        '''

        # By having the data in a deterministic order we can do useful tests
        # when we serialize the data and can assume it's still in the same
        # order as it was added.
        self.data = collections.OrderedDict()

        self.config_ids = {}  # config -> id
        self.ids_config = {}  # id -> config
        self._n_id = 0

        self.cost_per_config = {} # config_id -> cost
        
        self.aggregate_func = aggregate_func

    def add(self, config, cost, time,
            status, instance_id=None,
            seed=None,
            additional_info=None):
        '''
        adds a data of a new target algorithm (TA) run;
        it will update data if the same key values are used 
        (config, instance_id, seed)

        Attributes
        ----------
            config : dict (or other type -- depending on config space module)
                parameter configuratoin
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
        '''

        config_id = self.config_ids.get(config)
        if config_id is None:
            self._n_id += 1
            self.config_ids[config] = self._n_id
            config_id = self.config_ids.get(config)
            self.ids_config[self._n_id] = config

        k = RunKey(config_id, instance_id, seed)
        v = RunValue(cost, time, status, additional_info)

        self.data[k] = v
        self.update_cost(config)

    def update_cost(self, config):
        '''
            store the performance of a configuration across the instances in self.cost_perf_config; 
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

    def get_cost(self, config):
        config_id = self.config_ids[config]
        return self.cost_per_config[config_id]

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
        InstanceSeedPair = collections.namedtuple("InstanceSeedPair",
                                                  ["instance", "seed"])
        config_id = self.config_ids.get(config)
        list_ = []
        for k in self.data:
            # TA will return ABORT if config. budget was exhausted and
            # we don't want to collect such runs to compute the cost of a configuration
            if config_id == k.config_id and self.data[k].status not in [StatusType.ABORT] : 
                ist = InstanceSeedPair(k.instance_id, k.seed)
                list_.append(ist)
        return list_

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

        configs = {id_: conf.get_dictionary()
                   for id_, conf in self.ids_config.items()}

        data = [([int(k.config_id),
                  str(k.instance_id) if k.instance_id is not None else None,
                  int(k.seed)], list(v))
                for k, v in self.data.items()]

        with open(fn, "w") as fp:
            json.dump({"data": data,
                       "configs": configs}, fp)

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
            all_data = json.load(fp)

        self.ids_config = {int(id_): Configuration(cs, values=values)
                           for id_, values in all_data["configs"].items()}


        self.config_ids = {Configuration(cs, values=values): int(id_)
                           for id_, values in all_data["configs"].items()}

        self._n_id = len(self.config_ids)
        
        self.data = {RunKey(int(k[0]), k[1], int(k[2])):
                     RunValue(float(v[0]), float(v[1]), v[2], v[3])
                     for k, v in all_data["data"]}

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

        # Configurations might be already known, but by a different ID. This
        # does not matter here because the add() method handles this
        # correctly by assigning an ID to unknown configurations and re-using
        #  the ID
        for key, value in new_runhistory.data.items():
            config_id, instance_id, seed = key
            cost, time, status, additional_info = value
            config = new_runhistory.ids_config[config_id]
            self.add(config=config, cost=cost, time=time,
                     status=status, instance_id=instance_id,
                     seed=seed, additional_info=additional_info)
