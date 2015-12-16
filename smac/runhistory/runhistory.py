import collections

__author__ = "Marius Lindauer"
__copyright__ = "Copyright 2015, ML4AAD"
__license__ = "BSD"
__maintainer__ = "Marius Lindauer"
__email__ = "lindauer@cs.uni-freiburg.de"
__version__ = "0.0.1"


class RunHistory(object):
    '''
         saves all run informations from target algorithm runs

        Attributes
        ----------
    '''

    def __init__(self):
        '''
        Constructor
        '''
        self.data = {}

        self.RunKey = collections.namedtuple(
            'RunKey', ['config_id', 'instance_id', 'seed'])

        self.RunValue = collections.namedtuple(
            'RunValue', ['cost', 'time', 'status', 'additional_info'])

        self.config_ids = {}  # config -> id
        self.ids_config = {}  # id -> config
        self._n_id = 0

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
            instance_id: int
                id of instance (default: None)
            seed: int
                random seed used by TA (default: None)
            additional_info: dict
                additional run infos (could include further returned
                information from TA or fields such as start time and host_id)
        '''

        # TODO: replace str casting of config when we have something hashable
        # as a config object
        config_id = self.config_ids.get(str(config))
        if config_id is None:
            self._n_id += 1
            self.config_ids[str(config)] = self._n_id
            config_id = self.config_ids.get(str(config))
            self.ids_config[self._n_id] = config

        k = self.RunKey(config_id, instance_id, seed)
        v = self.RunValue(cost, time, status, additional_info)

        self.data[k] = v

    def get_runs_for_config(self, config):
        '''
        given a configuration return all runs (instance, seed) of this config
        Attributes
        ----------
            config: Configuration from ConfigSpace
                parameter configuration
        Returns
        ----------
            list
        '''
        InstSeedTuple = collections.namedtuple(
            "Inst_Seed", ["instance", "seed", "time"])
        list_ = []
        for k in self.data:
            if config == self.ids_config[k.config_id]:
                ist = InstSeedTuple(k.instance_id, k.seed, self.data[k].time)
                list_.append(ist)
        return list_
