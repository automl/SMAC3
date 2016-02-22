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
        # TODO JTS: We might have to execute one config multiple times
        #           since the results can be noisy and then we can't simply
        #           overwrite the old config result here!
        config_id = self.config_ids.get(config.__repr__())
        if config_id is None:
            self._n_id += 1
            self.config_ids[config.__repr__()] = self._n_id
            config_id = self.config_ids.get(config.__repr__())
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
            list: tuples of instance, seed, time
        '''
        InstSeedTuple = collections.namedtuple(
            "Inst_Seed", ["instance", "seed", "time", "cost"])
        list_ = []
        for k in self.data:
            if config == self.ids_config[k.config_id]:
                ist = InstSeedTuple(
                    k.instance_id, k.seed, self.data[k].time, self.data[k].cost)
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
