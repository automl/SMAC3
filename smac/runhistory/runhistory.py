import collections
import json
import numpy

from smac.configspace import Configuration

__author__ = "Marius Lindauer"
__copyright__ = "Copyright 2015, ML4AAD"
__license__ = "GPLv3"
__maintainer__ = "Marius Lindauer"
__email__ = "lindauer@cs.uni-freiburg.de"
__version__ = "0.0.1"

MAXINT = 2 ** 31 - 1


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

        self.cost_per_config = {} # config_id -> cost

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

    def update_cost(self, config, cost):
        config_id = self.config_ids[config.__repr__()]
        self.cost_per_config[config_id] = cost

    def get_cost(self, config):
        config_id = self.config_ids[config.__repr__()]
        return self.cost_per_config[config_id]

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

    def save_json(self, fn="runhistory.json"):
        '''
        saves runhistory on disk

        Parameters
        ----------
        fn : str
            file name
        '''

        id_vec = dict([(id_, conf.get_array().tolist())
                       for id_, conf in self.ids_config.iteritems()])

        data = [(list(k), list(v)) for k, v in self.data.iteritems()]

        with open(fn, "w") as fp:
            json.dump({"data": data,
                       "id_config": id_vec}, fp)

    def load_json(self, fn, cs):
        '''
        loads runhistory from disk in json represantation
        Overwrites current runthistory!

        Parameters
        ----------
        fn: str
            file name to load from
        cs: ConfigSpace
            instance of configuration space
        '''

        with open(fn) as fp:
            all_data = json.load(fp)

        self.ids_config = dict([(int(id_), Configuration(
            cs, vector=numpy.array(vec))) for id_, vec in all_data["id_config"].iteritems()])


        self.config_ids = dict([(Configuration(
            cs, vector=numpy.array(vec)).__repr__(), id_) for id_, vec in all_data["id_config"].iteritems()])
        self._n_id = len(self.config_ids)
        
        self.data = dict([(self.RunKey(int(k[0]), k[1], int(k[2])),
                           self.RunValue(float(v[0]), float(v[1]), v[2], v[3]))
                          for k, v in all_data["data"]
                          ])
