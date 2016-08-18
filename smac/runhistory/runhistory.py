import collections
import json
import numpy

from smac.configspace import Configuration

__author__ = "Marius Lindauer"
__copyright__ = "Copyright 2015, ML4AAD"
__license__ = "3-clause BSD"
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

        # By having the data in a deterministic order we can do useful tests
        # when we serialize the data and can assume it's still in the same
        # order as it was added.
        self.data = collections.OrderedDict()

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
        list_ = []
        for k in self.data:
            if config == self.ids_config[k.config_id]:
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

        id_vec = dict([(id_, conf.get_array().tolist())
                       for id_, conf in self.ids_config.items()])

        data = [([int(k.config_id),
                  str(k.instance_id) if k.instance_id is not None else None,
                  int(k.seed)], list(v))
                for k, v in self.data.items()]

        with open(fn, "w") as fp:
            json.dump({"data": data,
                       "id_config": id_vec}, fp)

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

        self.ids_config = dict([(int(id_), Configuration(
            cs, vector=numpy.array(vec))) for id_, vec in all_data["id_config"].items()])


        self.config_ids = dict([(Configuration(
            cs, vector=numpy.array(vec)).__repr__(), id_) for id_, vec in all_data["id_config"].items()])
        self._n_id = len(self.config_ids)
        
        self.data = dict([(self.RunKey(int(k[0]), k[1], int(k[2])),
                           self.RunValue(float(v[0]), float(v[1]), v[2], v[3]))
                          for k, v in all_data["data"]
                          ])

    def update_from_json(self, fn, cs):
        """Update the current runhistory by adding new runs from a json file.

        Parameters
        ----------
        fn : str
            file name to load from
        cs : ConfigSpace
            instance of configuration space
        """

        new_runhistory = RunHistory()
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
