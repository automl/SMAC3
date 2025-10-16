import logging
import shutil
from logging import Logger

from ConfigSpace import ConfigurationSpace, Configuration

from smac import Scenario, AlgorithmConfigurationFacade
import numpy as np

from smac.main.exceptions import ConfigurationSpaceExhaustedException
from smac.utils.configspace import get_config_hash
from smac.utils.logging import get_logger
"""
The Adaptive Capping Intensifier test tests SMAC capabilities to work with the adaptive capping feature being enabled.
To this end, we create a train function mockup which also logs every single evaluation requested by SMAC to ensure that
it follows the corresponding rules, i.e., evaluates the right configurations, switches the incumbent in the right moment
and also allocates the right amount of budget. This test logs any unexpected evaluation and prints a list of evaluations
(stored in the `expected_behavior_violations`) to facilitate debugging and informing what behavior exactly did not match  
the expectation of adaptive capping.
"""
__copyright__ = "Copyright 2025, automl.org"
__license__ = "3-clause BSD"

logger = get_logger(__name__)

class TrainMockup:
    """
    Class for logging the behavior of SMAC regarding calls to the run function. In particular, we log which candidates
    are evaluated for what instances and what runtime costs have been sampled from a random distribution.
    """

    def __init__(self):
        # counter for referring to evaluation number
        self.log_counter = 0

        # log list and map for easier access of performance data
        self.log_list = []
        self.log_map = {}

        # current incumbent configuration + hash
        self.incumbent = None

        # trace of incumbents over time
        self.incumbent_trace = []

        # accept/reject event log
        self.event_log = []

        # list of rejected challengers
        self.rejected_challengers = []

        # flag validating whether expected behavior got violated
        self.expected_behavior_violated = False

        # list of explanations what erroneous behavior got observed
        self.expected_behavior_violations = []
        np.random.seed(42)

    def train(self, config:Configuration, instance: str, cutoff: int, seed: int = 0):
        print("Cutoff configured: ", cutoff)
        self.log_counter += 1
        config_hash = get_config_hash(config)
        rand_perf = np.random.random_integers(low=1, high=20)
        censored = rand_perf > cutoff
        if censored:
            rand_perf = cutoff + 1

        # check whether config needed to be rejected already
        if config_hash in self.rejected_challengers:
            self.expected_behavior_violated = True

            # search for first rejection in event log
            reject_log = None
            for log in self.event_log:
                if log[1] == "reject" and log[2] == config_hash:
                    reject_log = log
                    break

            self.expected_behavior_violations += [f"{self.log_counter}: Configuration {config_hash} was already "
                                                  f"rejected here {reject_log}"]

        # specify log entry (id, config hash, instance, performance)
        log = (self.log_counter, config_hash, instance, rand_perf)

        # ensure config has an entry in the log map and store observed performance, add log entry to list
        if config_hash not in self.log_map:
            self.log_map[config_hash] = {}
        self.log_map[config_hash][instance] = rand_perf
        self.log_list += [log]
        logger.debug(f"Train: {log}")

        # if incumbent is none so far, we have a new incumbent, i.e., the initial one
        if self.incumbent is None:
            self.incumbent = config_hash
            logger.debug(f"Incumbent initially set to config {self.incumbent}")
        else:
            instances_evaluated = self.log_map[config_hash].keys()
            n_challenger = len(instances_evaluated)
            n_incumbent = len(self.log_map[self.incumbent])
            c_challenger = np.array([self.log_map[config_hash][instance] for instance in instances_evaluated]).sum()
            c_incumbent = np.array([self.log_map[self.incumbent][instance] for instance in instances_evaluated]).sum()

            logger.debug(f"evaluated challenger {config_hash} on budget {n_challenger} with performance "
                         f"{c_challenger} and incumbent {self.incumbent} was evaluated on budget {n_incumbent} "
                         f"showing performance {c_incumbent}.")

            if n_challenger >= n_incumbent and c_challenger < c_incumbent:
                self.event_log += [(self.log_counter, "accept", config_hash)]
                self.incumbent = config_hash
            elif c_challenger > c_incumbent:
                self.rejected_challengers += [config_hash]
                self.event_log += [(self.log_counter, "reject", config_hash)]

        return rand_perf

    def get_violation_report(self):
        report = "The following violations occurred:\n"
        report += "\n".join(self.expected_behavior_violations)
        return report

def get_basic_setup(train, num_configs = 10, num_instances = 10, num_trials=30):
    # generate config space with num_configs many different configurations
    cs = ConfigurationSpace({"p1": ["v"+str(i) for i in range(num_configs)], })
    cs.seed(42)
    # generate instance set with num_instances many instances
    instances = ["i"+str(i) for i in range(num_instances)]
    # setup scenario with generated config space, instances, and the given number of trials
    scenario = Scenario(cs, deterministic=True, n_trials=num_trials, instances=instances, seed=44, adaptive_capping=True, runtime_cutoff=500)
    return AlgorithmConfigurationFacade(scenario, train)

def test_incumbent_switch() -> None:
    """
    Test whether the incumbents switch at the right time, i.e., whenever there is a challenger meeting the dominance
    criteria.

    """
    # remove smac3 output folder to ensure proper execution of the test
    shutil.rmtree("./smac3_output", ignore_errors=True)
    shutil.rmtree("./smac3_output_test", ignore_errors=True)

    # setup test environment
    tm = TrainMockup()
    smac = get_basic_setup(tm.train)

    # activate logging
    l: Logger = get_logger("smac.intensifier.abstract_intensifier")
    l.setLevel(5)
    l: Logger = get_logger("smac.intensifier.intensifier")
    l.setLevel(10)

    # start smac run
    try:
        smac.optimize()
    except ConfigurationSpaceExhaustedException:
        pass

    assert tm.expected_behavior_violated is False, tm.get_violation_report()