#!/bin/python

__author__ = "Marius Lindauer"
__copyright__ = "Copyright 2015, ML4AAD"
__license__ = "3-clause BSD"
__maintainer__ = "Marius Lindauer"
__email__ = "lindauer@cs.uni-freiburg.de"
__version__ = "0.0.1"

import os
import sys
import inspect
import logging

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import numpy as np
from sklearn.cross_validation import KFold

cmd_folder = os.path.realpath(
    os.path.abspath(os.path.split(inspect.getfile(inspect.currentframe()))[0]))
cmd_folder = os.path.realpath(os.path.join(cmd_folder, ".."))
if cmd_folder not in sys.path:
    sys.path.append(cmd_folder)

from smac.scenario.scenario import Scenario
from smac.tae.execute_ta_run import StatusType
from smac.epm.rf_with_instances import RandomForestWithInstances
from smac.runhistory.runhistory import RunHistory
from smac.runhistory.runhistory2epm import RunHistory2EPM4LogCost
from ConfigSpace.io import pcs
from ConfigSpace.hyperparameters import CategoricalHyperparameter, \
    UniformFloatHyperparameter, UniformIntegerHyperparameter, Constant
from smac.epm.rfr_imputator import RFRImputator


class EPMImportance(object):

    def __init__(self, scenario, runhistory):
        '''
        Constructor

        Arguments
        ---------
        scenario: Scenario
            scenario object
        runhistory: RunHistory
            runhistory object to learn the EPM from it
        '''

        types = np.zeros(len(scen.cs.get_hyperparameters()),
                         dtype=np.uint)

        for i, param in enumerate(scen.cs.get_hyperparameters()):
            if isinstance(param, (CategoricalHyperparameter)):
                n_cats = len(param.choices)
                types[i] = n_cats

        if scen.feature_array is not None:
            types = np.hstack(
                (types, np.zeros((scen.feature_array.shape[1]))))

        types = np.array(types, dtype=np.uint)

        self.model = RandomForestWithInstances(types,
                                               scen.feature_array)

        self.num_params = len(scen.cs.get_hyperparameters())

        if scen.run_obj == "runtime":
            if scen.run_obj == "runtime":
                # if we log the performance data,
                # the RFRImputator will already get
                # log transform data from the runhistory
                cutoff = np.log10(scen.cutoff)
                threshold = np.log10(scen.cutoff *
                                     scen.par_factor)
            else:
                cutoff = scen.cutoff
                threshold = scen.cutoff * scen.par_factor

            imputor = RFRImputator(cs=scen.cs,
                                   rs=np.random.RandomState(42),
                                   cutoff=cutoff,
                                   threshold=threshold,
                                   model=self.model,
                                   change_threshold=0.01,
                                   max_iter=10)
            #TODO: Adapt runhistory2EPM object based on scenario
            rh2EPM = RunHistory2EPM4LogCost(scenario=scen,
                                    num_params=self.num_params,
                                    success_states=[StatusType.SUCCESS, ],
                                    impute_censored_data=True,
                                    impute_state=[StatusType.TIMEOUT, ],
                                    imputor=imputor,
                                    log_y=scen.run_obj == "runtime")
        else:
            rh2EPM = RunHistory2EPM4LogCost(scenario=self.scenario,
                                    num_params=self.num_params,
                                    success_states=None,
                                    impute_censored_data=False,
                                    impute_state=None,
                                    log_y=scen.run_obj == "runtime")

        self.X, self.Y = rh2EPM.transform(hist)

        self.types = types
        self.scen = scen
        self._MAX_P = min(10, self.num_params)

    def run(self):
        '''
            runs a 10-fold cross validation on trainings data from self.X and self.Y (converted runhistory)
            with parameter forward selection
        '''

        X = self.X
        y = self.Y

        kf = KFold(X.shape[0], n_folds=10)

        param_ids = range(self.num_params)
        used = []
        # always use all features
        used.extend(range(self.num_params, len(self.types)))

        for _ in range(self._MAX_P):
            scores = []
            for p in param_ids:

                used.append(p)
                X_l = X[:, used]

                model = RandomForestWithInstances(self.types[used],
                                                  self.scen.feature_array)

                rmses = []
                for train, test in kf:
                    X_train = X_l[train]
                    y_train = y[train]
                    X_test = X_l[test]
                    y_test = y[test]

                    model.train(X_train, y_train)
                    y_pred = model._predict(X_test)[0]

                    rmse = np.sqrt(np.mean((y_pred - y_test[:, 0])**2))
                    rmses.append(rmse)
                scores.append(np.mean(rmses))
                used.pop()
            best_indx = np.argmin(scores)
            best_score = scores[best_indx]
            p = param_ids.pop(best_indx)
            used.append(p)

            logging.info("%s : %.4f (RMSE)" % (p, best_score))

if __name__ == "__main__":

    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    req_opts = parser.add_argument_group("Required Options")
    req_opts.add_argument("--scenario_file", required=True,
                          help="scenario file in AClib format")
    req_opts.add_argument("--runhistory", required=True,
                          help="runhistory file")

    req_opts.add_argument("--verbose_level", default=logging.INFO,
                          choices=["INFO", "DEBUG"],
                          help="random seed")

    args_ = parser.parse_args()

    logging.basicConfig(level=args_.verbose_level)
    if args_.verbose_level == "DEBUG":
        logging.parent.level = 10

    scen = Scenario(args_.cenario_file)
    hist = RunHistory()
    hist.load_json(fn=args_.runhistoy, cs=scen.cs)

    epm_imp = EPMImportance(scenario=scen,
                            runhistory=hist)

    epm_imp.run()
