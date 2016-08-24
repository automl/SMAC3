#!/bin/python

__author__ = "Marius Lindauer"
__copyright__ = "Copyright 2016, ML4AAD"
__license__ = "3-clause BSD"
__maintainer__ = "Marius Lindauer"
__email__ = "lindauer@cs.uni-freiburg.de"
__version__ = "0.0.2"

import os
import sys
import inspect
import logging
import time
import math

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import numpy as np
from sklearn.decomposition import PCA
#from sklearn.ensemble import RandomForestRegressor

cmd_folder = os.path.realpath(
    os.path.abspath(os.path.split(inspect.getfile(inspect.currentframe()))[0]))
cmd_folder = os.path.realpath(os.path.join(cmd_folder, ".."))
if cmd_folder not in sys.path:
    sys.path.append(cmd_folder)

from smac.scenario.scenario import Scenario
from smac.runhistory.runhistory import RunHistory
from smac.epm.rf_with_instances import RandomForestWithInstances

from utils.set_up import convert_data

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

class ForwardSelection(object):

    def __init__(self, scenario: Scenario, runhistory: RunHistory, max_it: int=10):
        '''
        Constructor

        Parameters
        ---------
        scenario: Scenario
            scenario object
        runhistory: RunHistory
            runhistory object to learn the EPM from it
        max_it: int
            maximal number of iterations
        '''

        self.logger = logging.getLogger("ForwardSelection")

        self.X, self.Y, self.types = convert_data(scenario=scenario,
                                                   runhistory=runhistory)

        self.scen = scenario
        self.params = scen.cs.get_hyperparameters()
        self._MAX_P = min(max_it, len(self.params))

    def run(self, save_fn: str=None):
        '''
            forward selection on SMAC's EPM (RF) wrt configuration space
            to minimize the out-of-bag error returned by the RF

            Parameters
            ----------
            save_fn:str
                file name to save plot

            Returns
            -------
            list 
                tuples of parameter name and oob score
        '''

        importance_tuples = []
        X = self.X
        y = self.Y

        param_ids = list(range(len(self.params)))
        used = []
        # always use all features
        used.extend(range(len(self.params), len(self.types)))


        pca = PCA(n_components=min(7, len(self.types) - len(self.params)))
        self.scen.feature_array = pca.fit_transform(self.scen.feature_array)

        for _ in range(self._MAX_P):
            scores = []
            for p in param_ids:

                self.logger.debug(self.params[p])
                used.append(p)
                X_l = X[:, used]

                model = RandomForestWithInstances(self.types[used],
                                                  self.scen.feature_array)
                model.rf.compute_oob_error = True

                start = time.time()
                model.train(X_l, y)
                self.logger.debug("End Fit RF (sec %.2f; oob: %.4f)" % (
                    time.time() - start, model.rf.out_of_bag_error()))

                #==============================================================
                # start = time.time()
                # rf = RandomForestRegressor(n_estimators=30,
                #                            min_samples_split=3,
                #                            min_samples_leaf=3,
                #                            max_features=math.ceil(
                #                                (5. / 6.) * X_l.shape[1]),
                #                            max_leaf_nodes=1000,
                #                            max_depth=20, oob_score=True)
                # rf.fit(X_l, y.ravel())
                # self.logger.debug("End Fit Sklearn RF (sec %.2f, oob: %.4f))" % (
                #     time.time() - start, rf.oob_score_))
                #==============================================================

                score = model.rf.out_of_bag_error()
                scores.append(score)
                used.pop()

            best_indx = np.argmin(scores)
            best_score = scores[best_indx]
            p = param_ids.pop(best_indx)
            used.append(p)

            self.logger.info("%s : %.4f (OOB)" %
                             (self.params[p].name, best_score))
            importance_tuples.append((self.params[p].name, best_score))

        self.plot_importance(importance_tuples=importance_tuples, 
                             save_fn=save_fn)
        return importance_tuples

    def plot_importance(self, importance_tuples: list, save_fn=None):
        '''
            plot oob score as bar charts

            Parameters
            ----------
            importance_tuples:list
                list of tuples (parameter name, oob score)
            save_fn:str
                file name to save plot
        '''

        fig, ax = plt.subplots()
        scores = list(map(lambda x: x[1], importance_tuples))
        params = list(map(lambda x: x[0], importance_tuples))

        ind = np.arange(len(scores))
        bar_plot = ax.bar(ind, scores, color='b')

        ax.set_ylabel('Out-Of-Bag Error')
        ax.set_xticks(ind+0.5)
        ax.set_xticklabels(params, rotation=30, ha='right')

        plt.tight_layout()
        if save_fn:
            fig.savefig(save_fn)
        else:
            fig.show()


if __name__ == "__main__":

    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    req_opts = parser.add_argument_group("Required Options")
    req_opts.add_argument("--scenario_file", required=True,
                          help="scenario file in AClib format")
    req_opts.add_argument("--runhistory", required=True, nargs="+",
                          help="runhistory files")

    req_opts.add_argument("--verbose_level", default=logging.INFO,
                          choices=["INFO", "DEBUG"],
                          help="random seed")

    req_opts.add_argument("--save_fn", default="fw_importance.pdf",
                          help="file name of saved plot")

    args_ = parser.parse_args()

    logging.basicConfig(level=args_.verbose_level)
    # if args_.verbose_level == "DEBUG":
    #    logging.parent.level = 10

    scen = Scenario(args_.scenario_file)
    hist = RunHistory()
    for runhist_fn in args_.runhistory:
        hist.update_from_json(fn=runhist_fn, cs=scen.cs)

    fws = ForwardSelection(scenario=scen,
                           runhistory=hist)

    fws.run(save_fn=args_.save_fn)
