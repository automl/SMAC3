import os
import typing
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("PlotTraj")

import matplotlib.pyplot as plt

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

from smac.utils.io.traj_logging import TrajLogger
from smac.scenario.scenario import Scenario
from smac.facade.smac_ac_facade import SMAC4AC
from smac.configspace import convert_configurations_to_array

__author__ = "Marius Lindauer"
__copyright__ = "Copyright 2017, ML4AAD"
__license__ = "3-clause BSD"


def setup_SMAC_from_file(smac_out_dn: str,
                         add_dn: typing.List[str]):
    '''
        read all files from disk
        and initialize SMAC data structures with it

        Arguments
        ---------
        smac_out_dn: str
            output directory name of a SMAC run
        add_dn: typing.List[str]
            additional output directories of SMAC runs
            to extend runhistory
            (Assumption: All output directories correspond
            to the same scenario)

        Returns
        -------
        smac: SMAC()
            SMAC Facade object
        traj: typing.List
            list of trajectory entries (dictionaries)
    '''

    scenario_fn = os.path.join(smac_out_dn, "scenario.txt")
    scenario = Scenario(scenario_fn, {"output_dir": ""})
    smac = SMAC4AC(scenario=scenario)

    rh = smac.solver.runhistory
    rh.load_json(os.path.join(smac_out_dn, "runhistory.json"), cs=scenario.cs)

    for dn in add_dn:
        rh.update_from_json(fn=os.path.join(dn, "runhistory.json"), cs=scenario.cs)

    logger.info("Fit EPM on %d observations." % (len(rh.data)))
    X, Y = smac.solver.rh2EPM.transform(rh)
    smac.solver.model.train(X, Y)

    traj = TrajLogger.read_traj_aclib_format(fn=os.path.join(smac_out_dn, "traj_aclib2.json"),
                                             cs=scenario.cs)

    return smac, traj


def predict_perf_of_traj(traj, smac: SMAC4AC):
    '''
        predict the performance of all entries in the trajectory
        marginalized across all instances

        Arguments
        ---------
        smac: SMAC()
            SMAC Facade object
        traj: typing.List
            list of trajectory entries (dictionaries)

        Returns
        -------
        perfs: typing.List[float]
            list of performance values
        time_stamps: typing.List[float]
            list of time stamps -- in the same order as perfs
    '''

    logger.info("Predict performance of %d entries in trajectory." %
                (len(traj)))
    time_stamps = []
    perfs = []
    for entry in traj:
        config = entry["incumbent"]
        wc_time = entry["wallclock_time"]
        config_array = convert_configurations_to_array([config])
        m, v = smac.solver.model.predict_marginalized_over_instances(
            X=config_array)

        if smac.solver.scenario.run_obj == "runtime":
            p = 10 ** m[0, 0]
        else:
            p = m[0, 0]

        perfs.append(p)
        time_stamps.append(wc_time)

    return perfs, time_stamps


def plot(x: typing.List[float], y: typing.List[float], out_dir: str):
    '''
        plot x vs y and save in out_dir

        Arguments
        ---------
        x: typing.List[float]
            time stamps
        y:typing.List[float]
            predicted performance values
        out_dir: str
            output directory to save plot
    '''
    plt.plot(x, y)

    plt.semilogx()
    plt.ylabel("Average Cost")
    plt.xlabel("Configuration Time")
    plt.title("Predicted Performance of Incumbents over Time")

    out_fn = os.path.join(out_dir, "pred_perf_over_time.png")

    logger.info("Plot average performance and save at %s" % (out_fn))

    plt.savefig(out_fn)


if __name__ == "__main__":
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("--smac_output_dir", required=True,
                        help="Output directory of SMAC")
    parser.add_argument("--additional_data", nargs="*",
                        help="Further output directory of SMAC which is used to extend the runhistory")
    args_ = parser.parse_args()

    smac, traj = setup_SMAC_from_file(smac_out_dn=args_.smac_output_dir,
                                      add_dn=args_.additional_data)

    perfs, times = predict_perf_of_traj(traj=traj, smac=smac)

    plot(x=times, y=perfs, out_dir=args_.smac_output_dir)
