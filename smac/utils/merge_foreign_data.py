from smac.runhistory.runhistory import RunHistory, DataOrigin
from smac.scenario.scenario import Scenario
from smac.configspace import ConfigurationSpace

import typing

__copyright__ = "Copyright 2021, AutoML.org Freiburg-Hannover"
__license__ = "3-clause BSD"


def merge_foreign_data_from_file(scenario: Scenario,
                                 runhistory: RunHistory,
                                 in_scenario_fn_list: typing.List[str],
                                 in_runhistory_fn_list: typing.List[str],
                                 cs: ConfigurationSpace,) -> typing.Tuple[Scenario, RunHistory]:
    """Extend <scenario> and <runhistory> with runhistory data from another
    <in_scenario> assuming the same pcs, feature space, but different instances

    Parameters
    ----------
    scenario: Scenario
        original scenario -- feature dictionary will be extended
    runhistory: RunHistory
        original runhistory -- will be extended by further data points
    in_scenario_fn_list: typing.List[str]
        input scenario file names
    in_runhistory_fn_list: typing.List[str]
        list filenames of runhistory dumps
    cs: ConfigurationSpace
        parameter configuration space to read runhistory from file

    Returns
    -------
    scenario: Scenario
    runhistory: Runhistory
    """

    if not in_scenario_fn_list:
        raise ValueError("To read warmstart data from previous runhistories,"
                         " the corresponding scenarios are required. Use option --warmstart_scenario")
    scens = [Scenario(scenario=scen_fn, cmd_options={"output_dir": ""}) for scen_fn in in_scenario_fn_list]
    rhs = []
    for rh_fn in in_runhistory_fn_list:
        rh = RunHistory()
        rh.load_json(rh_fn, cs)
        rhs.append(rh)

    return merge_foreign_data(scenario, runhistory, in_scenario_list=scens, in_runhistory_list=rhs)


def merge_foreign_data(scenario: Scenario,
                       runhistory: RunHistory,
                       in_scenario_list: typing.List[Scenario],
                       in_runhistory_list: typing.List[RunHistory]) -> typing.Tuple[Scenario, RunHistory]:
    """Extend <scenario> and <runhistory> with runhistory data from another
    <in_scenario> assuming the same pcs, feature space, but different instances

    Parameters
    ----------
    scenario: Scenario
        original scenario -- feature dictionary will be extended
    runhistory: RunHistory
        original runhistory -- will be extended by further data points
    in_scenario_list: typing.List[Scenario]
        input scenario
    in_runhistory_list: typing.List[RunHistory]
        list of runhistories wrt <in_scenario>

    Returns
    -------
    scenario: Scenario
    runhistory: Runhistory
    """
    # add further instance features
    for in_scenario in in_scenario_list:
        if scenario.n_features != in_scenario.n_features:
            raise ValueError("Feature Space has to be the same for both scenarios (%d vs %d)." % (
                scenario.n_features, in_scenario.n_features))

        if scenario.cs != in_scenario.cs:  # type: ignore[attr-defined] # noqa F821
            raise ValueError("PCS of both scenarios have to be identical.")

        if scenario.cutoff != in_scenario.cutoff:  # type: ignore[attr-defined] # noqa F821
            raise ValueError("Cutoffs of both scenarios have to be identical.")

        scenario.feature_dict.update(in_scenario.feature_dict)

    # extend runhistory
    for rh in in_runhistory_list:
        runhistory.update(rh, origin=DataOrigin.EXTERNAL_DIFFERENT_INSTANCES)

    for date in runhistory.data:
        if scenario.feature_dict.get(date.instance_id) is None:
            raise ValueError(
                "Instance feature for \"%s\" was not found in scenario data." % (date.instance_id))

    runhistory.compute_all_costs(instances=scenario.train_insts)

    return scenario, runhistory
