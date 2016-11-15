from smac.runhistory.runhistory import RunHistory
from smac.scenario.scenario import Scenario 


def merge_foreign_data(scenario:Scenario, runhistory:RunHistory,
                       in_scenario:Scenario, in_runhistory_fn_list:list):
    '''
        extend <scenario> and <runhistory> with runhistory data from another <in_scenario> 
        assuming the same pcs, feature space, but different instances
        
        Arguments
        ---------
        scenario: Scenario
            original scenario -- feature dictionary will be extended
        runhistory: RunHistory
            original runhistory -- will be extended by further data points
        in_scenario: Scenario
            input scenario 
        in_runhistory_fn_list: list
            list of filenames of dumped runhistories wrt <in_scenario>
            
        Returns
        -------
            scenario, runhistory
    '''
    
    if scenario.n_features != in_scenario.n_features:
        raise ValueError("Feature Space has to be the same for both scenarios.")
    
    if scenario.cs != in_scenario.cs:
        raise ValueError("PCS of both scenarios have to be identical.")
    
    if scenario.cutoff != in_scenario.cutoff:
        raise ValueError("Cutoffs of both scenarios have to be identical.")
    
    # add further instance features
    scenario.feature_dict.update(in_scenario.feature_dict)
    
    for fn in in_runhistory_fn_list:
        runhistory.load_json(fn=fn, cs=scenario.cs)
    
    return scenario, runhistory
