def get_command_line_cmd(runargs, config):
    '''
    @contact:    lindauer@informatik.uni-freiburg.de, fh@informatik.uni-freiburg.de
    Returns the command line call string to execute the target algorithm (here: Spear).
    Args:
        runargs: a map of several optional arguments for the execution of the target algorithm.
                {
                  "instance": <instance>,
                  "specifics" : <extra data associated with the instance>,
                  "cutoff" : <runtime cutoff>,
                  "runlength" : <runlength cutoff>,
                  "seed" : <seed>
                }
        config: a mapping from parameter name to parameter value
    Returns:
        A command call list to execute the target algorithm.
    '''
    solver_binary = "target_algorithm/spear-python/Spear-32_1.2.1"
    cmd = "%s --seed %d --model-stdout --dimacs %s" %(solver_binary, runargs["seed"], runargs["instance"])       
    for name, value in config.items():
        cmd += " -%s %s" %(name,  value)
        
    return cmd
