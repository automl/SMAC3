#!/usr/bin/env python2.7
# encoding: utf-8

'''
spearWrapper -- AClib target algorithm warpper for SAT solver spear

@author:     Marius Lindauer, Chris Fawcett, Alex Fréchette, Frank Hutter
@copyright:  2014 AClib. All rights reserved.
@license:    GPL
@contact:    lindauer@informatik.uni-freiburg.de, fawcettc@cs.ubc.ca, afrechet@cs.ubc.ca, fh@informatik.uni-freiburg.de

example call (in aclib folder structure):
python -O SATCSSCWrapper.py --script ./cssc_wrapper/spearCSSCWrapper.py --runsolver-path ../../target_algorithms/runsolver/runsolver -- ../../instances/sat/data/SWV-Calysto/SWV-Calysto/DSPAM_v3.6.5__v1.1/dspam_vc9625.cnf "" 30.0 2147483647 1234 -sp-var-dec-heur 16 -sp-learned-clause-sort-heur 5 -sp-orig-clause-sort-heur 8 -sp-res-order-heur 5 -sp-clause-del-heur 2 -sp-phase-dec-heur 1 -sp-resolution 1 -sp-variable-decay 2 -sp-clause-decay 1.3 -sp-restart-inc 1.9 -sp-learned-size-factor 0.136079 -sp-learned-clauses-inc 1.1 -sp-clause-activity-inc 1.0555555555555556 -sp-var-activity-inc 1.2777777777777777 -sp-rand-phase-dec-freq 0.0010 -sp-rand-var-dec-freq 0.0010 -sp-rand-var-dec-scaling 1.1 -sp-rand-phase-scaling 1 -sp-max-res-lit-inc 2.3333333333333335 -sp-first-restart 43 -sp-res-cutoff-cls 4 -sp-res-cutoff-lits 1176 -sp-max-res-runs 3 -sp-update-dec-queue 1 -sp-use-pure-literal-rule 0

with checking solution:
python -O SATCSSCWrapper.py --sat-checker ./cssc_wrapper/SAT/SAT --sol-file ./test.txt  --script ./cssc_wrapper/spearCSSCWrapper.py --runsolver-path ../../target_algorithms/runsolver/runsolver -- ../../instances/sat/data/SWV-Calysto/SWV-Calysto/DSPAM_v3.6.5__v1.1/dspam_vc9625.cnf "" 30.0 2147483647 1234 -sp-var-dec-heur 16 -sp-learned-clause-sort-heur 5 -sp-orig-clause-sort-heur 8 -sp-res-order-heur 5 -sp-clause-del-heur 2 -sp-phase-dec-heur 1 -sp-resolution 1 -sp-variable-decay 2 -sp-clause-decay 1.3 -sp-restart-inc 1.9 -sp-learned-size-factor 0.136079 -sp-learned-clauses-inc 1.1 -sp-clause-activity-inc 1.0555555555555556 -sp-var-activity-inc 1.2777777777777777 -sp-rand-phase-dec-freq 0.0010 -sp-rand-var-dec-freq 0.0010 -sp-rand-var-dec-scaling 1.1 -sp-rand-phase-scaling 1 -sp-max-res-lit-inc 2.3333333333333335 -sp-first-restart 43 -sp-res-cutoff-cls 4 -sp-res-cutoff-lits 1176 -sp-max-res-runs 3 -sp-update-dec-queue 1 -sp-use-pure-literal-rule 0
'''

import sys
import re
import os
import imp
from subprocess import Popen, PIPE

from genericWrapper import AbstractWrapper

class SatCSSCWrapper(AbstractWrapper):
    '''
        Simple wrapper for a SAT solver (Spear)
    '''
    
    def __init__(self):
        '''
            Constructor
        '''
        AbstractWrapper.__init__(self)
        
        self.parser.add_argument("--script", dest="cssc_script", required=True, help="simple cssc script with only \"get_command_line_cmd(runargs, config)\"")
        self.parser.add_argument("--sol-file", dest="solubility_file", default=None, help="File with \"<instance> {SATISFIABLE|UNSATISFIABLE|UNKNOWN}\" ")
        self.parser.add_argument("--sat-checker", dest="sat_checker", default="./target_algorithms/sat/scripts/SAT", help="binary of SAT checker")

        self._instance = ""
        self.__cmd = ""
        
        self._FAILED_FILE = "failed_runs.txt" # in self._tmp_dir
        
    def get_command_line_args(self, runargs, config):
        '''
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
        
        ext_script = self.args.cssc_script
        if not os.path.isfile(ext_script):
            self._ta_status = "ABORT"
            self._ta_misc = "cssc script is missing - should have been at %s." % (ext_script)
            self._exit_code = 1
            sys.exit(1)
        
        loaded_script = imp.load_source("cssc", ext_script)
        
        
        cmd = loaded_script.get_command_line_cmd(runargs, config)

        # remember instance and cmd to verify the result later on
        self._instance = runargs["instance"] 
        self.__cmd = cmd

        return cmd

    def save_failed_cmd(self):
        # save command line call
        failed_file = os.path.join(self._tmp_dir, self._FAILED_FILE)
        with open(failed_file, "a") as fp:
            fp.write(self.__cmd+"\n")
            fp.flush()

    def process_results(self, filepointer, exit_code):
        '''
        Parse a results file to extract the run's status (SUCCESS/CRASHED/etc) and other optional results.
    
        Args:
            filepointer: a pointer to the file containing the solver execution standard out.
            exit_code : exit code of target algorithm
        Returns:
            A map containing the standard AClib run results. The current standard result map as of AClib 2.06 is:
            {
                "status" : <"SAT"/"UNSAT"/"TIMEOUT"/"CRASHED"/"ABORT">,
                "runtime" : <runtime of target algrithm>,
                "quality" : <a domain specific measure of the quality of the solution [optional]>,
                "misc" : <a (comma-less) string that will be associated with the run [optional]>
            }
            ATTENTION: The return values will overwrite the measured results of the runsolver (if runsolver was used). 
        '''
        self.print_d("reading solver results from %s" % (filepointer.name))
        data = str(filepointer.read())
        resultMap = {}

        # Make sure self._specific has an entry
        try:
            self._set_true_solubility()
        except ValueError:
            resultMap['status'] = 'ABORT'
            resultMap['misc'] = "SCENARIO BUG: Solubility of instance %s specified in both, instance specifics and true solubility file, but with different values" % self._instance
            return resultMap

        print("INFO: True solubility look-up yielded '%s'" % self._specifics)

        if self._ta_status == "TIMEOUT":
            resultMap['status'] = 'TIMEOUT'
            resultMap['misc'] = 'Runsolver returned TIMEOUT; disregard the rest of the output'
            return resultMap

        if re.search('s SATISFIABLE', data):
            # Solver returned "SATISFIABLE", trying to verify this
            resultMap['status'] = 'SAT'

            if not self.args.sat_checker:
                resultMap['misc'] = "SAT checker was not given; could not verify SAT"
            elif not os.path.isfile(self.args.sat_checker):
                resultMap['misc'] = "have not found %s; could not verify SAT" %(self.args.sat_checker)
            else:
                sat_checked = self._verify_SAT(filepointer)
                if sat_checked:
                    if self._specifics in ("UNSATISFIABLE", "20"):
                        # Solver managed to solve unsatisfiable instance
                        resultMap['status'] = 'ABORT'
                        resultMap['misc'] = "SCENARIO BUG: True solubility of instance %s was supposed to be UNSATISFIABLE, but we verifiably solved the instance as SATISFIABLE" % self._instance
                        self.save_failed_cmd()
                    return resultMap
                else:
                    # SAT checker returned false
                    resultMap['status'] = 'CRASHED'
                    resultMap['misc'] = "SOLVER BUG: solver returned a wrong model"
                    self.save_failed_cmd()
                    return resultMap

            # Could not use SAT checker, so we only compare to true solubility.
            if self._specifics in ("UNSATISFIABLE", "20"):
                resultMap['status'] = 'CRASHED'
                resultMap['misc'] = "SOLVER BUG: instance is UNSATISFIABLE but solver claimed it is SATISFIABLE"
                self.save_failed_cmd()

        elif re.search('s UNSATISFIABLE', data):
            # Solver returned 'UNSAT', verify this via true solubility
            resultMap['status'] = 'UNSAT'

            if self._specifics in ("SATISFIABLE", "10"):
                resultMap['status'] = 'CRASHED'
                resultMap['misc'] += "SOLVER BUG: instance is SATISFIABLE but solver claimed it is UNSATISFIABLE"
                self.save_failed_cmd()

        elif re.search('s UNKNOWN', data):
            resultMap['status'] = 'TIMEOUT'
            resultMap['misc'] = "Found s UNKNOWN line - interpreting as TIMEOUT"
            return resultMap
        elif re.search('INDETERMINATE', data):
            resultMap['status'] = 'TIMEOUT'
            resultMap['misc'] = "Found INDETERMINATE line - interpreting as TIMEOUT"
            return resultMap
        else:
            print(self._ta_status)
            resultMap['status'] = 'CRASHED'
            resultMap['misc'] = "Could not find usual SAT competition-formatted result string in %s" % data
        return resultMap

    def _verify_SAT(self, solver_output):
        '''
            verifies the model for self._instance 
            Args:
                solver_output: filepointer to solver output
            Returns:
                True if model was correct
                False if model was not correct
        '''
        cmd = [self.args.sat_checker, self._instance, solver_output.name]
        io = Popen(cmd, stdout=PIPE)
        out_, err_ = io.communicate()
        for line in out_.split("\n"):
            if "Solution verified" in line:
                self.print_d("Solution verified")
                return True
            elif "Wrong solution" in line:
                return False
        raise ValueError("%s did not work" % " ".join(cmd))

    def _set_true_solubility(self):
        '''
            Gets solubility from <self.args.solubility_file> and from instance specifics.
        '''
        sol_status = None
        if self.args.solubility_file and os.path.isfile(self.args.solubility_file):
            with open(self.args.solubility_file) as fp:
                for line in fp:
                    if line.startswith(self._instance):
                        line = line.strip("\n")
                        sol_status = line.split(" ")[1]
                        break
        if sol_status is None:
            # There is nothing in the solubility file we can confirm/reject
            if self.args.solubility_file is not None:
                print("INFO: solubility file %s was specified, but does not contain solubility of instance %s" % (self.args.solubility_file, self._instance))
            return

        if sol_status == self._specifics:
            # Solubility file and specifics agree on solubility
            pass
        elif sol_status in ("20", "UNSATISFIABLE") and self._specifics in ("20", "UNSATISFIABLE"):
            # Solubility file and specifics agree
            self._specifics = "UNSATISFIABLE"
        elif sol_status in ("10", "SATISFIABLE") and self._specifics in ("10", "SATISFIABLE"):
            # Solubility file and specifics agree
            self._specifics = "SATISFIABLE"
        elif sol_status in ("20", "UNSATISFIABLE") and self._specifics in ("10", "SATISFIABLE"):
            # Solubility file and specifics don't agree
            raise ValueError("self.specifics says 'SATISFIABLE', solubility says 'UNSATISFIABLE'")
        elif sol_status in ("10", "SATISFIABLE") and self._specifics in ("20", "UNSATISFIABLE"):
            # Solubility file and specifics don't agree
            raise ValueError("self.specifics says 'UNSATISFIABLE', solubility says 'SATISFIABLE'")
        elif self._specifics not in ("20", "UNSATISFIABLE", "10", "SATISFIABLE") and sol_status in ("20", "UNSATISFIABLE", "10", "SATISFIABLE"):
            self._specifics = sol_status
        elif self._specifics in ("20", "UNSATISFIABLE", "10", "SATISFIABLE"):
            pass
        else:
            self._specifics = "UNKNOWN"

        return
        

if __name__ == "__main__":
    wrapper = SatCSSCWrapper()
    wrapper.main()    
