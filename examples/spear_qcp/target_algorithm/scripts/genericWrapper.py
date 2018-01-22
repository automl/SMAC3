#!/usr/bin/env python2.7
# encoding: utf-8

'''
genericWrapper -- template for an AClib target algorithm wrapper
abstract methods for generation of callstring and parsing of solver output 
@author:     Marius Lindauer, Chris Fawcett, Alex Fréchette, Frank Hutter  
@copyright:  2014 AClib. All rights reserved.
@license:    GPL
@contact:    lindauer@informatik.uni-freiburg.de, fawcettc@cs.ubc.ca, afrechet@cs.ubc.ca, fh@informatik.uni-freiburg.de

@note: example call: python src/generic_wrapper/spearWrapper.py --runsolver ./target_algorithms/runsolver/runsolver-3.3.4/src/runsolver -- <instance> <instance specific> <cutoff> <runlength> <seed>
@warning:  use "--" after the last additional argument of the wrapper to deactivate prefix matching! 
'''

import sys
import os
import signal
import time
import re
import random
import traceback
import shutil
import json
import logging
from argparse import ArgumentParser, RawDescriptionHelpFormatter
from subprocess import Popen, PIPE
from tempfile import NamedTemporaryFile, mkdtemp

__all__ = []
__version__ = 0.1
__authors__ = 'Marius Lindauer, Chris Fawcett, Alex Fréchette, Frank Hutter'
__date__ = '2014-03-06'
__updated__ = '2014-03-21'

def signalHandler(signum, frame):
    sys.exit(2)

class AbstractWrapper(object):
    '''
        abstract solver wrapper
    '''
    
    def __init__(self):
        '''
            Constructor
        '''
        #program_name = os.path.basename(sys.argv[0])
        program_version = "v%s" % __version__
        program_build_date = str(__updated__)
        program_version_message = "%%(prog)s %s (%s)" % (program_version, program_build_date)
        program_shortdesc = __import__("__main__").__doc__.split("\n")[1]
        program_license = '''%s
    
          Created by %s on %s.
          Copyright 2014 - AClib. All rights reserved.
          
          Licensed under the GPLv2
          http://www.gnu.org/licenses/gpl-2.0.html
          
          Distributed on an "AS IS" basis without warranties
          or conditions of any kind, either express or implied.
        
          USAGE
        ''' % (program_shortdesc, str(__authors__), str(__date__))
        #self.parser = ArgumentParser(description=program_license, formatter_class=RawDescriptionHelpFormatter, add_help=False)
        self.parser = OArgumentParser()
        self.args = None

        self.RESULT_MAPPING = {'SUCCESS': "SAT"} 
        self._watcher_file = None
        self._solver_file = None

        self._instance = ""
        self._specifics = ""
        self._cutoff = 0.0
        self._runlength = 0
        self._seed = 0
        self._config_dict = {}
        
        self._exit_code = None
        
        self._runsolver = None
        self._mem_limit = 2048
        self._tmp_dir = None
        self._tmp_dir_algo = None

        self._crashed_if_non_zero_status = True
        
        self._subprocesses = []
        
        self._DEBUG = True
        self._DELAY2KILL = 2

        self._ta_status = "EXTERNALKILL"
        self._ta_runtime = 999999999.0
        self._ta_runlength = -1
        self._ta_quality = -1
        self._ta_exit_code = None
        self._ta_misc = ""
        

    def print_d(self, str_):
        if self._DEBUG:
            print(str_)
        
    def main(self, argv=None): 
        ''' parse command line'''
        if argv is None:
            argv = sys.argv
        else:
            sys.argv.extend(argv)
    
        try:
            signal.signal(signal.SIGTERM, signalHandler)
            signal.signal(signal.SIGQUIT, signalHandler)
            signal.signal(signal.SIGINT, signalHandler)

            # Setup argument parser
            
            self.parser.add_argument("--runsolver-path", dest="runsolver", default="./target_algorithm/runsolver/runsolver", help="path to runsolver binary (if None, the runsolver is deactivated)")
            self.parser.add_argument("--temp-file-dir", dest="tmp_dir", default=None, help="directory for temporary files (relative to -exec-dir in SMAC scenario). If 'NONE' use $TMPDIR if available, otherwise './'")
            self.parser.add_argument("--temp-file-dir-algo", dest="tmp_dir_algo", default=True, type=bool, help="create a directory for temporary files from target algo") #TODO: set default to False
            self.parser.add_argument("--mem-limit", dest="mem_limit", default=self._mem_limit, type=int, help="memory limit in MB")
            self.parser.add_argument("--internal", dest="internal", default=False, type=bool, help="skip calling an external target algorithm")
            self.parser.add_argument("--log", dest="log", default=True, type=bool, help="logs all runs in \"target_algo_runs.json\" in --temp-file-dir")
            self.parser.add_argument("--ext-callstring", dest="ext_callstring", default=None, help="Command to get call string via external program;" +
                                                                                             "your programm gets a file with"+
                                                                                             "first line: instance name,"+
                                                                                             "second line: seed"+
                                                                                             "further lines: paramter name, paramater value;"+ 
                                                                                             "output: one line with callstring for target algorithm")
            self.parser.add_argument("--ext-parsing", dest="ext_parsing", default=None, help="Command to use an external program to parse the output of your target algorihm;" +
                                                                                           "only paramter: name of output file;"+
                                                                                           "output of your progam:"+
                                                                                           "status: SAT|UNSAT|TIMEOUT|CRASHED\n"+
                                                                                           "quality: <integer>\n"+
                                                                                           "misc: <string>")
            self.parser.add_argument("--help", dest="show_help", default=False, type=bool, help="shows help")

            # Process arguments
            self.args, target_args = self.parser.parse_cmd(sys.argv[1:])
            args = self.args
           
            if args.show_help:
                self.parser.print_help()
                self._ta_status = "ABORT"
                self._ta_misc = "help was requested..."
                self._exit_code = 1
                sys.exit(1)

            if args.runsolver != "None" and not os.path.isfile(args.runsolver) and not args.internal:
                self._ta_status = "ABORT"
                self._ta_misc = "runsolver is missing - should have been at %s." % (args.runsolver)
                self._exit_code = 1
                sys.exit(1)
            else:
                self._runsolver = args.runsolver
                self._mem_limit = args.mem_limit

            if args.tmp_dir is None:
                if "TMPDIR" in os.environ:
                    args.tmp_dir = os.environ["TMPDIR"]
                else:
                    args.tmp_dir = "."
            
            if not os.path.isdir(args.tmp_dir):
                self._ta_status = "ABORT"
                self._ta_misc = "temp directory is missing - should have been at %s." % (args.tmp_dir)
                self._exit_code = 1
                sys.exit(1)
            else:
                self._tmp_dir = args.tmp_dir
            
            if len(target_args) < 5:
                self._ta_status = "ABORT"
                self._ta_misc = "some required TA parameters (instance, specifics, cutoff, runlength, seed) missing - was [%s]." % (" ".join(target_args))
                self._exit_code = 1
                sys.exit(1)
                
            self._config_dict = self.build_parameter_dict(target_args)

            if args.tmp_dir_algo:
                try: 
                    self._tmp_dir_algo = mkdtemp(dir="/tmp/")
                except OSError:
                    sys.stderr.write("Creating directory for temporary files failed")
                    pass
            
            runargs = {
                        "instance": self._instance,
                        "specifics" : self._specifics,
                        "cutoff" : self._cutoff,
                        "runlength" : self._runlength,
                        "seed" : self._seed,
                        "tmp" : self._tmp_dir_algo
                      }
            
            if args.ext_callstring:
                target_cmd = self.get_command_line_args_ext(runargs=runargs, config=self._config_dict, ext_call=args.ext_callstring)
            else:
                target_cmd = self.get_command_line_args(runargs=runargs, config=self._config_dict)
                      
            target_cmd = target_cmd.split(" ")
            target_cmd = filter(lambda x: x != "", target_cmd)
            
 
            if not args.internal:
                self.call_target(target_cmd)
                self.read_runsolver_output()
               
            try: 
                if "core" in os.listdir("."):
                    os.remove("core")
            except: 
                traceback.print_exc()
                
            if args.ext_parsing:
                resultMap = self.process_results_ext(self._solver_file, {"exit_code" : self._ta_exit_code}, ext_call=args.ext_parsing)
            else:
                resultMap = self.process_results(self._solver_file, {"exit_code" : self._ta_exit_code})
            
            if ('status' in resultMap):
                self._ta_status = self.RESULT_MAPPING.get(resultMap['status'],resultMap['status'])
            if ('runtime' in resultMap):
                self._ta_runtime = resultMap['runtime']
            if ('quality' in resultMap):
                self._ta_quality = resultMap['quality']
            if 'misc' in resultMap and not self._ta_misc:
                self._ta_misc = resultMap['misc']
            if 'misc' in resultMap and self._ta_misc:
                self._ta_misc +=  " - " + resultMap['misc']
                
            # if still no status was determined, something went wrong and output files should be kept
            if self._ta_status is "EXTERNALKILL":
                self._ta_status = "CRASHED"
            sys.exit()
        except (KeyboardInterrupt, SystemExit):
            self.cleanup()
            self.print_result_string()
            #traceback.print_exc()
            if self._ta_exit_code:
                sys.exit(self._ta_exit_code)
            elif self._exit_code:
                sys.exit(self._exit_code)
            else:
                sys.exit(0)
        
    def build_parameter_dict(self, arg_list):
        '''
            Reads all arguments which were not parsed by ArgumentParser,
            extracts all meta information
            and builds a mapping: parameter name -> parameter value
            Format Assumption: <instance> <specifics> <runtime cutoff> <runlength> <seed> <solver parameters>
            Args:
                list of all options not parsed by ArgumentParser
        '''
        self._instance = arg_list[0]
        self._specifics = arg_list[1]
        self._cutoff = int(float(arg_list[2]) + 1) # runsolver only rounds down to integer
        self._ta_runtime = self._cutoff
        self._runlength = int(arg_list[3])
        self._seed = int(arg_list[4])
        
        params = arg_list[5:]
        if (len(params)/2)*2 != len(params):
            self._ta_status = "ABORT"
            self._ta_misc = "target algorithm parameter list MUST have even length - found %d arguments." % (len(params))
            self.print_d(" ".join(params))
            self._exit_code = 1
            sys.exit(1)
        
        return dict((name, value.strip("'")) for name, value in zip(params[::2], params[1::2]))
        
    def call_target(self, target_cmd):
        '''
            extends the target algorithm command line call with the runsolver
            and executes it
            Args:
                list of target cmd (from getCommandLineArgs)
        '''
        logging.warning('genericWrapper: falling back to non-deterministic behaviour')
        random_id = random.randint(0, 1000000)
        self._watcher_file = NamedTemporaryFile(suffix=".log", prefix="watcher-%d-" %(random_id), dir=self._tmp_dir, delete=False)
        self._solver_file = NamedTemporaryFile(suffix=".log", prefix="solver-%d-" %(random_id), dir=self._tmp_dir, delete=False)
        
        runsolver_cmd = []
        if self._runsolver != "None":
            runsolver_cmd = [self._runsolver, "-M", self._mem_limit, "-C", self._cutoff,
                             "-w", self._watcher_file.name,
                             "-o", self._solver_file.name]
        
        runsolver_cmd.extend(target_cmd)
        #for debugging
        self.print_d("Calling runsolver. Command-line:")
        self.print_d(" ".join(map(str,runsolver_cmd)))

        # run
        try:
            if self._runsolver != "None":
                if "\"" in runsolver_cmd: # if there are quotes in the call, we cannot split it individual list elements. we have to call it via shell as a string; problematic solver: SparrowToRiss
                    runsolver_cmd = " ".join(map(str, runsolver_cmd))
                    io = Popen(runsolver_cmd, shell=True, preexec_fn=os.setpgrp,universal_newlines=True)
                else:
                    io = Popen(map(str, runsolver_cmd), shell=False, preexec_fn=os.setpgrp,universal_newlines=True)
            else:
                io = Popen(map(str, runsolver_cmd), stdout=self._solver_file, shell=False, preexec_fn=os.setpgrp,universal_newlines=True)
            self._subprocesses.append(io)
            io.wait()
            self._subprocesses.remove(io)
            if io.stdout:
                io.stdout.flush()
        except OSError:
            self._ta_status = "ABORT"
            self._ta_misc = "execution failed: %s"  % (" ".join(map(str,runsolver_cmd)))
            self._exit_code = 1 
            sys.exit(1)
            
        self._solver_file.seek(0)

    def float_regex(self):
        return '[+-]?\d+(?:\.\d+)?(?:[eE][+-]\d+)?'

    def read_runsolver_output(self):
        '''
            reads self._watcher_file, 
            extracts runtime
            and returns if memout or timeout found
        ''' 
        if self._runsolver == "None":
            self._ta_exit_code = 0
            return
        
        self.print_d("Reading runsolver output from %s" % (self._watcher_file.name))
        data = str(self._watcher_file.read())

        if (re.search('runsolver_max_cpu_time_exceeded', data) or re.search('Maximum CPU time exceeded', data)):
            self._ta_status = "TIMEOUT"

        if (re.search('runsolver_max_memory_limit_exceeded', data) or re.search('Maximum VSize exceeded', data)):
            self._ta_status = "TIMEOUT"
            self._ta_misc = "memory limit was exceeded"
           
        cpu_pattern1 = re.compile('runsolver_cputime: (%s)' % (self.float_regex()))
        cpu_match1 = re.search(cpu_pattern1, data)
            
        cpu_pattern2 = re.compile('CPU time \\(s\\): (%s)' % (self.float_regex()))
        cpu_match2 = re.search(cpu_pattern2, data)

        if (cpu_match1):
            self._ta_runtime = float(cpu_match1.group(1))
        if (cpu_match2):
            self._ta_runtime = float(cpu_match2.group(1))

        exitcode_pattern = re.compile('Child status: ([0-9]+)')
        exitcode_match = re.search(exitcode_pattern, data)

        if (exitcode_match):
            self._ta_exit_code = int(exitcode_match.group(1))
            

    def print_result_string(self):
        
        if self.args and self.args.log:
            #if not os.path.isfile("target_algo_runs.csv"):
            #    with open("target_algo_runs.csv", "a") as fp:
            #        fp.write("instance,seed,status,performance,config,[misc]\n")
            with open("target_algo_runs.json", "a") as fp:
                out_dict = {"instance": self._instance,
                            "seed": self._seed,
                            "status": self._ta_status,
                            "time": self._ta_runtime,
                            "config": self._config_dict,
                            "misc": self._ta_misc}
                json.dump(out_dict, fp)
                fp.write("\n")
                fp.flush()
        
        sys.stdout.write("Result for ParamILS: %s, %s, %s, %s, %s" % (self._ta_status, str(self._ta_runtime), str(self._ta_runlength), str(self._ta_quality), str(self._seed)))
        if (len(self._ta_misc) > 0):
            sys.stdout.write(", %s" % (self._ta_misc))
        print('')
        sys.stdout.flush()
        
    def cleanup(self):
        '''
            cleanup if error occurred or external signal handled
        '''
        if (len(self._subprocesses) > 0):
            print("killing the target run!")
            try:
                for sub in self._subprocesses:
                    #sub.terminate()
                    Popen(["pkill","-TERM", "-P",str(sub.pid)])
                    self.print_d("Wait %d seconds ..." % (self._DELAY2KILL))
                    time.sleep(self._DELAY2KILL)
                    if sub.returncode is None: # still running
                        sub.kill()

                self.print_d("done... If anything in the subprocess tree fork'd a new process group, we may not have caught everything...")
                self._ta_misc = "forced to exit by signal or keyboard interrupt."
                self._ta_runtime = self._cutoff
            except (OSError, KeyboardInterrupt, SystemExit):
                self._ta_misc = "forced to exit by multiple signals/interrupts."
                self._ta_runtime = self._cutoff

        if (self._ta_status is "ABORT" or self._ta_status is "CRASHED"):
            if (len(self._ta_misc) == 0):
                if self._ta_exit_code:
                    self._ta_misc = 'Problem with run. Exit code was %d.' % (self._ta_exit_code)
                else:
                    self._ta_misc = 'Problem with run. Exit code was N/A.'

            if (self._watcher_file and self._solver_file):
                self._ta_misc = self._ta_misc + '; Preserving runsolver output at %s - preserving target algorithm output at %s' % (self._watcher_file.name or "<none>", self._solver_file.name or "<none>")

        try:
            if (self._watcher_file):
                self._watcher_file.close()
            if (self._solver_file):
                self._solver_file.close()

            if (self._ta_status is not "ABORT" and self._ta_status is not "CRASHED"):
                os.remove(self._watcher_file.name)
                os.remove(self._solver_file.name)
                
            if self._tmp_dir_algo:
                shutil.rmtree(self._tmp_dir_algo)
                
        except (OSError, KeyboardInterrupt, SystemExit):
            self._ta_misc = "problems removing temporary files during cleanup."
        except AttributeError:
            pass #in internal mode, these files are not generated
    
        if self._ta_status is "EXTERNALKILL":
            self._ta_status = "CRASHED"
            self._exit_code = 3

    def get_command_line_args(self, runargs, config):
        '''
        Returns the command call list containing arguments to execute the implementing subclass' solver.
        The default implementation delegates to get_command_line_args_ext. If this is not implemented, a
        NotImplementedError will be raised.
    
        Args:
            runargs: a map of any non-configuration arguments required for the execution of the solver.
            config: a mapping from parameter name (with prefix) to parameter value.
        Returns:
            A command call list to execute a target algorithm.
        '''
        raise NotImplementedError()

    def get_command_line_args_ext(self, runargs, config, ext_call):
        '''
        When production of the target algorithm is done from a source other than python,
        override this method to return a command call list to execute whatever you need to produce the command line.

        Args:
            runargs: a map of any non-configuration arguments required for the execution of the solver.
            config: a mapping from parameter name (with prefix) to parameter value.
            ext_call: string to call external program to get callstring of target algorithm
        Returns:
            A command call list to execute the command producing a single line of output containing the solver command string
        '''
        callstring_in = NamedTemporaryFile(suffix=".csv", prefix="callstring", dir=self._tmp_dir, delete=False)
        callstring_in.write("%s\n" %(runargs["instance"]))
        callstring_in.write("%d\n" %(runargs["seed"]))
        for name,value in config.items():
            callstring_in.write("%s,%s\n" %(name,value))
        callstring_in.flush()
        
        cmd = ext_call.split(" ")
        cmd.append(callstring_in.name)
        self.print_d(" ".join(cmd))
        try:
            io = Popen(cmd, shell=False, preexec_fn=os.setpgrp, stdout=PIPE, universal_newlines=True)
            self._subprocesses.append(io)
            out_, _ = io.communicate()
            self._subprocesses.remove(io)
        except OSError:
            self._ta_misc = "failed to run external program for output parsing : %s" %(" ".join(cmd))
            self._ta_runtime = self._cutoff
            self._exit_code = 2
            sys.exit(2)
        if not out_ :
            self._ta_misc = "external program for output parsing yielded empty output: %s" %(" ".join(cmd))
            self._ta_runtime = self._cutoff
            self._exit_code = 2
            sys.exit(2)
        callstring_in.close()
        os.remove(callstring_in.name)
        self._instance = runargs["instance"]
        return out_.strip('\n\r\b')
    
    def process_results(self, filepointer, out_args):
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
        raise NotImplementedError()

    def process_results_ext(self, filepointer, out_args, ext_call):
        '''
        Args:
            filepointer: a pointer to the file containing the solver execution standard out.
            exit_code : exit code of target algorithm
        Returns:
            A map containing the standard AClib run results. The current standard result map as of AClib 2.06 is:
            {
                "status" : <"SAT"/"UNSAT"/"TIMEOUT"/"CRASHED"/"ABORT">,
                "quality" : <a domain specific measure of the quality of the solution [optional]>,
                "misc" : <a (comma-less) string that will be associated with the run [optional]>
            }
        '''
        
        cmd = ext_call.split(" ")
        cmd.append(filepointer.name)
        self.print_d(" ".join(cmd))
        try:
            io = Popen(cmd, shell=False, preexec_fn=os.setpgrp, stdout=PIPE, universal_newlines=True)
            self._subprocesses.append(io)
            out_, _ = io.communicate()
            self._subprocesses.remove(io)
        except OSError:
            self._ta_misc = "failed to run external program for output parsing"
            self._ta_runtime = self._cutoff
            self._exit_code = 2
            sys.exit(2)
        
        result_map = {}
        for line in out_.split("\n"):
            if line.startswith("status:"):
                result_map["status"] = line.split(":")[1].strip(" ")
            elif line.startswith("quality:"):
                result_map["quality"] = line.split(":")[1].strip(" ")
            elif line.startswith("misc:"):
                result_map["misc"] = line.split(":")[1]
        
        return result_map
        
        
class Arguments():
    '''
       parsed arguments 
    '''
    
    def __init__(self):
        '''
            Constructor
        '''
        
class OArgumentParser(object):
    '''
       my own argument parser... 
       problem with the standard argument parser is the prefix-matching when using parse_known_args() 
    '''
    
    def __init__(self):
        '''
            Constructor
        '''
        self.options = {}
        self.required = []
        self.args = Arguments()
        
    def add_argument(self, parameter_name, dest, default=None, help="", type=str, required=False):
        '''
            adds arguments to parse from command line
            Args:
                parameter_name: name of parameter
                dest: destination in returned Argument() object
                default: default value
                help: help output if --help
        '''
        setattr(self.args, dest, default)
        self.options[parameter_name] = {"dest": dest, "default": default, "help": help, "type": str}
        if required:
            self.required.append(parameter_name)
        
    def print_help(self):
        '''
            print help message
        '''
        print("")
        print("Help:")
        for name_, dict_ in self.options.items(): 
            print("\t %-20s \t %s (default: %s)" %(name_, str(dict_["help"]), str(dict_["default"])))
        print("")
        sys.exit(0)
            
    def parse_cmd(self, args):
        '''
             parse command line
        '''
        unknown_args = []
        iterator_args = iter(args)
        while True:
            try:
                name = next(iterator_args)
            except StopIteration:
                break
        #for name, value in zip(args[::2], args[1::2]):
            #if name in ["--help"]:
            #    self.print_help()
            if self.options.get(name):
                try:
                    value = next(iterator_args)
                except StopIteration:
                    sys.stderr.write("%s is missing some value\n" %(name))
                    sys.exit(2)
                dict_ = self.options.get(name)
                setattr(self.args, dict_["dest"], dict_["type"](value))
                if name in self.required:
                    self.required.remove(name)
            else: 
                unknown_args.append(name)
        
        if self.required:
            print("The following arguments are required:")
            for name_ in self.required:
                print("\t%s" %(name_))
        
        return self.args, unknown_args
    
#===============================================================================
# if __name__ == "__main__":
#     sys.exit(main())
#===============================================================================
