#!/usr/bin/env ruby

require 'fileutils'

def float_regexp()
        return '[+-]?\d+(?:\.\d+)?(?:[eE][+-]\d+)?';
end

def get_true_solubility
	File.open($true_solubility_file){|file|
		while line = file.gets
			entries = line.split
			if entries[0].eql?($orig_input_file)
				return entries[1]
			end
		end
	}
end

if ARGV.length < 6
    puts "Usage: ruby generic_solver_wrapper <solver_directory> <instance_file_name>
<instance_specifics (string in \"quotes\"> <cutoff_time> <cutoff_length> <seed> <params to be
passed on, as -param_name param_value pairs."
    exit -1
end

#=== Read inputs.
solver_directory_name = ARGV[0]
input_file = ARGV[1]
instance_specifics = ARGV[2]
timeout = [ARGV[3].to_f,1].max
cutoff_length = ARGV[4].to_f
seed = ARGV[5].to_i


#=== Require solver-specific callstring file, and generate absolute paths.
require File.expand_path("#{solver_directory_name}/callstring_generator.rb")
$true_solubility_file = File.expand_path("./instances/sat/true_solubility.txt") 
$orig_input_file = input_file
runsolver_executable = File.expand_path("./target_algorithms/runsolver/runsolver")
SAT_executable = File.expand_path("./target_algorithms/sat/scripts/SAT")

if (!File.exists?("./SCRATCH"))
	FileUtils.mkdir("./SCRATCH")
end

randnum = rand
tmp_file = File.expand_path("./SCRATCH/solver_output_#{randnum}.txt")
tmp_runsolve = File.expand_path("./SCRATCH/runsolver_output_#{randnum}.txt")
tmp_checking_outfilename = File.expand_path("./SCRATCH/result_check_#{randnum}.txt")


#=== Form hashmap of partial parameter configuration
remainingArgs = ARGV[6..(ARGV.length-1)]
param_hashmap = {}
remainingArgs.each_index { |i|
        if (i % 2 == 0)
		param_name = remainingArgs[i]
		param_name = param_name.sub(/^-/, "")   #remove leading -
		param_value = remainingArgs[i+1]
		# Remove all quotes:
		param_value = param_value.gsub(/'/,"")
		param_hashmap[param_name] = param_value
	end
}		


#=== If instance is zipped, unzip it. 
created_instance_file=nil
if input_file =~ /.bz2$/ 
	instance_filename=" spear_instance#{rand}.cnf"
	system "bunzip2 --stdout #{input_file} > #{instance_filename}"
	created_instance_file=1
	instance_created = instance_filename
else
	instance_filename=input_file
end

Signal.trap("TERM") {
        #=== Respond to termination by deleting temporary file and crashing.
        begin
                puts "Result for ParamILS: CRASHED, 0, 0, 0, #{$seed}"
	        File.delete(tmp_checking_outfilename)
	        File.delete(tmp_file)
        	File.delete(tmp_runsolve)
	        if created_instance_file
        	        File.delete(instance_created)
	        end
        ensure
                Process.exit 1
        end
}


#=== Go into solver directory and call the solver.
memout="3000"
solver_callstring = get_solver_callstring(instance_filename, seed, param_hashmap)
cmd = "#{runsolver_executable} --timestamp -w #{tmp_runsolve} -o #{tmp_file} -C #{timeout} -M #{memout} #{solver_callstring}"
exec_cmd = "#{cmd}"
STDERR.puts "Call to runsolver: #{exec_cmd}"
system exec_cmd
#runsolver_exitcode = $?.exitstatus


#=== Parse algorithm output to extract relevant information for ParamILS.
solved = "CRASHED"
runtime = nil
witness=nil
File.open(tmp_file){|file|
    while line = file.gets
        if line =~ /s UNSATISFIABLE/
            solved = "UNSAT"
        end
        if line =~ /s SATISFIABLE/
            solved = "SAT"
        end
   end
}

#=== Parse runsolver output to get runtime and to count successes past the cutoff as timeouts.
File.open(tmp_runsolve){|file|
    while line = file.gets
        if line =~ /runsolver_max_cpu_time_exceeded/ or line =~ /Maximum CPU time exceeded/
                solved = "TIMEOUT"
        end
        if line =~ /CPU time \(s\): (#{float_regexp})/
            runtime = $1.to_f
        end
    end
}

#=== Check correctness of solution (solubility status and witness)
if solved == "SAT" or solved == "UNSAT"
	# Check the solubility status against the correct one, call differences crashed.
	true_solubility = get_true_solubility
        true_solubility = instance_specifics
        puts "#{true_solubility} #{solved}"
	solved = "CRASHED" unless ((true_solubility == "SATISFIABLE" && solved == "SAT") || true_solubility == "UNSATISFIABLE" && solved == "UNSAT") || true_solubility == "UNKNOWN"
end


if solved == "SAT" # check the witness
	#solved = "CRASHED" # if we can't verify it we'll call it crashed.
=begin
# Can't easily get the actual exitcode, so using the manual parsing below instead.
	checking_cmd = "#{SAT_executable} #{instance_filename} #{tmp_file}"
	puts "Checking the solution with cmd: #{checking_cmd}"
	system checking_cmd

	p $?
	exitcode = $?.exitstatus
	p exitcode
	if exitcode == 10 || exitcode == 11
		solved = "SAT"
	end
=end

	checking_cmd = "#{SAT_executable} #{instance_filename} #{tmp_file} > #{tmp_checking_outfilename}"
	puts "Checking the solution with cmd: #{checking_cmd}"
	system checking_cmd
	File.open(tmp_checking_outfilename){|file|
		while line = file.gets
			if line =~ /Solution verified./
				solved = "SAT" 
			end
		end
	}
	
end

#=== Output for configurators.
puts "Result for ParamILS: #{solved}, #{runtime}, 0, 0, #{seed}"


unless solved == "CRASHED"
	#=== Tidy up.
	File.delete(tmp_checking_outfilename) if solved == "SAT"
	File.delete(tmp_file)
	File.delete(tmp_runsolve)
	if created_instance_file
        	File.delete(instance_created)
	end
end
