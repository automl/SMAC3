import re
import tempfile
import os


RUNHISTORY_FILEPATTERN = '.runhistory_%d.json'
RUNHISTORY_RE = r'\.runhistory_[0-9]*\.json'


def read(run_history, output_directory, configuration_space, logger):
    """Update runhistory with run results from concurrent runs of pSMAC.

    Parameters
    ----------
    run_history : smac.runhistory.RunHistory
        RunHistory object to be updated with run information from runhistory
        objects stored in the output directory.

    output_directory : str
        SMAC output directory. This function will search the output directory
        for files matching the runhistory regular expression.

    configuration_space : ConfigSpace.ConfigurationSpace
        A ConfigurationSpace object to check if loaded configurations are valid.

    logger : logging.Logger

    """
    numruns_in_runhistory = len(run_history.data)
    initial_numruns_in_runhistory = numruns_in_runhistory

    files_in_output_directory = os.listdir(output_directory)
    for file_in_output_directory in files_in_output_directory:
        match = re.match(RUNHISTORY_RE, file_in_output_directory)
        if match:
            runhistory_file = os.path.join(output_directory,
                                           file_in_output_directory)
            run_history.update_from_json(runhistory_file,
                                         configuration_space)

            new_numruns_in_runhistory = len(run_history.data)
            difference = new_numruns_in_runhistory - numruns_in_runhistory
            logger.debug('Shared model mode: Loaded %d new runs from %s' %
                         (difference, file_in_output_directory))
            numruns_in_runhistory = new_numruns_in_runhistory

    difference = numruns_in_runhistory - initial_numruns_in_runhistory
    logger.debug('Shared model mode: Finished loading new runs, found %d new '
                 'runs.' % difference)


def write(run_history, output_directory, num_run):
    """Write the runhistory to the output directory.

    Overwrites previously outputted runhistories.

    Parameters
    ----------
    run_history : smac.runhistory.RunHistory
        RunHistory object to be saved.

    output_directory : str

    run_run : int
        ID of the current SMAC run.

    """

    output_filename = os.path.join(output_directory,
                                   RUNHISTORY_FILEPATTERN % num_run)

    with tempfile.NamedTemporaryFile('wb', dir=output_directory,
                                     delete=False) as fh:
        temporary_filename = fh.name

    run_history.save_json(temporary_filename)
    os.rename(temporary_filename, output_filename)