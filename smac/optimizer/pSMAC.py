import re
import tempfile
import os
import typing
import logging
import glob

from smac.runhistory.runhistory import RunHistory
from smac.configspace import ConfigurationSpace

RUNHISTORY_FILEPATTERN = 'runhistory.json'
RUNHISTORY_RE = r'runhistory\.json$'
VALIDATEDRUNHISTORY_RE = r'validated_runhistory\.json$'


def read(run_history: RunHistory,
         output_dirs: typing.Union[str, typing.List[str]],
         configuration_space: ConfigurationSpace,
         logger: logging.Logger):
    """Update runhistory with run results from concurrent runs of pSMAC.

    Parameters
    ----------
    run_history : smac.runhistory.RunHistory
        RunHistory object to be updated with run information from runhistory
        objects stored in the output directory.
    output_dirs : typing.Union[str,typing.List[str]]
        List of SMAC output directories
        or Linux path expression (str) which will be casted into a list with
        glob.glob(). This function will search the output directories
        for files matching the runhistory regular expression.
    configuration_space : ConfigSpace.ConfigurationSpace
        A ConfigurationSpace object to check if loaded configurations are valid.
    logger : logging.Logger
    """
    numruns_in_runhistory = len(run_history.data)
    initial_numruns_in_runhistory = numruns_in_runhistory

    if isinstance(output_dirs, str):
        parsed_output_dirs = glob.glob(output_dirs)
        if glob.glob(os.path.join(output_dirs, "run_*")):
            parsed_output_dirs += glob.glob(os.path.join(output_dirs, "run_*"))
    else:
        parsed_output_dirs = output_dirs

    for output_directory in parsed_output_dirs:
        for file_in_output_directory in os.listdir(output_directory):
            match = re.match(RUNHISTORY_RE, file_in_output_directory)
            valid_match = re.match(VALIDATEDRUNHISTORY_RE, file_in_output_directory)
            if match or valid_match:
                runhistory_file = os.path.join(output_directory,
                                               file_in_output_directory)
                run_history.update_from_json(runhistory_file,
                                             configuration_space)

                new_numruns_in_runhistory = len(run_history.data)
                difference = new_numruns_in_runhistory - numruns_in_runhistory
                logger.debug('Shared model mode: Loaded %d new runs from %s' %
                             (difference, runhistory_file))
                numruns_in_runhistory = new_numruns_in_runhistory

    difference = numruns_in_runhistory - initial_numruns_in_runhistory
    logger.info('Shared model mode: Finished loading new runs, found %d new '
                 'runs.' % difference)


def write(run_history: RunHistory, output_directory: str,
          logger: logging.Logger):
    """Write the runhistory to the output directory.

    Overwrites previously outputted runhistories.

    Parameters
    ----------
    run_history : ~smac.runhistory.runhistory.RunHistory
        RunHistory object to be saved.

    output_directory : str

    logger : logging.Logger
    """

    output_filename = os.path.join(output_directory, RUNHISTORY_FILEPATTERN)

    logging.debug("Saving runhistory to %s" %(output_filename))

    with tempfile.NamedTemporaryFile('wb', dir=output_directory,
                                     delete=False) as fh:
        temporary_filename = fh.name

    run_history.save_json(temporary_filename, save_external=False)
    os.rename(temporary_filename, output_filename)
