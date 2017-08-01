import os
import shutil

import typing
from ConfigSpace import ConfigurationSpace

from smac.configspace import pcs_new as pcs


class OutputWriter(object):
    """Writing scenario to file."""

    def __init__(self):
        pass

    def write_scenario_file(self, scenario):
        """Write scenario to a file (format is compatible with input_reader).
        Will overwrite if file exists. If you have arguments that need special
        parsing when saving, specify so in the _parse_argument-function.
        Creates output-dir if necessesary.

        Parameters
        ----------
            scenario: Scenario
                Scenario to be written to file

        Returns
        -------
            status: False or None
                False indicates that writing process failed
        """
        if scenario.output_dir is None or scenario.output_dir == "":
            scenario.logger.info("No output directory for scenario logging "
                                 "specified -- scenario will not be logged.")
            return False
        # Create output-dir if necessary
        if not os.path.isdir(scenario.output_dir):
            scenario.logger.debug("Output directory does not exist! Will be "
                              "created.")
            try:
                os.makedirs(scenario.output_dir)
            except OSError:
                raise OSError("Could not make output directory: "
                              "{}.".format(scenario.output_dir))

        # options_dest2name maps scenario._arguments from dest -> name
        options_dest2name = {(scenario._arguments[v]['dest'] if
            scenario._arguments[v]['dest'] else v) : v for v in scenario._arguments}

        # Write all options into "output_dir/scenario.txt"
        path = os.path.join(scenario.output_dir, "scenario.txt")
        scenario.logger.debug("Writing scenario-file to {}.".format(path))
        with open(path, 'w') as fh:
            for key in options_dest2name:
                new_value = self._parse_argument(scenario, key, getattr(scenario, key))
                if new_value is not None:
                    fh.write("{} = {}\n".format(options_dest2name[key], new_value))

    def _parse_argument(self, scenario, key: str, value):
        """Some values of the scenario-file need to be changed upon writing,
        such as the 'ta' (target algorithm), due to it's callback. Also,
        the configspace, features, train_inst- and test-inst-lists are saved
        to output_dir, if they exist.

        Parameters:
        -----------
            scenario: Scenario
                Scenario-file to be written
            key: string
                Name of the attribute in scenario-file
            value: Any
                Corresponding attribute

        Returns:
        --------
            new value: string
                The altered value, to be written to file

        Sideeffects:
        ------------
          - copies files pcs_fn, train_inst_fn, test_inst_fn and feature_fn to
            output if possible, creates the files from attributes otherwise
        """
        if key in ['pcs_fn', 'train_inst_fn', 'test_inst_fn', 'feature_fn',
                   'output_dir']:
            # Copy if file exists, else write to new file
            if value is not None and os.path.isfile(value):
                try:
                    return shutil.copy(value, scenario.output_dir)
                except shutil.SameFileError:
                    return value  # File is already in output_dir
            elif key == 'pcs_fn' and scenario.cs is not None:
                new_path = os.path.join(scenario.output_dir, "configspace.pcs")
                self.write_pcs_file(scenario.cs, new_path)
            elif key == 'train_inst_fn' and scenario.train_insts != [None]:
                new_path = os.path.join(scenario.output_dir, 'train_insts.txt')
                self.write_inst_file(scenario.train_insts, new_path)
            elif key == 'test_inst_fn' and scenario.test_insts != [None]:
                new_path = os.path.join(scenario.output_dir, 'test_insts.txt')
                self.write_inst_file(scenario.test_insts, new_path)
            elif key == 'feature_fn' and scenario.feature_dict != {}:
                new_path = os.path.join(scenario.output_dir, 'features.txt')
                self.write_inst_features_file(scenario.n_features,
                                              scenario.feature_dict, new_path)
            else:
                return None
            # New value -> new path
            return new_path
        elif key == 'ta' and value is not None:
            # Reversing the callback on 'ta' (shlex.split)
            return " ".join(value)
        elif key in ['train_insts', 'test_insts', 'cs', 'feature_dict']:
            # No need to log, recreated from files
            return None
        else:
            return value

    def write_inst_file(self, insts: typing.List[str], fn: str):
        """Writes instance-list to file.

        Parameters
        ----------
            insts: list<string>
                 Instance list to be written
            fn: string
                 Output path
        """
        with open(fn, 'w') as fh:
            fh.write("\n".join(insts))

    def write_inst_features_file(self, n_features: int, feat_dict, fn: str):
        """Writes features to file.

        Parameters
        ----------
            n_features: int
                 Number of features
            feat_dict: dict
                 Features to be written
            fn: string
                 File name of instance feature file
        """
        header = "Instance, " + ", ".join(
            ["feature"+str(i) for i in range(n_features)]) + "\n"
        body = [", ".join([inst] + [str(f) for f in feat_dict[inst]]) + "\n"
                for inst in feat_dict]
        with open(fn, 'w') as fh:
            fh.write(header + "".join(body))

    def write_pcs_file(self, cs: ConfigurationSpace, fn: str):
        """Writing ConfigSpace to file.

        Parameters
        ----------
            cs: ConfigurationSpace
                 Config-space to be written
            fn: string
                 Output-file-path
        """
        with open(fn, 'w') as fh:
            fh.write(pcs.write(cs))
