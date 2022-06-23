import typing

import os
import shutil
import warnings

from smac.configspace import ConfigurationSpace, json, pcs_new
from smac.utils.logging import PickableLoggerAdapter

if typing.TYPE_CHECKING:
    from smac.scenario.scenario import Scenario

__copyright__ = "Copyright 2021, AutoML.org Freiburg-Hannover"
__license__ = "3-clause BSD"


class OutputWriter(object):
    """Writing scenario to file."""

    def __init__(self) -> None:
        self.logger = PickableLoggerAdapter(name=self.__module__ + "." + self.__class__.__name__)

    def write_scenario_file(self, scenario: "Scenario") -> None:
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
        if scenario.output_dir_for_this_run is None or scenario.output_dir_for_this_run == "":
            scenario.logger.info(
                "No output directory for scenario logging " "specified -- scenario will not be logged."
            )
            return

        # Create output-dir if necessary
        if not os.path.isdir(scenario.output_dir_for_this_run):
            scenario.logger.debug("Output directory does not exist! Will be " "created.")
            try:
                os.makedirs(scenario.output_dir_for_this_run)
            except OSError:
                scenario.logger.debug("Could not make output directory.", exc_info=True)
                raise OSError("Could not make output directory: " "{}.".format(scenario.output_dir_for_this_run))

        # options_dest2name maps scenario._arguments from dest -> name
        options_dest2name = {
            (scenario._arguments[v]["dest"] if scenario._arguments[v]["dest"] else v): v.lstrip("-").replace("-", "_")
            for v in scenario._arguments
        }

        # Write all options into "output_dir/scenario.txt"
        path = os.path.join(scenario.output_dir_for_this_run, "scenario.txt")
        scenario.logger.debug("Writing scenario-file to {}.".format(path))
        with open(path, "w") as fh:
            for key in options_dest2name:
                key = key.lstrip("-").replace("-", "_")
                new_value = self._parse_argument(scenario, key, getattr(scenario, key))

                # Make array to string again
                if key == "multi_objectives" and isinstance(new_value, list):
                    new_value = ",".join(new_value)

                if new_value is not None:
                    fh.write("{} = {}\n".format(options_dest2name[key], new_value))

    def _parse_argument(self, scenario: "Scenario", key: str, value: typing.Any) -> typing.Any:
        """Some values of the scenario-file need to be changed upon writing,
        such as the 'ta' (target algorithm), due to it's callback. Also,
        the configspace, features, train_inst- and test-inst-lists are saved
        to output_dir, if they exist.

        Sideeffects - copies files pcs_fn, train_inst_fn, test_inst_fn and feature_fn to
        output if possible, creates the files from attributes otherwise

        Parameters
        ----------
            scenario: Scenario
                Scenario-file to be written
            key: string
                Name of the attribute in scenario-file
            value: Any
                Corresponding attribute

        Returns
        -------
            new value: string
                The altered value, to be written to file

        """
        if key in ["pcs_fn", "train_inst_fn", "test_inst_fn", "feature_fn"]:
            # Copy if file exists, else write to new file
            if value is not None and os.path.isfile(value):
                try:
                    assert scenario.output_dir_for_this_run is not None  # please mypy
                    new_path = shutil.copy(value, scenario.output_dir_for_this_run)
                except shutil.SameFileError:
                    new_path = value  # File is already in output_dir
                # For .pcs-file, also save with the same basename as json and use json-path!
                if key == "pcs_fn" and scenario.cs is not None and value.endswith(".pcs"):  # type: ignore[attr-defined] # noqa F821
                    file_name = os.path.splitext(os.path.basename(value))[0]
                    assert scenario.output_dir_for_this_run is not None  # please mypy
                    new_path = os.path.join(scenario.output_dir_for_this_run, file_name + ".json")
                    self.save_configspace(scenario.cs, new_path, "json")  # type: ignore[attr-defined] # noqa F821
                    scenario.logger.debug("Setting the pcs_fn-attr of written scenario from %s to %s", value, new_path)
            elif key == "pcs_fn" and scenario.cs is not None:  # type: ignore[attr-defined] # noqa F821
                try:
                    assert scenario.output_dir_for_this_run is not None  # please mypy
                    pcs_path = os.path.join(scenario.output_dir_for_this_run, "configspace.pcs")
                    self.save_configspace(scenario.cs, pcs_path, "pcs_new")  # type: ignore[attr-defined] # noqa F821
                except (TypeError, ValueError) as e:
                    if isinstance(e, TypeError):
                        self.logger.error(
                            "Could not write pcs file to disk." " ConfigSpace not compatible with (new) pcs format."
                        )
                    else:
                        warnings.warn(f"{e}. No pcs file will be generated in the output.")

                assert scenario.output_dir_for_this_run is not None  # please mypy
                new_path = os.path.join(scenario.output_dir_for_this_run, "configspace.json")
                self.save_configspace(scenario.cs, new_path, "json")  # type: ignore[attr-defined] # noqa F821
            elif key == "train_inst_fn" and scenario.train_insts != [None]:
                assert scenario.output_dir_for_this_run is not None  # please mypy
                new_path = os.path.join(scenario.output_dir_for_this_run, "train_insts.txt")
                self.write_inst_file(scenario.train_insts, new_path)
            elif key == "test_inst_fn" and scenario.test_insts != [None]:
                assert scenario.output_dir_for_this_run is not None  # please mypy
                new_path = os.path.join(scenario.output_dir_for_this_run, "test_insts.txt")
                self.write_inst_file(scenario.test_insts, new_path)
            elif key == "feature_fn" and scenario.feature_dict != {}:
                assert scenario.output_dir_for_this_run is not None  # please mypy
                new_path = os.path.join(scenario.output_dir_for_this_run, "features.txt")
                self.write_inst_features_file(scenario.n_features, scenario.feature_dict, new_path)
            else:
                return None
            # New value -> new path
            return new_path
        elif key == "ta" and value is not None:
            # Reversing the callback on 'ta' (shlex.split)
            return " ".join(value)
        elif key in ["train_insts", "test_insts", "cs", "feature_dict"]:
            # No need to log, recreated from files
            return None

        return value

    def write_inst_file(self, insts: typing.List[str], fn: str) -> None:
        """Writes instance-list to file.

        Parameters
        ----------
            insts: list<string>
                 Instance list to be written
            fn: string
                 Output path
        """
        with open(fn, "w") as fh:
            fh.write("\n".join(insts))

    def write_inst_features_file(
        self,
        n_features: int,
        feat_dict: typing.Dict[str, typing.Iterable[float]],
        fn: str,
    ) -> None:
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
        header = "Instance, " + ", ".join(["feature" + str(i) for i in range(n_features)]) + "\n"
        body = [", ".join([inst] + [str(f) for f in feat_dict[inst]]) + "\n" for inst in feat_dict]
        with open(fn, "w") as fh:
            fh.write(header + "".join(body))

    def save_configspace(self, cs: ConfigurationSpace, fn: str, output_format: str) -> None:
        """Writing ConfigSpace to file.

        Parameters
        ----------
            cs : ConfigurationSpace
                Config-space to be written
            fn : str
                Output-file-path
            output_format : str
                Output format of the configuration space file. Currently,
                ``json`` and ``pcs_new`` are supported.
        """
        writers = {"pcs_new": pcs_new.write, "json": json.write}

        writer = writers.get(output_format)
        if writer:
            with open(fn, "w") as fh:
                fh.write(writer(cs))
        else:
            raise ValueError(
                "Configuration space output format %s not supported. "
                "Please choose one of %s" % (output_format, set(writers.keys()))
            )
