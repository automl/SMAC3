from enum import Enum


class IntensifierStage(Enum):
    """Class to define different stages of intensifier."""

    RUN_FIRST_CONFIG = 0  # to replicate the old initial design
    RUN_INCUMBENT = 1  # Lines 3-7
    RUN_CHALLENGER = 2  # Lines 8-17
    RUN_BASIS = 3

    # helpers to determine what type of run to process
    # A challenger is assumed to be processed if the stage
    # is not from first_config or incumbent
    PROCESS_FIRST_CONFIG_RUN = 4
    PROCESS_INCUMBENT_RUN = 5
