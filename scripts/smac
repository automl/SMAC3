#!/usr/bin/env python

import logging
import sys
import os
import inspect
cmd_folder = os.path.realpath(os.path.abspath(os.path.split(inspect.getfile( inspect.currentframe() ))[0]))
cmd_folder = os.path.realpath(os.path.join(cmd_folder, ".."))
if cmd_folder not in sys.path:
    sys.path.insert(0,cmd_folder)

from smac.smac_cli import SMACCLI


if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)

    smac = SMACCLI()
    smac.main_cli()
