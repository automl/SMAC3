#!/usr/bin/env python
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int)
parser.add_argument("--instance", type=str)
parser.add_argument("--instance_features", type=str)
parser.add_argument("--x0", type=int)

args = parser.parse_args()

print(f"cost={args.x0}; status=SUCCESS; additional_info=blub")
