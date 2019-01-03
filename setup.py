#!/usr/bin/env python3
from setuptools import setup

with open('requirements.txt') as fh:
    requirements = fh.read()
requirements = requirements.split('\n')
requirements = [requirement.strip() for requirement in requirements]

setup(
    python_requires=">=3.5.2",
    install_requires=requirements,
    test_suite="nose.collector",
    tests_require=["mock", "nose"]
)
