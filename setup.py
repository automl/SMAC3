#!/usr/bin/env python3
import json
import os
from setuptools import setup

__copyright__ = "Copyright 2021, AutoML.org Freiburg-Hannover"
__license__ = "3-clause BSD"


with open('requirements.txt') as fh:
    requirements = [line.strip() for line in fh.readlines()]
with open('extras_require.json') as fh:
    extras_require = json.load(fh)
    extras_require['all'] = set(sum(extras_require.values(), []))


def get_version():
    version_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "smac", "__init__.py")
    for line in open(version_file, encoding='utf-8'):
        if line.startswith("__version__"):
            version = line.split("=")[1].strip().replace("'", "").replace('"', '')
            return version
    raise RuntimeError("Unable to find version string in %s" % version_file)


def get_author():
    version_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "smac", "__init__.py")
    for line in open(version_file, encoding='utf-8'):
        if line.startswith("__author__"):
            version = line.split("=")[1].strip().replace("'", "").replace('"', '')
            return version
    raise RuntimeError("Unable to find author string in %s" % version_file)


setup(
    python_requires=">=3.7",
    install_requires=requirements,
    extras_require=extras_require,
    package_data={'smac': ['requirements.txt', 'extras_require.json']},
    author=get_author(),
    version=get_version(),
    test_suite="nose.collector",
    tests_require=["mock", "nose"],
    long_description_content_type='text/markdown',
)
