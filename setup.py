import setuptools
import sys

import smac


with open('requirements.txt') as fh:
    requirements = fh.read()
requirements = requirements.split('\n')
requirements = [requirement.strip() for requirement in requirements]

with open("smac/__version__.py") as fh:
    version = fh.readlines()[-1].split()[-1].strip("\"'")

if sys.version_info < (3, 5, 2):
    raise ValueError('Unsupported Python version %s found. SMAC3 requires Python 3.5.2 or higher.' % sys.version_info)

setuptools.setup(
    name="smac",
    version=version,
    author=smac.AUTHORS,
    author_email="fh@cs.uni-freiburg.de",
    description=("SMAC3, a Python implementation of 'Sequential Model-based "
                 "Algorithm Configuration'."),
    license="3-clause BSD",
    keywords="machine learning algorithm configuration hyperparameter "
             "optimization tuning",
    url="",
    entry_points={
        'console_scripts': ['smac=smac.smac_cli:cmd_line_call'],
    },
    packages=setuptools.find_packages(exclude=['test', 'source']),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Topic :: Utilities",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: BSD License",
    ],
    platforms=['Linux'],
    install_requires=requirements,
    python_requires='>=3.5.2',
    tests_require=['mock',
                   'nose'],
    test_suite='nose.collector'
)
