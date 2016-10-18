import os
import setuptools

import smac


requirements = ['setuptools',
                'numpy>=1.6.1',
                'scipy>=0.13.1',
                'pyrfr',
                'ConfigSpace>=0.2.1',
                'pynisher>=0.4.1']

setuptools.setup(
    name="smac",
    version=smac.VERSION,
    author=smac.AUTHORS,
    author_email="fh@cs.uni-freiburg.de",
    description=("SMAC3, a Python implementation of 'Sequential Model-based "
                 "Algorithm Configuration'."),
    license="3-clause BSD",
    keywords="machine learning algorithm configuration hyperparameter "
             "optimization tuning",
    url="",
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
    tests_require=['mock',
                   'nose'],
    test_suite='nose.collector'
)
