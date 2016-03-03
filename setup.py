import os
import setuptools


requirements = ['setuptools',
                'numpy>=1.9.1',
                'scipy>=0.16.1',
                'pyrfr',
                'ConfigSpace']

setuptools.setup(
    name="smac",
    version="0.0.1dev",
    author="Marius Lindauer, Matthias Feurer, Katharina Eggensperger, Aaron Klein, Stefan Falkner and Frank Hutter",
    author_email="fh@cs.uni-freiburg.de",
    description=("SMAC3, a python implementation of 'Sequential Model-based "
                 "Algorithm Configuration'."),
    license="GPLv3",
    keywords="machine learning algorithm configuration hyperparameter "
             "optimization tuning",
    url="",
    packages=setuptools.find_packages(exclude=['test', 'source']),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Topic :: Utilities",
        "License :: OSI Approved :: GPLv3 License",
    ],
    platforms=['Linux'],
    install_requires=requirements,
    test_suite='nose.collector'
)
