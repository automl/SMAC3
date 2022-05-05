import os

import setuptools

from smac import (
    author,
    author_email,
    description,
    package_name,
    project_urls,
    url,
    version,
)

HERE = os.path.dirname(os.path.realpath(__file__))


def read_file(filepath: str) -> str:
    """
    Read in a files contents

    Parameters
    ----------
    filepath : str
        The name of the file.

    Returns
    -------
    str
        The contents of the file.
    """

    with open(filepath, "r", encoding="utf-8") as fh:
        return fh.read()


extras_require = {
    "dev": [
        "setuptools",
        "types-setuptools",
        # Test
        "pytest>=4.6",
        "pytest-cov",
        "pytest-xdist",
        "pytest-timeout",
        # Docs
        "automl-sphinx-theme>=0.1.9",
        # Others
        "mypy",
        "isort",
        "black",
        "pydocstyle",
        "flake8",
        "pre-commit",
    ],
}

setuptools.setup(
    name=package_name,
    author=author,
    author_email=author_email,
    description=description,
    long_description=read_file(os.path.join(HERE, "README.md")),
    long_description_content_type="text/markdown",
    license="BSD 3-Clause License",
    url=url,
    project_urls=project_urls,
    version=version,
    packages=setuptools.find_packages(exclude=["tests"]),
    include_package_data=True,
    python_requires=">=3.7",
    install_requires=[
        "numpy>=1.7.1",
        "scipy>=1.7.0",
        "psutil",
        "pynisher>=0.4.1",
        "ConfigSpace>=0.5.0",
        "joblib",
        "scikit-learn>=0.22.0",
        "pyrfr>=0.8.0",
        "dask",
        "distributed",
        "emcee>=3.0.0",
    ],
    extras_require=extras_require,
    test_suite="pytest",
    platforms=["Linux"],
    classifiers=[
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Natural Language :: English",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
