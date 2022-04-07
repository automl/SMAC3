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
        "setuptools>=62.0.0",
        "types-setuptools>=57.4.12",
        # Test
        "pytest>=4.6",
        "pytest-cov",
        "pytest-xdist",
        "pytest-timeout",
        # Docs
        "automl-sphinx-theme>=0.1.8",
        # Others
        "mypy>=0.942",
        "isort>=5.10.1",
        "black>=22.3.0",
        "pydocstyle>=6.1.1",
        "flake8>=4.0.1",
        "pre-commit>=2.18.1",
    ]
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
    python_requires=">=3.8",
    install_requires=read_file(os.path.join(HERE, "requirements.txt")).split("\n"),
    extras_require=extras_require,
    test_suite="pytest",
    platforms=["Linux"],
    classifiers=[
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
