import importlib
import pkg_resources
import re
import typing
from packaging.version import Version

__copyright__ = "Copyright 2021, AutoML.org Freiburg-Hannover"
__license__ = "3-clause BSD"


SUBPATTERN = (
    r"((?P<operation%d>==|>=|>|<)(?P<version%d>(\d+)?(\.[a-zA-Z0-9]+)?(\.\d+)?))"
)
RE_PATTERN = re.compile(
    r"^(?P<name>[\w\-]+)%s?(,%s)?$" % (SUBPATTERN % (1, 1), SUBPATTERN % (2, 2))
)


def verify_packages(packages: typing.Union[typing.List[str], str]) -> None:
    if not packages:
        return
    if isinstance(packages, str):
        packages = packages.splitlines()

    for package in packages:
        if not package:
            continue

        match = RE_PATTERN.match(package)
        if match:
            name = match.group("name")
            for group_id in range(1, 3):
                if "operation%d" % group_id in match.groupdict():
                    operation = match.group("operation%d" % group_id)
                    version = match.group("version%d" % group_id)
                    _verify_package(name, operation, version)

        else:
            raise ValueError("Unable to read requirement: %s" % package)


def _verify_package(name: str, operation: str, version: str) -> None:
    try:
        distribution = pkg_resources.get_distribution(name)
        installed_version = Version(distribution.version)
    except pkg_resources.DistributionNotFound:
        try:
            module = importlib.import_module(name)
            installed_version = Version(module.__version__)  # type: ignore[attr-defined] # noqa F821
        except ImportError:
            raise MissingPackageError(name)

    if not operation:
        return

    required_version = Version(version)

    if operation == "==":
        check = required_version == installed_version
    elif operation == ">":
        check = installed_version > required_version
    elif operation == "<":
        check = installed_version < required_version
    elif operation == ">=":
        check = (
            installed_version > required_version
            or installed_version == required_version
        )
    elif operation == "<=":
        check = (
            installed_version < required_version
            or installed_version == required_version
        )
    else:
        raise NotImplementedError("operation '%s' is not supported" % operation)
    if not check:
        raise IncorrectPackageVersionError(
            name, installed_version, operation, required_version
        )


class MissingPackageError(Exception):
    error_message = "Mandatory package '{name}' not found!"

    def __init__(self, package_name: str) -> None:
        self.package_name = package_name
        super(MissingPackageError, self).__init__(
            self.error_message.format(name=package_name)
        )


class IncorrectPackageVersionError(Exception):
    error_message = (
        "'{name} {installed_version}' version mismatch ({operation}{required_version})"
    )

    def __init__(
        self,
        package_name: str,
        installed_version: Version,
        operation: str,
        required_version: Version,
    ) -> None:
        self.package_name = package_name
        self.installed_version = installed_version
        self.operation = operation
        self.required_version = required_version
        message = self.error_message.format(
            name=package_name,
            installed_version=installed_version,
            operation=operation,
            required_version=required_version,
        )
        super(IncorrectPackageVersionError, self).__init__(message)
