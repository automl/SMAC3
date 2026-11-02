import ast
from pathlib import Path


def read_metadata(file: str = "smac/__init__.py") -> dict[str, str]:
    # Parse instead of import so that the script works without SMAC's dependencies installed
    metadata = {}
    with open(file) as f:
        for node in ast.parse(f.read()).body:
            if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        metadata[target.id] = node.value.value

    return metadata


def replace(file: str, find: str, replace: str) -> None:
    new_lines = []
    with open(file) as f:
        for line in f.readlines():
            line = line.replace(find, replace)
            new_lines += [line]

    # Empty file
    with open(file, "w") as f:
        f.write("")

    # Save file with new lines
    with open(file, "a") as f:
        for line in new_lines:
            f.write(line)


def replace_if_starts_with(file: str, starts_with: str, replace: str = "") -> None:
    new_lines = []
    with open(file) as f:
        for line in f.readlines():
            if line.startswith(starts_with):
                line = replace
                if len(replace) > 0:
                    line = line + "\n"

            new_lines += [line]

    # Empty file
    with open(file, "w") as f:
        f.write("")

    # Save file with new lines
    with open(file, "a") as f:
        for line in new_lines:
            f.write(line)


def check(file: str) -> None:
    has_copyright = False
    has_license = False

    copyright = "__copyright__"
    license = "__license__"

    with open(file) as f:
        for line in f.readlines():
            if line.startswith(copyright):
                has_copyright = True

            if line.startswith(license):
                has_license = True

    if not has_copyright:
        print(f"{file} does not have copyright.")

    if not has_license:
        print(f"{file} does not have license.")


def replace_copyright_notice(file: str, notice: str, url: str) -> None:
    # Only touches our own notice (identified by the url), e.g. not the original 2015 notice in LICENSE.txt
    new_lines = []
    with open(file) as f:
        for line in f.readlines():
            if line.startswith("Copyright (c)") and url in line:
                line = notice + "\n"

            new_lines += [line]

    with open(file, "w") as f:
        f.writelines(new_lines)


if __name__ == "__main__":
    metadata = read_metadata()
    year, holder, url = metadata["copyright_year"], metadata["copyright_holder"], metadata["url"]

    for directory in ["smac", "tests", "examples"]:
        for path in Path(directory).rglob("*.py"):
            filename = str(path)
            if filename == "smac/__init__.py":
                continue

            replace_if_starts_with(filename, "__copyright__ =", f'__copyright__ = "Copyright {year}, {holder}"')
            replace_if_starts_with(filename, "__author__ =")
            replace_if_starts_with(filename, "__maintainer__ =")
            replace_if_starts_with(filename, "__version__ =")
            replace_if_starts_with(filename, "__email__ =")

            if directory == "smac":
                check(filename)

    replace_copyright_notice("LICENSE.txt", f"Copyright (c) {year}, {holder} ({url})", url)
    replace_copyright_notice("README.md", f"Copyright (c) {year}, [{holder}]({url})", url)
