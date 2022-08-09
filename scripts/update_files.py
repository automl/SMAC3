from pathlib import Path


def replace(file, starts_with, replace="") -> None:
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


def check(file) -> None:
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


if __name__ == "__main__":

    for path in Path("smac").rglob("*.py"):
        filename = str(path)
        if filename == "smac/__init__.py":
            continue

        replace(filename, "__copyright__ =", '__copyright__ = "Copyright 2022, automl.org"')
        replace(filename, "__author__ =")
        replace(filename, "__maintainer__ =")
        replace(filename, "__version__ =")
        replace(filename, "__email__ =")
        check(filename)
