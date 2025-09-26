# How to Create a New Release
Export version numer, e.g.
```bash
export VERSION="2.4.0"
```
If you do not use `uv`, remove `uv` from the commands.

1. Refresh main.
    ```bash
    git checkout main
    git pull
    ```

1. Checkout development branch:
    ```bash
    git checkout development
    git pull
    ```

1. Run `make tests` to ensure everything works. If tests run through, proceed.
    ```bash
    make tests
    ```

1. Create new branch from development with name e.g. `v${VERSION}`:
    ```bash
    git branch v${VERSION}
    git checkout v${VERSION}
    ```

1. Merge main into branch
    ```bash
    git merge main
    ```

1. Check `CHANGELOG.md` whether the version number is correct and the order is fine.
1. Replace version numbers everywhere: in `CITATION.cff`, `__init__.py`.
1. Create a PR to merge branch `v${VERSION}` into `main`.
1. Test installation with a fresh environment, see `test_package.sh`.
1. Merge PR if tests are fine and installation is fine.
1. Create release, add notes from changelog.
1. Update doc link.
1. Deploy github pages (replace version in the following command):
    ```bash
    mike deploy "v${VERSION}" latest -u -p --title "v${VERSION} (latest)"
    ```

1. Upload to testpypi:
    ```bash
    python -m twine upload --repository testpypi dist/*
    ```

1. Test from testpypi:
    ```bash
    uv pip uninstall smac
    uv pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ smac==${VERSION}
    python -c 'import smac'
    ```
    If this is fine, proceed.

1. Upload to pypi: 
    ```bash
    python -m twine upload dist/*
    ```