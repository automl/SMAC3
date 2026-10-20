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
1. Create a PR to merge branch `v${VERSION}` into `main`. As description you can use the changelog notes.
1. Test installation with a fresh environment, see `test_package.sh`.
1. Merge PR if tests are fine and installation is fine.
1. Create release, add notes from changelog:
    - On GitHub, go to **Releases → Draft a new release**.
    - Tag `v${VERSION}`, targeting `main` (the commit the PR was just merged into).
    - Title: `v${VERSION}`.
    - Paste the corresponding `CHANGELOG.md` section as the release notes.
    - Publish.

1. Update doc link. Check the repo's **Website** field (GitHub Settings → General → Website) still points at `https://automl.github.io/SMAC3/latest/`.

1. Deploy github pages (uses `${VERSION}` exported above):
    ```bash
    mike deploy "v${VERSION}" latest -u -p --title "v${VERSION} (latest)"
    ```
    This only touches the `gh-pages` branch (builds and pushes the docs site, moves the `latest` alias). After pushing, GitHub Pages can take a few minutes to actually serve the update — a stale-looking site right after deploy isn't necessarily a failed deploy; check `https://automl.github.io/SMAC3/versions.json` for the new version before assuming something's wrong.

1. Build the distribution:
    ```bash
    make build
    ```
    (Run `make clean-build` first if `dist/` might still contain artifacts from a previous release — `twine upload dist/*` below uploads everything it finds there.)

1. Upload to testpypi:
    ```bash
    python -m twine upload --repository testpypi dist/*
    ```

1. Test from testpypi:
    ```bash
    uv pip uninstall smac
    uv pip install --index-strategy unsafe-best-match --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ smac==${VERSION}
    python -c 'import smac'
    ```
    If this is fine, proceed.

1. Upload to pypi: 
    ```bash
    python -m twine upload dist/*
    ```