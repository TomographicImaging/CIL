# GitHub Actions

There is a single github action file with multiple jobs, which builds both the conda package and documentation, and optionally publishes the documentation: [build](./build.yml)

The jobs are:

- `test-cuda`
  + uses our self-hosted (STFC Cloud) CUDA-enabled runners to run GPU tests
  + `TESTS_FORCE_GPU=1 python -m unittest discover -v -k tigre -k TIGRE -k astra -k ASTRA -k gpu -k GPU ./Wrappers/Python/test`
- `test`
  + uses default (GitHub-hosted) runners to run tests on the min & max supported Python & NumPy versions
  + `python -m unittest discover -v ./Wrappers/Python/test`
- conda
  + uses `mambabuild` to build the conda package (saved as a build artifact named `cil-package`)
- docs
  + uses `docs/docs_environment.yml` plus `make -C docs` to build the documentation (saved as a build artifact named `DocumentationHTML`)
  + renders to the `gh-pages` branch on `master` (nightly) pushes or on tag (release) pushes
    * this in turn is hosted at <https://tomographicimaging.github.io/CIL> as per <https://github.com/TomographicImaging/CIL/settings/pages>
- docker
  + builds a docker image from [`Dockerfile`](../../Dockerfile) (pushed to `ghcr.io/tomographicimaging/cil` as per [the CIL README section on Docker](../../README.md#docker))

Details on some of these jobs are given below.

## conda

When opening or modifying a pull request to `master`, a single variant is built and tested. This variant is for linux with `python=3.11` and `numpy=1.25`.

> [!NOTE]
> The action does not publish to conda, instead this is done by jenkins. We will eventually move from jenkins to conda-forge instead.
> When pushing to `master` or creating an [annotated tag](https://git-scm.com/book/en/v2/Git-Basics-Tagging), *all* variants are built and tested.

It looks for conda-build dependencies in the channels listed [here](./build.yml#L118). If you add any new dependencies, the appropriate channels need to be added to this line.

> [!TIP]
> The `conda` job builds the `*.tar.bz2` package and uploads it as an artifact called `cil-package`.
> It can be found by going to the "Actions" tab, and selecting the appropriate run of `.github/workflows/build.yml`, or by clicking on the tick on the action in the "All checks have passed/failed" section of a PR. When viewing the "Summary" for the run of the action, there is an "Artifact" section at the bottom of the page.
> Clicking on `cil-package` allows you to download a zip folder containing the `*.tar.bz2` file.

## docs

This github action builds and optionally publishes the documentation located in [docs/source](../../docs/source).

The [docs](./build.yml#L124) job:

- creates a `miniconda` environment from [requirements-test.yml](../../scripts/requirements-test.yml) and [docs_environment.yml](../../docs/docs_environment.yml)
- `cmake` builds & installs CIL into the `miniconda` environment
- builds the HTML documentation with `sphinx`
- uploads a `DocumentationHTML` artifact (which can be downloaded to view locally for debugging)
- pushes the HTML documentation to the `gh-pages` branch
  + only if pushing to `master` or tagging (skipped if pushing to a branch or a PR)

> [!TIP]
> The `docs` job builds the documentation and uploads it as an artifact called `DocumentationHTML`.
> It can be found by going to the "Actions" tab, and selecting the appropriate run of `.github/workflows/build.yml`, or by clicking on the tick on the action in the "All checks have passed/failed" section of a PR. When viewing the "Summary" for the run of the action, there is an "Artifact" section at the bottom of the page.
> Clicking on `DocumentationHTML` allows you to download a zip folder containing the built HTML files. This allows you to preview the documentation site before it is published.
