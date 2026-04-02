# GitHub Actions

## [build.yml](./build.yml)
This github action file has multiple jobs, which build both the conda package and documentation, and optionally publish the documentation.

The jobs are:

- `test-cuda`
  + uses our self-hosted (STFC Cloud) CUDA-enabled runners to run GPU tests
  + `TESTS_FORCE_GPU=1 python -m unittest discover -v -k tigre -k TIGRE -k astra -k ASTRA -k Astra -k gpu -k GPU ./Wrappers/Python/test`
- `test`
  + uses default (GitHub-hosted) runners to run tests on the min & max supported Python & NumPy versions
  + `python -m unittest discover -v ./Wrappers/Python/test`
- `conda`
  + uses `mambabuild` to build the conda package (saved as a build artifact named `cil-py3.m-OS`)
- `docs`
  + uses `docs/docs_environment.yml` plus `make -C docs` to build the documentation (saved as a build artifact named `DocumentationHTML`)
  + renders to the `gh-pages` branch on `master` (nightly) pushes or on tag (release) pushes
    * this in turn is hosted at <https://tomographicimaging.github.io/CIL> as per <https://github.com/TomographicImaging/CIL/settings/pages>
- `docker`
  + builds a docker image from [`Dockerfile`](../../Dockerfile) (pushed to `ghcr.io/tomographicimaging/cil` as per [the CIL README section on Docker](../../README.md#docker))

Details on some of these jobs are given below.

> [!TIP]
> To skip tests, include [one of the following](https://docs.github.com/en/actions/managing-workflow-runs-and-deployments/managing-workflow-runs/skipping-workflow-runs) in your commit message: `[skip ci]`, `[ci skip]`, `[no ci]`, `[skip actions]` or `[actions skip]`.

### conda

When opening or modifying a pull request to `master`, two variants are built and tested (for linux with minimum & maximum supported `python` & `numpy` versions).

> [!NOTE]
> When pushing to `master` or creating an [annotated tag](https://git-scm.com/book/en/v2/Git-Basics-Tagging), *all* variants are built and tested.
>
> To test *all* variants on other branches, use [the `Run workflow` button on the web UI](https://github.com/TomographicImaging/CIL/actions/workflows/build.yml).

<!-- <br/> -->

> [!NOTE]
> The action publishes `ccpi` as well as `https://tomography.stfc.ac.uk/conda/` conda channels. We will eventually move to conda-forge instead.

It looks for conda-build dependencies in the channels listed [here](./build.yml#L118). If you add any new dependencies, the appropriate channels need to be added to this line.

> [!TIP]
> The `conda` job builds the `*.tar.bz2` package and uploads it as an artifact called `cil-py3.m-OS`.
> It can be found by going to the "Actions" tab, and selecting the appropriate run of `.github/workflows/build.yml`, or by clicking on the tick on the action in the "All checks have passed/failed" section of a PR. When viewing the "Summary" for the run of the action, there is an "Artifact" section at the bottom of the page.
> Clicking on `cil-py3.m-OS` allows you to download a zip folder containing the `*.tar.bz2` file.

### docs

This github action builds and optionally publishes the documentation located in [docs/source](../../docs/source).

The [docs](./build.yml#L124) job:

- creates a `miniconda` environment from [requirements-test.yml](../../scripts/requirements-test.yml) and [docs_environment.yml](../../docs/docs_environment.yml)
- `cmake` builds & installs CIL into the `miniconda` environment
  + builds the HTML documentation with `sphinx`
- installs ruby dependencies from [`Gemfile`](../../docs/Gemfile)
  + builds the HTML landing page with `jekyll`
- uploads a `DocumentationHTML` artifact (which can be downloaded to view locally for debugging)
  + pushes the HTML documentation to the `gh-pages` branch
    * only if pushing to `master` or tagging (skipped if pushing to a branch or a PR)

> [!TIP]
> The `docs` job builds the documentation and uploads it as an artifact called `DocumentationHTML`.
> It can be found by going to the "Actions" tab, and selecting the appropriate run of `.github/workflows/build.yml`, or by clicking on the tick on the action in the "All checks have passed/failed" section of a PR. When viewing the "Summary" for the run of the action, there is an "Artifact" section at the bottom of the page.
> Click on `DocumentationHTML` to download a zip archive of the built HTML files.
> It must be extracted into a `CIL` subfolder to properly render locally:
>
> ```sh
> mkdir CIL
> unzip -d CIL DocumentationHTML.zip
> python -m http.server
> ```
>
> Then open a browser and navigate to <http://localhost:8000/CIL/> to view the documentation.

## [skip.yml](./skip.yml)

This action prevents the build jobs from running if only text files have been modified in the PR.
