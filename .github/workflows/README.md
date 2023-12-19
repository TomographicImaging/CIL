# GitHub Actions

There is a single github action file with multiple jobs, which builds both the conda package and documentation, and optionally publishes the documentation: [build](https://github.com/TomographicImaging/CIL/blob/master/.github/workflows/build.yml)

## Building the Conda Package: conda job

This github action builds and tests the conda package, by using the [conda-package-publish-action](https://github.com/TomographicImaging/conda-package-publish-action)

When pushing to master or creating an [annotated tag](https://git-scm.com/book/en/v2/Git-Basics-Tagging), *all* variants are built and tested.

When opening or modifying a pull request to master, a single variant is built and tested. This variant is for linux with `python=3.9` and `numpy=1.22`.

The action does not publish to conda, instead this is done by jenkins. This is because github-actions do not have a GPU.

It looks for conda-build dependencies in the channels listed [here](https://github.com/TomographicImaging/CIL/blob/master/.github/workflows/build.yml#L49). If you add any new dependencies, the appropriate channels need to be added to this line.

An artifact of the resulting tar.bz2 file is made available in the 'Summary' section of the action. It is called `cil-package`. This is used by the **docs** job. It can be found by going to the ‘Actions’ tab, and selecting the appropriate run of `.github/workflows/build.yml`, or by clicking on the tick on the action in the "All checks have passed/failed" section of a PR. When viewing the summary for the run of the action, there is an `Artifact` section at the bottom of the page. Clicking on `cil-package` allows you to download a zip folder containing the tar.bz2 file.

## Building/Pushing the Docker Image: docker job

TODO

## Building/Publishing Documentation: docs job

This github action builds and optionally publishes the documentation located in [docs/source](https://github.com/TomographicImaging/CIL/tree/master/docs/source). To do this it uses a forked version of the [build-sphinx-action](https://github.com/lauramurgatroyd/build-sphinx-action).

The [docs](https://github.com/TomographicImaging/CIL/blob/master/.github/workflows/build.yml#L59) job:

- creates a miniconda environment from [docs_environment.yml](https://github.com/TomographicImaging/CIL/blob/master/.github/workflows/docs/docs_environment.yml)
- installs cil into the miniconda environment, using the tar.bz2 artifact (cil-package) created in the **conda** job
- builds the documentation with sphinx
- uses upload-artifact to upload the html files: `HTMLDocumentation`, which can be downloaded to view
- pushes the html files to the `nightly` folder on the gh-pages branch

If opening or modifying a pull request to master, `docs` is run, but the final gh-pages step is skipped.
If pushing to master or tagging, the documentation is built *and* published to gh-pages.

### Viewing Built Documentation

The `docs` job builds the documentation and uploads it as an artifact, in a folder named `DocumentationHTML`.
This can be found by going to the ‘Actions’ tab, and selecting the appropriate run of `.github/workflows/build.yml`, or by clicking on the tick on the action in the "All checks have passed/failed" section of a PR.

When viewing the `Summary` for the run of the action, there is an `Artifact` section at the bottom of the page.
Clicking on `DocumentationHTML` allows you to download a zip folder containing the built html files. This allows you to preview the documentation site before it is published.

### Publication of the Documentation

The documentation is hosted on the [github site](https://tomographicimaging.github.io/CIL/) associated with the repository.
This is built from the [gh-pages branch](https://github.com/TomographicImaging/CIL/tree/gh-pages).

If you are an admin of the CIL repository you are able to see the settings for the site by going to `Settings->Pages`.

To publish the documentation, the publish job of the gh-action pushes the documentation changes to the `gh-pages` branch.
Any push to this branch automatically updates the github site.

### Initial Setup of the Docs Site & Action

To get the action to work I first had to:

1. [Create a gh-pages branch](https://gist.github.com/ramnathv/2227408) - note this only worked in bash, not windows command line.
2. [Set the source](https://github.com/TomographicImaging/CIL/settings/pages) for our github pages to be the gh-pages branch.

I followed the examples on the [sphinx build action page](https://github.com/marketplace/actions/sphinx-build), specifically this [example workflow](https://github.com/ammaraskar/sphinx-action-test/blob/master/.github/workflows/default.yml)
