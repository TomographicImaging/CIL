# GitHub Actions

## Building the Conda Package: [conda_build](https://github.com/TomographicImaging/CIL/blob/master/.github/workflows/conda_build.yml)
This github action builds and tests the conda package.

If pushing to master or tagging, *all* variants are built and tested.

If opening or modifying a pull request to master, a single variant is built and tested.

The action does not publish to conda, instead this is done by jenkins. This is because github-actions do not have a GPU.

## Building/Publishing Documentation: [docs_build_and_publish](https://github.com/TomographicImaging/CIL/blob/master/.github/workflows/docs_build_and_publish.yml)

This github action builds and optionally publishes the documentation located in [docs/source](https://github.com/TomographicImaging/CIL/tree/master/docs/source). 

The github action has two jobs:
[build](https://github.com/TomographicImaging/CIL/blob/39b6f7a722afb6d5f0e2d47a99ce8266378c2a65/.github/workflows/docs_build_and_publish.yml#L12)
and [publish](https://github.com/TomographicImaging/CIL/blob/39b6f7a722afb6d5f0e2d47a99ce8266378c2a65/.github/workflows/docs_build_and_publish.yml#L40).

If opening or modifying a pull request to master, `build` is run, but not `publish`.
If pushing to master or tagging, the documentation is built *and* published (both the `build` and `publish` jobs are run).

### Viewing Built Documentation
The `build` job builds the documentation and uploads it as an [artifact](https://github.com/TomographicImaging/CIL/blob/39b6f7a722afb6d5f0e2d47a99ce8266378c2a65/.github/workflows/docs_build_and_publish.yml#L36),
in a folder named `DocumentationHTML`.
This can be found by going to the ‘Actions’ tab, and selecting the appropriate run of `.github/workflows/docs_build_and_publish.yml`.

When viewing the summary for the run of the action, there is an `Artifact` section at the bottom of the page.
Clicking on `DocumentationHTML` allows you to download a zip folder containing the built html files. This allows you to preview

### Publication of the Documentation
The documentation is hosted on the [github site](https://tomographicimaging.github.io/CIL/) associated with the repository.
This is built from the [gh-pages branch](https://github.com/TomographicImaging/CIL/tree/gh-pages). 

If you are an admin of the CIL repository you are able to see the settings for the site by going to `Settings->Pages`.

To publish the documentation, the publish job of the gh-action pushes the documentation changes to the `gh-pages` branch.
Any push to this branch automatically updates the github site.


