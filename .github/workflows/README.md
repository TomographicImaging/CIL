# GitHub Actions

## Building the Conda Package: [conda_build](https://github.com/TomographicImaging/CIL/blob/master/.github/workflows/conda_build.yml)
This github action builds and tests the conda package, by using the [conda-package-publish-action](https://github.com/paskino/conda-package-publish-action)

When pushing to master or creating an [annotated tag](https://git-scm.com/book/en/v2/Git-Basics-Tagging), *all* variants are built and tested.

When opening or modifying a pull request to master, a single variant is built and tested. This variant is for linux with `python=3.7` and `numpy=1.18`.

The action does not publish to conda, instead this is done by jenkins. This is because github-actions do not have a GPU.

## Building/Publishing Documentation: [docs_build_and_publish](https://github.com/TomographicImaging/CIL/blob/master/.github/workflows/docs_build_and_publish.yml)

This github action builds and optionally publishes the documentation located in [docs/source](https://github.com/TomographicImaging/CIL/tree/master/docs/source). 

The github action has two jobs:
1. [build](https://github.com/TomographicImaging/CIL/blob/39b6f7a722afb6d5f0e2d47a99ce8266378c2a65/.github/workflows/docs_build_and_publish.yml#L12): 
-  builds the documentation with sphinx
-  uses upload-artifact to upload the html files which may then be used by **publish**

2. [publish](https://github.com/TomographicImaging/CIL/blob/39b6f7a722afb6d5f0e2d47a99ce8266378c2a65/.github/workflows/docs_build_and_publish.yml#L40):
-  uses download-artifact to retrieve the built html files
-  pushes the html files to the `nightly` folder on the gh-pages branch

If opening or modifying a pull request to master, `build` is run, but not `publish`.
If pushing to master or tagging, the documentation is built *and* published (both the `build` and `publish` jobs are run).

### Viewing Built Documentation
The `build` job builds the documentation and uploads it as an [artifact](https://github.com/TomographicImaging/CIL/blob/39b6f7a722afb6d5f0e2d47a99ce8266378c2a65/.github/workflows/docs_build_and_publish.yml#L36),
in a folder named `DocumentationHTML`.
This can be found by going to the ‘Actions’ tab, and selecting the appropriate run of `.github/workflows/docs_build_and_publish.yml`.

When viewing the summary for the run of the action, there is an `Artifact` section at the bottom of the page.
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

