# CIL - Core Imaging Library

[![CI-master](https://github.com/TomographicImaging/CIL/actions/workflows/build.yml/badge.svg)](https://github.com/TomographicImaging/CIL/actions/workflows/build.yml) ![conda-ver](https://anaconda.org/ccpi/cil/badges/version.svg) ![conda-date](https://anaconda.org/ccpi/cil/badges/latest_release_date.svg) [![conda-plat](https://anaconda.org/ccpi/cil/badges/platforms.svg) ![conda-dl](https://anaconda.org/ccpi/cil/badges/downloads.svg)](https://anaconda.org/ccpi/cil)

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/TomographicImaging/CIL-Demos/HEAD?urlpath=lab/tree/binder%2Findex.ipynb)

The Core Imaging Library (CIL) is an open-source Python framework for tomographic imaging with particular emphasis on reconstruction of challenging datasets. Conventional filtered backprojection reconstruction tends to be insufficient for highly noisy, incomplete, non-standard or multichannel data arising for example in dynamic, spectral and in situ tomography. CIL provides an extensive modular optimisation framework for prototyping reconstruction methods including sparsity and total variation regularisation, as well as tools for loading, preprocessing and visualising tomographic data.

## Documentation

The documentation for CIL can be accessed [here](https://tomographicimaging.github.io/CIL).

## Installation of CIL

### Conda

Binary installation of CIL can be achieved with `conda`.

We recommend using either [`miniconda`](https://docs.conda.io/projects/miniconda/en/latest) or [`miniforge`](https://github.com/conda-forge/miniforge), which are both minimal installers for `conda`. We also recommend a `conda` version of at least `23.10` for quicker installation.

Install a new minimal environment with CIL using:

```sh
conda create --name cil - c https://software.repos.intel.com/python/conda -c conda-forge -c ccpi cil=24.3.0 
```
A number of additional dependencies are required for specific functionality in CIL, these should be added to your environment as necessary. See the dependency table below for details.


#### Binary packages and dependencies
While building the CIL package we test with specific versions of dependencies. These are listed in the [build.yml](https://github.com/TomographicImaging/CIL/blob/master/.github/workflows/build.yml) GitHub workflow and [environment-test.yml](https://github.com/TomographicImaging/CIL/blob/master/scripts/requirements-test.yml). The following table tries to resume the tested versions of CIL and its required and optional dependencies. If you use these packages as a backend please remember to cite them in addition to CIL.

| Package | Tested Version |  Conda install command | Description | License |
|----|----|--------|--------|----|
| [Python](https://www.python.org/) | >=3.10,<=3.12 |  `conda-forge::python>=3.10,<=3.12` | | [PSLF](https://docs.python.org/3/license.html) |
| [Numpy](https://github.com/numpy/numpy) |  >=1.23,<2.0.0 |  `conda-forge::numpy>=1.23,<2.0.0` | | [BSD](https://numpy.org/doc/stable/license.html)|
| **Optional dependencies** |||| 
| [IPP](https://www.intel.com/content/www/us/en/developer/tools/oneapi/ipp.html#gs.gxwq5p) | 2021.12 | `conda install -c https://software.repos.intel.com/python/conda ipp=2021.12`| The Intel Integrated Performance Primitives Library (required for the CIL recon class) |[Intel Simplified Software License](http://www.intel.com/content/www/us/en/developer/articles/license/end-user-license-agreement.html) |
| [ASTRA toolbox](http://www.astra-toolbox.com) | 2.1 |  CPU: `conda-forge::astra-toolbox=2.1=py*` <br> GPU: `conda-forge::astra-toolbox=2.1=cuda*` | CT projectors, FBP and FDK. |[GPLv3](https://github.com/astra-toolbox/astra-toolbox?tab=GPL-3.0-1-ov-file#readme) |
| [TIGRE](https://github.com/CERN/TIGRE) | 2.6 |  `ccpi::tigre=2.6` | CT projectors, FBP and FDK. |[BSD](https://github.com/CERN/TIGRE?tab=BSD-3-Clause-1-ov-file#readme) |
| [CCPi Regularisation Toolkit](https://github.com/TomographicImaging/CCPi-Regularisation-Toolkit) |  24.0.1 | `ccpi::ccpi-regulariser=24.0.1` | Toolbox of regularisation methods. |[Apache v2](https://github.com/TomographicImaging/CCPi-Regularisation-Toolkit?tab=Apache-2.0-1-ov-file#readme) |
| [TomoPhantom](https://github.com/dkazanc/TomoPhantom) | 22.0.0 | `ccpi::tomophantom=22.0.0` |  Generates phantoms to use as test data. |[Apache v2](https://github.com/dkazanc/TomoPhantom?tab=Apache-2.0-1-ov-file#readme) |
| [ipykernel](https://github.com/ipython/ipykernel) |    |  `conda-forge::ipykernel` | Provides the IPython kernel to run Jupyter notebooks | [BSD](https://github.com/ipython/ipykernel?tab=BSD-3-Clause-1-ov-file) |
| [ipywidgets](https://github.com/jupyter-widgets/ipywidgets) |    |  `conda-forge::ipywidgets` | Enables visulisation tools within jupyter noteboooks | [BSD](https://github.com/jupyter-widgets/ipywidgets?tab=BSD-3-Clause-1-ov-file) |

We maintain an environment file with the required packages to run the [CIL demos](https://github.com/TomographicImaging/CIL-Demos) which you can use to create a new environment. This will have specific and tested versions of all dependencies that are outlined in the table above: 

```sh
conda env create -f https://tomographicimaging.github.io/scripts/env/cil_demos.yml
```

### Docker

Finally, CIL can be run via a Jupyter Notebook enabled Docker container:

```sh
docker run --rm --gpus all -p 8888:8888 -it ghcr.io/tomographicimaging/cil:latest
```

> [!TIP]
> docker tag | CIL branch/tag
> :---|:---
> `latest` | [latest tag `v*.*.*`](https://github.com/TomographicImaging/CIL/releases/latest)
> `YY.M` | latest tag `vYY.M.*`
> `YY.M.m` | tag `vYY.M.m`
> `master` | `master`
> only build & test (no tag) | CI (current commit)
>
> See [`ghcr.io/tomographicimaging/cil`](https://github.com/TomographicImaging/CIL/pkgs/container/cil) for a full list of tags.

<!-- <br/> -->

> [!NOTE]
> GPU support requires [`nvidia-container-toolkit`](https://github.com/NVIDIA/nvidia-container-toolkit) and an NVIDIA GPU.
> Omit the `--gpus all` to run without GPU support.

<!-- <br/> -->

> [!IMPORTANT]
> Folders can be shared with the correct (host) user permissions using
> `--user $(id -u) --group-add users -v /local/path:/container/path`
> where `/local/path` is an existing directory on your local (host) machine which will be mounted at `/container/path` in the docker container.

<!-- <br/> -->

> [!TIP]
> See [jupyter-docker-stacks](https://jupyter-docker-stacks.readthedocs.io/en/latest/using/common.html) for more information.

## Getting Started with CIL

### CIL Training

We typically run training courses at least twice a year - check <https://ccpi.ac.uk/training/> for our upcoming events!

### CIL on binder

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/TomographicImaging/CIL-Demos/HEAD?urlpath=lab/tree/binder%2Findex.ipynb)

Jupyter Notebooks usage examples without any local installation are provided in [Binder](https://mybinder.org/v2/gh/TomographicImaging/CIL-Demos/HEAD?urlpath=lab/tree/binder%2Findex.ipynb). Please click the launch binder icon above. For more information, go to [CIL-Demos](https://github.com/TomographicImaging/CIL-Demos) and [https://mybinder.org](https://mybinder.org).

### CIL Videos

- [PyCon DE & PyData Berlin 2022](https://2022.pycon.de), Apr 2022: [Abstract](https://2022.pycon.de/program/GSLJUY), [Video](https://www.youtube.com/watch?v=Xd4erPj0uEs), [Material](https://github.com/TomographicImaging/CIL-Demos/blob/main/binder/PyData22_deblurring.ipynb)
- [Training School for the Synergistic Image Reconstruction Framework (SIRF) and Core Imaging Library (CIL)](https://www.ccpsynerbi.ac.uk/SIRFCIL2021), Jun 2021: [Videos](https://www.youtube.com/playlist?list=PLTuAla-OP8WVNPWZfis6BRsWFq_S0bqvp), [Material](https://github.com/TomographicImaging/CIL-Demos/tree/main/training/2021_Fully3D)
- [Synergistic Reconstruction Symposium](https://www.ccpsynerbi.ac.uk/symposium2019), Nov 2019: [Slides](https://www.ccppetmr.ac.uk/sites/www.ccppetmr.ac.uk/files/Papoutsellis%202.pdf), [Videos](https://www.youtube.com/playlist?list=PLyxAZuV8tuKsOY4DTDzy04DRrwkxBkTYh), [Material](https://github.com/TomographicImaging/CIL-Demos/tree/main/training/2019_SynergisticSymposium)

## Building CIL from source code

### Getting the code

In case of development it is useful to be able to build the software directly. You should clone this repository as

```sh
git clone --recurse-submodule git@github.com:TomographicImaging/CIL
```

The use of `--recurse-submodule` is necessary if the user wants the examples data to be fetched (they are needed by the unit tests). We have moved such data, previously hosted in this repo at `Wrappers/Python/data` to the [CIL-data](https://github.com/TomographicImaging/CIL-Data) repository and linked it to this one as submodule. If the data is not available it can be fetched in an already cloned repository as

```sh
git submodule update --init --recursive
```

### Building with `pip`

#### Install Dependencies

To create a conda environment with all the dependencies for building CIL run the following shell script:

```sh
bash ./scripts/create_local_env_for_cil_development.sh
```

Or with the CIL build and test dependencies:

```sh
bash ./scripts/create_local_env_for_cil_development.sh -t
```

And then install CIL in to this environment using `pip`.

Alternatively, one can use the `scripts/requirements-test.yml` to create a conda environment with all the
appropriate dependencies on any OS, using the following command:

```sh
conda env create -f ./scripts/requirements-test.yml
```

#### Build CIL

A C++ compiler is required to build the source code. Let's suppose that the user is in the source directory, then the following commands should work:

```sh
pip install --no-deps .
```

If not installing inside a conda environment, then the user might need to set the locations of optional libraries:

```sh
pip install . -Ccmake.define.IPP_ROOT="<path_to_ipp>" -Ccmake.define.OpenMP_ROOT="<path_to_openmp>"
```

### Building with Docker

In the repository root, simply update submodules and run `docker build`:

```sh
git submodule update --init --recursive
docker build . -t ghcr.io/tomographicimaging/cil
```

### Testing

One installed, CIL functionality can be tested using the following command:

```sh
export TESTS_FORCE_GPU=1  # optional, makes GPU test failures noisy
python -m unittest discover -v ./Wrappers/Python/test
```

## Citing CIL

If you use CIL in your research, please include citations to **both** the software on Zenodo, and a CIL paper:

E. Pasca, J. S. Jørgensen, E. Papoutsellis, E. Ametova, G. Fardell, K. Thielemans, L. Murgatroyd, M. Duff and H. Robarts (2023) <br>
Core Imaging Library (CIL) <br>
Zenodo [software archive] <br>
**DOI:** https://doi.org/10.5281/zenodo.4746198 <br>

In most cases, the first CIL paper will be the appropriate choice:

J. S. Jørgensen, E. Ametova, G. Burca, G. Fardell, E. Papoutsellis, E. Pasca, K. Thielemans, M. Turner, R. Warr, W. R. B. Lionheart and P. J. Withers (2021) <br>
Core Imaging Library - Part I: a versatile Python framework for tomographic imaging. <br>
Phil. Trans. R. Soc. A. 379: 20200192. <br>
**DOI:** https://doi.org/10.1098/rsta.2020.0192 <br>
**Code:** https://github.com/TomographicImaging/Paper-2021-RSTA-CIL-Part-I <br>

However, if your work is more closely related to topics covered in our second CIL paper then please additionally or alternatively reference the second paper:

E. Papoutsellis, E. Ametova, C. Delplancke, G. Fardell, J. S. Jørgensen, E. Pasca, M. Turner, R. Warr, W. R. B. Lionheart and P. J. Withers (2021) <br>
Core Imaging Library - Part II: multichannel reconstruction for dynamic and spectral tomography. <br>
Phil. Trans. R. Soc. A. 379: 20200193. <br>
**DOI:** https://doi.org/10.1098/rsta.2020.0193) <br>
**Code:** https://github.com/TomographicImaging/Paper-2021-RSTA-CIL-Part-II <br>
