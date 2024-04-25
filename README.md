# CIL - Core Imaging Library

Master | Development | Conda binaries
-|-|-
[![CI-master](https://anvil.softeng-support.ac.uk/jenkins/buildStatus/icon?job=CILsingle/CCPi-Framework)](https://anvil.softeng-support.ac.uk/jenkins/job/CILsingle/job/CCPi-Framework) | [![CI-dev](https://anvil.softeng-support.ac.uk/jenkins/buildStatus/icon?job=CILsingle/CCPi-Framework-dev)](https://anvil.softeng-support.ac.uk/jenkins/job/CILsingle/job/CCPi-Framework-dev) | ![conda-ver](https://anaconda.org/ccpi/cil/badges/version.svg) ![conda-date](https://anaconda.org/ccpi/cil/badges/latest_release_date.svg) [![conda-plat](https://anaconda.org/ccpi/cil/badges/platforms.svg) ![conda-dl](https://anaconda.org/ccpi/cil/badges/downloads.svg)](https://anaconda.org/ccpi/cil)

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/TomographicImaging/CIL-Demos/HEAD?urlpath=lab/tree/binder%2Findex.ipynb)

The Core Imaging Library (CIL) is an open-source Python framework for tomographic imaging with particular emphasis on reconstruction of challenging datasets. Conventional filtered backprojection reconstruction tends to be insufficient for highly noisy, incomplete, non-standard or multichannel data arising for example in dynamic, spectral and in situ tomography. CIL provides an extensive modular optimisation framework for prototyping reconstruction methods including sparsity and total variation regularisation, as well as tools for loading, preprocessing and visualising tomographic data.

## Documentation

The documentation for CIL can be accessed [here](https://tomographicimaging.github.io/CIL).

## Installation of CIL

### Conda

Binary installation of CIL can be achieved with `conda`.

We recommend using either [`miniconda`](https://docs.conda.io/projects/miniconda/en/latest) or [`miniforge`](https://github.com/conda-forge/miniforge), which are both minimal installers for `conda`. We also recommend a `conda` version of at least `23.10` for quicker installation.

Install a new environment using:

```sh
conda create --name cil -c conda-forge -c intel -c ccpi cil=24.0.0
```

To install CIL and the additional packages and plugins needed to run the [CIL demos](https://github.com/TomographicImaging/CIL-Demos) install the environment with:

```sh
conda create --name cil -c conda-forge -c intel -c ccpi cil=24.0.0 astra-toolbox=*=cuda* tigre ccpi-regulariser tomophantom ipywidgets
```

where:
- `astra-toolbox` enables CIL support for [ASTRA toolbox](http://www.astra-toolbox.com) CPU projector (2D Parallel beam only) (GPLv3 license)
- `astra-toolbox=*=cuda*` (requires an NVIDIA GPU) enables CIL support for [ASTRA toolbox](http://www.astra-toolbox.com) GPU projectors (GPLv3 license)
- `tigre` (requires an NVIDIA GPU) enables support for [TIGRE](https://github.com/CERN/TIGRE) toolbox projectors (BSD license)
- `ccpi-regulariser` is the [CCPi Regularisation Toolkit](https://github.com/TomographicImaging/CCPi-Regularisation-Toolkit)
- `tomophantom` can generate phantoms to use as test data [Tomophantom](https://github.com/dkazanc/TomoPhantom)
- `ipywidgets` enables visulisation tools within jupyter noteboooks

### Dependencies

CIL's [optimised FDK/FBP](https://github.com/TomographicImaging/CIL/discussions/1070) `recon` module requires:

1. the Intel [Integrated Performance Primitives](https://www.intel.com/content/www/us/en/developer/tools/oneapi/ipp.html#gs.gxwq5p) Library ([license](https://www.intel.com/content/dam/develop/external/us/en/documents/pdf/intel-simplified-software-license-version-august-2021.pdf)) which can be installed via conda from the `intel` [channel](https://anaconda.org/intel/ipp).
2. [TIGRE](https://github.com/CERN/TIGRE), which can be installed via conda from the `ccpi` channel.

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
git clone --recurse-submodule git@github.com:TomographicImaging/CIL.git
```

The use of `--recurse-submodule` is necessary if the user wants the examples data to be fetched (they are needed by the unit tests). We have moved such data, previously hosted in this repo at `Wrappers/Python/data` to the [CIL-data](https://github.com/TomographicImaging/CIL-Data) repository and linked it to this one as submodule. If the data is not available it can be fetched in an already cloned repository as

```sh
git submodule update --init --recursive
```

### Build dependencies

To create a conda environment with all the dependencies for building CIL run the following shell script:

```sh
bash scripts/create_local_env_for_cil_development.sh
```

Or with the CIL build and test dependencies:

```sh
bash scripts/create_local_env_for_cil_development.sh -t
```

And then install CIL in to this environment using CMake.

Alternatively, one can use the `scripts/requirements-test.yml` to create a conda environment with all the
appropriate dependencies on any OS, using the following command:

```sh
conda env create -f scripts/requirements-test.yml
```

### Build with CMake

CMake and a C++ compiler are required to build the source code. Let's suppose that the user is in the source directory, then the following commands should work:

```sh
cmake -S . -B ./build -DCMAKE_INSTALL_PREFIX=<install_directory>
cmake --build ./build --target install
```

If targeting an active conda environment then the `<install_directory>` can be set to the `CONDA_PREFIX` environment variable (e.g. `${CONDA_PREFIX}` in Bash, or `%CONDA_PREFIX%` in the Anaconda Prompt on Windows).

If not installing to a conda environment then the user will also need to set the locations of the IPP library and includes, and the path to CIL.

By default the location of the IPP library and includes is `${CMAKE_INSTALL_PREFIX}/lib` and `${CMAKE_INSTALL_PREFIX}/include` respectively. To pass the location of the IPP library and headers please pass the following parameters:

```sh
cmake -S . -B ./build -DCMAKE_INSTALL_PREFIX=<install_directory> -DIPP_LIBRARY=<path_to_ipp_library> -DIPP_INCLUDE=<path_to_ipp_includes>
```

The user will then need to add the path `<install_directory>/lib` to the environment variable `PATH` or `LD_LIBRARY_PATH`, depending on system OS.


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
