
| Master | Development | Anaconda binaries |
|--------|-------------|-------------------|
| [![Build Status](https://anvil.softeng-support.ac.uk/jenkins/buildStatus/icon?job=CILsingle/CCPi-Framework)](https://anvil.softeng-support.ac.uk/jenkins/job/CILsingle/job/CCPi-Framework/) | [![Build Status](https://anvil.softeng-support.ac.uk/jenkins/buildStatus/icon?job=CILsingle/CCPi-Framework-dev)](https://anvil.softeng-support.ac.uk/jenkins/job/CILsingle/job/CCPi-Framework-dev/) |![conda version](https://anaconda.org/ccpi/cil/badges/version.svg) ![conda last release](https://anaconda.org/ccpi/cil/badges/latest_release_date.svg) [![conda platforms](https://anaconda.org/ccpi/cil/badges/platforms.svg) ![conda dowloads](https://anaconda.org/ccpi/cil/badges/downloads.svg)](https://anaconda.org/ccpi/cil) |

# CIL - Core Imaging Library

The Core Imaging Library (CIL) is an open-source Python framework for tomographic imaging with particular emphasis on reconstruction of challenging datasets. Conventional filtered backprojection reconstruction tends to be insufficient for highly noisy, incomplete, non-standard or multichannel data arising for example in dynamic, spectral and in situ tomography. CIL provides an extensive modular optimization framework for prototyping reconstruction methods including sparsity and total variation regularization, as well as tools for loading, preprocessing and visualizing tomographic data.

## CIL on binder

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/TomographicImaging/CIL-Demos/HEAD?urlpath=lab/tree/binder%2Findex.ipynb)

Jupyter Notebooks usage examples without any local installation are provided in [Binder](https://mybinder.org/v2/gh/TomographicImaging/CIL-Demos/HEAD?urlpath=lab/tree/binder%2Findex.ipynb). Please click the launch binder icon above. For more information, go to [CIL-Demos](https://github.com/TomographicImaging/CIL-Demos) and [https://mybinder.org](https://mybinder.org).

## Documentation

The documentation for CIL can be accessed [here](https://tomographicimaging.github.io/CIL).

## Installation

Binary installation of CIL can be done with `conda`. Install a new environment using:

```bash
conda create --name cil -c conda-forge -c intel -c ccpi cil
```

To install CIL and the additional packages and plugins needed to run the [CIL demos](https://github.com/TomographicImaging/CIL-Demos) install the environment with:

```bash

conda create --name cil -c conda-forge -c intel -c astra-toolbox/label/dev -c ccpi cil cil-astra ccpi-regulariser tigre tomophantom=1.4.10
```

where,

```ccpi-regulariser``` will give you access to the [CCPi Regularisation Toolkit](https://github.com/vais-ral/CCPi-Regularisation-Toolkit).

```cil-astra``` will give you access to the CIL wrappers to the [ASTRA toolbox](http://www.astra-toolbox.com/) projectors (GPLv3 license).

```tomophantom``` [Tomophantom](https://github.com/dkazanc/TomoPhantom) will allow you to generate phantoms to use as test data.

```tigre``` will allow you to use CIL wrappers to the [TIGRE](https://github.com/CERN/TIGRE) toolbox projectors (BSD license).

```cudatoolkit``` If you have GPU drivers compatible with more recent CUDA versions you can modify this package selector (installing tigre via conda requires 9.2).

### Dependency

CIL's [optimised FDK/FBP](https://github.com/TomographicImaging/CIL/discussions/1070) `recon` module requires:
1. the Intel [Integrated Performance Primitives](https://www.intel.com/content/www/us/en/developer/tools/oneapi/ipp.html#gs.gxwq5p) Library ([license](https://www.intel.com/content/dam/develop/external/us/en/documents/pdf/intel-simplified-software-license-version-august-2021.pdf)) which can be installed via conda from the `intel` [channel](https://anaconda.org/intel/ipp).
2. [TIGRE](https://github.com/CERN/TIGRE), which can be installed via conda from the `ccpi` channel.

## Building from source code 

### Getting the code

In case of development it is useful to be able to build the software directly. You should clone this repository as
```bash

git clone --recurse-submodule git@github.com:TomographicImaging/CIL.git
```
The use of `--recurse-submodule` is necessary if the user wants the examples data to be fetched (they are needed by the unit tests). We have moved such data, previously hosted in this repo at `Wrappers/Python/data` to the [CIL-data](https://github.com/TomographicImaging/CIL-Data) repository and linked it to this one as submodule. If the data is not available it can be fetched in an already cloned repository as
```bash

git submodule update --init
```

### Build with CMake
CMake and a C++ compiler are required to build the source code. Let's suppose that the user is in the source directory, then the following commands should work:

```bash

mkdir build
cd build
cmake .. -DCMAKE_INSTALL_PREFIX=<install_directory>
cmake --build . --target install
```

The user then needs to add the path to `<install_directory>/lib` where the library is installed to the environment variable `PATH` or `LD_LIBRARY_PATH`, depending on system


## References

[1] Jørgensen JS et al. 2021 [Core Imaging Library Part I: a versatile python framework for tomographic imaging](https://doi.org/10.1098/rsta.2020.0192). Phil. Trans. R. Soc. A 20200192. [**Code.**](https://github.com/TomographicImaging/Paper-2021-RSTA-CIL-Part-I) [Pre-print](https://arxiv.org/abs/2102.04560)

[2] Papoutsellis E et al. 2021 [Core Imaging Library - Part II: multichannel reconstruction for dynamic and spectral
tomography](https://doi.org/10.1098/rsta.2020.0193). Phil. Trans. R. Soc. A 20200193. [**Code.**](https://github.com/TomographicImaging/Paper-2021-RSTA-CIL-Part-II) [Pre-print](https://arxiv.org/abs/2102.06126)



