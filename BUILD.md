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

