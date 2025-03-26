# Mac build

Binaries for mac (apple silicon) are currently not available. You can manually build and install CIL for mac following these instructions. 

## Known Issues

1. Some underlying CIL code depends on Intel IPP library which specifically targets Intel CPU architecture. This is optional and will not be available on macs
2. TIGRE and ASTRA plugins are developed in CUDA and use NVIDIA GPU. Mac laptops do not have NVIDIA GPUs, so you would only be able to do basic 2D parallel/fan beam with ASTRA.

## Build instructions

To build on a mac you can follow the instructions for Linux at 
https://github.com/TomographicImaging/CIL#building-cil-from-source-code

It is suggested to:
1. clone the CIL repository and checkout the `macbuild` branch
2. create a conda environment with the `scripts/requirements-osx.yml` environment file
3. create a build directory as suggested in the build [instructions](https://github.com/TomographicImaging/CIL#building-cil-from-source-code)


```sh
cmake -S . -B ./build -DCMAKE_INSTALL_PREFIX=${CONDA_PREFIX} -DPython_EXECUTABLE=${CONDA_PREFIX}/bin/python
cmake --build ./build --target install --config RelWithDebInfo
```

This should install CIL in your environment.