#!/usr/bin/env bash
set -euxo pipefail
numpy='1.24'
python='3.10'
name=cil
test_deps=0
cil_ver=''
while getopts hn:p:e:tv: option ; do
  case "${option}" in
  n) numpy="${OPTARG}" ;;
  p) python="${OPTARG}" ;;
  e) name="${OPTARG}" ;;
  t) test_deps=1 ;;
  v) test_deps=1 ; cil_ver="${OPTARG}" ;;
  h)
    echo "Usage: $0 [-n numpy_version] [-p python_version] [-e environment_name] [-t] [-v cil_version]"
    echo 'Where `-t` installs test dependencies, and `-v cil_version` implies `-t`'
    exit
    ;;
  *)
    echo "Wrong option passed. Use the -h option to get some help." >&2
    exit 1
    ;;
  esac
done

echo "Numpy $numpy"
echo "Python $python"
echo "Environment name $name"

conda_args=(create --name="$name"
  python="$python"
  numpy="$numpy"
  cmake'>=3.16'
  dxchange
  h5py
  ipp
  ipp-devel
  ipp-include
  libgcc-ng
  matplotlib
  numba
  olefile
  packaging
  pillow
  python-wget
  pywavelets
  scikit-image
  scipy
  tqdm
)
if test -n "$cil_ver"; then
  echo "CIL version $cil_ver"
  conda_args+=(cil="${cil_ver}")
fi

if test $test_deps = 0; then
  conda_args+=(-c conda-forge -c intel -c defaults --override-channels)
else
  conda_args+=(
    astra-toolbox'>=1.9.9.dev5,<2.1'
    ccpi-regulariser=22.0.0
    cil-data
    cvxpy
    ipywidgets
    setuptools
    tigre=2.4
    tomophantom=2.0.0
    -c conda-forge
    -c intel
    -c ccpi/label/dev
    -c ccpi
    -c astra-toolbox
    -c astra-toolbox/label/dev
    --override-channels
  )
fi

conda "${conda_args[@]}"

# Authors: CIL Developers (https://github.com/TomographicImaging/CIL/blob/master/NOTICE.txt)
# Copyright 2020 United Kingdom Research and Innovation
# Copyright 2020 The University of Manchester
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
