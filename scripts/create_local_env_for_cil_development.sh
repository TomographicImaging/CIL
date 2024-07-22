#!/usr/bin/env bash

# Copyright 2020 United Kingdom Research and Innovation
# Copyright 2020 The University of Manchester
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Authors:
# CIL Developers, listed at: https://github.com/TomographicImaging/CIL/blob/master/NOTICE.txt

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
  ipp'>=2021.10'
  ipp-devel'>=2021.10'
  ipp-include'>=2021.10'
  libgcc-ng
  libstdcxx-ng
  matplotlib
  numba
  olefile'>=0.46'
  packaging
  pillow
  python-wget
  pywavelets
  scikit-image
  scipy
  tqdm
  zenodo_get'>1.5.1'
)
if test -n "$cil_ver"; then
  echo "CIL version $cil_ver"
  conda_args+=(cil="${cil_ver}")
fi

if test $test_deps = 0; then
  conda_args+=(-c conda-forge -c https://software.repos.intel.com/python/conda -c defaults --override-channels)
else
  conda_args+=(
    astra-toolbox=2.1=cuda*
    ccpi-regulariser=24.0.1
    cil-data
    cvxpy
    ipywidgets
    packaging
    python-wget
    setuptools
    scikit-image
    tigre=2.6
    tomophantom=2.0.0
    -c conda-forge
    -c https://software.repos.intel.com/python/conda
    -c ccpi/label/dev
    -c ccpi
    --override-channels
  )
fi

conda "${conda_args[@]}"
