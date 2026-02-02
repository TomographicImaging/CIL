#!/usr/bin/env bash
set -euxo pipefail

extra_args="-G Ninja"
if test $(python -c "from __future__ import print_function; import platform; print(platform.system())") = Darwin ; then
  echo "Darwin"
  extra_args="$extra_args -DOPENMP_LIBRARIES=${CONDA_PREFIX}/lib -DOPENMP_INCLUDES=${CONDA_PREFIX}/include"
else
  extra_args="$extra_args -DIPP_ROOT=${CONDA_PREFIX}"
fi

export SETUPTOOLS_SCM_PRETEND_VERSION_FOR_CIL="${PKG_VERSION}"
if test "${GIT_DESCRIBE_NUMBER}" != "0"; then
  export SETUPTOOLS_SCM_PRETEND_VERSION_FOR_CIL="${PKG_VERSION}.dev${GIT_DESCRIBE_NUMBER}+${GIT_DESCRIBE_HASH}"
fi
pip install . --no-deps -Ccmake.args="${extra_args}"
