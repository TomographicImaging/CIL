#!/usr/bin/env bash
set -euxo pipefail

if test $(python -c "from __future__ import print_function; import platform; print(platform.system())") = Darwin ; then
  echo "Darwin"
  extra_args="-DOPENMP_LIBRARIES=${CONDA_PREFIX}/lib -DOPENMP_INCLUDES=${CONDA_PREFIX}/include"
else
  echo "something else"
  extra_args=""
fi

export SETUPTOOLS_SCM_PRETEND_VERSION_FOR_CIL="${PKG_VERSION}"
if test "${GIT_DESCRIBE_NUMBER}" != "0"; then
  export SETUPTOOLS_SCM_PRETEND_VERSION_FOR_CIL="${PKG_VERSION}.dev${GIT_DESCRIBE_NUMBER}+${GIT_DESCRIBE_HASH}"
fi
cmake ${RECIPE_DIR}/../ $extra_args \
                        -DCONDA_BUILD=ON \
                        -DCMAKE_BUILD_TYPE=RelWithDebInfo \
                        -DLIBRARY_LIB=$CONDA_PREFIX/lib \
                        -DLIBRARY_INC=$CONDA_PREFIX/include \
                        -DCMAKE_INSTALL_PREFIX=$PREFIX
cmake --build . --target install
