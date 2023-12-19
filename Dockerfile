# sources:
# - https://github.com/jupyter/docker-stacks
# - https://github.com/TomographicImaging/CIL#installation-of-cil
# consumers:
# - https://github.com/stfc/cloud-docker-images
# TODO: use `ubuntu-22.04` after `python=3.11` is supported, vis.
# https://github.com/TomographicImaging/CIL/issues/1490
FROM jupyter/tensorflow-notebook:ubuntu-20.04
LABEL org.opencontainers.image.source=https://github.com/TomographicImaging/CIL
LABEL org.opencontainers.image.description="Core Imaging Library"
LABEL org.opencontainers.image.licenses="Apache-2.0"

# CUDA-specific packages
ARG CIL_EXTRA_PACKAGES=astra-toolbox tigre
# build & runtime dependencies
# TODO: sync with scripts/requirements-test.yml
# TODO: sync with scripts/create_local_env_for_cil_development.sh
RUN mamba install -y -c conda-forge -c intel -c ccpi \
  $CIL_EXTRA_PACKAGES \
  _openmp_mutex ccpi-regulariser cmake dxchange h5py libgcc-ng libstdcxx-ng numba numpy pillow pip pywavelets setuptools tomophantom tqdm \
  python-wget scikit-image packaging \
  "cil-data>=21.3.0" "ipp>=2021.10" "ipp-devel>=2021.10" "ipp-include>=2021.10" "ipywidgets<8" "matplotlib>=3.3.0" "olefile>=0.46" "scipy>=1.4.0" \
  jupyter-server-proxy \
  && mamba clean -a -y -f \
  && fix-permissions "${CONDA_DIR}" \
  && fix-permissions "/home/${NB_USER}"

# NB: trailing `/` is required
ENV TENSORBOARD_PROXY_URL=/user-redirect/proxy/6006/

# build & install CIL
COPY --chown="${NB_USER}" . src
RUN mkdir build && cd build \
  && cmake ../src -DCMAKE_BUILD_TYPE="Release" -DCONDA_BUILD=ON -DCMAKE_INSTALL_PREFIX="${CONDA_DIR}" \
  && cmake --build . --target install \
  && cd .. && rm -rf src build \
  && fix-permissions "${CONDA_DIR}"
