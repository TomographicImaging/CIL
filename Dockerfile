# sources:
# - https://github.com/jupyter/docker-stacks
# - https://github.com/TomographicImaging/CIL#installation-of-cil
# consumers:
# - https://github.com/stfc/cloud-docker-images
# TODO: use `ubuntu-22.04` after `python=3.11` is supported, vis.
# https://github.com/TomographicImaging/CIL/issues/1490
FROM jupyter/tensorflow-notebook:ubuntu-20.04
LABEL org.opencontainers.image.source=https://github.com/TomographicImaging/CIL
# tigre: BSD-3-Clause, astra-toolbox: GPL-3.0
LABEL org.opencontainers.image.licenses="Apache-2.0 AND BSD-3-Clause AND GPL-3.0"

# CUDA-specific packages
ARG CIL_EXTRA_PACKAGES=tigre astra-toolbox
# build & runtime dependencies
# TODO: sync scripts/create_local_env_for_cil_development.sh, scripts/requirements-test.yml, recipe/meta.yaml (e.g. missing libstdcxx-ng _openmp_mutex pip)?
# vis. https://github.com/TomographicImaging/CIL/pull/1590
COPY --chown="${NB_USER}" scripts/requirements-test.yml environment.yml
# channel_priority: https://stackoverflow.com/q/58555389
RUN sed -ri '/tigre|astra-toolbox/d' environment.yml \
  && for pkg in jupyter-server-proxy $CIL_EXTRA_PACKAGES; do echo "  - $pkg" >> environment.yml; done \
  && conda config --env --set channel_priority strict \
  && for ch in defaults ccpi intel conda-forge; do conda config --env --add channels $ch; done \
  && mamba env update -n base \
  && mamba clean -a -y -f \
  && rm environment.yml \
  && fix-permissions "${CONDA_DIR}" /home/${NB_USER}

# NB: trailing `/` is required
ENV TENSORBOARD_PROXY_URL=/user-redirect/proxy/6006/

# build & install CIL
COPY --chown="${NB_USER}" . src
RUN mkdir build && cd build \
  && cmake ../src -DCMAKE_BUILD_TYPE=RelWithDebInfo -DCONDA_BUILD=ON -DCMAKE_INSTALL_PREFIX="${CONDA_DIR}" \
  && cmake --build . --target install \
  && cd .. && rm -rf src build \
  && fix-permissions "${CONDA_DIR}" /home/${NB_USER}
