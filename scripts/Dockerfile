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

ARG CIL_VERSION=23.1.0
# CUDA-specific packages
ARG CIL_EXTRA_PACKAGES=astra-toolbox tigre
RUN mamba install -y -c conda-forge -c intel -c ccpi \
  $CIL_EXTRA_PACKAGES \
  cil=$CIL_VERSION ccpi-regulariser tomophantom "ipywidgets<8" \
  jupyter-server-proxy \
  && mamba clean -a -y -f \
  && fix-permissions "${CONDA_DIR}" \
  && fix-permissions "/home/${NB_USER}"

# NB: trailing `/` is required
ENV TENSORBOARD_PROXY_URL=/user-redirect/proxy/6006/
