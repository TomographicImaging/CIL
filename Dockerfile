# sources:
# - https://github.com/jupyter/docker-stacks
# - https://github.com/TomographicImaging/CIL#installation-of-cil
# consumers:
# - harbor.stfc.ac.uk/imaging-tomography/cil
FROM quay.io/jupyter/tensorflow-notebook:ubuntu-24.04
LABEL org.opencontainers.image.source=https://github.com/TomographicImaging/CIL
# tigre: BSD-3-Clause, astra-toolbox: GPL-3.0
LABEL org.opencontainers.image.licenses="Apache-2.0 AND BSD-3-Clause AND GPL-3.0"

# CUDA-specific packages
ARG CIL_EXTRA_PACKAGES="ccpi::tigre=3.1.3 astra-toolbox::astra-toolbox=2.4"
# build & runtime dependencies
# TODO: sync scripts/cil_development.yml, recipe.yaml
COPY --chown="${NB_USER}" scripts/cil_development.yml environment.yml
RUN sed -ri '/tigre|astra-toolbox| python /d' environment.yml \
  && for pkg in 'jupyter-server-proxy>4.1.0' $CIL_EXTRA_PACKAGES; do echo "  - $pkg" >> environment.yml; done \
  && mamba env update -n base -f environment.yml \
  && mamba clean -a -y -f \
  && rm environment.yml \
  && fix-permissions "${CONDA_DIR}" /home/${NB_USER}

# NB: trailing `/` is required
ENV TENSORBOARD_PROXY_URL=/user-redirect/proxy/6006/

# build & install CIL
COPY --chown="${NB_USER}" . src
ENV CIL_FORCE_IPP=ON
RUN pip install ./src && rm -rf src \
  && fix-permissions "${CONDA_DIR}" /home/${NB_USER}
