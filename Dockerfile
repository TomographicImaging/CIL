# sources:
# - https://github.com/jupyter/docker-stacks
# - https://github.com/TomographicImaging/CIL#installation-of-cil
# consumers:
# - harbor.stfc.ac.uk/imaging-tomography/cil
ARG BASE_IMAGE=quay.io/jupyter/tensorflow-notebook:cuda-python-3.12
FROM ${BASE_IMAGE} AS base
LABEL org.opencontainers.image.source=https://github.com/TomographicImaging/CIL
# tigre: BSD-3-Clause, astra-toolbox: GPL-3.0
LABEL org.opencontainers.image.licenses="Apache-2.0 AND BSD-3-Clause AND GPL-3.0"

# CUDA-specific packages
ARG CIL_EXTRA_PACKAGES="ccpi/label/dev::tigre astra-toolbox::astra-toolbox ccpi::ccpi-regulariser cuda-version>=12"
# build & runtime dependencies
# TODO: sync scripts/create_local_env_for_cil_development.sh, scripts/requirements-test.yml, recipe/meta.yaml (e.g. missing libstdcxx-ng _openmp_mutex pip)?
# vis. https://github.com/TomographicImaging/CIL/pull/1590
USER root
COPY scripts/requirements-test.yml /opt/environment.yml
RUN sed -ri -e '/tigre| python |cuda/d' /opt/environment.yml \
  && for pkg in 'jupyter-server-proxy>4.1.0' $CIL_EXTRA_PACKAGES; do echo "  - $pkg" >> /opt/environment.yml; done \
  && mamba env update -n base --file /opt/environment.yml \
  && mamba clean -a -y -f \
  && fix-permissions "${CONDA_DIR}" /home/${NB_USER}

# NB: trailing `/` is required
ENV TENSORBOARD_PROXY_URL=/user-redirect/proxy/6006/

FROM base AS dev
RUN apt-get update -qq && apt-get install -yqq --no-install-recommends \
  bash-completion \
  && rm -rf /var/lib/apt/lists/*
COPY docs/docs_environment.yml .
RUN mamba install conda-merge \
  && conda-merge /opt/environment.yml docs_environment.yml > environment.yml \
  && mamba env update -n base --file environment.yml \
  && mamba clean -a -y -f \
  && rm {docs_,}environment.yml \
  && fix-permissions "${CONDA_DIR}" /home/${NB_USER}
# https://github.com/rbenv/rbenv
RUN apt-get update -qq && apt-get install -yqq --no-install-recommends \
  zlib1g-dev libffi-dev libyaml-dev \
  && rm -rf /var/lib/apt/lists/*
USER ${NB_USER}
ENV RBENV_VERSION=3.4.8
RUN git clone https://github.com/rbenv/rbenv.git ~/.rbenv \
  && ~/.rbenv/bin/rbenv init \
  && export PATH="$HOME/.rbenv/bin:$PATH" \
  && git clone https://github.com/rbenv/ruby-build.git "$(rbenv root)"/plugins/ruby-build \
  && rbenv install $RBENV_VERSION && rbenv global $RBENV_VERSION

FROM base AS cil
# build & install CIL
COPY . src
RUN pip install ./src && rm -rf src \
  && fix-permissions "${CONDA_DIR}" /home/${NB_USER}
USER ${NB_USER}
