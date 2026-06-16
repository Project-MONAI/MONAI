# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# To build with a different base image
# please run `docker build` using the `--build-arg PYTORCH_IMAGE=...` flag.
ARG PYTORCH_IMAGE=nvcr.io/nvidia/pytorch:25.12-py3
FROM ${PYTORCH_IMAGE}

LABEL maintainer="monai.contact@gmail.com"

# TODO: remark for issue [revise the dockerfile](https://github.com/zarr-developers/numcodecs/issues/431)
RUN if [[ $(uname -m) =~ "aarch64" ]]; then \
      export CFLAGS="-O3" && \
      export DISABLE_NUMCODECS_SSE2=true && \
      export DISABLE_NUMCODECS_AVX2=true && \
      pip install numcodecs; \
    fi

WORKDIR /opt/monai

# Patch NVIDIA's pip constraint file:
#   - keep the base image's numpy pin if it has one (nv25.03 pins numpy==1.26.4 as its
#     torch was compiled against NumPy 1.x; nv25.12 ships an empty constraint.txt)
#   - add setuptools<71 (newer setuptools removed pkg_resources needed by legacy setup.py)
#   - pin urllib3>=2 so that notebook cells doing !pip install legacy packages (e.g.
#     bentoml==0.13.1) cannot downgrade urllib3 to 1.x, which would break requests,
#     huggingface_hub, gdown, transformers, mlflow, etc. for every subsequent notebook
#   - remove jupytext/isort pins so the tutorial runner can install its required versions
RUN (grep '^numpy' /etc/pip/constraint.txt || true) > /tmp/new_constraints.txt \
  && printf 'setuptools<71\nurllib3>=2\n' >> /tmp/new_constraints.txt \
  && cp /tmp/new_constraints.txt /etc/pip/constraint.txt

# install full deps
COPY requirements.txt requirements-min.txt requirements-dev.txt /tmp/
RUN cp /tmp/requirements.txt /tmp/req.bak \
  && awk '!/torch/' /tmp/requirements.txt > /tmp/tmp && mv /tmp/tmp /tmp/requirements.txt \
  && python -m pip install --upgrade --no-cache-dir --no-build-isolation pip wheel wheel-stub \
  && python -m pip install --no-cache-dir --no-build-isolation -r /tmp/requirements-dev.txt \
  && python -m pip install --no-cache-dir --no-build-isolation papermill jupytext autopep8 autoflake ipywidgets

# compile ext and remove temp files
# TODO: remark for issue [revise the dockerfile #1276](https://github.com/Project-MONAI/MONAI/issues/1276)
# please specify exact files and folders to be copied -- else, basically always, the Docker build process cannot cache
# this or anything below it and always will build from at most here; one file change leads to no caching from here on...

COPY LICENSE CHANGELOG.md CODE_OF_CONDUCT.md CONTRIBUTING.md README.md versioneer.py setup.py setup.cfg runtests.sh MANIFEST.in ./
COPY tests ./tests
COPY monai ./monai

RUN BUILD_MONAI=1 FORCE_CUDA=1 python setup.py develop \
  && rm -rf build __pycache__

# NGC Client
WORKDIR /opt/tools
ARG NGC_CLI_URI="https://ngc.nvidia.com/downloads/ngccli_linux.zip"
RUN wget -q ${NGC_CLI_URI} && unzip ngccli_linux.zip && chmod u+x ngc-cli/ngc && \
    find ngc-cli/ -type f -exec md5sum {} + | LC_ALL=C sort | md5sum -c ngc-cli.md5 && \
    rm -rf ngccli_linux.zip ngc-cli.md5
ENV PATH=${PATH}:/opt/tools:/opt/tools/ngc-cli
RUN apt-get update \
  && DEBIAN_FRONTEND="noninteractive" apt-get install -y libopenslide0  \
  && rm -rf /var/lib/apt/lists/*
# append /opt/tools to runtime path for NGC CLI to be accessible from all file system locations
ENV PATH=${PATH}:/opt/tools
ENV POLYGRAPHY_AUTOINSTALL_DEPS=1


WORKDIR /opt/monai
