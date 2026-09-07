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

ENV BUILD_MONAI=1

# TODO: remark for issue [revise the dockerfile](https://github.com/zarr-developers/numcodecs/issues/431)
RUN if [[ $(uname -m) =~ "aarch64" ]]; then \
      export CFLAGS="-O3" && \
      export DISABLE_NUMCODECS_SSE2=true && \
      export DISABLE_NUMCODECS_AVX2=true && \
      pip install numcodecs; \
    fi

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

# TODO: remark for issue [revise the dockerfile #1276](https://github.com/Project-MONAI/MONAI/issues/1276)
# please specify exact files and folders to be copied -- else, basically always, the Docker build process cannot cache
# this or anything below it and always will build from at most here; one file change leads to no caching from here on...
COPY LICENSE CHANGELOG.md CODE_OF_CONDUCT.md CONTRIBUTING.md README.md versioneer.py setup.py pyproject.toml runtests.sh MANIFEST.in ./
COPY tests ./tests
COPY monai ./monai

# Need to install build requirements explicitly so that no-build-isolation can be used. This is needed to make pip build
# against the included version of PyTorch, rather than install a new version in the isolated environment. Constraint
# files will not work for this image which installed things like PyTorch through files which are no longer present.
RUN python monai/config/print_dependencies.py build-system | xargs -d '\n' pip install --no-cache-dir --no-build-isolation \
  && FORCE_CUDA=1 pip install --no-cache-dir --no-build-isolation -e .[all,testing]
