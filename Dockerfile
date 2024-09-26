# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# To build with a different base image
# please run `docker build` using the `--build-arg PYTORCH_IMAGE=...` flag.
ARG PYTORCH_IMAGE=nvcr.io/nvidia/pytorch:23.08-py3
FROM ${PYTORCH_IMAGE}

LABEL maintainer="monai.contact@gmail.com"

WORKDIR /opt/monai

# Install full dependencies, remove PyTorch from requirements to avoid conflicts
COPY requirements.txt requirements-min.txt requirements-dev.txt /tmp/
RUN cp /tmp/requirements.txt /tmp/req.bak \
  && awk '!/torch/' /tmp/requirements.txt > /tmp/tmp && mv /tmp/tmp /tmp/requirements.txt \
  && python -m pip install --upgrade --no-cache-dir pip \
  && python -m pip install --no-cache-dir -r /tmp/requirements-dev.txt

# Compile extensions and clean up temporary files
COPY LICENSE CHANGELOG.md CODE_OF_CONDUCT.md CONTRIBUTING.md README.md versioneer.py setup.py setup.cfg runtests.sh MANIFEST.in ./
COPY tests ./tests
COPY monai ./monai
RUN BUILD_MONAI=1 FORCE_CUDA=1 python setup.py develop \
  && rm -rf build __pycache__

# NGC Client setup
WORKDIR /opt/tools
ARG NGC_CLI_URI="https://ngc.nvidia.com/downloads/ngccli_linux.zip"
RUN wget -q ${NGC_CLI_URI} \
  && unzip ngccli_linux.zip && chmod u+x ngc-cli/ngc \
  && find ngc-cli/ -type f -exec md5sum {} + | LC_ALL=C sort | md5sum -c ngc-cli.md5 \
  && rm -rf ngccli_linux.zip ngc-cli.md5
ENV PATH=${PATH}:/opt/tools:/opt/tools/ngc-cli

# Install additional dependencies
RUN apt-get update \
  && DEBIAN_FRONTEND="noninteractive" apt-get install -y libopenslide0  \
  && rm -rf /var/lib/apt/lists/*

# Append /opt/tools to runtime path for NGC CLI to be accessible from all file system locations
ENV PATH=${PATH}:/opt/tools

WORKDIR /opt/monai
