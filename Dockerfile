# Copyright 2020 - 2021 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

ARG PYTORCH_IMAGE=nvcr.io/nvidia/pytorch:20.10-py3

FROM ${PYTORCH_IMAGE} as base

MAINTAINER Wenqi Li <wenqil@nvidia.com>, Nic Ma <nma@nvidia.com>, Daniel Schulz <danielschulz2005@hotmail.com>

WORKDIR /opt/monai
ENV PATH=/opt/tools:${PATH}

# install full deps
COPY requirements.txt requirements-min.txt requirements-dev.txt /tmp/
RUN cp /tmp/requirements.txt /tmp/req.bak \
  && awk '!/torch/' /tmp/requirements.txt > /tmp/tmp && mv /tmp/tmp /tmp/requirements.txt \
  && python -m pip install --no-cache-dir --use-feature=2020-resolver -r /tmp/requirements-dev.txt

# compile ext and remove temp files
COPY . .
RUN BUILD_MONAI=1 FORCE_CUDA=1 python setup.py develop \
  && rm -rf build __pycache__

# NGC Client
WORKDIR /opt/tools
ARG NGC_CLI_SHA256_HASH="03c86402449469a6d12ac2fbbf5c8b460bdbed4f1e01f692e07eb1c144f6de0f"
ARG NGC_CLI_URI="https://ngc.nvidia.com/downloads/ngccli_cat_linux.zip"
RUN wget -q ${NGC_CLI_URI} && \
    # check integrity of downloaded archive using SHA256 hash; append "-s" option to supress print oneliner
    echo "${NGC_CLI_SHA256_HASH}  ./ngccli_cat_linux.zip" | sha256sum -c && \
    unzip ngccli_cat_linux.zip && chmod u+x ngc && \
    rm -rf ngccli_cat_linux.zip ngc.md5
WORKDIR /opt/monai
