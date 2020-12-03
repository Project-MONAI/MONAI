# Copyright 2020 MONAI Consortium
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

WORKDIR /opt/monai
ENV PATH=/opt/tools:$PATH

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
RUN wget -q https://ngc.nvidia.com/downloads/ngccli_cat_linux.zip && \
    unzip ngccli_cat_linux.zip && chmod u+x ngc && \
    rm -rf ngccli_cat_linux.zip ngc.md5
WORKDIR /opt/monai
