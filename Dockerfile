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
COPY . .

ENV PATH=/opt/tools:$PATH

# ignore "torch" in requirements.txt, as we prefer PYTORCH_IMAGE
RUN cp requirements.txt req.bak \
  && awk '!/torch/' requirements.txt > tmp && mv tmp requirements.txt \
  && python -m pip install --no-cache-dir -U pip wheel \
  && python -m pip install --no-cache-dir -r requirements-dev.txt \
  && mv req.bak requirements.txt \
  && BUILD_MONAI=1 FORCE_CUDA=1 python setup.py develop
# restored the original requirements.txt so that the version string is clean

# NGC Client
WORKDIR /opt/tools
RUN wget -q https://ngc.nvidia.com/downloads/ngccli_cat_linux.zip && \
    unzip ngccli_cat_linux.zip && chmod u+x ngc && \
    rm -rf ngccli_cat_linux.zip ngc.md5
WORKDIR /opt/monai
