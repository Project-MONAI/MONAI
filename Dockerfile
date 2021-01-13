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

FROM ${PYTORCH_IMAGE}

LABEL maintainer="monai.miccai2019@gmail.com"

WORKDIR /opt/monai

# install full deps
COPY requirements.txt requirements-min.txt requirements-dev.txt /tmp/
RUN cp /tmp/requirements.txt /tmp/req.bak \
  && awk '!/torch/' /tmp/requirements.txt > /tmp/tmp && mv /tmp/tmp /tmp/requirements.txt \
  && python -m pip install --no-cache-dir --use-feature=2020-resolver -r /tmp/requirements-dev.txt

# compile ext and remove temp files
# TODO: remark for issue [revise the dockerfile #1276](https://github.com/Project-MONAI/MONAI/issues/1276)
# please specify exact files and folders to be copied -- else, basically always, the Docker build process cannot cache
# this or anything below it and always will build from at most here; one file change leads to no caching from here on...

COPY LICENSE setup.py setup.cfg versioneer.py runtests.sh .gitignore .gitattributes README.md MANIFEST.in ./
COPY tests ./tests
COPY monai ./monai
COPY .git ./.git
RUN BUILD_MONAI=1 FORCE_CUDA=1 python setup.py develop \
  && rm -rf build __pycache__

# NGC Client
WORKDIR /opt/tools
ARG NGC_CLI_URI="https://ngc.nvidia.com/downloads/ngccli_cat_linux.zip"
RUN wget -q ${NGC_CLI_URI} && \
    unzip ngccli_cat_linux.zip && chmod u+x ngc && \
    md5sum -c ngc.md5 && \
    rm -rf ngccli_cat_linux.zip ngc.md5
# append /opt/tools to runtime path for NGC CLI to be accessible from all file system locations
ENV PATH=${PATH}:/opt/tools
WORKDIR /opt/monai
