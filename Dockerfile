# Start from the NVIDIA PyTorch image
ARG PYTORCH_IMAGE=nvcr.io/nvidia/pytorch:24.08-py3
FROM ${PYTORCH_IMAGE}

LABEL maintainer="monai.contact@gmail.com"

# Install system dependencies and set up the environment
RUN apt-get update && DEBIAN_FRONTEND="noninteractive" apt-get install -y \
    libopenslide0 wget unzip && \
    rm -rf /var/lib/apt/lists/*

# Install necessary Python packages
COPY requirements.txt requirements-min.txt requirements-dev.txt /tmp/
RUN cp /tmp/requirements.txt /tmp/req.bak \
  && awk '!/torch/' /tmp/requirements.txt > /tmp/tmp && mv /tmp/tmp /tmp/requirements.txt \
  && python -m pip install --upgrade pip \
  && python -m pip install --no-cache-dir -r /tmp/requirements-dev.txt

# Set up MONAI source
COPY LICENSE CHANGELOG.md CODE_OF_CONDUCT.md CONTRIBUTING.md README.md versioneer.py setup.py setup.cfg runtests.sh MANIFEST.in ./
COPY tests ./tests
COPY monai ./monai

# Build MONAI and clean up
RUN python setup.py develop && rm -rf build __pycache__

# Install flake8 and mypy for linting and type checking
RUN pip install flake8 mypy

# Run code format checks
RUN flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics || { echo "flake8 failed"; exit 1; }

# Run mypy for type checking
RUN mypy . --ignore-missing-imports || { echo "mypy failed"; exit 1; }

# Install NGC CLI tools
WORKDIR /opt/tools
ARG NGC_CLI_URI="https://ngc.nvidia.com/downloads/ngccli_linux.zip"
RUN wget -q ${NGC_CLI_URI} && unzip ngccli_linux.zip && chmod u+x ngc-cli/ngc \
    && rm -rf ngccli_linux.zip

# Add NGC CLI to PATH
ENV PATH=${PATH}:/opt/tools:/opt/tools/ngc-cli
