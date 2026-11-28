# Installation Guide

## Table of Contents

- [Installation Guide](#installation-guide)
	- [Table of Contents](#table-of-contents)
	- [GPU-enabled installation (CUDA and CuPy)](#gpu-enabled-installation-cuda-and-cupy)
	- [GPU-enabled installation (AMD ROCm)](#gpu-enabled-installation-amd-rocm)
	- [From PyPI](#from-pypi)
		- [Milestone release](#milestone-release)
		- [Weekly preview release](#weekly-preview-release)
		- [Uninstall the packages](#uninstall-the-packages)
	- [From conda-forge](#from-conda-forge)
	- [From GitHub](#from-github)
		- [Option 1 (as a part of your system-wide module):](#option-1-as-a-part-of-your-system-wide-module)
		- [Option 2 (editable installation):](#option-2-editable-installation)
	- [Validating the install](#validating-the-install)
	- [MONAI version string](#monai-version-string)
	- [From DockerHub](#from-dockerhub)
	- [Installing the recommended dependencies](#installing-the-recommended-dependencies)

---

MONAI's core functionality is written in Python 3 (>= 3.11) and only requires [Numpy](https://numpy.org/) and [PyTorch](https://pytorch.org/).

The package is currently distributed via Github as the primary source code repository,
and the Python package index (PyPI). The pre-built Docker images are made available on DockerHub.

To install optional features such as handling the NIfTI files using
[Nibabel](https://nipy.org/nibabel/), or building workflows using [PyTorch
Ignite](https://pytorch.org/ignite/), please follow the instructions:

- [Installing the recommended dependencies](#installing-the-recommended-dependencies)

---

## GPU-enabled installation (CUDA and CuPy)

The installation commands below usually end up installing the CPU variant of PyTorch. To install GPU-enabled PyTorch:

1. Install the latest NVIDIA driver.
1. Check the [PyTorch Official Guide](https://pytorch.org/get-started/locally/) for the recommended CUDA versions. For Pip packages, PyTorch wheels already bundle the CUDA runtime, so you only need to pick the CUDA version matching your driver from the selector and install with the provided command. You do not need to manually download CUDA or set `CUDA_PATH`.
1. Continue to follow the guide and install PyTorch.
1. Install MONAI using one of the ways described below.

Installing GPU-enabled PyTorch is enough to run models and transforms on the GPU. Some transforms,
however, additionally use [CuPy](https://cupy.dev/) for GPU-accelerated array operations (for example
when a transform converts a CUDA tensor via `convert_to_cupy`). If CuPy is not installed, these code
paths raise `OptionalImportError: import cupy (No module named 'cupy')`.

MONAI provides a dedicated `cupy` extra that installs a compatible CuPy build:

```bash
pip install 'monai[cupy]'
```

The `cucim` extra installs [cuCIM](https://github.com/rapidsai/cucim) (`cucim-cu12` or `cucim-cu13`
depending on your Python version), which is a separate GPU image-processing library and does not
install CuPy.

If you prefer to install CuPy directly, note that the PyPI package name is CUDA-version specific
(e.g. `cupy-cuda12x` for CUDA 12.x, `cupy-cuda13x` for CUDA 13.x) rather than plain `cupy`. See the
[CuPy installation guide](https://docs.cupy.dev/en/stable/install.html) for the correct package for
your CUDA toolkit.

---

## GPU-enabled installation (AMD ROCm)

On AMD GPUs MONAI runs on the [ROCm](https://rocm.docs.amd.com/) build of PyTorch. The ROCm packages are
served from AMD's package indexes rather than PyPI, so the index URLs are given on the command line.

1. Install the ROCm SDK with the ROCm builds of PyTorch and Torchvision, naming the architecture of your GPU
   (`gfx942` for the MI300 series, `gfx950` for the MI350 series), then unpack the SDK's development files:

   ```bash
   pip install --index-url https://stable.repo.amd.com/rocm/whl-next/ \
       "rocm[libraries,devel,device-gfx950]==10.0.*" "torch[device-gfx950]" "torchvision[device-gfx950]"
   rocm-sdk init
   ```

1. Install MONAI with the AMD extras, adding AMD's extensions index for the same ROCm series:

   ```bash
   pip install --extra-index-url https://pypi.amd.com/rocm-10.0.0/simple/ 'monai[all_amd]'
   ```

Three extras are provided for ROCm:

- `hipcim` installs [hipCIM](https://rocm.docs.amd.com/projects/hipCIM/en/latest/) (`amd-hipcim`), the ROCm
  build of cuCIM.
- `amd_cupy` installs `amd-cupy`, the ROCm build of CuPy.
- `all_amd` is the counterpart of `all`: the optional dependencies that do not need NVIDIA hardware, together
  with `hipcim` and `amd_cupy`.

`amd-hipcim` provides the same `cucim` module as `cucim-cu12`/`cucim-cu13`, and `amd-cupy` the same `cupy`
module as `cupy-cuda13x`, so MONAI code using either works unchanged. For the same reason only one of each
pair can be installed in an environment: do not combine the AMD extras with `all`, `cucim`, `cupy` or
`nvimgcodec`.

A few points to be aware of:

- `amd-cupy` and `amd-hipcim` are currently released for ROCm 10.0. Keep the ROCm SDK on that series, as the
  `==10.0.*` pin above does.
- Install Torchvision from the ROCm index in the first step. If it is left to the second step, `pip` takes
  the PyPI build, which is not built for ROCm.
- `amd-cupy` and `amd-hipcim` are served from AMD's extensions index, so the `--extra-index-url` option is
  required in the second step.
- `amd-hipcim` is currently released for CPython 3.12 on x86_64 Linux.
- `Dockerfile.rocm` in the repository root sets up a complete environment, including the compiler and library
  paths needed to build the C++/HIP extensions.

---

## From PyPI

### Milestone release

To install the [current milestone release](https://pypi.org/project/monai/):

```bash
pip install monai
```

MONAI supports the extras syntax such as `pip install 'monai[nibabel]'`. The options are

```text
amd_cupy, clearml, cucim, cupy, einops, fire, gdown, h5py, hipcim, huggingface_hub, hyena, ignite, imagecodecs, itk, jsonschema, lmdb, matplotlib, metrics_reloaded, mlflow, nibabel, nni, nvimgcodec, onnx, openslide, optuna, pandas, pillow, polygraphy, psutil, pyamg, pybind11, pydicom, pynrrd, pynvml, pyyaml, requests, segment_anything, scipy, skimage, tensorboard, tensorboardX, tifffile, torchio, torchvision, tqdm, transformers, zarr
```

which correspond to the packages: `amd-cupy`, `clearml`, `cucim` (`cucim-cu12` or `cucim-cu13`), `cupy-cuda13x`, `einops`, `fire`, `gdown`, `h5py`, `amd-hipcim`, `huggingface_hub`, `nvsubquadratic`, `omegaconf`, `pytorch-ignite`, `imagecodecs`, `itk`, `jsonschema`, `lmdb`, `matplotlib`, `MetricsReloaded`, `mlflow`, `nibabel`, `nni`, `filelock`, `nvidia-nvimgcodec-cu13`, `onnx`, `onnxruntime`, `onnx_graphsurgeon`, `onnxscript`, `openslide-python`, `openslide-bin`, `optuna`, `pandas`, `pillow`, `polygraphy`, `psutil`, `pyamg`, `pybind11`, `pydicom`, `pynrrd`, `nvidia-ml-py`, `pyyaml`, `requests`, `segment_anything`, `scipy`, `scikit-image`, `tensorboard`, `tensorboardX`, `tifffile`, `torchio`, `torchvision`, `tqdm`, `transformers`, `zarr`.

The AlexNet, VGG, and SqueezeNet options in `PerceptualLoss` require `torchvision`
(`pip install "monai[torchvision]"`). The external `lpips` package is no longer required.
ImageNet feature weights are downloaded by Torchvision. LPIPS calibration weights are
downloaded from the official [MONAI Hugging Face repository](https://huggingface.co/MONAI/checkpoints),
verified with SHA-256, and cached under the Torch Hub `checkpoints` directory on first use.
`PerceptualLoss(cache_dir=...)` selects the Torch Hub cache location for both sets of weights.
With `pretrained=False`, calibration weights are not downloaded; ImageNet feature weights
are still required.
The existing `monai[lpips]` extra is retained as an alias for `monai[torchvision]`.

Almost all of these can be installed together with the `all` option. For development on MONAI, this should be accompanied by `testing` which will install the testing static checking packages. Cupy and `nvimgcodec` are omitted from `all` since the choice between
Cuda 12 and 13 versions of the libraries can't be resolved when installing and must be manually installed.
On AMD ROCm use `all_amd` in place of `all`, see [GPU-enabled installation (AMD ROCm)](#gpu-enabled-installation-amd-rocm).

The `nvimgcodec` extra installs GPU-accelerated DICOM decoding for `NvImgCodecPydicomReader`
(`pip install 'monai[nvimgcodec]'`). It is Linux-only in the extra definition; CUDA 13 is the
default. CUDA 12 users should install matching `cupy-cuda12x` and `nvidia-nvimgcodec-cu12` wheels.

The `hyena` extra pulls in [`nvsubquadratic`](https://github.com/NVIDIA-BioNeMo/nvSubquadratic),
required by `HyenaNDUNETR` / `HyenaMixer` / `HyenaTransformerBlock` (subquadratic
O(N log N) alternatives to windowed self-attention). Install with
`pip install 'monai[hyena]'`.

The command  `pip install 'monai[all,hyena,testing]'` installs almost all the optional dependencies.

When installing MONAI, the compiled extensions are not compiled by default. Set the environment variable `BUILD_MONAI` to `1` when invoking `pip` to compile these, see below for details.

### Weekly preview release

To install the [weekly preview release](https://pypi.org/project/monai-weekly/):

```bash
pip install monai-weekly
```

The weekly build is released to PyPI every Sunday with a pre-release build number `dev[%y%U]`.
To report any issues on the weekly preview, please include the version information:

```bash
python -c "import monai; print(monai.__version__)"
```

Coexistence of package `monai` and `monai-weekly` in a system may cause namespace conflicts
and `ImportError`.
This is usually a result of running both `pip install monai` and `pip install monai-weekly`
without uninstalling the existing one first.
To address this issue, please uninstall both packages, and retry the installation.

### Uninstall the packages

The packages installed using `pip install` could be removed by:

```bash
pip uninstall -y monai
pip uninstall -y monai-weekly
```

## From conda-forge

To install the [current milestone release](https://anaconda.org/conda-forge/monai):

```bash
conda install -c conda-forge monai
```

## From GitHub

(_If you have installed the
PyPI release version using `pip install monai`, please run `pip uninstall
monai` before using the commands from this section. Because `pip` by
default prefers the milestone release_.)

The milestone versions are currently planned and released every few months. As the
codebase is under active development, you may want to install MONAI from GitHub
for the latest features:

### Option 1 (as a part of your system-wide module):

```bash
pip install git+https://github.com/Project-MONAI/MONAI
```

or, to build with MONAI C++/CUDA extensions:

```bash
BUILD_MONAI=1 pip install git+https://github.com/Project-MONAI/MONAI
```

On Windows the inline `BUILD_MONAI=1 pip install ...` form is not supported by
`cmd.exe` or PowerShell. Set the environment variable first, then run either
install command shown above:

```bat
:: cmd.exe
set BUILD_MONAI=1
pip install git+https://github.com/Project-MONAI/MONAI
```

```powershell
# PowerShell
$env:BUILD_MONAI="1"
pip install git+https://github.com/Project-MONAI/MONAI
```

To build the extensions, if the system environment already has a version of PyTorch installed, `--no-build-isolation` might be preferred:

```bash
BUILD_MONAI=1 pip install --no-build-isolation git+https://github.com/Project-MONAI/MONAI
```

When using build isolation (pip's default behaviour), a version of PyTorch must be installed which may not be the same as an existing install. This can cause the compiled libraries to be built against an ABI-incompatible PyTorch and thus not function at runtime. Building without isolation requires the current environment to have the necessary building libraries already installed. See the `build-system` section of `pyproject.toml` for these libraries, or use the following to install them in a bash environment:

```bash
python monai/config/print_dependencies.py build-system | xargs -d '\n' pip install --no-build-isolation
```

An alternative solution is to use built constraints during installation:

```bash
pip freeze | grep torch > constraints.txt
pip install --build-constraint constraints.txt git+https://github.com/Project-MONAI/MONAI
```

this command will download and install the current `dev` branch of [MONAI from
GitHub](https://github.com/Project-MONAI/MONAI).

This documentation website by default shows the information for the latest version.

### Option 2 (editable installation):

To install an editable version of MONAI, it is recommended to clone the codebase directly:

```bash
git clone https://github.com/Project-MONAI/MONAI.git
```

This command will create a `MONAI/` folder in your current directory.
You can install it by running:

```bash
cd MONAI/
pip install -e .
# or pip install -e '.[all,testing]' to include most of the dependencies
```

or, to build with MONAI C++/CUDA extensions and install:

```bash
cd MONAI/
BUILD_MONAI=1 pip install -e .
# for MacOS
BUILD_MONAI=1 CC=clang CXX=clang++ pip install -e .
```

On Windows set the environment variable before running `pip install -e .`:

```bat
:: cmd.exe
cd MONAI/
set BUILD_MONAI=1
pip install -e .
```

```powershell
# PowerShell
cd MONAI/
$env:BUILD_MONAI="1"
pip install -e .
```

If the compiled extensions were built by pip against a different version of PyTorch than the one in your environment, you may need to run the above with the `--no-build-isolation` flag to force the use of that version, or use the `--build-constraint` method.

To uninstall the package please run:

```bash
cd MONAI/
pip uninstall -y monai

# to further clean up the MONAI/ folder (Bash script)
./runtests.sh --clean
```

Alternatively, simply adding the root directory of the cloned source code (e.g., `/workspace/Documents/MONAI`) to your `$PYTHONPATH`
and the codebase is ready to use (without the additional features of MONAI C++/CUDA extensions).

> The C++/CUDA extension features are currently experimental, a pre-compiled version is made available via
> [the recent docker image releases](https://hub.docker.com/r/projectmonai/monai).
> Building the extensions from source may require [Ninja](https://ninja-build.org/) and [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit).
> By default, CUDA extension is built if `torch.cuda.is_available()`. It's possible to force building by
> setting `FORCE_CUDA=1` environment variable.

## Validating the install

You can verify the installation by:

```bash
python -c "import monai; monai.config.print_config()"
```

If the installation is successful, this command will print out the MONAI version information, and this confirms the core
modules of MONAI are ready-to-use.

## MONAI version string

The MONAI version string shows the current status of your local installation. For example:

```
MONAI version: 0.1.0+144.g52c763d.dirty
```

- `0.1.0` indicates that your installation is based on the `0.1.0` milestone release.
- `+144` indicates that your installation is 144 git commits ahead of the milestone release.
- `g52c763d` indicates that your installation corresponds to the git commit hash `52c763d`.
- `dirty` indicates that you have modified the codebase locally, and the codebase is inconsistent with `52c763d`.

## From DockerHub

Make sure you have installed the NVIDIA driver and Docker 19.03+ for your Linux distribution.
Note that you do not need to install the CUDA toolkit on the host, but the driver needs to be installed.
Please find out more information on [nvidia-docker](https://github.com/NVIDIA/nvidia-docker).

Assuming that you have the Nvidia driver and Docker 19.03+ installed, running the following command will
download and start a container with the latest version of MONAI. The latest `dev` branch of MONAI from GitHub
is included in the image.

```bash
docker run --gpus all --rm -ti --ipc=host projectmonai/monai:latest
```

You can also run a milestone release docker image by specifying the image tag, for example:

```
docker run --gpus all --rm -ti --ipc=host projectmonai/monai:0.1.0
```

## Installing the recommended dependencies

By default, the installation steps will only download and install the minimal requirements of MONAI.
Optional dependencies can be installed using [the extras syntax](https://packaging.python.org/tutorials/installing-packages/#installing-setuptools-extras) to support additional features.

For example, to install MONAI with Nibabel and Scikit-image support:

```bash
git clone https://github.com/Project-MONAI/MONAI.git
cd MONAI/
pip install -e '.[nibabel,skimage]'
```

Alternatively, to install all optional dependencies:

```bash
git clone https://github.com/Project-MONAI/MONAI.git
cd MONAI/
pip install -e ".[all]"
```

To install all optional dependencies with `pip` based on MONAI development environment settings without installing
MONAI itself:

```bash
git clone https://github.com/Project-MONAI/MONAI.git
cd MONAI/
python monai/config/print_dependencies.py \* > requirements.txt
pip install -r requirements.txt
```

To install all optional dependencies with `conda` based on MONAI development environment settings (`environment-dev.yml`;
this will install PyTorch as well as `pytorch-cuda`, please follow https://pytorch.org/get-started/locally/#start-locally for more details about installing PyTorch):

```bash
git clone https://github.com/Project-MONAI/MONAI.git
cd MONAI/
conda create -n <name> python=<ver>  # eg 3.11
conda env update -n <name> -f environment-dev.yml
```
