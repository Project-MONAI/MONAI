
# What's new in 1.6.1 🎉🎉

This is primarily a security release with fixes for vulnerabilities identified through security alerts and tools.

## Security Fixes

A number of recent security alerts are addressed in this release: GHSA-vm9c-7j6g-c7mm, GHSA-8f32-8649-rv87, 
GHSA-x6pr-233j-x5cw, GHSA-wvpx-5qmp-46g3, GHSA-2wx3-8x3w-r8qv, GHSA-x4pc-gj5h-3pq7, GHSA-hhh4-h52m-fqh6, and GHSA-6hp3-vr39-rqw8.
These will be published after the release and will be marked as being fixed by version 1.6.1.

Actions permissions were updated to only allow read access when this is all that's required. A few other changes were
made relating to CodeQL identified issues (#9098).

## Added Features

A number of additions are included with this security release:

* `NaViT` (`monai.networks.nets.NaViT`): Native Resolution Vision Transformer with Patch n' Pack, supporting variable-resolution 2D and 3D inputs. Implements factorized positional embeddings, token dropout, attention pooling, and QK normalization, based on ["Patch n' Pack: NaViT, a Vision Transformer for any Aspect Ratio and Resolution"](https://arxiv.org/abs/2307.06304).
* `HyenaMixer`, `HyenaTransformerBlock`, and `DepthwiseFFTConv{2,3}d` in `monai.networks.blocks`: subquadratic O(N log N) alternatives to windowed self-attention, backed by the HyenaND operator from the optional `nvsubquadratic` package. `HyenaNDUNETR` (`monai.networks.nets.HyenaNDUNETR`): thin `SwinUNETR` subclass with a `get_variant(name)` classmethod for the three Hyena variants (`HHHH`, `HAHA`, `HHAA`) from the NeurIPS 2026 paper "Native Multi-Dimensional Subquadratic Operators via Input Dependent Long Convolutions" (paper id 26539).
* Add GPU-accelerated Dicom image decoding through the added `NvImgCodecPydicomReader` reader class. Use the `[nvimgcodec]` extra to install CUDA 13 dependencies.

## CI and Building Fixes

MONAI was updated in this release to use the `pyproject.toml` file exclusively for building and package definition, thus
the requirements text files and `setup.*` files have been removed. An added script `monai/config/print_dependencies.py` is
used to recreate the requirements files when needed, see the docstring for this file for use or see how the Dockerfile
uses it for no-build-isolation installation of MONAI. 

With the current version of `pip` the default behaviour is to build MONAI in an isolated environment. This can install
a version of PyTorch that varies from the one in the target environment if already present, this is a problem when compiling
the CUDA extensions as these versions may not be ABI-compatible. An installation with the pip flag `--no-build-isolation`
is needed in this case, for example in the Dockerfile:

```dockerfile
RUN python monai/config/print_dependencies.py build-system | xargs -d '\n' pip install --no-cache-dir --no-build-isolation \
  && FORCE_CUDA=1 pip install --no-cache-dir --no-build-isolation -e .[all,testing]
```

This uses the dependencies script to install the build dependencies then builds MONAI. If build issues are encountered 
you can use this two-step method to setup your environment correctly then install without isolation.
