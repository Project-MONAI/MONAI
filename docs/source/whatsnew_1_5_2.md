
# What's new in 1.5.2

This is a minor update for MONAI to address a security concern and adds a new network architecture.

- Security fix to address advisory [GHSA-9rg3-9pvr-6p27](https://github.com/Project-MONAI/MONAI/security/advisories/GHSA-9rg3-9pvr-6p27).

## New Network: NaViT (Native Resolution Vision Transformer)

`NaViT` (`monai.networks.nets.NaViT`) is a Vision Transformer that removes the fixed-resolution constraint of
standard ViT by packing multiple variable-size images into a single sequence per batch element
("Patch n' Pack"), based on [Dehghani et al., 2023](https://arxiv.org/abs/2307.06304).

Key features:

- **Variable-resolution inputs**: images of different spatial sizes can be processed in the same batch without
  resizing or padding to a fixed resolution.
- **Patch n' Pack**: multiple images are concatenated into one sequence per batch element, with a per-image
  attention mask preventing cross-image attention.
- **Factorised positional embeddings**: separate learnable tables for each spatial axis are summed, allowing
  generalisation to resolutions not seen during training.
- **Token dropout**: a configurable fraction of patch tokens can be randomly dropped during training,
  acting as a form of masked-image modelling.
- **Attention pooling**: a learned query vector attends over each image's tokens to produce a fixed-size
  per-image representation, cleanly handling variable numbers of images per packed sequence.
- **QK normalisation**: RMS normalisation on queries and keys for training stability (ViT-22B style).
- **2D and 3D support**: works for standard images `(C, H, W)` and volumetric data `(C, H, W, D)`.

```python
from monai.networks.nets import NaViT
import torch

# 3D single-channel (e.g. CT) classification
net = NaViT(
    image_size=96, patch_size=16, num_classes=2,
    hidden_size=768, mlp_dim=3072, num_layers=12, num_heads=12,
    in_channels=1, spatial_dims=3,
)
# Each inner list is a group of images packed into one sequence
volumes = [
    [torch.randn(1, 96, 96, 96), torch.randn(1, 64, 64, 64)],
    [torch.randn(1, 80, 96, 80)],
]
logits = net(volumes)  # shape: (3, 2)
```
