# Networks

## Layout

- `nets/` holds full architectures (`basic_unet.py`, `attentionunet.py`, and
  many more), one file per architecture family.
- `blocks/` holds reusable building blocks, `layers/` holds single layers and
  the layer factories, `schedulers/` holds diffusion noise schedulers, and
  `utils.py` holds helpers shared by this package only.

## Things that are easy to get wrong

- Most networks are parameterized by `spatial_dims` (1, 2, or 3). Do not write
  2D-only logic into a shared block; use the dimension-agnostic helpers in
  `layers/` and `monai.networks.utils`.
- Activations, normalizations, and dropouts are commonly passed as names or
  tuples and resolved through the factories in `monai/networks/layers/`. Keep
  that pattern instead of hard-coding a module class, so configs and bundles
  can still build the network.
- Channel arguments have an established order in MONAI signatures:
  `spatial_dims`, `in_channels`, `out_channels` first. Follow the neighboring
  networks when adding a new one.
- A new network is public API: export it in `nets/__init__.py` and in
  `monai/networks/__init__.py`, add it to `docs/source/networks.rst`, and add
  a test under `tests/networks/` that builds it in each supported
  `spatial_dims` and runs one forward pass on CPU-sized inputs.
