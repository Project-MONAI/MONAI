# AGENTS.md

Guidance for AI coding assistants (and the humans reviewing their work) in this
repository. The file closest to the code being edited takes precedence: start
here, then read the `AGENTS.md` in the subpackage you are changing.

## Project overview

MONAI (Medical Open Network for AI) is a PyTorch-based, open-source framework
for deep learning in healthcare imaging, part of the PyTorch Ecosystem. It
provides domain-optimized data transforms, networks, losses, metrics, training
engines, and a bundle configuration system, with research and clinical users.
Supported Python and PyTorch versions live in `pyproject.toml`; do not copy
version numbers into other files, link there instead.

## Repository map

| Path | Purpose |
|------|---------|
| `monai/transforms` | array and dictionary transforms, lazy resampling, invertibility |
| `monai/networks` | networks, building blocks, layers, schedulers |
| `monai/data` | datasets, data loaders, image readers and writers, `MetaTensor` |
| `monai/losses` | loss functions |
| `monai/metrics` | evaluation metrics, mostly cumulative across batches |
| `monai/engines` | training and evaluation workflows, built on PyTorch Ignite |
| `monai/handlers` | Ignite handlers: checkpointing, logging, metrics output |
| `monai/inferers` | sliding window and other inference strategies |
| `monai/bundle` | JSON/YAML configuration system and bundle scripts |
| `monai/apps` | application packages: detection, DeepEdit, pathology, and more |
| `monai/auto3dseg` | Auto3DSeg algorithms; the runner lives in `monai/apps/auto3dseg` |
| `monai/fl` | federated learning client and utilities |
| `monai/visualize` | visualization and interpretability tools |
| `monai/optimizers` | optimizers and learning rate utilities |
| `monai/csrc` | optional C++/CUDA extensions |
| `monai/utils` | shared utilities; check here before writing a new helper |
| `tests` | unit and integration tests, mirroring the `monai` tree |
| `docs` | Sphinx documentation sources |

## Where the detail lives

| File | Covers |
|------|--------|
| [`monai/transforms/AGENTS.md`](monai/transforms/AGENTS.md) | array/dict transform pairs, randomness, lazy and inverse transforms |
| [`monai/networks/AGENTS.md`](monai/networks/AGENTS.md) | network, block, and layer conventions |
| [`monai/data/AGENTS.md`](monai/data/AGENTS.md) | datasets, caching, readers, test data |
| [`monai/losses/AGENTS.md`](monai/losses/AGENTS.md) | loss conventions, shared with metrics |
| [`monai/metrics/AGENTS.md`](monai/metrics/AGENTS.md) | cumulative metrics lifecycle |
| [`monai/bundle/AGENTS.md`](monai/bundle/AGENTS.md) | config syntax, references, bundle scripts |
| [`monai/apps/AGENTS.md`](monai/apps/AGENTS.md) | application packages and downloaded data |
| [`monai/auto3dseg/AGENTS.md`](monai/auto3dseg/AGENTS.md) | Auto3DSeg pipeline and generated bundles |
| [`monai/engines/AGENTS.md`](monai/engines/AGENTS.md) | engines and handlers |
| [`monai/csrc/AGENTS.md`](monai/csrc/AGENTS.md) | C++/CUDA extension build |
| [`tests/AGENTS.md`](tests/AGENTS.md) | test layout, minimal tests, skip helpers |
| [`docs/AGENTS.md`](docs/AGENTS.md) | documentation sources and API lists |

## Setup and commands

Verified in a fresh checkout of `dev` (Python 3.12, CPU-only PyTorch):

```bash
./runtests.sh --help                        # list every option of the test runner
./runtests.sh --setup                       # install git pre-commit hooks (black, isort, ruff, DCO check)
./runtests.sh --autofix                     # auto-format code with isort, black, and Ruff
./runtests.sh --ruff                        # ruff checks only, a fast pre-push check
python -m pytest tests/utils/test_look_up_option.py -q   # run a single test module
```

For the full pre-PR test suite, see the "Unit testing" section of
`CONTRIBUTING.md`. Do not add a command to the list above unless you have
actually run it in this repository.

## Repo-wide conventions

- Pull requests target the `dev` branch. Name branches `[ticket_id]-[task_name]`,
  for example `9199-introduce-agents-md`, based on the latest `dev`.
- Every commit needs a DCO sign-off: `git commit -s`. The sign-off identity is
  the human author who takes responsibility for the change.
- Line length is 120. Docstrings are Google style. Public APIs carry type hints.
- American English spelling (`normalize`, `visualize`), as in PyTorch.
- New source files start with the Apache 2.0 license header used across the
  repo (copy it from a neighboring file).
- Optional dependencies are only reached through `monai.utils.optional_import`;
  never hard-import an optional package at module top level.
- A new public API is not finished until it is exported (the module's `__all__`
  and the package `__init__.py`) and added to the matching page in
  `docs/source/*.rst`.
- Deprecations go through `monai.utils.deprecated` / `deprecated_arg`, with
  `since` and `removed` versions, following the process in `CONTRIBUTING.md`.
- Tests mirror the source tree under `tests/`; a change without a test is
  treated as broken.
- Before writing a utility, search `monai/utils` and the subpackage's own
  `utils.py`; MONAI already has many shared helpers.

## AI-assisted contributions

MONAI accepts tool-assisted contributions under the Tool-Assisted Contribution
Policy in `CONTRIBUTING.md`: the submitting author must review and understand
everything submitted, sign the DCO for it, and label AI-generated content in
issues and pull requests. These `AGENTS.md` files exist so that assisted work
arrives closer to mergeable. When a change alters a convention written down in
one of these files, update that file in the same pull request.
