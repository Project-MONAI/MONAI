# AGENTS.md

Notes for AI coding assistants working in this repository. Human contributors should read
[CONTRIBUTING.md](CONTRIBUTING.md) — this file only adds the machine-facing details and links back
to it rather than repeating it.

## Project overview

MONAI is an open-source framework for deep learning in medical imaging, built on PyTorch. It ships
transforms, networks, losses, metrics, data loading, engines and bundle workflows for 2D and 3D
medical images. Supported Python and PyTorch versions are declared in
[pyproject.toml](pyproject.toml) and [setup.py](setup.py); read them there instead of copying
version numbers into prose.

## Repository map

| Path | Purpose |
|------|---------|
| `monai/transforms` | array, dictionary (`*d`) and random (`Rand*`) transforms, lazy resampling, invertible transforms |
| `monai/networks` | `nets`, `blocks`, `layers`, `schedulers` |
| `monai/data` | datasets, image readers, `MetaTensor` |
| `monai/losses` | loss functions |
| `monai/metrics` | metrics and the confusion-matrix machinery |
| `monai/bundle` | YAML workflow configuration and the scripts that run it |
| `monai/apps` | applications built on the frameworks (pathology, radiology, ...) |
| `monai/auto3dseg` | automated 3D segmentation |
| `monai/engines` | engine base classes |
| `monai/handlers` | handler base classes and the concrete handlers |
| `monai/fl` | federated learning |
| `monai/inferers` | inference helpers |
| `monai/optimizers` | optimizers and learning-rate schedulers |
| `monai/visualize` | visualization helpers |
| `monai/utils` | shared helpers, `optional_import`, deprecation decorators |
| `monai/csrc` | optional C++/CUDA extensions |
| `monai/config` | the `python -m monai.config` command line entry point |
| `monai/_extensions` | optional compiled extensions loaded at runtime |
| `tests` | unit tests, mirroring the `monai` tree |
| `docs` | Sphinx sources, `docs/source/*.rst` |

## Where the detail lives

| File | Covers |
|------|--------|
| [monai/transforms/AGENTS.md](monai/transforms/AGENTS.md) | transform families, `MetaTensor`, invertibility, random state, lazy resampling |
| [tests/AGENTS.md](tests/AGENTS.md) | test layout, `min_tests.py`, skip decorators, running a single test |

## Setup and commands

```bash
./runtests.sh --setup                 # install the pre-commit hooks (black, isort, ruff, DCO sign-off)
./runtests.sh --autofix               # format the working tree (isort + black)
./runtests.sh --codeformat            # style and static analysis checks
./runtests.sh --quick --unittests     # fast local verification
./runtests.sh -f -u --net --coverage  # what CI runs before a pull request
python -m tests.losses.test_dice_loss  # run a single test module
```

`./runtests.sh -h` lists every option.

## Repo-wide conventions

- Pull requests target the `dev` branch, and every commit carries a `Signed-off-by` line
  (`git commit -s`). See [CONTRIBUTING.md](CONTRIBUTING.md#submitting-pull-requests).
- Line length is 120; `black`, `isort` and `ruff` all read it from
  [pyproject.toml](pyproject.toml).
- New source files start with the Apache 2.0 header quoted in
  [CONTRIBUTING.md](CONTRIBUTING.md#licensing-information).
- Optional dependencies are only ever reached through `monai.utils.optional_import`
  (`monai/utils/module.py`); never import an optional package at module scope.
- A new public symbol is added to `__all__` in the module that defines it, re-exported from the
  package `__init__.py`, and listed in the matching `docs/source/<area>.rst`.
- Deprecations go through `monai.utils.deprecated` and `monai.utils.deprecated_arg`
  (`monai/utils/deprecate_utils.py`) with `since` and `removed` versions.
- Tests mirror the source tree: `monai/transforms/spatial/array.py` is covered by
  `tests/transforms/test_*.py`.
- Use American English spelling, and f-strings rather than `%` formatting. See
  [CONTRIBUTING.md](CONTRIBUTING.md#utility-functions).
- Prefer the shared helpers in `monai/utils` and `monai/networks/utils.py` over re-implementing
  them.

## AI-assisted contributions

MONAI accepts tool-assisted contributions under the conditions set out in
[CONTRIBUTING.md](CONTRIBUTING.md#tool-assisted-contribution-policy): the submitter reviews the
content, takes responsibility for it, signs off the work, and labels AI-generated content. The
policy is not repeated here.

## Do not

- Edit generated files: `monai/_version.py` and the versioneer output.
- Commit data, weights or binaries. Test data is downloaded at test time, see
  [CONTRIBUTING.md](CONTRIBUTING.md#testing-data).
- Change CI workflows unless the task is about them.
- Add personal tooling configuration, machine-specific paths or secrets to the repository.
