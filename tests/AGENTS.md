# AGENTS.md — `tests`

Notes for AI coding assistants changing the MONAI test suite. The repository-wide rules live in the
root [AGENTS.md](../AGENTS.md); human contributors should read
[CONTRIBUTING.md](../CONTRIBUTING.md).

## Layout

`tests/` mirrors the `monai` package: `tests/transforms`, `tests/networks`, `tests/data`,
`tests/losses`, `tests/metrics`, `tests/bundle`, `tests/apps`, `tests/auto3dseg`,
`tests/engines`, `tests/handlers`, `tests/inferers`, `tests/optimizers`, `tests/utils`,
`tests/visualize`, plus `tests/integration` for the end-to-end workflows.

File names follow the convention in [CONTRIBUTING.md](../CONTRIBUTING.md#unit-testing):

- `test_[module_name].py` — a unit test module.
- `test_[module_name]_dist.py` — a unit test module that needs a distributed environment.
- `test_integration_[workflow_name].py` — an end-to-end test.

## Running tests

```bash
python -m tests.losses.test_dice_loss   # a single test module
./runtests.sh --quick --unittests       # fast local verification
./runtests.sh -f -u --net --coverage    # recommended before opening a pull request
QUICKTEST=true python -m tests.min_tests  # the minimal suite, without optional dependencies
```

`./runtests.sh -h` lists every option.

## Traps

**Optional dependencies decide how a test is written.** Reach an optional package through
`monai.utils.optional_import` and skip the test when it is missing, either with the
`SkipIfNoModule` / `SkipIfModule` decorators in `tests/test_utils.py` or with the
`skipUnless(has_matplotlib, ...)` pattern. When a whole module needs an optional package, add its
name to `exclude_cases` in `tests/min_tests.py` so the minimal runner skips it; that list is
checked for duplicates, so add the name once.

**Test data is not in the repository.** The URLs live in
`tests/testing_data/data_config.json` and are read with `testing_data_config`, and the download
itself goes through `download_url_or_skip_test` — both in `tests/test_utils.py`. The `--net` runs
exercise the download path. Never add images, weights or other binaries to the repository, and do
not edit `tests/testing_data` to make a test pass.

**Import the helpers from `tests.test_utils`.** The shared helpers — `skip_if_quick`,
`skip_if_downloading_fails`, `skip_if_no_cpp_extension`, `skip_if_no_cuda`, `skip_if_windows`,
`skip_if_darwin`, `assert_allclose`, `dict_product` — live in `tests/test_utils.py`; the
`tests/utils` package holds the tests of `monai.utils` itself.

**`skip_if_quick` is the marker of a slow test.** A test that takes more than a couple of seconds
is expected to carry it, otherwise `--quick` and the minimal runner pay for it.

**A new test module should stay small in its dependencies.** The recommended set is the Python
standard library, `torch`, `numpy`, `coverage` and `parameterized`; anything else has to be
declared in `pyproject.toml` and, when it is optional, excluded from the minimal run as described
above.
