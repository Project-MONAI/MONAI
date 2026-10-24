# Tests

## Layout

Tests mirror the source tree: `tests/transforms`, `tests/data`,
`tests/losses`, and so on, plus cross-cutting modules at this level.
Unit tests are named `test_[module_name].py`; files ending in `_dist.py`
need a distributed, GPU-based environment; integration tests are named
`test_integration_[workflow_name].py`. Shared helpers live in
`tests/test_utils.py` and the `tests/utils` package.

## Things that are easy to get wrong

- Tests must run in the minimal environment too. Build new test modules from
  the standard library, `torch`, `numpy`, `coverage`, and `parameterized`
  only. If a test module needs another package, add its file name to
  `exclude_cases` in `tests/min_tests.py`, or the minimal CI runner will
  try to execute it and fail on import.
- Slow tests are skipped in quick runs with the `skip_if_quick` decorator
  from `tests/test_utils.py`. Mark genuinely slow tests instead of shrinking
  them until they stop testing anything.
- Do not commit test images or other binary data. Small inputs are generated
  in code; real data is fetched through the `tests.utils` helpers from the
  MONAI-extra-test-data releases (see `CONTRIBUTING.md`, "Testing data").
- Prefer `parameterized.expand` case tables over copy-pasted test methods,
  and assert on values, not only on output shapes: shape-only tests keep
  passing when the numbers go wrong.
- Deterministic tests only: seed random transforms (`set_random_state`) and
  torch generators explicitly, or the test will flake under a different
  worker count.
