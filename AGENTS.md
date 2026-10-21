# AGENTS.md

Guidance for AI coding assistants (Claude Code, Codex, Copilot agent mode, Cursor, Gemini CLI, Jules, Aider, ...)
working in this repository. See [CONTRIBUTING.md](CONTRIBUTING.md) for the full human contributor guide and
[the AI/tool-assisted contribution policy](CONTRIBUTING.md) for what is and isn't acceptable when using these
tools — this file is about *how the codebase works*, not *whether AI assistance is allowed*.

## Project overview

MONAI is a PyTorch-based, open-source framework for deep learning in healthcare imaging. See
[setup.cfg](setup.cfg) / [pyproject.toml](pyproject.toml) for the supported Python and PyTorch versions.

## Repository map

| Path | Purpose |
|------|---------|
| `monai/` | the library itself: transforms, networks, losses, metrics, data, bundle, apps, engines, etc. |
| `tests/` | unit and integration tests, mirroring `monai/`'s layout |
| `docs/` | Sphinx documentation sources (`docs/source/*.rst`, `*.md`) |
| `runtests.sh` | the test and style-check entry point used in CI and locally |

## Required workflow

- Pull requests target the `dev` branch, not `main`.
- Every commit needs a DCO sign-off (`git commit -s`, i.e. a `Signed-off-by:` trailer). The DCO app fails PRs
  with commits missing it; see [Signing your work](CONTRIBUTING.md#signing-your-work).
- Format and lint with `./runtests.sh --autofix` (runs `isort` and `black`) before committing. If pre-commit
  hooks are installed (`./runtests.sh --setup`), the same checks also run on every `git commit`.

## Code conventions

- Optional third-party dependencies are imported lazily via `monai.utils.optional_import`, not a bare
  `import`, so the package stays usable without every extra installed.
- A new public API needs a corresponding entry in `docs/source/*.rst` (or the matching `.md` file for guides),
  not just a docstring.
- A slow test belongs in `tests/min_tests.py`'s exclusion list, not in the default fast suite.

## What stays out of this file

Package-specific traps (e.g. for `monai/transforms`, `monai/networks`, `monai/bundle`) are intentionally not
covered here — this root file only carries rules that apply repo-wide. Nested, package-level `AGENTS.md` files
can follow in later PRs if the maintainers want that detail; see the discussion on #9199.
