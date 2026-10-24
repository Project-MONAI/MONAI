# Bundle and the configuration system

## How configs work

`ConfigParser` (`config_parser.py`) reads JSON and YAML configs. Inside a
config:

- `@name` references another config item (nested keys are joined with `::`,
  as in `@preprocessing::transforms::keys`).
- `$...` is a Python expression, evaluated when the config is parsed.
- A dict with `_target_` is instantiated as that class or function.
  `_disabled_` skips instantiation, `_requires_` lists references that must
  be resolved first, `_mode_` changes how the callable is invoked. These
  keys are defined in `config_item.py`; do not add new underscore keys
  without updating `ConfigComponent` and `docs/source/config_syntax.md`.

## Things that are easy to get wrong

- Reference resolution order matters. `ReferenceResolver` resolves items on
  demand and caches them; changing resolution code can quietly alter every
  bundle in the model zoo, so run the bundle tests even for small changes.
- Configs are user-facing files, often edited by hand. Error messages should
  name the config ID that failed; a bare `KeyError` from deep in the parser
  is a bug report waiting to happen.
- Command line entry points live in `scripts.py` and run through
  `python -m monai.bundle` (`run`, `download`, `ckpt_export`, and others).
  Their arguments map onto config overrides; keep the CLI and the config
  syntax in sync.
- Bundle metadata follows the specification in
  `docs/source/mb_specification.rst` and `mb_properties.rst`; `properties.py`
  holds the key definitions.
