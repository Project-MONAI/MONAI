# MONAI Bundle Configuration

The `monai.bundle` module supports building Python-based workflows via structured configurations.

The main benefits are threefold:

- it provides good readability and usability by separating system parameter settings from the Python code.
- it describes workflow at a relatively high level and allows for different low-level implementations.
- learning paradigms at a higher level such as federated learning and AutoML can be decoupled from the component details.

Content:

- [A basic example](#a-basic-example)
- [Syntax examples explained](#syntax-examples-explained)
  - [`@` to interpolate with Python objects](#to-interpolate-with-python-objects)
  - [`$` to evaluate as Python expressions](#to-evaluate-as-python-expressions)
  - [`%` to textually replace configuration elements](#to-textually-replace-configuration-elements)
  - [`_target_` (`_disabled_` and `_requires_`) to instantiate a Python object](#instantiate-a-python-object)
- [The command line interface](#the-command-line-interface)
- [Recommendations](#recommendations)

## A basic example

Components as part of a workflow can be specified using `JSON` or `YAML` syntax, for example, a network architecture
definition could be stored in a `demo_config.json` file with the following content:

```json
{
  "demo_net": {
    "_target_": "monai.networks.nets.BasicUNet",
    "spatial_dims": 3,
    "in_channels": 1,
    "out_channels": 2,
    "features": [16, 16, 32, 32, 64, 64]
  }
}
```

or alternatively, in `YAML` format (`demo_config.yaml`):

```yaml
demo_net:
  _target_: monai.networks.nets.BasicUNet
  spatial_dims: 3
  in_channels: 1
  out_channels: 2
  features: [16, 16, 32, 32, 64, 64]
```

The configuration parser can instantiate the component as a Python object:

```py
>>> from monai.bundle import ConfigParser
>>> config = ConfigParser()
>>> config.read_config("demo_config.json")
>>> net = config.get_parsed_content("demo_net", instantiate=True)
BasicUNet features: (16, 16, 32, 32, 64, 64).
>>> print(type(net))
<class 'monai.networks.nets.basic_unet.BasicUNet'>
```

or additionally, tune the input parameters then instantiate the component:

```py
>>> config["demo_net"]["features"] = [32, 32, 32, 64, 64, 64]
>>> net = config.get_parsed_content("demo_net", instantiate=True)
BasicUNet features: (32, 32, 32, 64, 64, 64).
```

For more details on the `ConfigParser` API, please see https://docs.monai.io/en/latest/bundle.html#config-parser.

## Syntax examples explained

A few characters and keywords are interpreted beyond the plain texts, here are examples of the syntax:

### To interpolate with Python objects

```json
"@preprocessing#transforms#keys"
```

_Description:_ `@` character indicates a reference to another configuration value defined at `preprocessing#transforms#keys`.
where `#` indicates a sub-structure of this configuration file.

```json
"@preprocessing#1"
```

_Description:_ `1` is interpreted as an integer, which is used to index (zero-based indexing) the `preprocessing` sub-structure.

### To evaluate as Python expressions

```json
"$print(42)"
```

_Description:_ `$` is a special character to indicate evaluating `print(42)` at runtime.

```json
"$[i for i in @datalist]"
```

_Description:_ Create a list at runtime using the values in `datalist` as input.

```json
"$from torchvision.models import resnet18"
```

_Description:_ `$` followed by an import statement is handled slightly differently from the
Python expressions. The imported module `resnet18` will be available as a global variable
to the other configuration sections. This is to simplify the use of external modules in the configuration.

### To textually replace configuration elements

```json
"%demo_config.json#demo_net#in_channels"
```

_Description:_ `%` character indicates a macro to replace the current configuration element with the texts at `demo_net#in_channels` in the
`demo_config.json` file. The replacement is done before instantiating or evaluating the components.

### Instantiate a Python object

```json
{
  "demo_name":{
    "_target_": "my.python.module.Class",
    "args1": "string",
    "args2": 42}
}
```

_Description:_ This dictionary defines an object with a reference name `demo_name`, with an instantiable type
specified at `_target_` and with input arguments `args1` and `args2`.
This dictionary will be instantiated as a Pytorch object at runtime.

`_target_` is a required key by monai bundle syntax for the Python object name.
`args1` and `args2` should be compatible with the Python object to instantiate.

```json
{
  "component_name": {
    "_target_": "my.module.Class",
    "_requires_": "@cudnn_opt",
    "_disabled_": "true"}
}
```

_Description:_ `_requires_` and `_disabled_` are optional keys.
`_requires_` specifies references (string starts with `@`) or
Python expression that will be evaluated/instantiated before `_target_` object is instantiated.
It is useful when the component does not explicitly depend on the other ConfigItems via
its arguments, but requires the dependencies to be instantiated/evaluated beforehand.
`_disabled_` specifies a flag to indicate whether to skip the instantiation.

## The command line interface

In addition to the Pythonic APIs, a few command line interfaces (CLI) are provided to interact with the bundle.
The primary usage is:
```bash
python -m monai.bundle COMMANDS
```

where `COMMANDS` is one of the following: `run`, `verify_metadata`, `ckpt_export`, ...
(please see `python -m monai.bundle --help` for a list of available options).

The CLI supports flexible use cases, such as overriding configs at runtime and predefining arguments in a file.
To display a usage page for a command, for example `run`:
```bash
python -m monai.bundle run -- --help
```

The support is provided by [Python Fire](https://github.com/google/python-fire), please
make sure the optional dependency is installed, for example,
using `pip install monai[fire]` or `pip install fire`.
Details on the CLI argument parsing is provided in the
[Python Fire Guide](https://github.com/google/python-fire/blob/master/docs/guide.md#argument-parsing).

## Recommendations
- Both `YAML` and `JSON` are supported, but the advanced features of these formats are not supported.
- Using meaningful names for the configuration elements can improve the readability.
- While it is possible to build complex configurations with the bundle syntax,
  simple structures with sparse uses of expressions or references are preferred.
- For `$import <module>` in the configuration, please make sure there are instructions for the users to install
  the `<module>` if it is not a (optional) dependency of MONAI.
- As "#" and "$" might be interpreted differently by the `shell` or `CLI` tools, may need to add escape characters
  or quotes for them in the command line, like: `"\$torch.device('cuda:1')"`, `"'train_part#trainer'"`.
