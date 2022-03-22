# MONAI Bundle Configuration

The `monai.bundle` module supports building Python-based workflows via structured configurations.

The main benefits are threefold:
  - it provides good readability and usability by separating system parameter settings from the Python code.
  - it describes workflow at a relatively high level and allows for different low-level implementations.
  - learning paradigms at a higher level such as federated learning and AutoML can be decoupled from the component details.

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

The configuration parser can instantiate the components as a Python object:
```py
>>> from monai.bundle import ConfigParser
>>> config = ConfigParser()
>>> config.read_config("demo_config.json")
>>> net = config.get_parsed_content("demo_net", instantiate=True)
BasicUNet features: (16, 16, 32, 32, 64, 64).
>>> print(type(net))
<class 'monai.networks.nets.basic_unet.BasicUNet'>
```
or additionally tune the input parameters then :
```py
>>> config["demo_net"]["features"] = [32, 32, 32, 64, 64, 64]
>>> net = config.get_parsed_content("demo_net", instantiate=True)
BasicUNet features: (32, 32, 32, 64, 64, 64).
```
