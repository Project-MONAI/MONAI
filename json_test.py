from functools import partial
import numpy as np
import torch
import json


JSON_NUMPY_TYPE = "numpy"
JSON_TORCH_TYPE = "torch"


class SafeEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return {"__type__": JSON_NUMPY_TYPE, "dtype": str(obj.dtype), "data": obj.tolist()}

        if isinstance(obj, torch.Tensor):
            return {
                "__type__": JSON_TORCH_TYPE,
                "dtype": str(obj.dtype).replace("torch.", ""),
                "data": obj.detach().cpu().numpy().tolist(),
            }

        return super().default(obj)


def json_decode_hook(item):
    datatype = item.get("__type__", "")
    if datatype == JSON_NUMPY_TYPE:
        return np.array(item["data"], dtype=np.dtype(item["dtype"]))
    elif datatype == JSON_TORCH_TYPE:
        return torch.tensor(item["data"], dtype=getattr(torch, item["dtype"]))
    else:
        return item


json_dumps = partial(json.dumps, cls=SafeEncoder)
json_loads = partial(json.loads, object_hook=json_decode_hook)

item = {
    "np": np.random.rand(5, 5),
    "pt": torch.rand(4, 4),
    "int": 123,
    "list": [1, 2, 3],
    "tuple": (1, 2, 3),
    "dict": {"one": 1, "two": 2},
}

print(item)
print()

dump_item = json_dumps(item)

loaded_item = json_loads(dump_item)

print(loaded_item)
