from inspect import getmembers, isclass

from monai import transforms
from monai.transforms import MapTransform, Transform
from monai.transforms.transform import NumpyTransform, TorchOrNumpyTransform, TorchTransform


class Colours:
    red = "91"
    green = "92"
    yellow = "93"
    light_purple = "94"
    purple = "95"
    cyan = "96"
    light_gray = "97"
    black = "98"


def print_colour(t, colour):
    print(f"\033[{colour}m{t}\033[00m")


tr_total = 0
tr_t_or_np = 0
tr_t = 0
tr_np = 0
tr_todo = 0
tr_uncategorised = 0
for n, obj in getmembers(transforms):
    if isclass(obj) and issubclass(obj, Transform) and not issubclass(obj, MapTransform):
        if n in [
            "Transform",
            "InvertibleTransform",
            "Lambda",
            "LoadImage",
            "Compose",
            "RandomizableTransform",
            "ToPIL",
            "ToCupy",
        ]:
            continue
        tr_total += 1
        if issubclass(obj, TorchOrNumpyTransform):
            tr_t_or_np += 1
            print_colour(f"TorchOrNumpy:  {n}", Colours.green)
        elif issubclass(obj, TorchTransform):
            tr_t += 1
            print_colour(f"Torch:         {n}", Colours.green)
        elif issubclass(obj, NumpyTransform):
            tr_np += 1
            print_colour(f"Numpy:         {n}", Colours.yellow)
        else:
            tr_uncategorised += 1
            print_colour(f"Uncategorised: {n}", Colours.red)
print("Total number of transforms:", tr_total)
print_colour(f"Number of TorchOrNumpyTransform: {tr_t_or_np}", Colours.green)
print_colour(f"Number of TorchTransform: {tr_t}", Colours.green)
print_colour(f"Number of NumpyTransform: {tr_np}", Colours.yellow)
print_colour(f"Number of uncategorised: {tr_uncategorised}", Colours.red)
