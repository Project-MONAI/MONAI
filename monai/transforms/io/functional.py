import json

import numpy as np

import torch

from monai.data.meta_tensor import MetaTensor

from monai.utils.enums import KindKeys


def load_geometry(file, image, origin):
    """
    Load geometry from a file and optionally map it to another coordinate space.
    """

    geometry = json.load(file)
    geometry_schema = geometry.get("schema", None)
    if geometry_schema is None:
        raise ValueError("Geometry import issue: missing 'schema' entry")
    elif "geometry" not in geometry_schema:
        raise ValueError(f"Geometry import issue: 'schema' entry must contain 'geometry' key, got: {geometry_schema}")

    if "points" not in geometry:
        raise ValueError("Geometry import issue: missing 'points' entry")

    points = geometry["points"]
    if not isinstance(points, list):
        raise ValueError(f"Geometry import issue: 'points' entry must be a list, got: {type(points)}")

    if len(points) > 0:
        first_len = None
        for p in points:
            if first_len is None:
                first_len = len(p)
            if len(p) != first_len:
                raise ValueError(f"Geometry import issue: 'points' entry contains inconsistent point lengths")

    points = np.asarray(points)
    points = np.concatenate((points, np.ones((points.shape[0], 1))), axis=1)
    points = torch.as_tensor(points, dtype=torch.float32)
    points = MetaTensor(points)
    points.kind = KindKeys.POINT

    return points
