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
                raise ValueError("Geometry import issue: 'points' entry contains inconsistent point lengths")

    points = np.asarray(points)
    points = np.concatenate((points, np.ones((points.shape[0], 1))), axis=1)
    points = torch.as_tensor(points, dtype=torch.float32)
    points = MetaTensor(points)
    points.kind = KindKeys.POINT

    return points

"""
{
            "schema": {
                "geometry": "point"
            },
            "points": [
                [0, 0, 0],
                [0, 1, 0],
                [0, 0, 1],
                [0, 1, 1],
                [1, 0, 0],
                [1, 1, 0],
                [1, 0, 1],
                [1, 1, 1],
            ]
        }
"""

def save_geometry(data, file, image, origin):
    """
    Load geometry from a file and optionally map it to another coordinate space.
    """
    if not isinstance(data, MetaTensor):
        raise ValueError(f"Geometry export issue: data must be a MetaTensor, got: {type(data)}")
    if data.kind != KindKeys.POINT:
        raise ValueError(f"Geometry export issue: geometry must be a point {KindKeys.POINT}")
    geometry = data.detach().cpu().numpy()
    geometry = geometry[:, :-1].tolist()

    schema = {
        "schema": {
            "geometry": "point"
        },
        "points":
            geometry
    }

    geometry = json.dump(schema, file)
    return None
