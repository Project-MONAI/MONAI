import json


def load_geometry(file, image, origin):
    """
    Load geometry from a file and optionally map it to another coordinate space.
    """
    with open(file, "r") as f:
        geometry = json.load(f)
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
