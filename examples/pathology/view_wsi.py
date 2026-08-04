#!/usr/bin/env python
# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Sample application for a typical WSI workflow with MONAI.

There is no existing end-to-end DICOM/WSI demo in this repository (only library APIs and
tests). This script fills that gap for interactive inspection and tile extraction.

Typical workflow:
  1. Open a slide (SVS/TIFF folder or DICOM WSI folder/files).
  2. Inspect pyramid levels, sizes, and mpp.
  3. Extract a region/tile (or a sliding-window grid) and save PNG images for viewing.

Examples:
  # Inspect a DICOM WSI folder (wsidicom backend)
  python examples/pathology/view_wsi.py info \\
      --path wsi_datasets/PixelMed_20190212/CMU1_DICOMOriginalRGBJPEGWithBigTIFFAndPyramid \\
      --backend wsidicom

  # Extract one tile at level 0 and save PNG
  python examples/pathology/view_wsi.py tile \\
      --path /path/to/slide.svs --backend cucim \\
      --location 1000 2000 --size 256 256 --level 0 \\
      --output ./wsi_tiles

  # Save a thumbnail (full slide at a lower pyramid level)
  python examples/pathology/view_wsi.py thumbnail \\
      --path /path/to/dicom_wsi_folder --backend wsidicom --level 2 \\
      --output ./wsi_tiles

  # Sliding-window tiles over the slide
  python examples/pathology/view_wsi.py grid \\
      --path /path/to/slide.svs --backend openslide \\
      --size 512 512 --level 1 --overlap 0.0 --max-tiles 16 \\
      --output ./wsi_tiles
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np
from _nvimgcodec_probe import NvImgCodecProbe, add_log_level_argument, configure_logging, gpu_memory_report
from PIL import Image

from monai.data import WSIReader
from monai.utils import optional_import


def _channel_last(patch: np.ndarray) -> np.ndarray:
    """Convert CHW RGB/RGBA patch to HWC for PIL."""
    if patch.ndim != 3:
        raise ValueError(f"Expected a 3D patch (C, H, W), got shape {patch.shape}")
    if patch.shape[0] in (1, 3, 4):
        return np.moveaxis(patch, 0, -1)
    if patch.shape[-1] in (1, 3, 4):
        return patch
    raise ValueError(f"Cannot infer channel axis from patch shape {patch.shape}")


def _save_patch(patch: np.ndarray, out_path: Path) -> None:
    arr = _channel_last(np.asarray(patch))
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    Image.fromarray(arr).save(out_path)


def _build_reader(args: argparse.Namespace) -> WSIReader:
    kwargs: dict = {"backend": args.backend, "dtype": np.uint8, "mode": "RGB"}
    if args.backend == "wsidicom":
        # Prefer stable CPU decode unless the user explicitly enables GPU plugin registration.
        kwargs["register_nvimgcodec"] = args.register_nvimgcodec
        kwargs["prefer_pydicom_decoder"] = True
        kwargs["num_threads"] = args.num_threads
    elif args.backend == "cucim":
        kwargs["num_workers"] = args.num_threads
    return WSIReader(**kwargs)


def _open_wsi(reader: WSIReader, path: str):
    wsi = reader.read(path)
    if isinstance(wsi, (list, tuple)):
        if not wsi:
            raise RuntimeError(f"No WSI objects opened from: {path}")
        if len(wsi) > 1:
            print(f"Opened {len(wsi)} WSI objects; using the first one.", file=sys.stderr)
        return wsi[0]
    return wsi


def _available_levels(reader: WSIReader, wsi) -> list[int]:
    """
    Return pyramid indices that are actually present.

    Some DICOM WSI pyramids are sparse: ``highest_level`` can exceed the set of
    stored levels (wsidicom reports the theoretical 1x1 level). Prefer enumerating
    concrete levels when the backend exposes them.
    """
    pyramid = getattr(wsi, "pyramid", None)
    if pyramid is not None and hasattr(pyramid, "levels"):
        indices = []
        for level_obj in pyramid.levels:
            idx = getattr(level_obj, "level", None)
            if idx is not None:
                indices.append(int(idx))
        if indices:
            return sorted(set(indices))

    indices = []
    for level in range(reader.get_level_count(wsi)):
        try:
            reader.get_size(wsi, level)
        except Exception:
            continue
        indices.append(level)
    return indices


def _describe_wsi(reader: WSIReader, wsi, path: str) -> dict:
    level_indices = _available_levels(reader, wsi)
    levels = []
    for level in level_indices:
        height, width = reader.get_size(wsi, level)
        entry = {
            "level": level,
            "height": height,
            "width": width,
            "downsample": reader.get_downsample_ratio(wsi, level),
        }
        try:
            mpp_y, mpp_x = reader.get_mpp(wsi, level)
            entry["mpp"] = {"y": mpp_y, "x": mpp_x}
        except Exception as exc:  # backends may lack mpp metadata
            entry["mpp"] = f"unavailable ({exc})"
        try:
            entry["power"] = reader.get_power(wsi, level)
        except Exception:
            entry["power"] = None
        levels.append(entry)

    return {
        "path": str(path),
        "backend": reader.backend,
        "level_count": len(level_indices),
        "reported_level_count": reader.get_level_count(wsi),
        "levels": levels,
    }


def cmd_info(args: argparse.Namespace) -> int:
    reader = _build_reader(args)
    wsi = _open_wsi(reader, args.path)
    info = _describe_wsi(reader, wsi, args.path)
    print(json.dumps(info, indent=2))
    return 0


def cmd_tile(args: argparse.Namespace) -> int:
    reader = _build_reader(args)
    wsi = _open_wsi(reader, args.path)
    location = (args.location[0], args.location[1])
    size = (args.size[0], args.size[1]) if args.size else None
    patch, meta = reader.get_data(wsi, location=location, size=size, level=args.level, mpp=args.mpp, power=args.power)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / args.name
    _save_patch(patch, out_path)
    print(json.dumps({"saved": str(out_path), "shape": list(patch.shape), "meta": _jsonable(meta)}, indent=2))
    return 0


def cmd_thumbnail(args: argparse.Namespace) -> int:
    reader = _build_reader(args)
    wsi = _open_wsi(reader, args.path)
    # Full-slide read at the requested level (or mpp); size=None uses the full level size.
    patch, meta = reader.get_data(wsi, location=(0, 0), size=None, level=args.level, mpp=args.mpp, power=args.power)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / args.name
    _save_patch(patch, out_path)
    print(json.dumps({"saved": str(out_path), "shape": list(patch.shape), "meta": _jsonable(meta)}, indent=2))
    return 0


def cmd_grid(args: argparse.Namespace) -> int:
    from monai.inferers import WSISlidingWindowSplitter

    reader_kwargs: dict = {}
    if args.level is not None:
        reader_kwargs["level"] = args.level
    if args.mpp is not None:
        reader_kwargs["mpp"] = args.mpp
    if args.power is not None:
        reader_kwargs["power"] = args.power
    if args.backend == "wsidicom":
        reader_kwargs["register_nvimgcodec"] = args.register_nvimgcodec
        reader_kwargs["prefer_pydicom_decoder"] = True
        reader_kwargs["num_threads"] = args.num_threads

    splitter = WSISlidingWindowSplitter(
        patch_size=tuple(args.size), overlap=args.overlap, reader=args.backend, **reader_kwargs
    )
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    saved = []
    for i, (patch, location) in enumerate(splitter(args.path)):
        if args.max_tiles is not None and i >= args.max_tiles:
            break
        arr = np.asarray(patch)
        if arr.ndim == 4:  # BCHW from splitter
            arr = arr[0]
        name = f"tile_{i:04d}_y{location[0]}_x{location[1]}.png"
        out_path = out_dir / name
        _save_patch(arr, out_path)
        saved.append({"index": i, "location": list(location), "path": str(out_path)})

    print(json.dumps({"n_tiles": len(saved), "tiles": saved}, indent=2))
    return 0


def _benchmark_locations(
    reader: WSIReader, wsi, level: int, size: tuple[int, int], count: int
) -> list[tuple[int, int]]:
    """Generate raster-scan locations in the level-0 coordinate frame."""
    base_h, base_w = reader.get_size(wsi, 0)
    downsample = reader.get_downsample_ratio(wsi, level)
    footprint_h = max(1, int(round(size[0] * downsample)))
    footprint_w = max(1, int(round(size[1] * downsample)))
    locations = []
    for y in range(0, max(1, base_h - footprint_h + 1), footprint_h):
        for x in range(0, max(1, base_w - footprint_w + 1), footprint_w):
            locations.append((y, x))
            if len(locations) == count:
                return locations
    return locations or [(0, 0)]


def _wsidicom_decoder_name(wsi) -> str | None:
    """Report which wsidicom decoder class handled the slide's tiles."""
    try:
        for level in wsi.pyramid.levels:
            instances = level.instances
            # ``instances`` maps instance UID to instance.
            for instance in instances.values() if hasattr(instances, "values") else instances:
                decoder = getattr(getattr(instance, "image_data", None), "decoder", None)
                if decoder is not None:
                    return type(decoder).__name__
    except Exception:
        return None
    return None


def _set_wsidicom_frame_cache(size_bytes: int) -> None:
    """
    Resize wsidicom's frame caches.

    wsidicom caches decoded frames (100 MB by default), so re-reading a region returns
    cached pixels without decoding. Setting this to 0 makes every read decode, which is
    required to compare decoder throughput.
    """
    wsidicom_config, has_wsidicom = optional_import("wsidicom", name="config")
    if not has_wsidicom:
        return
    for name in ("decoded_frame_cache_size", "encoded_frame_cache_size"):
        if hasattr(type(wsidicom_config.settings), name):
            setattr(wsidicom_config.settings, name, size_bytes)


def cmd_benchmark(args: argparse.Namespace) -> int:
    """Benchmark opening a slide and decoding a reproducible raster of tiles."""
    if args.backend == "wsidicom" and args.frame_cache_bytes is not None:
        # Must happen before the slide is opened; caches are created per image.
        _set_wsidicom_frame_cache(args.frame_cache_bytes)

    open_started = time.perf_counter()
    reader = _build_reader(args)
    wsi = _open_wsi(reader, args.path)
    open_seconds = time.perf_counter() - open_started

    level = 0 if args.level is None else args.level
    size = (args.size[0], args.size[1])
    locations = _benchmark_locations(reader, wsi, level, size, max(args.iterations + args.warmup, 1))

    def read_tile(location: tuple[int, int]) -> np.ndarray:
        patch, _ = reader.get_data(wsi, location=location, size=size, level=level)
        return np.asarray(patch)

    for i in range(args.warmup):
        read_tile(locations[i % len(locations)])

    # Instrument after warm-up so counts reflect only measured reads.
    probe = NvImgCodecProbe().start() if args.backend == "wsidicom" else None

    durations = []
    checksum = 0
    first_patch = None
    for i in range(args.iterations):
        started = time.perf_counter()
        patch = read_tile(locations[(args.warmup + i) % len(locations)])
        durations.append(time.perf_counter() - started)
        checksum = (checksum + int(patch.sum(dtype=np.uint64))) % (2**32)
        if first_patch is None:
            first_patch = patch

    if args.output and first_patch is not None:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        _save_patch(first_patch, out_path)

    total_seconds = sum(durations)
    sorted_durations = sorted(durations)
    p95_index = max(0, int(np.ceil(0.95 * len(sorted_durations))) - 1)
    megapixels = args.iterations * size[0] * size[1] / 1_000_000
    result = {
        "path": args.path,
        "backend": args.backend,
        "nvimgcodec_requested": args.backend == "wsidicom" and args.register_nvimgcodec,
        "num_threads": args.num_threads,
        "level": level,
        "tile_size": list(size),
        "warmup": args.warmup,
        "iterations": args.iterations,
        "unique_locations": len(locations),
        "frame_cache_bytes": args.frame_cache_bytes,
        "open_seconds": open_seconds,
        "decode_total_seconds": total_seconds,
        "tile_seconds": {
            "mean": statistics.fmean(durations),
            "median": statistics.median(durations),
            "p95": sorted_durations[p95_index],
            "min": min(durations),
            "max": max(durations),
        },
        "tiles_per_second": args.iterations / total_seconds,
        "megapixels_per_second": megapixels / total_seconds,
        "checksum_uint32": checksum,
    }
    if probe is not None:
        probe.stop()
        result["wsidicom_decoder"] = _wsidicom_decoder_name(wsi)
        result["nvimgcodec"] = probe.report()
        if probe.successes:
            result["gpu_memory"] = gpu_memory_report()
    if args.output:
        result["first_tile"] = args.output
    print(json.dumps(result, indent=2))
    return 0


def _jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, Path):
        return str(obj)
    return obj


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--path", required=True, help="Path to a WSI file, or a DICOM WSI folder / file list for --backend wsidicom."
    )
    parser.add_argument(
        "--backend",
        default="wsidicom",
        choices=["wsidicom", "cucim", "openslide", "tifffile"],
        help="WSI reader backend (default: wsidicom).",
    )
    parser.add_argument("--num-threads", type=int, default=1, help="Reader worker/thread count where supported.")
    parser.add_argument(
        "--register-nvimgcodec",
        action="store_true",
        help="For wsidicom: register nvImageCodec GPU decoder plugin (optional).",
    )
    add_log_level_argument(parser)


def _add_resolution_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--level", type=int, default=None, help="Pyramid level (mutually exclusive with --mpp/--power)."
    )
    parser.add_argument("--mpp", type=float, default=None, help="Target microns-per-pixel resolution.")
    parser.add_argument("--power", type=int, default=None, help="Target objective power (not available for DICOM WSI).")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="MONAI sample app: inspect WSI pyramids and extract tiles for viewing.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_info = sub.add_parser("info", help="Print pyramid levels, sizes, and mpp.")
    _add_common_args(p_info)
    p_info.set_defaults(func=cmd_info)

    p_tile = sub.add_parser("tile", help="Extract one region/tile and save as PNG.")
    _add_common_args(p_tile)
    _add_resolution_args(p_tile)
    p_tile.add_argument(
        "--location", type=int, nargs=2, default=[0, 0], metavar=("Y", "X"), help="Top-left in level-0 frame."
    )
    p_tile.add_argument(
        "--size", type=int, nargs=2, default=[256, 256], metavar=("H", "W"), help="Patch size at the chosen level."
    )
    p_tile.add_argument("--output", default="./wsi_tiles", help="Output directory.")
    p_tile.add_argument("--name", default="tile.png", help="Output filename.")
    p_tile.set_defaults(func=cmd_tile)

    p_thumb = sub.add_parser("thumbnail", help="Save a full-slide thumbnail at a given level/mpp.")
    _add_common_args(p_thumb)
    _add_resolution_args(p_thumb)
    p_thumb.add_argument("--output", default="./wsi_tiles", help="Output directory.")
    p_thumb.add_argument("--name", default="thumbnail.png", help="Output filename.")
    p_thumb.set_defaults(func=cmd_thumbnail)

    p_grid = sub.add_parser("grid", help="Extract a sliding-window grid of tiles.")
    _add_common_args(p_grid)
    _add_resolution_args(p_grid)
    p_grid.add_argument("--size", type=int, nargs=2, default=[256, 256], metavar=("H", "W"), help="Tile size.")
    p_grid.add_argument("--overlap", type=float, default=0.0, help="Fractional overlap between tiles.")
    p_grid.add_argument("--max-tiles", type=int, default=16, help="Stop after this many tiles (None for all).")
    p_grid.add_argument("--output", default="./wsi_tiles", help="Output directory.")
    p_grid.set_defaults(func=cmd_grid)

    p_benchmark = sub.add_parser("benchmark", help="Benchmark slide opening and tile decoding.")
    _add_common_args(p_benchmark)
    p_benchmark.add_argument("--level", type=int, default=0, help="Pyramid level.")
    p_benchmark.add_argument(
        "--size", type=int, nargs=2, default=[512, 512], metavar=("H", "W"), help="Decoded tile size."
    )
    p_benchmark.add_argument("--iterations", type=int, default=100, help="Number of measured tile reads.")
    p_benchmark.add_argument("--warmup", type=int, default=10, help="Number of unmeasured warm-up tile reads.")
    p_benchmark.add_argument(
        "--frame-cache-bytes",
        type=int,
        default=None,
        help="For wsidicom: frame cache size. Use 0 to force decoding on every read.",
    )
    p_benchmark.add_argument("--output", help="Optionally save the first measured tile to this PNG path.")
    p_benchmark.set_defaults(func=cmd_benchmark)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    configure_logging(args.log_level)

    if args.command == "benchmark" and (args.iterations < 1 or args.warmup < 0):
        parser.error("--iterations must be at least 1 and --warmup must not be negative")

    # Soft dependency checks with clearer errors than import-time crashes.
    if args.backend == "wsidicom":
        _, has_wsidicom = optional_import("wsidicom")
        if not has_wsidicom:
            print("Backend 'wsidicom' requires: pip install wsidicom", file=sys.stderr)
            return 1
    elif args.backend == "cucim":
        _, has_cucim = optional_import("cucim")
        if not has_cucim:
            print("Backend 'cucim' requires: pip install cucim", file=sys.stderr)
            return 1
    elif args.backend == "openslide":
        _, has_osl = optional_import("openslide")
        if not has_osl:
            print("Backend 'openslide' requires: pip install openslide-python", file=sys.stderr)
            return 1

    if not Path(args.path).exists():
        print(f"Path does not exist: {args.path}", file=sys.stderr)
        return 1

    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
