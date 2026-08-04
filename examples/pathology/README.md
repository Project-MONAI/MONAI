# Pathology / WSI sample apps

MONAI ships WSI support as library APIs (`monai.data.WSIReader`, datasets, and
`WSISlidingWindowSplitter`). There was previously no in-repo end-to-end sample for
opening a slide and extracting tiles for viewing. This folder provides that workflow.

## `view_wsi.py`

CLI for a typical WSI inspection workflow:

1. Open a slide (SVS/TIFF or DICOM WSI folder)
2. Inspect pyramid levels / sizes / mpp
3. Extract a region or sliding-window tiles and save PNGs

### Dependencies

Install the backend you need:

```bash
# DICOM WSI
pip install wsidicom pydicom pillow

# Optional GPU-accelerated DICOM tile decode
pip install cupy nvidia-nvimgcodec-cu13

# Other backends
pip install cucim            # or tifffile+imagecodecs
pip install openslide-python openslide-bin
```

### DICOM WSI (PixelMed sample)

If you downloaded NEMA WG26 PixelMed data under `wsi_datasets/`:

```bash
SLIDE=wsi_datasets/PixelMed_20190212/CMU1_DICOMOriginalRGBJPEGWithBigTIFFAndPyramid

# From the MONAI repo root (so local monai is importable)
export PYTHONPATH=.

python examples/pathology/view_wsi.py info --path "$SLIDE" --backend wsidicom

python examples/pathology/view_wsi.py tile \
  --path "$SLIDE" --backend wsidicom \
  --location 0 0 --size 512 512 --level 0 \
  --output ./wsi_tiles

# PixelMed CMU1 sample exposes only pyramid level 0 as a stored level
python examples/pathology/view_wsi.py thumbnail \
  --path "$SLIDE" --backend wsidicom --level 0 \
  --output ./wsi_tiles

python examples/pathology/view_wsi.py grid \
  --path "$SLIDE" --backend wsidicom \
  --size 256 256 --level 0 --max-tiles 8 \
  --output ./wsi_tiles
```

Open the PNGs under `./wsi_tiles` to view extracted tiles.

### Compare wsidicom/nvImageCodec with OpenSlide

The `openslide` backend uses the `openslide-python` package and supports file-based
formats such as SVS and pyramidal TIFF. OpenSlide does not read DICOM WSI directories,
so a meaningful comparison requires equivalent DICOM and SVS/TIFF representations of
the same slide.

Run the same tile workload against each representation:

```bash
# CPU DICOM decoding
python examples/pathology/view_wsi.py benchmark \
  --path "$DICOM_SERIES" --backend wsidicom \
  --size 512 512 --level 0 --warmup 10 --iterations 100

# GPU DICOM decoding through the nvImageCodec pydicom plugin
python examples/pathology/view_wsi.py benchmark \
  --path "$DICOM_SERIES" --backend wsidicom --register-nvimgcodec \
  --size 512 512 --level 0 --warmup 10 --iterations 100

# OpenSlide decoding of the corresponding SVS/TIFF file
python examples/pathology/view_wsi.py benchmark \
  --path "$OPENSLIDE_FILE" --backend openslide \
  --size 512 512 --level 0 --warmup 10 --iterations 100
```

The JSON output separates slide-open time from measured tile-decode time and reports
mean, median, p95, throughput, and a checksum. The benchmark traverses distinct tile
locations in raster order rather than repeatedly reading only one cached region.

For the `wsidicom` backend it also reports which decoder was selected, how many frames
nvImageCodec decoded successfully, and the GPU memory held by the process:

```json
"wsidicom_decoder": "PydicomDecoder",
"nvimgcodec": {
  "frame_decodes_succeeded": 40,
  "frame_decodes_failed": 0,
  "first_error": null,
  "nvimgcodec_used": true
},
"gpu_memory": { "this_process_mb": 476.0, "device_name": "NVIDIA RTX 6000 Ada Generation" }
```

Check `frame_decodes_succeeded` rather than assuming `--register-nvimgcodec` took effect.
pydicom tries each decoder plugin in turn and, when one raises, logs the error and falls
back to the next, so a plugin can be invoked for every frame while producing no GPU
pixels. `frame_decodes_failed` with a non-null `first_error` identifies that case.

`gpu_memory` is sampled inside the process because `nvidia-smi` polls about once per
second and usually misses a short benchmark run entirely.

Pass `--log-level DEBUG` to trace plugin registration, decoder selection, and per-frame
color-space decisions. Decode failures are logged at `ERROR`, so they appear at the default
level.

### Measuring decode rather than caching

wsidicom caches decoded frames (100 MB by default), so repeated reads of the same region
return cached pixels and measure almost no decoding. Pass `--frame-cache-bytes 0` to force
a decode on every read:

```bash
python examples/pathology/view_wsi.py benchmark \
  --path "$DICOM_SERIES" --backend wsidicom --register-nvimgcodec \
  --size 2048 2048 --level 0 --warmup 2 --iterations 10 \
  --frame-cache-bytes 0
```

## `check_nvimgcodec_usage.py`

Diagnoses the GPU DICOM decode path when tiles appear to decode on the CPU. It reports each
dependency separately, whether the plugin registered, and how many frames it decoded during
a real tile read. The exit status is non-zero when nvImageCodec was not used.

```bash
python examples/pathology/check_nvimgcodec_usage.py --path "$DICOM_SERIES"
```

Failure modes it distinguishes:

- `cupy` reported unavailable: nvImageCodec registration is skipped entirely, since the
  plugin requires CuPy. A partially removed CuPy install shows up here as an import error
  even though `pip list` reports the package.
- Registration succeeds but no frames are decoded: wsidicom selected a different decoder,
  so frames never reach pydicom's plugin stack.
- Frames are attempted but all fail (`frame_decodes_failed` > 0): the plugin raised and
  pydicom fell back to a CPU plugin. `first_error` gives the reason. An incomplete
  `nvidia-nvimgcodec` install reports `module 'nvidia.nvimgcodec' has no attribute
  'Decoder'`, because the package directory is importable as a namespace package while the
  compiled bindings are missing. Verify a wheel is fully installed with:

```bash
pip install --force-reinstall --no-cache-dir 'nvidia-nvimgcodec-cu13[all]'
```
