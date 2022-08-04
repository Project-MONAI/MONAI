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

import os
import pathlib
import tempfile
import textwrap
from copy import deepcopy
from glob import glob
from typing import TYPE_CHECKING, Callable

import numpy as np
import torch

from monai.apps import download_and_extract
from monai.transforms import (
    EnsureChannelFirstd,
    Affine,
    Affined,
    AsDiscrete,
    Compose,
    Flip,
    Flipd,
    LoadImaged,
    MapTransform,
    Orientation,
    Orientationd,
    Rand3DElastic,
    Rand3DElasticd,
    RandFlip,
    RandFlipd,
    Randomizable,
    RandRotate,
    RandRotated,
    RandZoom,
    RandZoomd,
    Rotate,
    Rotate90,
    Rotate90d,
    Rotated,
    ScaleIntensity,
    ScaleIntensityd,
    SpatialPadd,
    Zoom,
    Zoomd,
)
from monai.transforms.croppad.array import (
    BorderPad,
    CenterScaleCrop,
    CenterSpatialCrop,
    CropForeground,
    DivisiblePad,
    RandCropByLabelClasses,
    RandCropByPosNegLabel,
    RandScaleCrop,
    RandSpatialCrop,
    RandSpatialCropSamples,
    RandWeightedCrop,
    ResizeWithPadOrCrop,
    SpatialCrop,
    SpatialPad,
)
from monai.transforms.croppad.dictionary import (
    BorderPadd,
    CenterScaleCropd,
    CenterSpatialCropd,
    CropForegroundd,
    DivisiblePadd,
    RandCropByLabelClassesd,
    RandCropByPosNegLabeld,
    RandScaleCropd,
    RandSpatialCropd,
    RandSpatialCropSamplesd,
    RandWeightedCropd,
    ResizeWithPadOrCropd,
    SpatialCropd,
)
from monai.transforms.intensity.array import (
    AdjustContrast,
    ForegroundMask,
    GaussianSharpen,
    GaussianSmooth,
    GibbsNoise,
    HistogramNormalize,
    KSpaceSpikeNoise,
    MaskIntensity,
    NormalizeIntensity,
    RandAdjustContrast,
    RandBiasField,
    RandCoarseDropout,
    RandCoarseShuffle,
    RandGaussianNoise,
    RandGaussianSharpen,
    RandGaussianSmooth,
    RandGibbsNoise,
    RandHistogramShift,
    RandKSpaceSpikeNoise,
    RandRicianNoise,
    RandScaleIntensity,
    RandShiftIntensity,
    RandStdShiftIntensity,
    SavitzkyGolaySmooth,
    ScaleIntensityRange,
    ScaleIntensityRangePercentiles,
    ShiftIntensity,
    StdShiftIntensity,
    ThresholdIntensity,
)
from monai.transforms.intensity.dictionary import (
    AdjustContrastd,
    ForegroundMaskd,
    GaussianSharpend,
    GaussianSmoothd,
    GibbsNoised,
    HistogramNormalized,
    KSpaceSpikeNoised,
    MaskIntensityd,
    NormalizeIntensityd,
    RandAdjustContrastd,
    RandBiasFieldd,
    RandCoarseDropoutd,
    RandCoarseShuffled,
    RandGaussianNoised,
    RandGaussianSharpend,
    RandGaussianSmoothd,
    RandGibbsNoised,
    RandHistogramShiftd,
    RandKSpaceSpikeNoised,
    RandRicianNoised,
    RandScaleIntensityd,
    RandShiftIntensityd,
    RandStdShiftIntensityd,
    SavitzkyGolaySmoothd,
    ScaleIntensityRanged,
    ScaleIntensityRangePercentilesd,
    ShiftIntensityd,
    StdShiftIntensityd,
    ThresholdIntensityd,
)
from monai.transforms.post.array import KeepLargestConnectedComponent, LabelFilter, LabelToContour
from monai.transforms.post.dictionary import AsDiscreted, KeepLargestConnectedComponentd, LabelFilterd, LabelToContourd
from monai.transforms.smooth_field.array import (
    RandSmoothDeform,
    RandSmoothFieldAdjustContrast,
    RandSmoothFieldAdjustIntensity,
)
from monai.transforms.smooth_field.dictionary import (
    RandSmoothDeformd,
    RandSmoothFieldAdjustContrastd,
    RandSmoothFieldAdjustIntensityd,
)
from monai.transforms.spatial.array import (
    GridDistortion,
    Rand2DElastic,
    RandAffine,
    RandAxisFlip,
    RandGridDistortion,
    RandRotate90,
    Resize,
    Spacing,
)
from monai.transforms.spatial.dictionary import (
    GridDistortiond,
    Rand2DElasticd,
    RandAffined,
    RandAxisFlipd,
    RandGridDistortiond,
    RandRotate90d,
    Resized,
    Spacingd,
)
from monai.utils.enums import CommonKeys
from monai.utils.module import optional_import

if TYPE_CHECKING:
    import matplotlib.pyplot as plt

    has_matplotlib = True

else:
    plt, has_matplotlib = optional_import("matplotlib.pyplot")


def get_data(keys):
    """Get the example data to be used.

    Use MarsAtlas as it only contains 1 image for quick download and
    that image is parcellated.
    """
    cache_dir = os.environ.get("MONAI_DATA_DIRECTORY") or tempfile.mkdtemp()
    fname = "MarsAtlas-MNI-Colin27.zip"
    url = "https://www.dropbox.com/s/ndz8qtqblkciole/" + fname + "?dl=1"
    out_path = os.path.join(cache_dir, "MarsAtlas-MNI-Colin27")
    zip_path = os.path.join(cache_dir, fname)

    download_and_extract(url, zip_path, out_path)

    image, label = sorted(glob(os.path.join(out_path, "*.nii")))

    data = {CommonKeys.IMAGE: image, CommonKeys.LABEL: label}

    transforms = Compose(
        [LoadImaged(keys), EnsureChannelFirstd(keys), ScaleIntensityd(CommonKeys.IMAGE), Rotate90d(keys, spatial_axes=[0, 2])]
    )
    data = transforms(data)
    max_size = max(data[keys[0]].shape)
    padder = SpatialPadd(keys, (max_size, max_size, max_size))
    return padder(data)


def update_docstring(code_path, transform_name):
    """
    Find the documentation for a given transform and if it's missing,
    add a pointer to the transform's example image.
    """
    with open(code_path) as f:
        contents = f.readlines()
    doc_start = None
    for i, line in enumerate(contents):
        # find the line containing start of the transform documentation
        if "`" + transform_name + "`" in line:
            doc_start = i
            break
    if doc_start is None:
        raise RuntimeError("Couldn't find transform documentation")

    # if image is already in docs, nothing to do
    image_line = doc_start + 2
    if ".. image" in contents[image_line]:
        return

    # add the line for the image and the alt text
    contents_orig = deepcopy(contents)
    contents.insert(
        image_line,
        ".. image:: https://github.com/Project-MONAI/DocImages/raw/main/transforms/" + transform_name + ".png\n",
    )
    contents.insert(image_line + 1, "    :alt: example of " + transform_name + "\n")

    # check that we've only added two lines
    if len(contents) != len(contents_orig) + 2:
        raise AssertionError

    # write the updated doc to overwrite the original
    with open(code_path, "w") as f:
        f.writelines(contents)


def pre_process_data(data, ndim, is_map, is_post):
    """If transform requires 2D data, then convert to 2D"""
    if ndim == 2:
        for k in keys:
            data[k] = data[k][..., data[k].shape[-1] // 2]

    if is_map:
        return data
    return data[CommonKeys.LABEL] if is_post else data[CommonKeys.IMAGE]


def get_2d_slice(image, view, is_label):
    """If image is 3d, get the central slice. If is already 2d, return as-is.
    If image is label, set 0 to np.nan.
    """
    if image.ndim == 2:
        out = image
    else:
        shape = image.shape
        slices = [slice(0, s) for s in shape]
        _slice = shape[view] // 2
        slices[view] = slice(_slice, _slice + 1)
        slices = tuple(slices)
        out = np.squeeze(image[slices], view)
    if is_label:
        out[out == 0] = np.nan
    return out


def get_stacked_2d_ims(im, is_label):
    """Get the 3 orthogonal views and stack them into 1 image.
    Requires that all images be same size, but this is taken care
    of by the `SpatialPadd` earlier.
    """
    return [get_2d_slice(im, i, is_label) for i in range(3)]


def get_stacked_before_after(before, after, is_label=False):
    """Stack before and after images into 1 image if 3d.
    Requires that before and after images be the same size.
    """
    return [get_stacked_2d_ims(d, is_label) for d in (before, after)]


def save_image(images, labels, filename, transform_name, transform_args, shapes, colorbar=False):
    """Save image to file, ensuring there's no whitespace around the edge."""
    plt.rcParams.update({"font.family": "monospace"})
    plt.style.use("dark_background")
    nrow = len(images)  # before and after (should always be 2)
    ncol = len(images[0])  # num orthogonal views (either 1 or 3)
    # roughly estimate the height_ratios of the first:second row
    hs = [float(r[0].shape[0]) for r in images]
    fig = plt.figure(tight_layout=True)
    spec = fig.add_gridspec(nrow, ncol, hspace=0, wspace=0, height_ratios=hs)
    for row in range(nrow):
        vmin = min(i.min() for i in images[row])
        vmax = max(i.max() for i in images[row])
        for col in range(ncol):
            ax = fig.add_subplot(spec[row, col])
            imshow = ax.imshow(images[row][col], cmap="gray", vmin=vmin, vmax=vmax)
            ax.set_aspect("equal")
            if colorbar and col == ncol - 1:
                plt.colorbar(imshow, ax=ax)
            if col == 0:
                y_label = "After" if row else "Before"
                y_label += ("\n" + shapes[row]) if shapes[0] != shapes[1] else ""
                ax.set_ylabel(y_label)
            # print yticks for the right most column
            if col != ncol - 1 or colorbar:
                ax.set_yticks([])
            else:
                ax.yaxis.tick_right()
                for n, label in enumerate(ax.yaxis.get_ticklabels()):
                    if n > 2:
                        label.set_visible(False)
            ax.set_xticks([])
            ax.set_frame_on(False)
            if labels is not None:
                ax.imshow(labels[row][col], cmap="hsv", alpha=0.9, interpolation="nearest")
    # title is e.g., Flipd(keys=keys, spatial_axis=0)
    title = transform_name + "("
    for k, v in transform_args.items():
        title += k + "="
        if isinstance(v, str):
            title += "'" + v + "'"
        elif isinstance(v, (np.ndarray, torch.Tensor)):
            title += "[array]"
        elif isinstance(v, Callable):
            title += "[callable]"
        else:
            title += str(v)
        title += ", "
    if len(transform_args) > 0:
        title = title[:-2]
    title += ")"
    # shorten the lines
    title = textwrap.fill(title, 50, break_long_words=False, subsequent_indent=" " * (len(transform_name) + 1))
    fig.suptitle(title, x=0.1, horizontalalignment="left")
    fig.savefig(filename)
    plt.close(fig)


def get_images(data, is_label=False):
    """Get image. If is dictionary, extract key. If is list, stack. If both dictionary and list, do both.
    Also return the image size as string to be used im the imshow. If it's a list, return `N x (H,W,D)`.
    """
    # If not a list, convert
    if not isinstance(data, list):
        data = [data]
    key = CommonKeys.LABEL if is_label else CommonKeys.IMAGE
    is_map = isinstance(data[0], dict)
    # length of the list will be equal to number of samples produced. This will be 1 except for transforms that
    # produce `num_samples`.
    data = [d[key] if is_map else d for d in data]
    data = [d[0] for d in data]  # remove channel component

    # for each sample, create a list of the orthogonal views. If image is 2d, length will be 1. If 3d, there
    # will be three orthogonal views
    num_samples = len(data)
    num_orthog_views = 3 if data[0].ndim == 3 else 1
    shape_str = (f"{num_samples} x " if num_samples > 1 else "") + str(data[0].shape)
    for i in range(num_samples):
        data[i] = [get_2d_slice(data[i], view, is_label) for view in range(num_orthog_views)]

    out = []
    if num_samples == 1:
        out = data[0]
    else:
        # we might need to panel the images. this happens if a transform produces e.g. 4 output images.
        # In this case, we create a 2-by-2 grid from them. Output will be a list containing n_orthog_views,
        # each element being either the image (if num_samples is 1) or the panelled image.
        nrows = int(np.floor(num_samples**0.5))
        for view in range(num_orthog_views):
            result = np.asarray([d[view] for d in data])
            nindex, height, width = result.shape
            ncols = nindex // nrows
            # only implemented for square number of images (e.g. 4 images goes to a 2-by-2 panel)
            if nindex != nrows * ncols:
                raise NotImplementedError
            # want result.shape = (height*nrows, width*ncols), have to be careful about striding
            result = result.reshape(nrows, ncols, height, width).swapaxes(1, 2).reshape(height * nrows, width * ncols)
            out.append(result)
    return out, shape_str


def create_transform_im(
    transform, transform_args, data, ndim=3, colorbar=False, update_doc=True, seed=0, is_post=False
):
    """Create an image with the before and after of the transform.
    Also update the transform's documentation to point to this image."""

    transform = transform(**transform_args)

    if not has_matplotlib:
        raise RuntimeError

    if isinstance(transform, Randomizable):
        # increment the seed for map transforms so they're different to the array versions.
        seed = seed + 1 if isinstance(transform, MapTransform) else seed
        transform.set_random_state(seed)

    out_dir = os.environ.get("MONAI_DOC_IMAGES")
    if out_dir is None:
        raise RuntimeError(
            "Please git clone https://github.com/Project-MONAI/DocImages"
            + " and then set the environment variable `MONAI_DOC_IMAGES`"
        )
    out_dir = os.path.join(out_dir, "transforms")

    # Path is transform name
    transform_name = transform.__class__.__name__
    out_fname = transform_name + ".png"
    out_file = os.path.join(out_dir, out_fname)

    is_map = isinstance(transform, MapTransform)
    data_in = pre_process_data(deepcopy(data), ndim, is_map, is_post)

    data_tr = transform(deepcopy(data_in))

    images_before, before_shape = get_images(data_in)
    images_after, after_shape = get_images(data_tr)
    images = (images_before, images_after)
    shapes = (before_shape, after_shape)

    labels = None
    if is_map:
        labels_before, *_ = get_images(data_in, is_label=True)
        labels_after, *_ = get_images(data_tr, is_label=True)
        labels = (labels_before, labels_after)

    save_image(images, labels, out_file, transform_name, transform_args, shapes, colorbar)

    if update_doc:
        base_dir = pathlib.Path(__file__).parent.parent.parent
        rst_path = os.path.join(base_dir, "docs", "source", "transforms.rst")
        update_docstring(rst_path, transform_name)


if __name__ == "__main__":

    keys = [CommonKeys.IMAGE, CommonKeys.LABEL]
    data = get_data(keys)
    create_transform_im(RandFlip, dict(prob=1, spatial_axis=1), data)
    create_transform_im(RandFlipd, dict(keys=keys, prob=1, spatial_axis=2), data)
    create_transform_im(Flip, dict(spatial_axis=1), data)
    create_transform_im(Flipd, dict(keys=keys, spatial_axis=2), data)
    create_transform_im(Orientation, dict(axcodes="RPI", image_only=True), data)
    create_transform_im(Orientationd, dict(keys=keys, axcodes="RPI"), data)
    create_transform_im(
        Rand3DElastic, dict(prob=1.0, sigma_range=(1, 2), magnitude_range=(0.5, 0.5), shear_range=(1, 1, 1)), data
    )
    create_transform_im(Affine, dict(shear_params=(0, 0.5, 0), image_only=True, padding_mode="zeros"), data)
    create_transform_im(
        Affined, dict(keys=keys, shear_params=(0, 0.5, 0), mode=["bilinear", "nearest"], padding_mode="zeros"), data
    )
    create_transform_im(RandAffine, dict(prob=1, shear_range=(0.5, 0.5), padding_mode="zeros"), data)
    create_transform_im(
        RandAffined,
        dict(keys=keys, prob=1, shear_range=(0.5, 0.5), mode=["bilinear", "nearest"], padding_mode="zeros"),
        data,
    )
    create_transform_im(
        Rand3DElastic, dict(sigma_range=(5, 7), magnitude_range=(50, 150), prob=1, padding_mode="zeros"), data
    )
    create_transform_im(
        Rand2DElastic, dict(prob=1, spacing=(20, 20), magnitude_range=(1, 2), padding_mode="zeros"), data, 2
    )
    create_transform_im(
        Rand2DElasticd,
        dict(
            keys=keys,
            prob=1,
            spacing=(20, 20),
            magnitude_range=(1, 2),
            padding_mode="zeros",
            mode=["bilinear", "nearest"],
        ),
        data,
        2,
    )
    create_transform_im(
        Rand3DElasticd,
        dict(
            keys=keys,
            sigma_range=(5, 7),
            magnitude_range=(50, 150),
            prob=1,
            padding_mode="zeros",
            mode=["bilinear", "nearest"],
        ),
        data,
    )
    create_transform_im(Rotate90, dict(spatial_axes=(1, 2)), data)
    create_transform_im(Rotate90d, dict(keys=keys, spatial_axes=(1, 2)), data)
    create_transform_im(RandRotate90, dict(prob=1), data)
    create_transform_im(RandRotate90d, dict(keys=keys, prob=1), data)
    create_transform_im(Rotate, dict(angle=0.1), data)
    create_transform_im(Rotated, dict(keys=keys, angle=0.1, mode=["bilinear", "nearest"]), data)
    create_transform_im(RandRotate, dict(prob=1, range_x=[0.4, 0.4]), data)
    create_transform_im(RandRotated, dict(keys=keys, prob=1, range_x=[0.4, 0.4], mode=["bilinear", "nearest"]), data)
    create_transform_im(Zoom, dict(zoom=0.6), data)
    create_transform_im(Zoomd, dict(keys=keys, zoom=1.3, mode=["area", "nearest"]), data)
    create_transform_im(RandZoom, dict(prob=1, min_zoom=0.6, max_zoom=0.8), data)
    create_transform_im(RandZoomd, dict(keys=keys, prob=1, min_zoom=1.3, max_zoom=1.5, mode=["area", "nearest"]), data)
    create_transform_im(ScaleIntensity, dict(minv=0, maxv=10), data, colorbar=True)
    create_transform_im(ScaleIntensityd, dict(keys=CommonKeys.IMAGE, minv=0, maxv=10), data, colorbar=True)
    create_transform_im(RandScaleIntensity, dict(prob=1.0, factors=(5, 10)), data, colorbar=True)
    create_transform_im(
        RandScaleIntensityd, dict(keys=CommonKeys.IMAGE, prob=1.0, factors=(5, 10)), data, colorbar=True
    )
    create_transform_im(DivisiblePad, dict(k=64), data)
    create_transform_im(DivisiblePadd, dict(keys=keys, k=64), data)
    create_transform_im(CropForeground, dict(), data)
    create_transform_im(CropForegroundd, dict(keys=keys, source_key=CommonKeys.IMAGE), data)
    create_transform_im(RandGaussianNoise, dict(prob=1, mean=0, std=0.1), data)
    create_transform_im(RandGaussianNoised, dict(keys=CommonKeys.IMAGE, prob=1, mean=0, std=0.1), data)
    create_transform_im(KSpaceSpikeNoise, dict(loc=(100, 100, 100), k_intensity=13), data)
    create_transform_im(KSpaceSpikeNoised, dict(keys=CommonKeys.IMAGE, loc=(100, 100, 100), k_intensity=13), data)
    create_transform_im(RandKSpaceSpikeNoise, dict(prob=1, intensity_range=(10, 13)), data)
    create_transform_im(
        RandKSpaceSpikeNoised,
        dict(keys=CommonKeys.IMAGE, global_prob=1, prob=1, common_sampling=True, intensity_range=(13, 15)),
        data,
    )
    create_transform_im(RandRicianNoise, dict(prob=1.0, mean=1, std=0.5), data)
    create_transform_im(RandRicianNoised, dict(keys=CommonKeys.IMAGE, prob=1.0, mean=1, std=0.5), data)
    create_transform_im(SavitzkyGolaySmooth, dict(window_length=5, order=1), data)
    create_transform_im(SavitzkyGolaySmoothd, dict(keys=CommonKeys.IMAGE, window_length=5, order=1), data)
    create_transform_im(GibbsNoise, dict(alpha=0.8), data)
    create_transform_im(GibbsNoised, dict(keys=CommonKeys.IMAGE, alpha=0.8), data)
    create_transform_im(RandGibbsNoise, dict(prob=1.0, alpha=(0.6, 0.8)), data)
    create_transform_im(RandGibbsNoised, dict(keys=CommonKeys.IMAGE, prob=1.0, alpha=(0.6, 0.8)), data)
    create_transform_im(ShiftIntensity, dict(offset=1), data, colorbar=True)
    create_transform_im(ShiftIntensityd, dict(keys=CommonKeys.IMAGE, offset=1), data, colorbar=True)
    create_transform_im(RandShiftIntensity, dict(prob=1.0, offsets=(10, 20)), data, colorbar=True)
    create_transform_im(
        RandShiftIntensityd, dict(keys=CommonKeys.IMAGE, prob=1.0, offsets=(10, 20)), data, colorbar=True
    )
    create_transform_im(StdShiftIntensity, dict(factor=10), data, colorbar=True)
    create_transform_im(StdShiftIntensityd, dict(keys=CommonKeys.IMAGE, factor=10), data, colorbar=True)
    create_transform_im(RandStdShiftIntensity, dict(prob=1.0, factors=(5, 10)), data, colorbar=True)
    create_transform_im(
        RandStdShiftIntensityd, dict(keys=CommonKeys.IMAGE, prob=1.0, factors=(5, 10)), data, colorbar=True
    )
    create_transform_im(RandBiasField, dict(prob=1, coeff_range=(0.2, 0.3)), data)
    create_transform_im(RandBiasFieldd, dict(keys=CommonKeys.IMAGE, prob=1, coeff_range=(0.2, 0.3)), data)
    create_transform_im(NormalizeIntensity, dict(subtrahend=0, divisor=10), data, colorbar=True)
    create_transform_im(NormalizeIntensityd, dict(keys=CommonKeys.IMAGE, subtrahend=0, divisor=10), data, colorbar=True)
    create_transform_im(ThresholdIntensity, dict(threshold=0.4, above=False, cval=0.9), data, colorbar=True)
    create_transform_im(
        ThresholdIntensityd, dict(keys=CommonKeys.IMAGE, threshold=0.4, above=False, cval=0.9), data, colorbar=True
    )
    create_transform_im(ScaleIntensityRange, dict(a_min=0, a_max=1, b_min=1, b_max=10), data, colorbar=True)
    create_transform_im(
        ScaleIntensityRanged, dict(keys=CommonKeys.IMAGE, a_min=0, a_max=1, b_min=1, b_max=10), data, colorbar=True
    )
    create_transform_im(ScaleIntensityRangePercentiles, dict(lower=5, upper=95, b_min=1, b_max=10), data, colorbar=True)
    create_transform_im(
        ScaleIntensityRangePercentilesd,
        dict(keys=CommonKeys.IMAGE, lower=5, upper=95, b_min=1, b_max=10),
        data,
        colorbar=True,
    )
    create_transform_im(AdjustContrast, dict(gamma=2), data, colorbar=True)
    create_transform_im(AdjustContrastd, dict(keys=CommonKeys.IMAGE, gamma=2), data, colorbar=True)
    create_transform_im(RandAdjustContrast, dict(prob=1, gamma=(1.5, 2)), data, colorbar=True)
    create_transform_im(RandAdjustContrastd, dict(keys=CommonKeys.IMAGE, prob=1, gamma=(1.5, 2)), data, colorbar=True)
    create_transform_im(MaskIntensity, dict(mask_data=data[CommonKeys.IMAGE], select_fn=lambda x: x > 0.3), data)
    create_transform_im(
        MaskIntensityd, dict(keys=CommonKeys.IMAGE, mask_key=CommonKeys.IMAGE, select_fn=lambda x: x > 0.3), data
    )
    create_transform_im(ForegroundMask, dict(invert=True), data)
    create_transform_im(ForegroundMaskd, dict(keys=CommonKeys.IMAGE, invert=True), data)
    create_transform_im(GaussianSmooth, dict(sigma=2), data)
    create_transform_im(GaussianSmoothd, dict(keys=CommonKeys.IMAGE, sigma=2), data)
    create_transform_im(RandGaussianSmooth, dict(prob=1.0, sigma_x=(1, 2)), data)
    create_transform_im(RandGaussianSmoothd, dict(keys=CommonKeys.IMAGE, prob=1.0, sigma_x=(1, 2)), data)
    create_transform_im(GaussianSharpen, dict(), GaussianSmoothd(CommonKeys.IMAGE, 2)(data))
    create_transform_im(GaussianSharpend, dict(keys=CommonKeys.IMAGE), GaussianSmoothd(CommonKeys.IMAGE, 2)(data))
    create_transform_im(RandGaussianSharpen, dict(prob=1), GaussianSmoothd(CommonKeys.IMAGE, 2)(data))
    create_transform_im(
        RandGaussianSharpend, dict(keys=CommonKeys.IMAGE, prob=1), GaussianSmoothd(CommonKeys.IMAGE, 2)(data)
    )
    create_transform_im(RandHistogramShift, dict(prob=1, num_control_points=3), data, colorbar=True)
    create_transform_im(
        RandHistogramShiftd, dict(keys=CommonKeys.IMAGE, prob=1, num_control_points=3), data, colorbar=True
    )
    create_transform_im(RandCoarseDropout, dict(prob=1, holes=200, spatial_size=20, fill_value=0), data)
    create_transform_im(
        RandCoarseDropoutd, dict(keys=CommonKeys.IMAGE, prob=1, holes=200, spatial_size=20, fill_value=0), data
    )
    create_transform_im(RandCoarseShuffle, dict(prob=1, holes=200, spatial_size=20), data)
    create_transform_im(RandCoarseShuffled, dict(keys=CommonKeys.IMAGE, prob=1, holes=200, spatial_size=20), data)
    create_transform_im(HistogramNormalize, dict(num_bins=10), data)
    create_transform_im(HistogramNormalized, dict(keys=CommonKeys.IMAGE, num_bins=10), data)
    create_transform_im(SpatialPad, dict(spatial_size=(300, 300, 300)), data)
    create_transform_im(SpatialPadd, dict(keys=keys, spatial_size=(300, 300, 300)), data)
    create_transform_im(BorderPad, dict(spatial_border=10), data)
    create_transform_im(BorderPadd, dict(keys=keys, spatial_border=10), data)
    create_transform_im(SpatialCrop, dict(roi_center=(75, 75, 75), roi_size=(100, 100, 100)), data)
    create_transform_im(SpatialCropd, dict(keys=keys, roi_center=(75, 75, 75), roi_size=(100, 100, 100)), data)
    create_transform_im(CenterSpatialCrop, dict(roi_size=(100, 100, 100)), data)
    create_transform_im(CenterSpatialCropd, dict(keys=keys, roi_size=(100, 100, 100)), data)
    create_transform_im(RandSpatialCrop, dict(roi_size=(100, 100, 100), random_size=False), data)
    create_transform_im(RandSpatialCropd, dict(keys=keys, roi_size=(100, 100, 100), random_size=False), data)
    create_transform_im(RandSpatialCropSamples, dict(num_samples=4, roi_size=(100, 100, 100), random_size=False), data)
    create_transform_im(
        RandSpatialCropSamplesd, dict(keys=keys, num_samples=4, roi_size=(100, 100, 100), random_size=False), data
    )
    create_transform_im(
        RandWeightedCrop, dict(spatial_size=(100, 100, 100), num_samples=4, weight_map=data[CommonKeys.IMAGE] > 0), data
    )
    create_transform_im(
        RandWeightedCropd, dict(keys=keys, spatial_size=(100, 100, 100), num_samples=4, w_key=CommonKeys.IMAGE), data
    )
    create_transform_im(
        RandCropByPosNegLabel,
        dict(spatial_size=(100, 100, 100), label=data[CommonKeys.LABEL], neg=0, num_samples=4),
        data,
    )
    create_transform_im(
        RandCropByPosNegLabeld,
        dict(keys=keys, spatial_size=(100, 100, 100), label_key=CommonKeys.LABEL, neg=0, num_samples=4),
        data,
    )
    create_transform_im(
        RandCropByLabelClasses,
        dict(
            spatial_size=(100, 100, 100), label=data[CommonKeys.LABEL] > 0, num_classes=2, ratios=[0, 1], num_samples=4
        ),
        data,
    )
    create_transform_im(
        RandCropByLabelClassesd,
        dict(
            keys=keys,
            spatial_size=(100, 100, 100),
            label_key=CommonKeys.LABEL,
            num_classes=2,
            ratios=[0, 1],
            num_samples=4,
        ),
        data,
    )
    create_transform_im(ResizeWithPadOrCrop, dict(spatial_size=(100, 100, 100)), data)
    create_transform_im(ResizeWithPadOrCropd, dict(keys=keys, spatial_size=(100, 100, 100)), data)
    create_transform_im(RandScaleCrop, dict(roi_scale=0.4), data)
    create_transform_im(RandScaleCropd, dict(keys=keys, roi_scale=0.4), data)
    create_transform_im(CenterScaleCrop, dict(roi_scale=0.4), data)
    create_transform_im(CenterScaleCropd, dict(keys=keys, roi_scale=0.4), data)
    create_transform_im(AsDiscrete, dict(to_onehot=None, threshold=10), data, is_post=True, colorbar=True)
    create_transform_im(AsDiscreted, dict(keys=CommonKeys.LABEL, to_onehot=None, threshold=10), data, is_post=True)
    create_transform_im(LabelFilter, dict(applied_labels=(1, 2, 3, 4, 5, 6)), data, is_post=True)
    create_transform_im(
        LabelFilterd, dict(keys=CommonKeys.LABEL, applied_labels=(1, 2, 3, 4, 5, 6)), data, is_post=True
    )
    create_transform_im(LabelToContour, dict(), data, is_post=True)
    create_transform_im(LabelToContourd, dict(keys=CommonKeys.LABEL), data, is_post=True)
    create_transform_im(Spacing, dict(pixdim=(5, 5, 5), image_only=True), data)
    create_transform_im(Spacingd, dict(keys=keys, pixdim=(5, 5, 5), mode=["bilinear", "nearest"]), data)
    create_transform_im(RandAxisFlip, dict(prob=1), data)
    create_transform_im(RandAxisFlipd, dict(keys=keys, prob=1), data)
    create_transform_im(Resize, dict(spatial_size=(100, 100, 100)), data)
    create_transform_im(Resized, dict(keys=keys, spatial_size=(100, 100, 100), mode=["area", "nearest"]), data)
    data_binary = deepcopy(data)
    data_binary[CommonKeys.LABEL] = (data_binary[CommonKeys.LABEL] > 0).astype(np.float32)
    create_transform_im(KeepLargestConnectedComponent, dict(applied_labels=1), data_binary, is_post=True, ndim=2)
    create_transform_im(
        KeepLargestConnectedComponentd, dict(keys=CommonKeys.LABEL, applied_labels=1), data_binary, is_post=True, ndim=2
    )
    create_transform_im(
        GridDistortion, dict(num_cells=3, distort_steps=[(1.5,) * 4] * 3, mode="nearest", padding_mode="zeros"), data
    )
    create_transform_im(
        GridDistortiond,
        dict(
            keys=keys, num_cells=3, distort_steps=[(1.5,) * 4] * 3, mode=["bilinear", "nearest"], padding_mode="zeros"
        ),
        data,
    )
    create_transform_im(RandGridDistortion, dict(num_cells=3, prob=1.0, distort_limit=(-0.1, 0.1)), data)
    create_transform_im(
        RandGridDistortiond,
        dict(keys=keys, num_cells=4, prob=1.0, distort_limit=(-0.2, 0.2), mode=["bilinear", "nearest"]),
        data,
    )
    create_transform_im(
        RandSmoothFieldAdjustContrast, dict(spatial_size=(217, 217, 217), rand_size=(10, 10, 10), prob=1.0), data
    )
    create_transform_im(
        RandSmoothFieldAdjustContrastd,
        dict(keys=keys, spatial_size=(217, 217, 217), rand_size=(10, 10, 10), prob=1.0),
        data,
    )
    create_transform_im(
        RandSmoothFieldAdjustIntensity,
        dict(spatial_size=(217, 217, 217), rand_size=(10, 10, 10), prob=1.0, gamma=(0.5, 4.5)),
        data,
    )
    create_transform_im(
        RandSmoothFieldAdjustIntensityd,
        dict(keys=keys, spatial_size=(217, 217, 217), rand_size=(10, 10, 10), prob=1.0, gamma=(0.5, 4.5)),
        data,
    )

    create_transform_im(
        RandSmoothDeform,
        dict(spatial_size=(217, 217, 217), rand_size=(10, 10, 10), prob=1.0, def_range=0.05, grid_mode="bilinear"),
        data,
    )
    create_transform_im(
        RandSmoothDeformd,
        dict(
            keys=keys,
            spatial_size=(217, 217, 217),
            rand_size=(10, 10, 10),
            prob=1.0,
            def_range=0.05,
            grid_mode="bilinear",
        ),
        data,
    )
