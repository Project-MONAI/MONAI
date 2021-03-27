# Copyright 2020 - 2021 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import os
from typing import Optional, Tuple

import numpy as np

from monai.metrics import compute_fp_tp_probs, compute_froc_curve_data, compute_froc_score
from monai.utils import optional_import

measure, _ = optional_import("skimage.measure")
openslide, _ = optional_import("openslide")
ndimage, _ = optional_import("scipy.ndimage")


def load_datalist(
    datalist_path: Optional[str] = None,
    prediction_key: str = "prediction",
    ground_truth_key: str = "ground_truth",
    prediciton_dir: Optional[str] = None,
    ground_truth_dir: Optional[str] = None,
):
    """
    This function is used to load the datalist.
    If

    Args:
        datalist_path: the path of the datalist in the form of a json file. The file should
            be a list of dictionaries, where each dictionary contains the prediction and
            ground truth paths of one image. Each prediction path should point to a csv file,
            and each ground truth path should point to a tif file.
            Defaults to None.
        prediction_key: the key of predictions in datalist dictionaries.
            Defaults to `"prediction"`.
        ground_truth_key: the key of ground truths in datalist dictionaries.
            Defaults to `"ground_truth"`.
        prediciton_dir: if the datalist json file is not provided, the directory of
            all prediction csv files should be provided. Defaults to None.
        ground_truth_dir: if the datalist json file is not provided,
            the directory of all ground truths tif files should be provided.
            Defaults to None.
    """
    if datalist_path is None:
        if prediciton_dir is None or ground_truth_dir is None:
            raise ValueError(
                "when datalist_path is None, prediciton_dir and \
                ground_truth_dir must be provided."
            )
        datalist = []
        pred_list = os.listdir(prediciton_dir)
        gt_list = os.listdir(ground_truth_dir)
        for filename in pred_list:
            data = {}
            if filename.endswith(".csv"):
                gt_filename = filename.split(".csv")[0] + ".tif"
                if gt_filename in gt_list:
                    data[prediction_key] = os.path.join(prediciton_dir, filename)
                    data[ground_truth_key] = os.path.join(ground_truth_dir, gt_filename)
                else:
                    print("missing {} in {}, skipping!".format(gt_filename, ground_truth_dir))
            if len(data) > 0:
                datalist.append(data)
    else:
        with open(datalist_path) as f:
            datalist = json.load(f)

    return datalist


def get_pred_result(prediction_path: str):
    """
    This function is used to get the pathology prediction results according to the
    original prediction file.

    Args:
        prediction_path: the ground truth file path, the path should point
            to a csv file. In each line of the csv file, there should have three
            values that represent probability, X-coordinates and  Y-coordinates.

    """
    x_coord, y_coord, probs = [], [], []
    csv_lines = open(prediction_path, "r").readlines()
    for i in range(len(csv_lines)):
        line = csv_lines[i]
        elems = line.rstrip().split(",")
        probs.append(float(elems[0]))
        x_coord.append(int(elems[1]))
        y_coord.append(int(elems[2]))

    return np.asarray(probs), np.asarray(x_coord), np.asarray(y_coord)


def compute_eval_mask(
    ground_truth_path: str,
    pixel_size: float = 0.243,
    annotation_size: int = 75,
    resolution_level: int = 5,
):
    """
    This function is used to compute the evaluation mask according to the
    original ground truth file.

    Args:
        ground_truth_path: the ground truth file path, the path should point
            to a tif file.
        pixel_size: the pixel size（in micrometer) of the image in level 0.
            Defaults to 0.243.
        annotation_size: the annotation size (in micrometer) of the ground truth.
            Defaults to 75.
        resolution_level: the level at which the ground truth is made.
            Defaults to 5.
    """
    slide = openslide.open_slide(ground_truth_path)
    dims = slide.level_dimensions[resolution_level]
    pixelarray = np.zeros(dims[0] * dims[1], dtype="uint")
    pixelarray = np.array(slide.read_region((0, 0), resolution_level, dims))

    neg = 255 - pixelarray[:, :, 0] * 255
    distance = ndimage.morphology.distance_transform_edt(neg)

    threshold = annotation_size / (pixel_size * pow(2, resolution_level) * 2)
    binary = distance < threshold
    filled_image = ndimage.morphology.binary_fill_holes(binary)
    evaluation_mask = measure.label(filled_image, connectivity=2)

    return evaluation_mask


def compute_itc_list(
    evaluation_mask: np.ndarray,
    pixel_size: float = 0.243,
    annotation_size: int = 75,
    resolution_level: int = 5,
    itc_diameter_threshold: int = 200,
):
    """
    This function is used to compute the list of labels containing
    Isolated Tumor Cells (ITC).

    Args:
        evaluation_mask: the evaluation mask in the form of numpy.ndarray.
        pixel_size: the pixel size（in micrometer) of the image in level 0.
            Defaults to 0.243.
        annotation_size: the annotation size (in micrometer) of the ground truth.
            Defaults to 75.
        resolution_level: the level at which the ground truth is made.
            Defaults to 5.
        itc_diameter_threshold: if a region's longest diameter (in micrometer) is
            less than this threshold, it is considered as an isolated tumor cell.
            Defaults to 200.
    """
    max_label = np.amax(evaluation_mask)
    properties = measure.regionprops(evaluation_mask, coordinates="rc")
    itc_list = []
    threshold = (itc_diameter_threshold + annotation_size) / (pixel_size * pow(2, resolution_level))
    for i in range(0, max_label):
        if properties[i].major_axis_length < threshold:
            itc_list.append(i + 1)

    return itc_list


class PathologyEvalFROC:
    """
    Evaluate with Free Response Operating Characteristic (FROC) score.

    Args:
        pixel_size: the pixel size （in micrometer) of the image in level 0.
            Defaults to 0.243.
        annotation_size: the annotation size (in micrometer) of the ground truth.
            Defaults to 75.
        itc_diameter_threshold: if a region's longest diameter (in micrometer) is
            less than this threshold, it is considered as an isolated tumor cell.
            Defaults to 200.
        resolution_level: the level at which the ground truth is made.
            Defaults to 5.
        eval_thresholds: the false positive rates for calculating the average sensitivity.
            Defaults to (0.25, 0.5, 1, 2, 4, 8) which is the same as the CAMELYON 16 Challenge.
    """

    def __init__(
        self,
        pixel_size: float = 0.243,
        annotation_size: int = 75,
        itc_diameter_threshold: int = 200,
        resolution_level: int = 5,
        eval_thresholds: Tuple = (0.25, 0.5, 1, 2, 4, 8),
    ) -> None:

        self.pixel_size = pixel_size
        self.annotation_size = annotation_size
        self.itc_diameter_threshold = itc_diameter_threshold
        self.resolution_level = resolution_level
        self.eval_thresholds = eval_thresholds

    def __call__(
        self,
        datalist_path: Optional[str] = None,
        prediction_key: str = "prediction",
        ground_truth_key: str = "ground_truth",
        prediciton_dir: Optional[str] = None,
        ground_truth_dir: Optional[str] = None,
    ):
        """
        Args:
            datalist_path: the path of the datalist in the form of a json file.
                The file should be a list of dictionaries, where each dictionary
                contains the prediction and ground truth paths of one image.
                Each prediction path should point to a csv file,
                and each ground truth path should point to a tif file.
                Defaults to None.
            prediction_key: the key of predictions in datalist dictionaries.
                Defaults to `"prediction"`.
            ground_truth_key: the key of ground truths in datalist dictionaries.
                Defaults to `"ground_truth"`.
            prediciton_dir: if the datalist json file is not provided,
                the directory of all prediction csv files should be provided.
                Defaults to None.
            ground_truth_dir: if the datalist json file is not provided,
                the directory of all ground truths tif files should be provided.
                Defaults to None.
        """
        datalist = load_datalist(
            datalist_path=datalist_path,
            prediction_key=prediction_key,
            ground_truth_key=ground_truth_key,
            prediciton_dir=prediciton_dir,
            ground_truth_dir=ground_truth_dir,
        )
        case_num = 0
        total_fp_probs, total_tp_probs = [], []
        total_num_targets = 0

        for data in datalist:
            prediction_path = data[prediction_key]
            ground_truth_path = data[ground_truth_key]

            # get prediction results
            probs, x_coord, y_coord = get_pred_result(prediction_path)
            # get evaluation mask
            evaluation_mask = compute_eval_mask(
                ground_truth_path=ground_truth_path,
                pixel_size=self.pixel_size,
                annotation_size=self.annotation_size,
                resolution_level=self.resolution_level,
            )
            # get itc list
            itc_list = compute_itc_list(
                evaluation_mask=evaluation_mask,
                pixel_size=self.pixel_size,
                annotation_size=self.annotation_size,
                resolution_level=self.resolution_level,
                itc_diameter_threshold=self.itc_diameter_threshold,
            )
            # get fp, tp
            fp_probs, tp_probs, num_targets = compute_fp_tp_probs(
                probs=probs,
                y_coord=y_coord,
                x_coord=x_coord,
                evaluation_mask=evaluation_mask,
                labels_to_exclude=itc_list,
                resolution_level=self.resolution_level,
            )
            total_fp_probs.append(fp_probs)
            total_tp_probs.append(tp_probs)
            total_num_targets += num_targets
            case_num += 1

        fps_per_image, total_sensitivity = compute_froc_curve_data(
            fp_probs=np.concatenate(total_fp_probs),
            tp_probs=np.concatenate(total_tp_probs),
            num_targets=total_num_targets,
            num_images=case_num,
        )

        froc_score = compute_froc_score(
            fps_per_image=fps_per_image,
            total_sensitivity=total_sensitivity,
            eval_thresholds=self.eval_thresholds,
        )

        return froc_score
