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

from typing import TYPE_CHECKING, Dict, List, Tuple

import numpy as np

from monai.apps.pathology.utils import PathologyProbNMS, compute_isolated_tumor_cells, compute_multi_instance_mask
from monai.data.image_reader import WSIReader
from monai.metrics import compute_fp_tp_probs, compute_froc_curve_data, compute_froc_score
from monai.utils import min_version, optional_import

if TYPE_CHECKING:
    from tqdm import tqdm

    has_tqdm = True
else:
    tqdm, has_tqdm = optional_import("tqdm", "4.47.0", min_version, "tqdm")

if not has_tqdm:

    def tqdm(x):
        return x


class LesionFROC:
    """
    Evaluate with Free Response Operating Characteristic (FROC) score.

    Args:
        data: either the list of dictionaries containing probability maps (inference result) and
            tumor mask (ground truth), as below, or the path to a json file containing such list.
            `{
            "prob_map": "path/to/prob_map_1.npy",
            "tumor_mask": "path/to/ground_truth_1.tiff",
            "level": 6,
            "pixel_spacing": 0.243
            }`
        grow_distance: Euclidean distance (in micrometer) by which to grow the label the ground truth's tumors.
            Defaults to 75, which is the equivalent size of 5 tumor cells.
        itc_diameter: the maximum diameter of a region (in micrometer) to be considered as an isolated tumor cell.
            Defaults to 200.
        eval_thresholds: the false positive rates for calculating the average sensitivity.
            Defaults to (0.25, 0.5, 1, 2, 4, 8) which is the same as the CAMELYON 16 Challenge.
        nms_sigma: the standard deviation for gaussian filter of non-maximal suppression. Defaults to 0.0.
        nms_prob_threshold: the probability threshold of non-maximal suppression. Defaults to 0.5.
        nms_box_size: the box size (in pixel) to be removed around the pixel for non-maximal suppression.
        image_reader_name: the name of library to be used for loading whole slide imaging, either CuCIM or OpenSlide.
            Defaults to CuCIM.

    Note:
        For more info on `nms_*` parameters look at monai.utils.prob_nms.ProbNMS`.

    """

    def __init__(
        self,
        data: List[Dict],
        grow_distance: int = 75,
        itc_diameter: int = 200,
        eval_thresholds: Tuple = (0.25, 0.5, 1, 2, 4, 8),
        nms_sigma: float = 0.0,
        nms_prob_threshold: float = 0.5,
        nms_box_size: int = 48,
        image_reader_name: str = "cuCIM",
    ) -> None:

        self.data = data
        self.grow_distance = grow_distance
        self.itc_diameter = itc_diameter
        self.eval_thresholds = eval_thresholds
        self.image_reader = WSIReader(image_reader_name)
        self.nms = PathologyProbNMS(sigma=nms_sigma, prob_threshold=nms_prob_threshold, box_size=nms_box_size)

    def prepare_inference_result(self, sample: Dict):
        """
        Prepare the probability map for detection evaluation.

        """
        # load the probability map (the result of model inference)
        prob_map = np.load(sample["prob_map"])

        # apply non-maximal suppression
        nms_outputs = self.nms(probs_map=prob_map, resolution_level=sample["level"])

        # separate nms outputs
        if nms_outputs:
            probs, x_coord, y_coord = zip(*nms_outputs)
        else:
            probs, x_coord, y_coord = [], [], []

        return np.array(probs), np.array(x_coord), np.array(y_coord)

    def prepare_ground_truth(self, sample):
        """
        Prepare the ground truth for evaluation based on the binary tumor mask

        """
        # load binary tumor masks
        img_obj = self.image_reader.read(sample["tumor_mask"])
        tumor_mask = self.image_reader.get_data(img_obj, level=sample["level"])[0][0]

        # calculate pixel spacing at the mask level
        mask_pixel_spacing = sample["pixel_spacing"] * pow(2, sample["level"])

        # compute multi-instance mask from a binary mask
        grow_pixel_threshold = self.grow_distance / (mask_pixel_spacing * 2)
        tumor_mask = compute_multi_instance_mask(mask=tumor_mask, threshold=grow_pixel_threshold)

        # identify isolated tumor cells
        itc_threshold = (self.itc_diameter + self.grow_distance) / mask_pixel_spacing
        itc_labels = compute_isolated_tumor_cells(tumor_mask=tumor_mask, threshold=itc_threshold)

        return tumor_mask, itc_labels

    def compute_fp_tp(self):
        """
        Compute false positive and true positive probabilities for tumor detection,
        by comparing the model outputs with the prepared ground truths for all samples

        """
        total_fp_probs, total_tp_probs = [], []
        total_num_targets = 0
        num_images = len(self.data)

        for sample in tqdm(self.data):
            probs, y_coord, x_coord = self.prepare_inference_result(sample)
            ground_truth, itc_labels = self.prepare_ground_truth(sample)
            # compute FP and TP probabilities for a pair of an image and an ground truth mask
            fp_probs, tp_probs, num_targets = compute_fp_tp_probs(
                probs=probs,
                y_coord=y_coord,
                x_coord=x_coord,
                evaluation_mask=ground_truth,
                labels_to_exclude=itc_labels,
                resolution_level=sample["level"],
            )
            total_fp_probs.extend(fp_probs)
            total_tp_probs.extend(tp_probs)
            total_num_targets += num_targets

        return np.array(total_fp_probs), np.array(total_tp_probs), total_num_targets, num_images

    def evaluate(self):
        """
        Evaluate the detection performance of a model based on the model probability map output,
        the ground truth tumor mask, and their associated metadata (e.g., pixel_spacing, level)
        """
        # compute false positive (FP) and true positive (TP) probabilities for all images
        fp_probs, tp_probs, num_targets, num_images = self.compute_fp_tp()

        # compute FROC curve given the evaluation of all images
        fps_per_image, total_sensitivity = compute_froc_curve_data(
            fp_probs=fp_probs, tp_probs=tp_probs, num_targets=num_targets, num_images=num_images
        )

        # compute FROC score give specific evaluation threshold
        froc_score = compute_froc_score(
            fps_per_image=fps_per_image, total_sensitivity=total_sensitivity, eval_thresholds=self.eval_thresholds
        )

        return froc_score
