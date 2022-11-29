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

from typing import Dict

__all__ = ["TCIA_LABEL_DICT"]

TCIA_LABEL_DICT: Dict[str, Dict[str, int]] = {
    "C4KC-KiTS": {"Kidney": 0, "Renal Tumor": 1},
    "NSCLC-Radiomics": {
        "Esophagus": 0,
        "GTV-1": 1,
        "Lungs-Total": 2,
        "Spinal-Cord": 3,
        "Lung-Left": 4,
        "Lung-Right": 5,
        "Heart": 6,
    },
    "NSCLC-Radiomics-Interobserver1": {
        "GTV-1auto-1": 0,
        "GTV-1auto-2": 1,
        "GTV-1auto-3": 2,
        "GTV-1auto-4": 3,
        "GTV-1auto-5": 4,
        "GTV-1vis-1": 5,
        "GTV-1vis-2": 6,
        "GTV-1vis-3": 7,
        "GTV-1vis-4": 8,
        "GTV-1vis-5": 9,
    },
    "QIN-PROSTATE-Repeatability": {"NormalROI_PZ_1": 0, "TumorROI_PZ_1": 1, "PeripheralZone": 2, "WholeGland": 3},
    "PROSTATEx": {
        "Prostate": 0,
        "Peripheral zone of prostate": 1,
        "Transition zone of prostate": 2,
        "Distal prostatic urethra": 3,
        "Anterior fibromuscular stroma of prostate": 4,
    },
}
