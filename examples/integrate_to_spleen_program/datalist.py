# Copyright 2020 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

datalist = {
    "description": "Spleen Segmentation",
    "labels": {
        "0": "background",
        "1": "spleen"
    },
    "licence": "CC-BY-SA 4.0",
    "modality": {
        "0": "CT"
    },
    "name": "Spleen",
    "numTest": 20,
    "numTraining": 41,
    "reference": "Memorial Sloan Kettering Cancer Center",
    "release": "1.0 06/08/2018",
    "tensorImageSize": "3D",
    "training": [
        {
            "image": "/workspace/data/medical/spleen/imagesTr/spleen_29.nii",
            "label": "/workspace/data/medical/spleen/labelsTr/spleen_29.nii"
        },
        {
            "image": "/workspace/data/medical/spleen/imagesTr/spleen_46.nii",
            "label": "/workspace/data/medical/spleen/labelsTr/spleen_46.nii"
        },
        {
            "image": "/workspace/data/medical/spleen/imagesTr/spleen_25.nii",
            "label": "/workspace/data/medical/spleen/labelsTr/spleen_25.nii"
        },
        {
            "image": "/workspace/data/medical/spleen/imagesTr/spleen_13.nii",
            "label": "/workspace/data/medical/spleen/labelsTr/spleen_13.nii"
        },
        {
            "image": "/workspace/data/medical/spleen/imagesTr/spleen_62.nii",
            "label": "/workspace/data/medical/spleen/labelsTr/spleen_62.nii"
        },
        {
            "image": "/workspace/data/medical/spleen/imagesTr/spleen_27.nii",
            "label": "/workspace/data/medical/spleen/labelsTr/spleen_27.nii"
        },
        {
            "image": "/workspace/data/medical/spleen/imagesTr/spleen_44.nii",
            "label": "/workspace/data/medical/spleen/labelsTr/spleen_44.nii"
        },
        {
            "image": "/workspace/data/medical/spleen/imagesTr/spleen_56.nii",
            "label": "/workspace/data/medical/spleen/labelsTr/spleen_56.nii"
        },
        {
            "image": "/workspace/data/medical/spleen/imagesTr/spleen_60.nii",
            "label": "/workspace/data/medical/spleen/labelsTr/spleen_60.nii"
        },
        {
            "image": "/workspace/data/medical/spleen/imagesTr/spleen_2.nii",
            "label": "/workspace/data/medical/spleen/labelsTr/spleen_2.nii"
        },
        {
            "image": "/workspace/data/medical/spleen/imagesTr/spleen_53.nii",
            "label": "/workspace/data/medical/spleen/labelsTr/spleen_53.nii"
        },
        {
            "image": "/workspace/data/medical/spleen/imagesTr/spleen_41.nii",
            "label": "/workspace/data/medical/spleen/labelsTr/spleen_41.nii"
        },
        {
            "image": "/workspace/data/medical/spleen/imagesTr/spleen_22.nii",
            "label": "/workspace/data/medical/spleen/labelsTr/spleen_22.nii"
        },
        {
            "image": "/workspace/data/medical/spleen/imagesTr/spleen_14.nii",
            "label": "/workspace/data/medical/spleen/labelsTr/spleen_14.nii"
        },
        {
            "image": "/workspace/data/medical/spleen/imagesTr/spleen_18.nii",
            "label": "/workspace/data/medical/spleen/labelsTr/spleen_18.nii"
        },
        {
            "image": "/workspace/data/medical/spleen/imagesTr/spleen_20.nii",
            "label": "/workspace/data/medical/spleen/labelsTr/spleen_20.nii"
        },
        {
            "image": "/workspace/data/medical/spleen/imagesTr/spleen_32.nii",
            "label": "/workspace/data/medical/spleen/labelsTr/spleen_32.nii"
        },
        {
            "image": "/workspace/data/medical/spleen/imagesTr/spleen_16.nii",
            "label": "/workspace/data/medical/spleen/labelsTr/spleen_16.nii"
        },
        {
            "image": "/workspace/data/medical/spleen/imagesTr/spleen_12.nii",
            "label": "/workspace/data/medical/spleen/labelsTr/spleen_12.nii"
        },
        {
            "image": "/workspace/data/medical/spleen/imagesTr/spleen_63.nii",
            "label": "/workspace/data/medical/spleen/labelsTr/spleen_63.nii"
        },
        {
            "image": "/workspace/data/medical/spleen/imagesTr/spleen_28.nii",
            "label": "/workspace/data/medical/spleen/labelsTr/spleen_28.nii"
        },
        {
            "image": "/workspace/data/medical/spleen/imagesTr/spleen_24.nii",
            "label": "/workspace/data/medical/spleen/labelsTr/spleen_24.nii"
        },
        {
            "image": "/workspace/data/medical/spleen/imagesTr/spleen_59.nii",
            "label": "/workspace/data/medical/spleen/labelsTr/spleen_59.nii"
        },
        {
            "image": "/workspace/data/medical/spleen/imagesTr/spleen_47.nii",
            "label": "/workspace/data/medical/spleen/labelsTr/spleen_47.nii"
        },
        {
            "image": "/workspace/data/medical/spleen/imagesTr/spleen_8.nii",
            "label": "/workspace/data/medical/spleen/labelsTr/spleen_8.nii"
        },
        {
            "image": "/workspace/data/medical/spleen/imagesTr/spleen_6.nii",
            "label": "/workspace/data/medical/spleen/labelsTr/spleen_6.nii"
        },
        {
            "image": "/workspace/data/medical/spleen/imagesTr/spleen_61.nii",
            "label": "/workspace/data/medical/spleen/labelsTr/spleen_61.nii"
        },
        {
            "image": "/workspace/data/medical/spleen/imagesTr/spleen_10.nii",
            "label": "/workspace/data/medical/spleen/labelsTr/spleen_10.nii"
        },
        {
            "image": "/workspace/data/medical/spleen/imagesTr/spleen_38.nii",
            "label": "/workspace/data/medical/spleen/labelsTr/spleen_38.nii"
        },
        {
            "image": "/workspace/data/medical/spleen/imagesTr/spleen_45.nii",
            "label": "/workspace/data/medical/spleen/labelsTr/spleen_45.nii"
        },
        {
            "image": "/workspace/data/medical/spleen/imagesTr/spleen_26.nii",
            "label": "/workspace/data/medical/spleen/labelsTr/spleen_26.nii"
        },
        {
            "image": "/workspace/data/medical/spleen/imagesTr/spleen_49.nii",
            "label": "/workspace/data/medical/spleen/labelsTr/spleen_49.nii"
        }
    ],
    "validation": [
        {
            "image": "/workspace/data/medical/spleen/imagesTr/spleen_19.nii",
            "label": "/workspace/data/medical/spleen/labelsTr/spleen_19.nii"
        },
        {
            "image": "/workspace/data/medical/spleen/imagesTr/spleen_31.nii",
            "label": "/workspace/data/medical/spleen/labelsTr/spleen_31.nii"
        },
        {
            "image": "/workspace/data/medical/spleen/imagesTr/spleen_52.nii",
            "label": "/workspace/data/medical/spleen/labelsTr/spleen_52.nii"
        },
        {
            "image": "/workspace/data/medical/spleen/imagesTr/spleen_40.nii",
            "label": "/workspace/data/medical/spleen/labelsTr/spleen_40.nii"
        },
        {
            "image": "/workspace/data/medical/spleen/imagesTr/spleen_3.nii",
            "label": "/workspace/data/medical/spleen/labelsTr/spleen_3.nii"
        },
        {
            "image": "/workspace/data/medical/spleen/imagesTr/spleen_17.nii",
            "label": "/workspace/data/medical/spleen/labelsTr/spleen_17.nii"
        },
        {
            "image": "/workspace/data/medical/spleen/imagesTr/spleen_21.nii",
            "label": "/workspace/data/medical/spleen/labelsTr/spleen_21.nii"
        },
        {
            "image": "/workspace/data/medical/spleen/imagesTr/spleen_33.nii",
            "label": "/workspace/data/medical/spleen/labelsTr/spleen_33.nii"
        },
        {
            "image": "/workspace/data/medical/spleen/imagesTr/spleen_9.nii",
            "label": "/workspace/data/medical/spleen/labelsTr/spleen_9.nii"
        }
    ],
    "test": [
        "/workspace/data/medical/spleen/imagesTs/spleen_15.nii",
        "/workspace/data/medical/spleen/imagesTs/spleen_23.nii",
        "/workspace/data/medical/spleen/imagesTs/spleen_1.nii",
        "/workspace/data/medical/spleen/imagesTs/spleen_42.nii",
        "/workspace/data/medical/spleen/imagesTs/spleen_50.nii",
        "/workspace/data/medical/spleen/imagesTs/spleen_54.nii",
        "/workspace/data/medical/spleen/imagesTs/spleen_37.nii",
        "/workspace/data/medical/spleen/imagesTs/spleen_58.nii",
        "/workspace/data/medical/spleen/imagesTs/spleen_39.nii",
        "/workspace/data/medical/spleen/imagesTs/spleen_48.nii",
        "/workspace/data/medical/spleen/imagesTs/spleen_35.nii",
        "/workspace/data/medical/spleen/imagesTs/spleen_11.nii",
        "/workspace/data/medical/spleen/imagesTs/spleen_7.nii",
        "/workspace/data/medical/spleen/imagesTs/spleen_30.nii",
        "/workspace/data/medical/spleen/imagesTs/spleen_43.nii",
        "/workspace/data/medical/spleen/imagesTs/spleen_51.nii",
        "/workspace/data/medical/spleen/imagesTs/spleen_36.nii",
        "/workspace/data/medical/spleen/imagesTs/spleen_55.nii",
        "/workspace/data/medical/spleen/imagesTs/spleen_57.nii",
        "/workspace/data/medical/spleen/imagesTs/spleen_34.nii"
    ]
}
