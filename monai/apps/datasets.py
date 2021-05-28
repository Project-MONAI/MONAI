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

import os
import sys
from typing import Callable, Dict, List, Optional, Sequence, Union

import numpy as np

from monai.apps.utils import download_and_extract
from monai.data import (
    CacheDataset,
    load_decathlon_datalist,
    load_decathlon_properties,
    partition_dataset,
    select_cross_validation_folds,
)
from monai.transforms import LoadImaged, Randomizable
from monai.utils import ensure_tuple

__all__ = ["MedNISTDataset", "DecathlonDataset", "CrossValidation"]


class MedNISTDataset(Randomizable, CacheDataset):
    """
    The Dataset to automatically download MedNIST data and generate items for training, validation or test.
    It's based on `CacheDataset` to accelerate the training process.

    Args:
        root_dir: target directory to download and load MedNIST dataset.
        section: expected data section, can be: `training`, `validation` or `test`.
        transform: transforms to execute operations on input data.
        download: whether to download and extract the MedNIST from resource link, default is False.
            if expected file already exists, skip downloading even set it to True.
            user can manually copy `MedNIST.tar.gz` file or `MedNIST` folder to root directory.
        seed: random seed to randomly split training, validation and test datasets, default is 0.
        val_frac: percentage of of validation fraction in the whole dataset, default is 0.1.
        test_frac: percentage of of test fraction in the whole dataset, default is 0.1.
        cache_num: number of items to be cached. Default is `sys.maxsize`.
            will take the minimum of (cache_num, data_length x cache_rate, data_length).
        cache_rate: percentage of cached data in total, default is 1.0 (cache all).
            will take the minimum of (cache_num, data_length x cache_rate, data_length).
        num_workers: the number of worker threads to use.
            if 0 a single thread will be used. Default is 0.

    Raises:
        ValueError: When ``root_dir`` is not a directory.
        RuntimeError: When ``dataset_dir`` doesn't exist and downloading is not selected (``download=False``).

    """

    resource = "https://drive.google.com/uc?id=13MhoPsNgI6qboJfLicFf_jNvsFUbIYsd"
    md5 = "0bc7306e7427e00ad1c5526a6677552d"
    compressed_file_name = "MedNIST.tar.gz"
    dataset_folder_name = "MedNIST"

    def __init__(
        self,
        root_dir: str,
        section: str,
        transform: Union[Sequence[Callable], Callable] = (),
        download: bool = False,
        seed: int = 0,
        val_frac: float = 0.1,
        test_frac: float = 0.1,
        cache_num: int = sys.maxsize,
        cache_rate: float = 1.0,
        num_workers: int = 0,
    ) -> None:
        if not os.path.isdir(root_dir):
            raise ValueError("Root directory root_dir must be a directory.")
        self.section = section
        self.val_frac = val_frac
        self.test_frac = test_frac
        self.set_random_state(seed=seed)
        tarfile_name = os.path.join(root_dir, self.compressed_file_name)
        dataset_dir = os.path.join(root_dir, self.dataset_folder_name)
        self.num_class = 0
        if download:
            download_and_extract(self.resource, tarfile_name, root_dir, self.md5)

        if not os.path.exists(dataset_dir):
            raise RuntimeError(
                f"Cannot find dataset directory: {dataset_dir}, please use download=True to download it."
            )
        data = self._generate_data_list(dataset_dir)
        if transform == ():
            transform = LoadImaged("image")
        CacheDataset.__init__(
            self, data, transform, cache_num=cache_num, cache_rate=cache_rate, num_workers=num_workers
        )

    def randomize(self, data: List[int]) -> None:
        self.R.shuffle(data)

    def get_num_classes(self) -> int:
        """Get number of classes."""
        return self.num_class

    def _generate_data_list(self, dataset_dir: str) -> List[Dict]:
        """
        Raises:
            ValueError: When ``section`` is not one of ["training", "validation", "test"].

        """
        class_names = sorted((x for x in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, x))))
        self.num_class = len(class_names)
        image_files = [
            [
                os.path.join(dataset_dir, class_names[i], x)
                for x in os.listdir(os.path.join(dataset_dir, class_names[i]))
            ]
            for i in range(self.num_class)
        ]
        num_each = [len(image_files[i]) for i in range(self.num_class)]
        image_files_list = []
        image_class = []
        class_name = []
        for i in range(self.num_class):
            image_files_list.extend(image_files[i])
            image_class.extend([i] * num_each[i])
            class_name.extend([class_names[i]] * num_each[i])

        data = []
        length = len(image_files_list)
        indices = np.arange(length)
        self.randomize(indices)

        test_length = int(length * self.test_frac)
        val_length = int(length * self.val_frac)
        if self.section == "test":
            section_indices = indices[:test_length]
        elif self.section == "validation":
            section_indices = indices[test_length : test_length + val_length]
        elif self.section == "training":
            section_indices = indices[test_length + val_length :]
        else:
            raise ValueError(
                f'Unsupported section: {self.section}, available options are ["training", "validation", "test"].'
            )

        for i in section_indices:
            data.append({"image": image_files_list[i], "label": image_class[i], "class_name": class_name[i]})

        return data


class DecathlonDataset(Randomizable, CacheDataset):
    """
    The Dataset to automatically download the data of Medical Segmentation Decathlon challenge
    (http://medicaldecathlon.com/) and generate items for training, validation or test.
    It will also load these properties from the JSON config file of dataset. user can call `get_properties()`
    to get specified properties or all the properties loaded.
    It's based on :py:class:`monai.data.CacheDataset` to accelerate the training process.

    Args:
        root_dir: user's local directory for caching and loading the MSD datasets.
        task: which task to download and execute: one of list ("Task01_BrainTumour", "Task02_Heart",
            "Task03_Liver", "Task04_Hippocampus", "Task05_Prostate", "Task06_Lung", "Task07_Pancreas",
            "Task08_HepaticVessel", "Task09_Spleen", "Task10_Colon").
        section: expected data section, can be: `training`, `validation` or `test`.
        transform: transforms to execute operations on input data.
            for further usage, use `AddChanneld` or `AsChannelFirstd` to convert the shape to [C, H, W, D].
        download: whether to download and extract the Decathlon from resource link, default is False.
            if expected file already exists, skip downloading even set it to True.
        val_frac: percentage of of validation fraction in the whole dataset, default is 0.2.
            user can manually copy tar file or dataset folder to the root directory.
        seed: random seed to randomly shuffle the datalist before splitting into training and validation, default is 0.
            note to set same seed for `training` and `validation` sections.
        cache_num: number of items to be cached. Default is `sys.maxsize`.
            will take the minimum of (cache_num, data_length x cache_rate, data_length).
        cache_rate: percentage of cached data in total, default is 1.0 (cache all).
            will take the minimum of (cache_num, data_length x cache_rate, data_length).
        num_workers: the number of worker threads to use.
            if 0 a single thread will be used. Default is 0.

    Raises:
        ValueError: When ``root_dir`` is not a directory.
        ValueError: When ``task`` is not one of ["Task01_BrainTumour", "Task02_Heart",
            "Task03_Liver", "Task04_Hippocampus", "Task05_Prostate", "Task06_Lung", "Task07_Pancreas",
            "Task08_HepaticVessel", "Task09_Spleen", "Task10_Colon"].
        RuntimeError: When ``dataset_dir`` doesn't exist and downloading is not selected (``download=False``).

    Example::

        transform = Compose(
            [
                LoadImaged(keys=["image", "label"]),
                AddChanneld(keys=["image", "label"]),
                ScaleIntensityd(keys="image"),
                ToTensord(keys=["image", "label"]),
            ]
        )

        val_data = DecathlonDataset(
            root_dir="./", task="Task09_Spleen", transform=transform, section="validation", seed=12345, download=True
        )

        print(val_data[0]["image"], val_data[0]["label"])

    """

    resource = {
        "Task01_BrainTumour": "https://msd-for-monai.s3-us-west-2.amazonaws.com/Task01_BrainTumour.tar",
        "Task02_Heart": "https://msd-for-monai.s3-us-west-2.amazonaws.com/Task02_Heart.tar",
        "Task03_Liver": "https://msd-for-monai.s3-us-west-2.amazonaws.com/Task03_Liver.tar",
        "Task04_Hippocampus": "https://msd-for-monai.s3-us-west-2.amazonaws.com/Task04_Hippocampus.tar",
        "Task05_Prostate": "https://msd-for-monai.s3-us-west-2.amazonaws.com/Task05_Prostate.tar",
        "Task06_Lung": "https://msd-for-monai.s3-us-west-2.amazonaws.com/Task06_Lung.tar",
        "Task07_Pancreas": "https://msd-for-monai.s3-us-west-2.amazonaws.com/Task07_Pancreas.tar",
        "Task08_HepaticVessel": "https://msd-for-monai.s3-us-west-2.amazonaws.com/Task08_HepaticVessel.tar",
        "Task09_Spleen": "https://msd-for-monai.s3-us-west-2.amazonaws.com/Task09_Spleen.tar",
        "Task10_Colon": "https://msd-for-monai.s3-us-west-2.amazonaws.com/Task10_Colon.tar",
    }
    md5 = {
        "Task01_BrainTumour": "240a19d752f0d9e9101544901065d872",
        "Task02_Heart": "06ee59366e1e5124267b774dbd654057",
        "Task03_Liver": "a90ec6c4aa7f6a3d087205e23d4e6397",
        "Task04_Hippocampus": "9d24dba78a72977dbd1d2e110310f31b",
        "Task05_Prostate": "35138f08b1efaef89d7424d2bcc928db",
        "Task06_Lung": "8afd997733c7fc0432f71255ba4e52dc",
        "Task07_Pancreas": "4f7080cfca169fa8066d17ce6eb061e4",
        "Task08_HepaticVessel": "641d79e80ec66453921d997fbf12a29c",
        "Task09_Spleen": "410d4a301da4e5b2f6f86ec3ddba524e",
        "Task10_Colon": "bad7a188931dc2f6acf72b08eb6202d0",
    }

    def __init__(
        self,
        root_dir: str,
        task: str,
        section: str,
        transform: Union[Sequence[Callable], Callable] = (),
        download: bool = False,
        seed: int = 0,
        val_frac: float = 0.2,
        cache_num: int = sys.maxsize,
        cache_rate: float = 1.0,
        num_workers: int = 0,
    ) -> None:
        if not os.path.isdir(root_dir):
            raise ValueError("Root directory root_dir must be a directory.")
        self.section = section
        self.val_frac = val_frac
        self.set_random_state(seed=seed)
        if task not in self.resource:
            raise ValueError(f"Unsupported task: {task}, available options are: {list(self.resource.keys())}.")
        dataset_dir = os.path.join(root_dir, task)
        tarfile_name = f"{dataset_dir}.tar"
        if download:
            download_and_extract(self.resource[task], tarfile_name, root_dir, self.md5[task])

        if not os.path.exists(dataset_dir):
            raise RuntimeError(
                f"Cannot find dataset directory: {dataset_dir}, please use download=True to download it."
            )
        self.indices: np.ndarray = np.array([])
        data = self._generate_data_list(dataset_dir)
        # as `release` key has typo in Task04 config file, ignore it.
        property_keys = [
            "name",
            "description",
            "reference",
            "licence",
            "tensorImageSize",
            "modality",
            "labels",
            "numTraining",
            "numTest",
        ]
        self._properties = load_decathlon_properties(os.path.join(dataset_dir, "dataset.json"), property_keys)
        if transform == ():
            transform = LoadImaged(["image", "label"])
        CacheDataset.__init__(
            self, data, transform, cache_num=cache_num, cache_rate=cache_rate, num_workers=num_workers
        )

    def get_indices(self) -> np.ndarray:
        """
        Get the indices of datalist used in this dataset.

        """
        return self.indices

    def randomize(self, data: List[int]) -> None:
        self.R.shuffle(data)

    def get_properties(self, keys: Optional[Union[Sequence[str], str]] = None):
        """
        Get the loaded properties of dataset with specified keys.
        If no keys specified, return all the loaded properties.

        """
        if keys is None:
            return self._properties
        if self._properties is not None:
            return {key: self._properties[key] for key in ensure_tuple(keys)}
        return {}

    def _generate_data_list(self, dataset_dir: str) -> List[Dict]:
        section = "training" if self.section in ["training", "validation"] else "test"
        datalist = load_decathlon_datalist(os.path.join(dataset_dir, "dataset.json"), True, section)
        return self._split_datalist(datalist)

    def _split_datalist(self, datalist: List[Dict]) -> List[Dict]:
        if self.section == "test":
            return datalist
        length = len(datalist)
        indices = np.arange(length)
        self.randomize(indices)

        val_length = int(length * self.val_frac)
        if self.section == "training":
            self.indices = indices[val_length:]
        else:
            self.indices = indices[:val_length]

        return [datalist[i] for i in self.indices]


class CrossValidation:
    """
    Cross validation dataset based on the general dataset which must have `_split_datalist` API.

    Args:
        dataset_cls: dataset class to be used to create the cross validation partitions.
            It must have `_split_datalist` API.
        nfolds: number of folds to split the data for cross validation.
        seed: random seed to randomly shuffle the datalist before splitting into N folds, default is 0.
        dataset_params: other additional parameters for the dataset_cls base class.

    Example of 5 folds cross validation training::

        cvdataset = CrossValidation(
            dataset_cls=DecathlonDataset,
            nfolds=5,
            seed=12345,
            root_dir="./",
            task="Task09_Spleen",
            section="training",
            download=True,
        )
        dataset_fold0_train = cvdataset.get_dataset(folds=[1, 2, 3, 4])
        dataset_fold0_val = cvdataset.get_dataset(folds=0)
        # execute training for fold 0 ...

        dataset_fold1_train = cvdataset.get_dataset(folds=[1])
        dataset_fold1_val = cvdataset.get_dataset(folds=[0, 2, 3, 4])
        # execute training for fold 1 ...

        ...

        dataset_fold4_train = ...
        # execute training for fold 4 ...

    """

    def __init__(
        self,
        dataset_cls,
        nfolds: int = 5,
        seed: int = 0,
        **dataset_params,
    ) -> None:
        if not hasattr(dataset_cls, "_split_datalist"):
            raise ValueError("dataset class must have _split_datalist API.")
        self.dataset_cls = dataset_cls
        self.nfolds = nfolds
        self.seed = seed
        self.dataset_params = dataset_params

    def get_dataset(self, folds: Union[Sequence[int], int]):
        """
        Generate dataset based on the specified fold indice in the cross validation group.

        Args:
            folds: index of folds for training or validation, if a list of values, concatenate the data.

        """
        nfolds = self.nfolds
        seed = self.seed

        class _NsplitsDataset(self.dataset_cls):  # type: ignore
            def _split_datalist(self, datalist: List[Dict]) -> List[Dict]:
                data = partition_dataset(data=datalist, num_partitions=nfolds, shuffle=True, seed=seed)
                return select_cross_validation_folds(partitions=data, folds=folds)

        return _NsplitsDataset(**self.dataset_params)
