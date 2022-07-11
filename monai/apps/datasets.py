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
import shutil
import sys
import warnings
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from monai.apps.utils import (
    download_and_extract,
    download_tcia_series_instance,
    get_ref_uuid,
    get_tcia_metadata,
    match_ref_uid_in_study,
)
from monai.config.type_definitions import PathLike
from monai.data import (
    CacheDataset,
    PydicomReader,
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
            If num_workers is None then the number returned by os.cpu_count() is used.
            If a value less than 1 is speficied, 1 will be used instead.
        progress: whether to display a progress bar when downloading dataset and computing the transform cache content.
        copy_cache: whether to `deepcopy` the cache content before applying the random transforms,
            default to `True`. if the random transforms don't modify the cached content
            (for example, randomly crop from the cached image and deepcopy the crop region)
            or if every cache item is only used once in a `multi-processing` environment,
            may set `copy=False` for better performance.
        as_contiguous: whether to convert the cached NumPy array or PyTorch tensor to be contiguous.
            it may help improve the performance of following logic.

    Raises:
        ValueError: When ``root_dir`` is not a directory.
        RuntimeError: When ``dataset_dir`` doesn't exist and downloading is not selected (``download=False``).

    """

    resource = "https://github.com/Project-MONAI/MONAI-extra-test-data/releases/download/0.8.1/MedNIST.tar.gz"
    md5 = "0bc7306e7427e00ad1c5526a6677552d"
    compressed_file_name = "MedNIST.tar.gz"
    dataset_folder_name = "MedNIST"

    def __init__(
        self,
        root_dir: PathLike,
        section: str,
        transform: Union[Sequence[Callable], Callable] = (),
        download: bool = False,
        seed: int = 0,
        val_frac: float = 0.1,
        test_frac: float = 0.1,
        cache_num: int = sys.maxsize,
        cache_rate: float = 1.0,
        num_workers: Optional[int] = 1,
        progress: bool = True,
        copy_cache: bool = True,
        as_contiguous: bool = True,
    ) -> None:
        root_dir = Path(root_dir)
        if not root_dir.is_dir():
            raise ValueError("Root directory root_dir must be a directory.")
        self.section = section
        self.val_frac = val_frac
        self.test_frac = test_frac
        self.set_random_state(seed=seed)
        tarfile_name = root_dir / self.compressed_file_name
        dataset_dir = root_dir / self.dataset_folder_name
        self.num_class = 0
        if download:
            download_and_extract(
                url=self.resource,
                filepath=tarfile_name,
                output_dir=root_dir,
                hash_val=self.md5,
                hash_type="md5",
                progress=progress,
            )

        if not dataset_dir.is_dir():
            raise RuntimeError(
                f"Cannot find dataset directory: {dataset_dir}, please use download=True to download it."
            )
        data = self._generate_data_list(dataset_dir)
        if transform == ():
            transform = LoadImaged("image")
        CacheDataset.__init__(
            self,
            data=data,
            transform=transform,
            cache_num=cache_num,
            cache_rate=cache_rate,
            num_workers=num_workers,
            progress=progress,
            copy_cache=copy_cache,
            as_contiguous=as_contiguous,
        )

    def randomize(self, data: np.ndarray) -> None:
        self.R.shuffle(data)

    def get_num_classes(self) -> int:
        """Get number of classes."""
        return self.num_class

    def _generate_data_list(self, dataset_dir: PathLike) -> List[Dict]:
        """
        Raises:
            ValueError: When ``section`` is not one of ["training", "validation", "test"].

        """
        dataset_dir = Path(dataset_dir)
        class_names = sorted(f"{x.name}" for x in dataset_dir.iterdir() if x.is_dir())  # folder name as the class name
        self.num_class = len(class_names)
        image_files = [[f"{x}" for x in (dataset_dir / class_names[i]).iterdir()] for i in range(self.num_class)]
        num_each = [len(image_files[i]) for i in range(self.num_class)]
        image_files_list = []
        image_class = []
        class_name = []
        for i in range(self.num_class):
            image_files_list.extend(image_files[i])
            image_class.extend([i] * num_each[i])
            class_name.extend([class_names[i]] * num_each[i])

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
        # the types of label and class name should be compatible with the pytorch dataloader
        return [
            {"image": image_files_list[i], "label": image_class[i], "class_name": class_name[i]}
            for i in section_indices
        ]


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
            user can manually copy tar file or dataset folder to the root directory.
        val_frac: percentage of of validation fraction in the whole dataset, default is 0.2.
        seed: random seed to randomly shuffle the datalist before splitting into training and validation, default is 0.
            note to set same seed for `training` and `validation` sections.
        cache_num: number of items to be cached. Default is `sys.maxsize`.
            will take the minimum of (cache_num, data_length x cache_rate, data_length).
        cache_rate: percentage of cached data in total, default is 1.0 (cache all).
            will take the minimum of (cache_num, data_length x cache_rate, data_length).
        num_workers: the number of worker threads to use.
            If num_workers is None then the number returned by os.cpu_count() is used.
            If a value less than 1 is speficied, 1 will be used instead.
        progress: whether to display a progress bar when downloading dataset and computing the transform cache content.
        copy_cache: whether to `deepcopy` the cache content before applying the random transforms,
            default to `True`. if the random transforms don't modify the cached content
            (for example, randomly crop from the cached image and deepcopy the crop region)
            or if every cache item is only used once in a `multi-processing` environment,
            may set `copy=False` for better performance.
        as_contiguous: whether to convert the cached NumPy array or PyTorch tensor to be contiguous.
            it may help improve the performance of following logic.

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
        root_dir: PathLike,
        task: str,
        section: str,
        transform: Union[Sequence[Callable], Callable] = (),
        download: bool = False,
        seed: int = 0,
        val_frac: float = 0.2,
        cache_num: int = sys.maxsize,
        cache_rate: float = 1.0,
        num_workers: int = 1,
        progress: bool = True,
        copy_cache: bool = True,
        as_contiguous: bool = True,
    ) -> None:
        root_dir = Path(root_dir)
        if not root_dir.is_dir():
            raise ValueError("Root directory root_dir must be a directory.")
        self.section = section
        self.val_frac = val_frac
        self.set_random_state(seed=seed)
        if task not in self.resource:
            raise ValueError(f"Unsupported task: {task}, available options are: {list(self.resource.keys())}.")
        dataset_dir = root_dir / task
        tarfile_name = f"{dataset_dir}.tar"
        if download:
            download_and_extract(
                url=self.resource[task],
                filepath=tarfile_name,
                output_dir=root_dir,
                hash_val=self.md5[task],
                hash_type="md5",
                progress=progress,
            )

        if not dataset_dir.exists():
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
        self._properties = load_decathlon_properties(dataset_dir / "dataset.json", property_keys)
        if transform == ():
            transform = LoadImaged(["image", "label"])
        CacheDataset.__init__(
            self,
            data=data,
            transform=transform,
            cache_num=cache_num,
            cache_rate=cache_rate,
            num_workers=num_workers,
            progress=progress,
            copy_cache=copy_cache,
            as_contiguous=as_contiguous,
        )

    def get_indices(self) -> np.ndarray:
        """
        Get the indices of datalist used in this dataset.

        """
        return self.indices

    def randomize(self, data: np.ndarray) -> None:
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

    def _generate_data_list(self, dataset_dir: PathLike) -> List[Dict]:
        # the types of the item in data list should be compatible with the dataloader
        dataset_dir = Path(dataset_dir)
        section = "training" if self.section in ["training", "validation"] else "test"
        datalist = load_decathlon_datalist(dataset_dir / "dataset.json", True, section)
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


class TciaDataset(Randomizable, CacheDataset):
    """
    The Dataset to automatically download the data from a public The Cancer Imaging Archive (TCIA) dataset
    and generate items for training, validation or test.

    This class is based on :py:class:`monai.data.CacheDataset` to accelerate the training process.

    Args:
        root_dir: user's local directory for caching and loading the TCIA dataset.
        collection: a TCIA dataset is defined as a collection. Please check the following list to browse
            the collection list (only public collections can be downloaded):
            https://www.cancerimagingarchive.net/collections/
        section: expected data section, can be: `training`, `validation` or `test`.
        transform: transforms to execute operations on input data.
            for further usage, use `AddChanneld` or `AsChannelFirstd` to convert the shape to [C, H, W, D].
        download: whether to download and extract the dataset, default is False.
            if expected file already exists, skip downloading even set it to True.
            user can manually copy tar file or dataset folder to the root directory.
        download_len: number of series that will be downloaded, the value should be larger than 0 or -1, where -1 means
            all series will be downloaded. Default is -1.
        picked_modality: modality that is used to do the first step download. Default is "SEG".
        modality_tag: tag of modality. Default is (0x0008, 0x0060).
        ref_series_uid_tag: tag of referenced Series Instance UID. Default is (0x0020, 0x000e).
        ref_sop_uid_tag: tag of referenced SOP Instance UID. Default is (0x0008, 0x1155).
        specific_tags: tags that will be loaded for "SEG" series. This argument will be used in
            `monai.data.PydicomReader`. Default is [(0x0008, 0x1115), (0x0008,0x1140), (0x3006, 0x0010),
            (0x0020,0x000D), (0x0010,0x0010), (0x0010,0x0020), (0x0020,0x0011), (0x0020,0x0012)].
        val_frac: percentage of of validation fraction in the whole dataset, default is 0.2.
        seed: random seed to randomly shuffle the datalist before splitting into training and validation, default is 0.
            note to set same seed for `training` and `validation` sections.
        cache_num: number of items to be cached. Default is `sys.maxsize`.
            will take the minimum of (cache_num, data_length x cache_rate, data_length).
        cache_rate: percentage of cached data in total, default is 1.0 (cache all).
            will take the minimum of (cache_num, data_length x cache_rate, data_length).
        num_workers: the number of worker threads to use.
            If num_workers is None then the number returned by os.cpu_count() is used.
            If a value less than 1 is speficied, 1 will be used instead.
        progress: whether to display a progress bar when downloading dataset and computing the transform cache content.
        copy_cache: whether to `deepcopy` the cache content before applying the random transforms,
            default to `True`. if the random transforms don't modify the cached content
            (for example, randomly crop from the cached image and deepcopy the crop region)
            or if every cache item is only used once in a `multi-processing` environment,
            may set `copy=False` for better performance.
        as_contiguous: whether to convert the cached NumPy array or PyTorch tensor to be contiguous.
            it may help improve the performance of following logic.

    Example::

        # collection is "C4KC-KiTS", picked_modality is "SEG"
        data = TciaDataset(
            root_dir="./", collection="C4KC-KiTS", section="validation", seed=12345, download=True
        )
        # collection is "Pancreatic-CT-CBCT-SEG", picked_modality is "RTSTRUCT"
        data = TciaDataset(
            root_dir="./", collection="Pancreatic-CT-CBCT-SEG", picked_modality="RTSTRUCT", download=True
        )

        print(data[0]["image"])

    """

    def __init__(
        self,
        root_dir: PathLike,
        collection: str,
        section: str,
        transform: Union[Sequence[Callable], Callable] = (),
        download: bool = False,
        download_len: int = -1,
        picked_modality: str = "SEG",
        modality_tag: Tuple = (0x0008, 0x0060),
        ref_series_uid_tag: Tuple = (0x0020, 0x000E),
        ref_sop_uid_tag: Tuple = (0x0008, 0x1155),
        specific_tags: Tuple = (
            (0x0008, 0x1115),
            (0x0008, 0x1140),
            (0x3006, 0x0010),
            (0x0020, 0x000D),
            (0x0010, 0x0010),
            (0x0010, 0x0020),
            (0x0020, 0x0011),
            (0x0020, 0x0012),
        ),
        seed: int = 0,
        val_frac: float = 0.2,
        cache_num: int = sys.maxsize,
        cache_rate: float = 0.0,
        num_workers: int = 1,
        progress: bool = True,
        copy_cache: bool = True,
        as_contiguous: bool = True,
    ) -> None:
        root_dir = Path(root_dir)
        if not root_dir.is_dir():
            raise ValueError("Root directory root_dir must be a directory.")
        self.section = section
        self.val_frac = val_frac
        self.picked_modality = picked_modality
        self.modality_tag = modality_tag
        self.ref_series_uid_tag = ref_series_uid_tag
        self.ref_sop_uid_tag = ref_sop_uid_tag

        self.set_random_state(seed=seed)
        download_dir = os.path.join(root_dir, collection)
        load_tags = list(specific_tags)
        load_tags += [modality_tag]
        self.load_tags = load_tags
        if download:
            seg_series_list = get_tcia_metadata(
                query=f"getSeries?Collection={collection}&Modality={picked_modality}", attribute="SeriesInstanceUID"
            )
            if download_len > 0:
                seg_series_list = seg_series_list[:download_len]
            if len(seg_series_list) == 0:
                raise ValueError(f"Cannot find data with collection: {collection} picked_modality: {picked_modality}")
            for series_uid in seg_series_list:
                self._download_series_reference_data(series_uid, download_dir)

        if not os.path.exists(download_dir):
            raise RuntimeError(f"Cannot find dataset directory: {download_dir}.")

        self.indices: np.ndarray = np.array([])
        self.datalist = self._generate_data_list(download_dir)

        if transform == ():
            transform = LoadImaged(reader="PydicomReader", keys=["image"])
        CacheDataset.__init__(
            self,
            data=self.datalist,
            transform=transform,
            cache_num=cache_num,
            cache_rate=cache_rate,
            num_workers=num_workers,
            progress=progress,
            copy_cache=copy_cache,
            as_contiguous=as_contiguous,
        )

    def get_indices(self) -> np.ndarray:
        """
        Get the indices of datalist used in this dataset.

        """
        return self.indices

    def randomize(self, data: np.ndarray) -> None:
        self.R.shuffle(data)

    def _download_series_reference_data(self, series_uid: str, download_dir: str):
        """
        First of all, download a series from TCIA according to `series_uid`.
        Then find all referenced series and download.
        """
        seg_first_dir = os.path.join(download_dir, "raw", series_uid)
        download_tcia_series_instance(
            series_uid=series_uid, download_dir=download_dir, output_dir=seg_first_dir, check_md5=False
        )
        dicom_files = [f for f in os.listdir(seg_first_dir) if f.endswith(".dcm")]
        # achieve series number and patient id from the first dicom file
        dcm_path = os.path.join(seg_first_dir, dicom_files[0])
        ds = PydicomReader(stop_before_pixels=True, specific_tags=self.load_tags).read(dcm_path)
        # (0x0010,0x0020) and (0x0010,0x0010), better to be contained in `specific_tags`
        patient_id = ds.PatientID if ds.PatientID else ds.PatientName
        if not patient_id:
            warnings.warn(f"unable to find patient name of dicom file: {dcm_path}, use 'patient' instead.")
            patient_id = "patient"
        # (0x0020,0x0011) and (0x0020,0x0012), better to be contained in `specific_tags`
        series_num = ds.SeriesNumber if ds.SeriesNumber else ds.AcquisitionNumber
        if not series_num:
            warnings.warn(f"unable to find series number of dicom file: {dcm_path}, use '0' instead.")
            series_num = 0

        series_num = str(series_num)
        seg_dir = os.path.join(download_dir, patient_id, series_num, self.picked_modality)
        dcm_dir = os.path.join(download_dir, patient_id, series_num, "IMAGE")

        # get ref uuid
        ref_uuid_list = []
        for dcm_file in dicom_files:
            dcm_path = os.path.join(seg_first_dir, dcm_file)
            ds = PydicomReader(stop_before_pixels=True, specific_tags=self.load_tags).read(dcm_path)
            if ds[self.modality_tag].value == self.picked_modality:
                ref_uuid = get_ref_uuid(
                    ds, find_sop=False, ref_series_uid_tag=self.ref_series_uid_tag, ref_sop_uid_tag=self.ref_sop_uid_tag
                )
                if ref_uuid == "":
                    ref_sop_uid = get_ref_uuid(
                        ds,
                        find_sop=True,
                        ref_series_uid_tag=self.ref_series_uid_tag,
                        ref_sop_uid_tag=self.ref_sop_uid_tag,
                    )
                    ref_uuid = match_ref_uid_in_study(ds.StudyInstanceUID, ref_sop_uid)
                if ref_uuid != "":
                    ref_uuid_list.append(ref_uuid)
        if len(ref_uuid_list) == 0:
            warnings.warn(f"Cannot find the referenced Series Instance UID from series: {series_uid}.")
        else:
            download_tcia_series_instance(
                series_uid=ref_uuid_list[0], download_dir=download_dir, output_dir=dcm_dir, check_md5=False
            )
        if not os.path.exists(seg_dir):
            shutil.copytree(seg_first_dir, seg_dir)

    def _generate_data_list(self, dataset_dir: PathLike) -> List[Dict]:
        # the types of the item in data list should be compatible with the dataloader
        dataset_dir = Path(dataset_dir)
        datalist = []
        patient_list = [f.name for f in os.scandir(dataset_dir) if f.is_dir() and f.name != "raw"]
        for patient_id in patient_list:
            series_list = [f.name for f in os.scandir(os.path.join(dataset_dir, patient_id)) if f.is_dir()]
            for series_num in series_list:
                image_path = os.path.join(dataset_dir, patient_id, series_num, "IMAGE")
                mask_path = os.path.join(dataset_dir, patient_id, series_num, self.picked_modality)
                if os.path.exists(image_path):
                    datalist.append({"image": image_path, self.picked_modality: mask_path})
                else:
                    datalist.append({self.picked_modality: mask_path})

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
            transform=train_transform,
            download=True,
        )
        dataset_fold0_train = cvdataset.get_dataset(folds=[1, 2, 3, 4])
        dataset_fold0_val = cvdataset.get_dataset(folds=0, transform=val_transform, download=False)
        # execute training for fold 0 ...

        dataset_fold1_train = cvdataset.get_dataset(folds=[0, 2, 3, 4])
        dataset_fold1_val = cvdataset.get_dataset(folds=1, transform=val_transform, download=False)
        # execute training for fold 1 ...

        ...

        dataset_fold4_train = ...
        # execute training for fold 4 ...

    """

    def __init__(self, dataset_cls, nfolds: int = 5, seed: int = 0, **dataset_params) -> None:
        if not hasattr(dataset_cls, "_split_datalist"):
            raise ValueError("dataset class must have _split_datalist API.")
        self.dataset_cls = dataset_cls
        self.nfolds = nfolds
        self.seed = seed
        self.dataset_params = dataset_params

    def get_dataset(self, folds: Union[Sequence[int], int], **dataset_params):
        """
        Generate dataset based on the specified fold indice in the cross validation group.

        Args:
            folds: index of folds for training or validation, if a list of values, concatenate the data.
            dataset_params: other additional parameters for the dataset_cls base class, will override
                the same parameters in `self.dataset_params`.

        """
        nfolds = self.nfolds
        seed = self.seed
        dataset_params_ = dict(self.dataset_params)
        dataset_params_.update(dataset_params)

        class _NsplitsDataset(self.dataset_cls):  # type: ignore
            def _split_datalist(self, datalist: List[Dict]) -> List[Dict]:
                data = partition_dataset(data=datalist, num_partitions=nfolds, shuffle=True, seed=seed)
                return select_cross_validation_folds(partitions=data, folds=folds)

        return _NsplitsDataset(**dataset_params_)
