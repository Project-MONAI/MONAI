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

import sys
import tempfile
import unittest
from copy import deepcopy
from os import path

import nibabel as nib
import numpy as np
import torch

from monai import data
from monai.auto3dseg import analyze_engine
from monai.auto3dseg.analyzer import (
    Analyzer,
    FgImageStatsCaseAnalyzer,
    FgImageStatsSummaryAnalyzer,
    ImageStatsCaseAnalyzer,
    ImageStatsSummaryAnalyzer,
    LabelStatsCaseAnalyzer,
    LabelStatsSummaryAnalyzer,
    FilenameCaseAnalyzer,
)
from monai.auto3dseg.analyze_engine import SegAnalyzeEngine
from monai.auto3dseg.data_analyzer import DataAnalyzer
from monai.auto3dseg.operations import Operations, SampleOperations, SummaryOperations
from monai.auto3dseg.utils import datafold_read, verify_report_format
from monai.bundle import ConfigParser
from monai.data import create_test_image_3d
from monai.data.meta_tensor import MetaTensor
from monai.data.utils import no_collation
from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    EnsureTyped,
    Lambdad,
    LoadImaged,
    Orientationd,
    SqueezeDimd,
    ToDeviced,
)
from numbers import Number

from monai.utils.enums import DATA_STATS

device = "cuda" if torch.cuda.is_available() else "cpu"
n_workers = 0 if sys.platform in ("win32", "darwin") else 2

fake_datalist = {
    "testing": [{"image": "val_001.fake.nii.gz"}, {"image": "val_002.fake.nii.gz"}],
    "training": [
        {"fold": 0, "image": "tr_image_001.fake.nii.gz", "label": "tr_label_001.fake.nii.gz"},
        {"fold": 0, "image": "tr_image_002.fake.nii.gz", "label": "tr_label_002.fake.nii.gz"},
        {"fold": 1, "image": "tr_image_001.fake.nii.gz", "label": "tr_label_001.fake.nii.gz"},
        {"fold": 1, "image": "tr_image_004.fake.nii.gz", "label": "tr_label_004.fake.nii.gz"},
    ],
}


class TestOperations(Operations):
    """
    Test example for user operation
    """

    def __init__(self) -> None:
        self.data = {"max": np.max, "mean": np.mean, "min": np.min}


class TestAnalyzer(Analyzer):
    """
    Test example for a simple Analyzer
    """

    def __init__(self, key, report_format, stats_name="test"):
        self.key = key
        super().__init__(stats_name, report_format)

    def __call__(self, data):
        d = dict(data)
        report = deepcopy(self.get_report_format())
        report["stats"] = self.ops["stats"].evaluate(d[self.key])
        d[self.stats_name] = report
        return d


class TestImageAnalyzer(Analyzer):
    """
    Test example for a simple Analyzer
    """

    def __init__(self, image_key="image", stats_name="test_image"):

        self.image_key = image_key
        report_format = {"test_stats": None}

        super().__init__(stats_name, report_format)
        self.update_ops("test_stats", TestOperations())

    def __call__(self, data):
        d = dict(data)
        report = deepcopy(self.get_report_format())
        report["test_stats"] = self.ops["test_stats"].evaluate(d[self.image_key])
        d[self.stats_name] = report
        return d


class TestDataAnalyzer(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.TemporaryDirectory()
        dataroot = self.test_dir.name

        # Generate a fake dataset
        for d in fake_datalist["testing"] + fake_datalist["training"]:
            im, seg = create_test_image_3d(39, 47, 46, rad_max=10)
            nib_image = nib.Nifti1Image(im, affine=np.eye(4))
            image_fpath = path.join(dataroot, d["image"])
            nib.save(nib_image, image_fpath)

            if "label" in d:
                nib_image = nib.Nifti1Image(seg, affine=np.eye(4))
                label_fpath = path.join(dataroot, d["label"])
                nib.save(nib_image, label_fpath)

        # write to a json file
        self.fake_json_datalist = path.join(dataroot, "fake_input.json")
        ConfigParser.export_config_file(fake_datalist, self.fake_json_datalist)

    def test_data_analyzer(self):
        dataroot = self.test_dir.name
        yaml_fpath = path.join(dataroot, "data_stats.yaml")
        analyser = DataAnalyzer(fake_datalist, dataroot, output_path=yaml_fpath, device=device, worker=n_workers)
        datastat = analyser.get_all_case_stats()

        assert len(datastat["stats_by_cases"]) == len(fake_datalist["training"])

    def test_data_analyzer_image_only(self):
        dataroot = self.test_dir.name
        yaml_fpath = path.join(dataroot, "data_stats.yaml")
        analyser = DataAnalyzer(
            fake_datalist, dataroot, output_path=yaml_fpath, device=device, worker=n_workers, label_key=None
        )
        datastat = analyser.get_all_case_stats()

        assert len(datastat["stats_by_cases"]) == len(fake_datalist["training"])

    def test_data_analyzer_from_yaml(self):
        dataroot = self.test_dir.name
        yaml_fpath = path.join(dataroot, "data_stats.yaml")
        analyser = DataAnalyzer(
            self.fake_json_datalist, dataroot, output_path=yaml_fpath, device=device, worker=n_workers
        )
        datastat = analyser.get_all_case_stats()

        assert len(datastat["stats_by_cases"]) == len(fake_datalist["training"])

    def test_basic_operation_class(self):
        op = TestOperations()
        test_data = np.random.rand(10, 10).astype(np.float64)
        test_ret_1 = op.evaluate(test_data)
        test_ret_2 = op.evaluate(test_data, axis=0)
        assert isinstance(test_ret_1, dict) and isinstance(test_ret_2, dict)
        assert ("max" in test_ret_1) and ("max" in test_ret_2)
        assert ("mean" in test_ret_1) and ("mean" in test_ret_2)
        assert ("min" in test_ret_1) and ("min" in test_ret_2)
        assert isinstance(test_ret_1['max'], np.float64)
        assert isinstance(test_ret_2['max'], np.ndarray)
        assert test_ret_1['max'].ndim == 0
        assert test_ret_2['max'].ndim == 1

    def test_sample_operations(self):
        op = SampleOperations()
        test_data_np = np.random.rand(10, 10).astype(np.float64)
        test_data_mt = MetaTensor(test_data_np, device=device)
        test_ret_np = op.evaluate(test_data_np)
        test_ret_mt = op.evaluate(test_data_mt)
        assert isinstance(test_ret_np['max'], Number)
        assert isinstance(test_ret_np['percentile'], list)
        assert isinstance(test_ret_mt['max'], Number)
        assert isinstance(test_ret_mt['percentile'], list)

        op.update({"sum": np.sum})
        test_ret_np = op.evaluate(test_data_np)
        assert "sum" in test_ret_np

    def test_summary_operations(self):
        op = SummaryOperations()
        test_dict = {
            "min": [0, 1, 2, 3],
            "max": [2, 3, 4, 5],
            "mean": [1, 2, 3, 4],
            "sum": [2, 4, 6, 8],
        }
        test_ret = op.evaluate(test_dict)
        assert isinstance(test_ret['max'], Number)
        assert isinstance(test_ret['min'], Number)

        op.update({"sum": np.sum})
        test_ret = op.evaluate(test_dict)
        assert "sum" in test_ret
        assert isinstance(test_ret['sum'], Number)

    def test_basic_analyzer_class(self):
        test_data = {}
        test_data['image_test'] = np.random.rand(10, 10)
        report_format = {"stats": None}
        user_analyzer = TestAnalyzer('image_test', report_format)
        user_analyzer.update_ops("stats", TestOperations())
        result = user_analyzer(test_data)
        assert result["test"]["stats"]["max"] == np.max(test_data['image_test'])
        assert result["test"]["stats"]["min"] == np.min(test_data['image_test'])
        assert result["test"]["stats"]["mean"] == np.mean(test_data['image_test'])

    def test_transform_analyzer_class(self):
        transform_list = [LoadImaged(keys=["image"]), TestImageAnalyzer(image_key="image")]
        transform = Compose(transform_list)
        dataroot = self.test_dir.name
        files, _ = datafold_read(self.fake_json_datalist, dataroot, fold=-1)
        ds = data.Dataset(data=files)
        self.dataset = data.DataLoader(ds, batch_size=1, shuffle=False, num_workers=0, collate_fn=no_collation)
        for batch_data in self.dataset:
            d = transform(batch_data[0])
            assert "test_image" in d
            assert "test_stats" in d["test_image"]
            assert "max" in d["test_image"]["test_stats"]
            assert "min" in d["test_image"]["test_stats"]
            assert "mean" in d["test_image"]["test_stats"]

    def test_image_stats_case_analyzer(self):
        analyzer = ImageStatsCaseAnalyzer(image_key="image")
        transform_list = [
            LoadImaged(keys=["image"]),
            EnsureChannelFirstd(keys=["image"]),  # this creates label to be (1,H,W,D)
            ToDeviced(keys=["image"], device=device, non_blocking=True),
            Orientationd(keys=["image"], axcodes="RAS"),
            EnsureTyped(keys=["image"], data_type="tensor"),
            analyzer,
        ]
        transform = Compose(transform_list)
        dataroot = self.test_dir.name
        files, _ = datafold_read(self.fake_json_datalist, dataroot, fold=-1)
        ds = data.Dataset(data=files)
        self.dataset = data.DataLoader(ds, batch_size=1, shuffle=False, num_workers=n_workers, collate_fn=no_collation)
        for batch_data in self.dataset:
            d = transform(batch_data[0])
            report_format = analyzer.get_report_format()
            assert verify_report_format(d["image_stats"], report_format)

    def test_foreground_image_stats_cases_analyzer(self):
        analyzer = FgImageStatsCaseAnalyzer(image_key="image", label_key="label")
        transform_list = [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),  # this creates label to be (1,H,W,D)
            ToDeviced(keys=["image", "label"], device=device, non_blocking=True),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            EnsureTyped(keys=["image", "label"], data_type="tensor"),
            Lambdad(keys=["label"], func=lambda x: torch.argmax(x, dim=0, keepdim=True) if x.shape[0] > 1 else x),
            SqueezeDimd(keys=["label"], dim=0),
            analyzer,
        ]
        transform = Compose(transform_list)
        dataroot = self.test_dir.name
        files, _ = datafold_read(self.fake_json_datalist, dataroot, fold=-1)
        ds = data.Dataset(data=files)
        self.dataset = data.DataLoader(ds, batch_size=1, shuffle=False, num_workers=n_workers, collate_fn=no_collation)
        for batch_data in self.dataset:
            d = transform(batch_data[0])
            report_format = analyzer.get_report_format()
            assert verify_report_format(d["image_foreground_stats"], report_format)

    def test_label_stats_case_analyzer(self):
        analyzer = LabelStatsCaseAnalyzer(image_key="image", label_key="label")
        transform_list = [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),  # this creates label to be (1,H,W,D)
            ToDeviced(keys=["image", "label"], device=device, non_blocking=True),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            EnsureTyped(keys=["image", "label"], data_type="tensor"),
            Lambdad(keys=["label"], func=lambda x: torch.argmax(x, dim=0, keepdim=True) if x.shape[0] > 1 else x),
            SqueezeDimd(keys=["label"], dim=0),
            analyzer,
        ]
        transform = Compose(transform_list)
        dataroot = self.test_dir.name
        files, _ = datafold_read(self.fake_json_datalist, dataroot, fold=-1)
        ds = data.Dataset(data=files)
        self.dataset = data.DataLoader(ds, batch_size=1, shuffle=False, num_workers=n_workers, collate_fn=no_collation)
        for batch_data in self.dataset:
            d = transform(batch_data[0])
            report_format = analyzer.get_report_format()
            assert verify_report_format(d["label_stats"], report_format)

    def test_filename_case_analyzer(self):
        analyzer_image = FilenameCaseAnalyzer("image", DATA_STATS.BY_CASE_IMAGE_PATH)
        analyzer_label = FilenameCaseAnalyzer("label", DATA_STATS.BY_CASE_IMAGE_PATH)
        transform_list = [
            LoadImaged(keys=["image", "label"]),
            analyzer_image,
            analyzer_label
        ]
        transform = Compose(transform_list)
        dataroot = self.test_dir.name
        files, _ = datafold_read(self.fake_json_datalist, dataroot, fold=-1)
        ds = data.Dataset(data=files)
        self.dataset = data.DataLoader(ds, batch_size=1, shuffle=False, num_workers=n_workers, collate_fn=no_collation)
        for batch_data in self.dataset:
            d = transform(batch_data[0])
            assert DATA_STATS.BY_CASE_IMAGE_PATH in d
            assert DATA_STATS.BY_CASE_IMAGE_PATH in d

    def test_filename_case_analyzer(self):
        analyzer_image = FilenameCaseAnalyzer("image", DATA_STATS.BY_CASE_IMAGE_PATH)
        analyzer_label = FilenameCaseAnalyzer(None, DATA_STATS.BY_CASE_IMAGE_PATH)
        transform_list = [
            LoadImaged(keys=["image"]),
            analyzer_image,
            analyzer_label
        ]
        transform = Compose(transform_list)
        dataroot = self.test_dir.name
        files, _ = datafold_read(self.fake_json_datalist, dataroot, fold=-1)
        ds = data.Dataset(data=files)
        self.dataset = data.DataLoader(ds, batch_size=1, shuffle=False, num_workers=n_workers, collate_fn=no_collation)
        for batch_data in self.dataset:
            d = transform(batch_data[0])
            assert DATA_STATS.BY_CASE_IMAGE_PATH in d
            assert d[DATA_STATS.BY_CASE_IMAGE_PATH] == ""

    def test_image_stats_summary_analyzer(self):
        summary_analyzer = ImageStatsSummaryAnalyzer("image_stats")

        transform_list = [
            LoadImaged(keys=["image"]),
            EnsureChannelFirstd(keys=["image"]),  # this creates label to be (1,H,W,D)
            ToDeviced(keys=["image"], device=device, non_blocking=True),
            Orientationd(keys=["image"], axcodes="RAS"),
            EnsureTyped(keys=["image"], data_type="tensor"),
            ImageStatsCaseAnalyzer(image_key="image"),
        ]
        transform = Compose(transform_list)
        dataroot = self.test_dir.name
        files, _ = datafold_read(self.fake_json_datalist, dataroot, fold=-1)
        ds = data.Dataset(data=files)
        self.dataset = data.DataLoader(ds, batch_size=1, shuffle=False, num_workers=n_workers, collate_fn=no_collation)
        stats = []
        for batch_data in self.dataset:
            stats.append(transform(batch_data[0]))
        summary_report = summary_analyzer(stats)
        report_format = summary_analyzer.get_report_format()
        assert verify_report_format(summary_report, report_format)

    def test_fg_image_stats_summary_analyzer(self):
        summary_analyzer = FgImageStatsSummaryAnalyzer("image_foreground_stats")

        transform_list = [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),  # this creates label to be (1,H,W,D)
            ToDeviced(keys=["image", "label"], device=device, non_blocking=True),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            EnsureTyped(keys=["image", "label"], data_type="tensor"),
            Lambdad(keys="label", func=lambda x: torch.argmax(x, dim=0, keepdim=True) if x.shape[0] > 1 else x),
            SqueezeDimd(keys=["label"], dim=0),
            FgImageStatsCaseAnalyzer(image_key="image", label_key="label"),
        ]
        transform = Compose(transform_list)
        dataroot = self.test_dir.name
        files, _ = datafold_read(self.fake_json_datalist, dataroot, fold=-1)
        ds = data.Dataset(data=files)
        self.dataset = data.DataLoader(ds, batch_size=1, shuffle=False, num_workers=n_workers, collate_fn=no_collation)
        stats = []
        for batch_data in self.dataset:
            stats.append(transform(batch_data[0]))
        summary_report = summary_analyzer(stats)
        report_format = summary_analyzer.get_report_format()
        assert verify_report_format(summary_report, report_format)

    def test_label_stats_summary_analyzer(self):
        summary_analyzer = LabelStatsSummaryAnalyzer("label_stats")

        transform_list = [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),  # this creates label to be (1,H,W,D)
            ToDeviced(keys=["image", "label"], device=device, non_blocking=True),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            EnsureTyped(keys=["image", "label"], data_type="tensor"),
            Lambdad(keys="label", func=lambda x: torch.argmax(x, dim=0, keepdim=True) if x.shape[0] > 1 else x),
            SqueezeDimd(keys=["label"], dim=0),
            LabelStatsCaseAnalyzer(image_key="image", label_key="label"),
        ]
        transform = Compose(transform_list)
        dataroot = self.test_dir.name
        files, _ = datafold_read(self.fake_json_datalist, dataroot, fold=-1)
        ds = data.Dataset(data=files)
        self.dataset = data.DataLoader(ds, batch_size=1, shuffle=False, num_workers=n_workers, collate_fn=no_collation)
        stats = []
        for batch_data in self.dataset:
            stats.append(transform(batch_data[0]))
        summary_report = summary_analyzer(stats)
        report_format = summary_analyzer.get_report_format()
        assert verify_report_format(summary_report, report_format)

    def test_analyzer_engine(self):
        analyze_engine = SegAnalyzeEngine("image", "label")
        keys = ["image", "label"]
        transform_list = [
            LoadImaged(keys=keys),
            EnsureChannelFirstd(keys=keys),  # this creates label to be (1,H,W,D)
            ToDeviced(keys=keys, device=device, non_blocking=True),
            Orientationd(keys=keys, axcodes="RAS"),
            EnsureTyped(keys=keys, data_type="tensor"),
            Lambdad(keys="label", func=lambda x: torch.argmax(x, dim=0, keepdim=True) if x.shape[0] > 1 else x),
            SqueezeDimd(keys=["label"], dim=0),
            analyze_engine,
        ]
        transform = Compose(transform_list)
        dataroot = self.test_dir.name
        files, _ = datafold_read(self.fake_json_datalist, dataroot, fold=-1)
        ds = data.Dataset(data=files)
        self.dataset = data.DataLoader(ds, batch_size=1, shuffle=False, num_workers=n_workers, collate_fn=no_collation)
        stats = []
        for batch_data in self.dataset:
            d = transform(batch_data[0])
            stats.append(d)
        report = analyze_engine.summarize(stats)
        assert str(DATA_STATS.IMAGE_STATS) in report
        assert str(DATA_STATS.FG_IMAGE_STATS) in report
        assert str(DATA_STATS.LABEL_STATS) in report


    def tearDown(self) -> None:
        self.test_dir.cleanup()


if __name__ == "__main__":
    unittest.main()
