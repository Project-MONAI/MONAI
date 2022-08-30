import os
import tempfile
import unittest
import nibabel as nib
import numpy as np

from monai.apps.auto3dseg import DataAnalyzer
from monai.apps.auto3dseg import BundleGen
from monai.bundle.config_parser import ConfigParser
from monai.data import create_test_image_3d



fake_datalist = {
    "testing": [{"image": "val_001.fake.nii.gz"}, {"image": "val_002.fake.nii.gz"}],
    "training": [
        {"fold": 0, "image": "tr_image_001.fake.nii.gz", "label": "tr_label_001.fake.nii.gz"},
        {"fold": 0, "image": "tr_image_002.fake.nii.gz", "label": "tr_label_002.fake.nii.gz"},
        {"fold": 0, "image": "tr_image_003.fake.nii.gz", "label": "tr_label_003.fake.nii.gz"},
        {"fold": 0, "image": "tr_image_004.fake.nii.gz", "label": "tr_label_004.fake.nii.gz"},
        {"fold": 1, "image": "tr_image_005.fake.nii.gz", "label": "tr_label_005.fake.nii.gz"},
        {"fold": 1, "image": "tr_image_006.fake.nii.gz", "label": "tr_label_006.fake.nii.gz"},
        {"fold": 1, "image": "tr_image_007.fake.nii.gz", "label": "tr_label_007.fake.nii.gz"},
        {"fold": 1, "image": "tr_image_008.fake.nii.gz", "label": "tr_label_008.fake.nii.gz"},
        {"fold": 2, "image": "tr_image_009.fake.nii.gz", "label": "tr_label_009.fake.nii.gz"},
        {"fold": 2, "image": "tr_image_010.fake.nii.gz", "label": "tr_label_010.fake.nii.gz"},
        {"fold": 2, "image": "tr_image_011.fake.nii.gz", "label": "tr_label_011.fake.nii.gz"},
        {"fold": 2, "image": "tr_image_012.fake.nii.gz", "label": "tr_label_012.fake.nii.gz"},
        {"fold": 3, "image": "tr_image_013.fake.nii.gz", "label": "tr_label_013.fake.nii.gz"},
        {"fold": 3, "image": "tr_image_014.fake.nii.gz", "label": "tr_label_014.fake.nii.gz"},
        {"fold": 3, "image": "tr_image_015.fake.nii.gz", "label": "tr_label_015.fake.nii.gz"},
        {"fold": 3, "image": "tr_image_016.fake.nii.gz", "label": "tr_label_016.fake.nii.gz"},
        {"fold": 4, "image": "tr_image_017.fake.nii.gz", "label": "tr_label_017.fake.nii.gz"},
        {"fold": 4, "image": "tr_image_018.fake.nii.gz", "label": "tr_label_018.fake.nii.gz"},
        {"fold": 4, "image": "tr_image_019.fake.nii.gz", "label": "tr_label_019.fake.nii.gz"},
        {"fold": 4, "image": "tr_image_020.fake.nii.gz", "label": "tr_label_020.fake.nii.gz"},
    ],
}

class TestEnsembleBuilder(unittest.TestCase):
    def setUp(self) -> None:
        self.test_dir = tempfile.TemporaryDirectory()
        self.dataroot = os.path.join(self.test_dir.name, 'dataroot')
        self.work_dir = os.path.join(self.test_dir.name, "workdir")
        self.da_output_yaml = os.path.join(self.work_dir, "datastats.yaml")
        self.data_src_cfg = os.path.join(self.work_dir, "data_src_cfg.yaml")

        if not os.path.isdir(self.dataroot):
            os.makedirs(self.dataroot)

        if not os.path.isdir(self.work_dir):
            os.makedirs(self.work_dir)

        # Generate a fake dataset
        for d in fake_datalist["testing"] + fake_datalist["training"]:
            im, seg = create_test_image_3d(39, 47, 46, rad_max=10)
            nib_image = nib.Nifti1Image(im, affine=np.eye(4))
            image_fpath = os.path.join(self.dataroot, d["image"])
            nib.save(nib_image, image_fpath)

            if "label" in d:
                nib_image = nib.Nifti1Image(seg, affine=np.eye(4))
                label_fpath = os.path.join(self.dataroot, d["label"])
                nib.save(nib_image, label_fpath)

        # write to a json file
        self.fake_json_datalist = os.path.join(self.dataroot, "fake_input.json")
        ConfigParser.export_config_file(fake_datalist, self.fake_json_datalist)

        progress_tracker = ["analysis", "configure"]

        if not os.path.isdir(self.work_dir):
            os.makedirs(self.work_dir)


        if "analysis" in progress_tracker:
            da = DataAnalyzer(
                self.fake_json_datalist,
                self.dataroot,
                output_path=self.da_output_yaml)

            datastat = da.get_all_case_stats()

        data_src = {
            "name": "fake_data",
            "task": "segmentation",
            "modality": "MRI",
            "datalist": self.fake_json_datalist,
            "dataroot": self.dataroot,
            "multigpu": False,
            "class_names": ["class1", "class2", "class3", "class4", "class5"]
        }


        ConfigParser.export_config_file(data_src, self.data_src_cfg)

        bundle_generator = BundleGen(
            data_stats_filename=self.da_output_yaml,
            data_lists_filename=self.data_src_cfg)

        bundle_generator.generate(self.work_dir, [0,1,2,3,4])

        # training
        history = bundle_generator.get_history()

        num_epoch = 4
        n_iter = num_epoch * len(fake_datalist["training"]) * 4 / 5
        n_iter_val = n_iter / 2
        for i, record in enumerate(history):
            for name, algo in record.items():
                algo.train(
                    num_iterations=n_iter,
                    num_iterations_per_validation=n_iter_val,
                    single_gpu=not data_src["multigpu"]
                )

    def test_data(self) -> None:
        pass

    def tearDown(self) -> None:
        pass

if __name__ == "__main__":
    unittest.main()
