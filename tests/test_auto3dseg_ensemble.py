import os
from monai.apps.auto3dseg import DataAnalyzer
from monai.apps.auto3dseg import BundleGen
from monai.bundle.config_parser import ConfigParser


import unittest

fake_datalist = {
    "testing": [{"image": "val_001.fake.nii.gz"}, {"image": "val_002.fake.nii.gz"}],
    "training": [
        {"fold": 0, "image": "tr_image_001.fake.nii.gz", "label": "tr_label_001.fake.nii.gz"},
        {"fold": 0, "image": "tr_image_002.fake.nii.gz", "label": "tr_label_002.fake.nii.gz"},
        {"fold": 1, "image": "tr_image_001.fake.nii.gz", "label": "tr_label_001.fake.nii.gz"},
        {"fold": 1, "image": "tr_image_004.fake.nii.gz", "label": "tr_label_004.fake.nii.gz"},
    ],
}

class TestEnsembleBuilder(unittest.TestCase):
    def setUp(self) -> None:
        pass
    
    def test_data_analyzer(self) -> None:
        progress_tracker = ["analysis", "configure"]

        work_dir = "/workspace/monai/monai-cache/run0"
        if not os.path.isdir(work_dir):
            os.makedirs(work_dir)

        dataroot = "/workspace/data/Task04_Hippocampus"
        datalist = "/workspace/data/msd_task04_hippocampus_folds.json"
        output_yaml = os.path.join(work_dir, "datastats.yaml")


        if "analysis" in progress_tracker:
            da = DataAnalyzer(datalist, dataroot, output_path=output_yaml)
            datastat = da.get_all_case_stats()

        data_src = {
            "name": "Task04_Hippocampus",
            "task": "segmentation",
            "modality": "MRI",
            "datalist": datalist,
            "dataroot": dataroot,
            "multigpu": True,
            "class_names": ["val_acc_ant", "val_acc_post"]
        }

        data_src_cfg = "/workspace/monai/monai-cache/run0/data_src_cfg.yaml"
        ConfigParser.export_config_file(data_src, data_src_cfg)

        bundle_generator = BundleGen(
            data_stats_filename=output_yaml, 
            data_lists_filename=data_src_cfg)

        bundle_generator.generate(work_dir, [0,1,2,3,4])

        # training
        history = bundle_generator.get_history()

        for i, record in enumerate(history):
            for name, algo in record.items():
                algo.train(num_iterations=1500)
    
    def tearDown(self) -> None:
        pass

if __name__ == "__main__":
    unittest.main()
