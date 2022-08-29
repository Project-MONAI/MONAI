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
        pass
    
    def tearDown(self) -> None:
        pass

if __name__ == "__main__":
    unittest.main()
