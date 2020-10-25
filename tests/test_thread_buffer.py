import time
import unittest

from monai.data import DataLoader, Dataset, ThreadBuffer
from monai.transforms import Compose, SimulateDelayd
from monai.utils import PerfContext


class TestDataLoader(unittest.TestCase):
    def setUp(self):
        super().setUp()

        self.datalist = [
            {"image": "spleen_19.nii.gz", "label": "spleen_label_19.nii.gz"},
            {"image": "spleen_31.nii.gz", "label": "spleen_label_31.nii.gz"},
        ]

        self.transform = Compose([SimulateDelayd(keys=["image", "label"], delay_time=0.1)])

    def test_values(self):
        dataset = Dataset(data=self.datalist, transform=self.transform)
        dataloader = DataLoader(dataset=dataset, batch_size=2, num_workers=0)

        tbuffer = ThreadBuffer(dataloader)

        for d in tbuffer:
            self.assertEqual(d["image"][0], "spleen_19.nii.gz")
            self.assertEqual(d["image"][1], "spleen_31.nii.gz")
            self.assertEqual(d["label"][0], "spleen_label_19.nii.gz")
            self.assertEqual(d["label"][1], "spleen_label_31.nii.gz")

    def test_time(self):
        dataset = Dataset(data=self.datalist * 2, transform=self.transform)  # contains data for 2 batches
        dataloader = DataLoader(dataset=dataset, batch_size=2, num_workers=0)

        tbuffer = ThreadBuffer(dataloader)

        with PerfContext() as pc:
            for _ in dataloader:
                time.sleep(0.5)  # each batch takes 0.8 s to generate on top of this time

        unbuffered_time = pc.total_time

        with PerfContext() as pc:
            for _ in tbuffer:
                time.sleep(0.5)  # while "computation" is happening the next batch is being generated, saving 0.4 s

        buffered_time = pc.total_time

        self.assertTrue(
            buffered_time < unbuffered_time,
            f"Buffered time {buffered_time} should be less than unbuffered time {unbuffered_time}",
        )


if __name__ == "__main__":
    unittest.main()
