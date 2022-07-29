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
import tempfile
import unittest

import torch

import monai.transforms as mt
from monai.data.video_dataset import CameraDataset, VideoFileDataset
from monai.utils.module import optional_import
from tests.utils import download_url_or_skip_test, testing_data_config

cv2, has_cv2 = optional_import("cv2")

if has_cv2:
    NUM_CAPTURE_DEVICES = CameraDataset.get_num_devices()

TRANSFORMS = mt.Compose([mt.AsChannelFirst(), mt.DivisiblePad(16), mt.ScaleIntensity(), mt.CastToType(torch.float32)])


class TestCameraDataset(unittest.TestCase):
    @unittest.skipUnless(has_cv2, "OpenCV required.")
    @unittest.skipIf(NUM_CAPTURE_DEVICES == 0, "At least one capture device required.")
    def test_camera_dataset(self):
        num_frames = 10
        ds = CameraDataset(0, TRANSFORMS, num_frames)
        frames = list(ds)
        self.assertEqual(len(frames), num_frames)
        for f in frames:
            self.assertTupleEqual(f.shape, frames[0].shape)

    @unittest.skipUnless(has_cv2, "OpenCV required.")
    @unittest.skipIf(NUM_CAPTURE_DEVICES == 0, "At least one capture device required.")
    def test_device_out_of_range(self):
        capture_device = NUM_CAPTURE_DEVICES + 1
        with self.assertRaises(RuntimeError):
            _ = CameraDataset(capture_device, TRANSFORMS, 0)

    @unittest.skipIf(has_cv2, "Only tested when OpenCV not installed.")
    def test_no_opencv_raises(self):
        with self.assertRaises():
            _ = CameraDataset(0, TRANSFORMS, 0)


class TestVideoFileDataset(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        super(__class__, cls).setUpClass()
        cls.filepath = os.path.join(tempfile.mkdtemp(), "endo.mp4")
        config = testing_data_config("videos", "endovis")
        download_url_or_skip_test(
            url=config["url"],
            filepath=cls.filepath,
            hash_val=config.get("hash_val"),
            hash_type=config.get("hash_type", "sha256"),
        )

    @unittest.skipUnless(has_cv2, "OpenCV required.")
    def test_video_file_dataset(self):
        for num_frames in (None,):
            ds = VideoFileDataset(self.filepath, TRANSFORMS, num_frames)
            known_num_frames_in_vid = 23
            self.assertEqual(len(ds), num_frames or known_num_frames_in_vid)
            total_num_frames = ds.get_num_frames()
            self.assertEqual(total_num_frames, known_num_frames_in_vid)

            frames = list(ds)
            self.assertEqual(len(frames), num_frames or total_num_frames)
            for f in frames:
                self.assertTupleEqual(f.shape, frames[0].shape)

    @unittest.skipIf(has_cv2, "Only tested when OpenCV not installed.")
    def test_no_opencv_raises(self):
        with self.assertRaises():
            _ = VideoFileDataset(self.filepath, TRANSFORMS, 0)


if __name__ == "__main__":
    unittest.main()
