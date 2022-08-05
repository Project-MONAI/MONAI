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
from contextlib import nullcontext
from typing import Type, Union

import torch

import monai.transforms as mt
from monai.data.dataloader import DataLoader
from monai.data.video_dataset import CameraDataset, VideoDataset, VideoFileDataset
from monai.utils.module import optional_import
from tests.utils import assert_allclose, download_url_or_skip_test, testing_data_config

cv2, has_cv2 = optional_import("cv2")


NUM_CAPTURE_DEVICES = CameraDataset.get_num_devices()
TRANSFORMS = mt.Compose([mt.AsChannelFirst(), mt.DivisiblePad(16), mt.ScaleIntensity(), mt.CastToType(torch.float32)])


class Base:
    class TestVideoDataset(unittest.TestCase):
        video_source: Union[int, str]
        ds: Type[VideoDataset]

        def get_ds(self, *args, **kwargs) -> VideoDataset:
            return self.ds(video_source=self.video_source, transform=TRANSFORMS, *args, **kwargs)  # type: ignore

        @unittest.skipIf(has_cv2, "Only tested when OpenCV not installed.")
        def test_no_opencv_raises(self):
            with self.assertRaises():
                _ = self.get_ds(max_num_frames=10)

        @unittest.skipUnless(has_cv2, "OpenCV required.")
        def test_multiprocessing(self):
            for multiprocessing in (True, False):
                for num_workers in (0, 2):
                    # we only expect an error if num_workers > 0 and not using
                    # multiprocessing
                    expect_raise = num_workers == 2 and not multiprocessing
                    ds = self.get_ds(max_num_frames=100, multiprocessing=multiprocessing)
                    dl = DataLoader(ds, num_workers=num_workers, batch_size=2)
                    with self.assertRaises((TypeError, RuntimeError)) if expect_raise else nullcontext():
                        _ = next(iter(dl))

        @unittest.skipUnless(has_cv2, "OpenCV required.")
        def test_multiple_sources(self, should_match: bool = True):
            ds1 = self.get_ds()
            ds2 = self.get_ds()
            if should_match:
                assert_allclose(ds1.get_frame(), ds2.get_frame())

        @unittest.skipUnless(has_cv2, "OpenCV required.")
        def test_dataset(self, known_num_frames=None, known_fps=None):
            num_frames = (10,) if known_num_frames is None else (10, None)
            for max_num_frames in num_frames:
                ds = self.get_ds(max_num_frames=max_num_frames)
                if known_fps is not None:
                    self.assertEqual(ds.get_fps(), known_fps)
                frames = list(ds)
                if max_num_frames is not None:
                    self.assertEqual(len(frames), max_num_frames)
                elif known_num_frames is not None:
                    self.assertEqual(len(frames), len(ds))
                for f in frames:
                    self.assertTupleEqual(f.shape, frames[0].shape)


@unittest.skipIf(NUM_CAPTURE_DEVICES == 0, "At least one capture device required.")
class TestCameraDataset(Base.TestVideoDataset):
    video_source = 0
    ds = CameraDataset

    @unittest.skipUnless(has_cv2, "OpenCV required.")
    def test_multiple_sources(self):
        super().test_multiple_sources(should_match=False)

    @unittest.skipUnless(has_cv2, "OpenCV required.")
    def test_device_out_of_range(self):
        capture_device = NUM_CAPTURE_DEVICES + 1
        with self.assertRaises(RuntimeError):
            _ = CameraDataset(capture_device, TRANSFORMS, 0)


class TestVideoFileDataset(Base.TestVideoDataset):
    ds = VideoFileDataset

    @classmethod
    def setUpClass(cls):
        super(__class__, cls).setUpClass()
        cls.video_source = os.path.join(tempfile.mkdtemp(), "endo.mp4")
        config = testing_data_config("videos", "endovis")
        download_url_or_skip_test(
            url=config["url"],
            filepath=cls.video_source,
            hash_val=config.get("hash_val"),
            hash_type=config.get("hash_type", "sha256"),
        )

    @unittest.skipUnless(has_cv2, "OpenCV required.")
    def test_dataset(self):
        known_fps = 2.0
        known_num_frames = 23
        super().test_dataset(known_num_frames, known_fps)
        self.assertEqual(self.get_ds().get_num_frames(), known_num_frames)


if __name__ == "__main__":
    unittest.main()
