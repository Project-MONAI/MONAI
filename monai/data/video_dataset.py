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
from typing import Any, Callable, Optional, Union

import numpy as np
from torch.utils.data import Dataset, IterableDataset

from monai.transforms.transform import apply_transform
from monai.utils.module import optional_import

cv2, has_cv2 = optional_import("cv2")

__all__ = ["VideoFileDataset", "CameraDataset"]


class VideoDataset:
    def __init__(self, video_source, transform, max_num_frames):
        self.cap = self.open_video(video_source)
        self.transform = transform
        self.max_num_frames = max_num_frames

    @staticmethod
    def open_video(video_source: Union[str, int]):
        """
        Use OpenCV to open a video source from either file or capture device.

        Args:
            video_source: filename or index referring to capture device.
        """
        if not has_cv2:
            raise RuntimeError("OpenCV not installed.")
        if isinstance(video_source, str) and not os.path.isfile(video_source):
            raise RuntimeError("Video file does not exist: " + video_source)
        cap = cv2.VideoCapture(video_source)
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {video_source}")
        return cap

    def get_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            raise RuntimeError("Failed to read frame.")
        # channel to front
        frame = np.moveaxis(frame, -1, 0)
        # BGR -> RGB
        frame = np.flip(frame, 0)
        return apply_transform(self.transform, frame) if self.transform is not None else frame

    def get_fps(self):
        """Get the FPS of the capture device."""
        return self.cap.get(cv2.CAP_PROP_FPS)


class VideoFileDataset(Dataset, VideoDataset):
    """
    Video dataset from file.

    This class requires that OpenCV be installed.

    Args:
        video_source: filename of video.
        transform: transform to be applied to each frame.
        max_num_frames: Max number of frames to iterate across. If `None` is passed,
            then the dataset will iterate until the end of the file.

    Raises:
        RuntimeError: OpenCV not installed.
    """

    def __init__(
        self, video_source: str, transform: Optional[Callable] = None, max_num_frames: Optional[int] = None
    ) -> None:
        VideoDataset.__init__(self, video_source, transform, max_num_frames)
        num_frames = self.get_num_frames()
        if max_num_frames is None or num_frames < max_num_frames:
            self.max_num_frames = num_frames
        else:
            self.max_num_frames = max_num_frames

    def get_num_frames(self) -> int:
        """
        Return the number of frames in a video file.

        Raises:
            RuntimeError: no frames found.
        """
        num_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if num_frames == 0:
            raise RuntimeError("0 frames found")
        return num_frames

    def __len__(self):
        return self.max_num_frames

    def __getitem__(self, index: int):
        """
        Fetch single data item from index.
        """
        if index >= self.max_num_frames:
            raise IndexError
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        return self.get_frame()


class CameraDataset(IterableDataset, VideoDataset):
    """
    Video dataset from a capture device (e.g., webcam).

    This class requires that OpenCV be installed.

    Args:
        video_source: index of capture device.
            `get_num_devices` can be used to determine possible devices.
        transform: transform to be applied to each frame.
        max_num_frames: Max number of frames to iterate across. If `None` is passed,
            then the dataset will iterate infinitely.

    Raises:
        RuntimeError: OpenCV not installed.
    """

    @staticmethod
    def get_num_devices() -> int:
        """Get number of possible devices detected by OpenCV that can be used for capture."""
        num_devices = 0
        while True:
            cap = cv2.VideoCapture(num_devices)
            if not cap.read()[0]:
                break
            num_devices += 1
            cap.release()
        return num_devices

    def get_next_frame(self) -> Any:
        """
        Get the next frame from the capture device. Apply transform and return.

        Raises:
            RuntimeError: failed to read a frame.
        """
        ret, frame = self.cap.read()
        if not ret:
            raise RuntimeError(f"Failed to read frame {frame}")
        return apply_transform(self.transform, frame) if self.transform is not None else frame

    def __iter__(self):
        frame_count = 0
        while True:
            frame = self.get_next_frame()
            frame_count += 1
            yield frame
            if self.max_num_frames is not None:
                if frame_count == self.max_num_frames:
                    break
