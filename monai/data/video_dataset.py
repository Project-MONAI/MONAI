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
from typing import Any, Callable, List, Optional, Union

from torch.utils.data import IterableDataset

from monai.utils.module import optional_import

cv2, has_cv2 = optional_import("cv2")

__all__ = ["VideoDataset", "VideoFileDataset", "CameraDataset"]


class VideoDataset(IterableDataset):
    """
    Abstract base class for video datasets. This combines videos from file and video
    from a capture device (e.g., a webcam).

    This class and inherited classes require OpenCV to be installed.

    Args:
        video_source: filename or index referring to capture device.
        transforms: transforms to be applied to each frame.
        max_num_frames: Max number of frames to iterate across. If `None` is passed,
            then the dataset will iterate until the end of the file or infinitely if
            a capture device is used.

    Raises:
        RuntimeError: OpenCV not installed.
    """

    def __init__(
        self, video_source: Union[str, int], transforms: Callable, max_num_frames: Optional[int] = None
    ) -> None:
        if not has_cv2:
            raise RuntimeError("OpenCV not installed.")

        self.max_num_frames = max_num_frames
        self.transforms = transforms
        self.cap = self.open_video(video_source)

    @staticmethod
    def open_video(video_source: Union[str, int]):
        """
        Use OpenCV to open a video source from either file or capture device.

        Args:
            video_source: filename or index referring to capture device.
        """
        if isinstance(video_source, str) and not os.path.isfile(video_source):
            raise RuntimeError("Video file does not exist: " + video_source)
        cap = cv2.VideoCapture(video_source)
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {video_source}")
        return cap

    def get_next_frame(self) -> Any:
        """
        Get the next frame from the capture device. Apply transforms and return.

        Raises:
            RuntimeError: failed to read a frame.
        """
        ret, frame = self.cap.read()
        if not ret:
            raise RuntimeError(f"Failed to read frame {frame}")
        return self.transforms(frame)

    def __iter__(self):
        frame_count = 0
        while True:
            frame = self.get_next_frame()
            frame_count += 1
            yield frame
            if self.max_num_frames is not None:
                if frame_count == self.max_num_frames:
                    break


class VideoFileDataset(VideoDataset):
    """
    Video dataset from file.

    This class requires that OpenCV be installed.

    Args:
        video_source: filename of video.
        transforms: transforms to be applied to each frame.
        max_num_frames: Max number of frames to iterate across. If `None` is passed,
            then the dataset will iterate until the end of the file.

    Raises:
        RuntimeError: OpenCV not installed.
    """

    def __init__(self, video_source: str, transforms: Callable, max_num_frames: Optional[int] = None) -> None:
        super().__init__(video_source, transforms, max_num_frames)
        num_frames = self.get_num_frames()
        if max_num_frames is None or num_frames < max_num_frames:
            self.max_num_frames = num_frames

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


class CameraDataset(VideoDataset):
    """
    Video dataset from a capture device (e.g., webcam).

    This class requires that OpenCV be installed.

    Args:
        video_source: index of capture device.
            `get_possible_devices` can be used to determine possible devices.
        transforms: transforms to be applied to each frame.
        max_num_frames: Max number of frames to iterate across. If `None` is passed,
            then the dataset will iterate infinitely.

    Raises:
        RuntimeError: OpenCV not installed.
    """

    def __init__(self, stream_device: int, transforms: Callable, max_num_frames: Optional[int] = None) -> None:
        super().__init__(stream_device, transforms, max_num_frames)

    @staticmethod
    def get_possible_devices() -> List[int]:
        """Get a list of possible devices detected by OpenCV that can be used for capture."""
        index = 0
        arr = []
        while True:
            cap = cv2.VideoCapture(index)
            if not cap.read()[0]:
                break
            else:
                arr.append(index)
            cap.release()
            index += 1
        return arr
