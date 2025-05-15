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

from __future__ import annotations

import os
import sys
import tempfile
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import numpy as np
from torch.utils.data import Dataset, IterableDataset

from monai.utils.enums import ColorOrder
from monai.utils.module import optional_import

__all__ = ["VideoDataset", "VideoFileDataset", "CameraDataset"]

if TYPE_CHECKING:
    import cv2

    has_cv2 = True
else:
    cv2, has_cv2 = None, None


def import_cv():
    """Import cv2. Put it inside a function to avoid webcam lights blinking on ``import monai``."""
    global cv2
    global has_cv2
    cv2, has_cv2 = optional_import("cv2")


class SuppressStderr:
    """Suppress stderr. Useful as OpenCV (and dependencies) can produce a lot of output."""

    def __enter__(self):
        self.errnull_file = open(os.devnull, "w")
        self.old_stderr_fileno_undup = sys.stderr.fileno()
        self.old_stderr_fileno = os.dup(sys.stderr.fileno())
        self.old_stderr = sys.stderr
        os.dup2(self.errnull_file.fileno(), self.old_stderr_fileno_undup)
        sys.stderr = self.errnull_file
        return self

    def __exit__(self, *_):
        sys.stderr = self.old_stderr
        os.dup2(self.old_stderr_fileno, self.old_stderr_fileno_undup)
        os.close(self.old_stderr_fileno)
        self.errnull_file.close()


class VideoDataset:
    # import inside class to avoid webcam blinking on ``import monai``.
    import_cv()

    def __init__(
        self,
        video_source: str | int,
        transform: Callable | None = None,
        max_num_frames: int | None = None,
        color_order: str = ColorOrder.RGB,
        multiprocessing: bool = False,
        channel_dim: int = 0,
    ) -> None:
        """
        Base video dataset.

        Args:
            video_source: filename of video.
            transform: transform to be applied to each frame.
            max_num_frames: Max number of frames to iterate across. If `None` is passed,
                then the dataset will iterate until the end of the file.
            color_order: Color order to return frame. Default is RGB.
            multiprocessing: If `True`, open the video source on the fly. This makes
                things process-safe, which is useful when combined with a DataLoader
                with `num_workers>0`. However, when using with `num_workers==0`, it
                makes sense to use `multiprocessing=False`, as the source will then
                only be opened once, at construction, which will be faster in those
                circumstances.
            channel_dim: OpenCV reads with the channel as the last dimension. Use this
                flag to move it elsewhere. By default this is zero, so the channel
                dimension is moved to the front.

        Raises:
            RuntimeError: OpenCV not installed.
            NotImplementedError: Unknown color order.
        """
        if not has_cv2:
            raise RuntimeError("OpenCV not installed.")
        if color_order not in ColorOrder:
            raise NotImplementedError

        self.color_order = color_order
        self.channel_dim = channel_dim
        self.video_source = video_source
        self.multiprocessing = multiprocessing
        if not multiprocessing:
            self.cap = self.open_video(video_source)
        self.transform = transform
        self.max_num_frames = max_num_frames

    @staticmethod
    def open_video(video_source: str | int):
        """
        Use OpenCV to open a video source from either file or capture device.

        Args:
            video_source: filename or index referring to capture device.

        Raises:
            RuntimeError: Source is a file but file not found.
            RuntimeError: Failed to open source.
        """
        if isinstance(video_source, str) and not os.path.isfile(video_source):
            raise RuntimeError("Video file does not exist: " + video_source)
        with SuppressStderr():
            cap = cv2.VideoCapture(video_source)
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {video_source}")
        return cap

    def _get_cap(self):
        """Return the cap. If multiprocessing, create a new one. Else return the one from construction time."""
        return self.open_video(self.video_source) if self.multiprocessing else self.cap

    def get_fps(self) -> int:
        """Get the FPS of the capture device."""
        return self._get_cap().get(cv2.CAP_PROP_FPS)  # type: ignore

    def get_frame(self) -> Any:
        """Get next frame. For a file, this will be the next frame, whereas for a camera
        source, it will be the next available frame."""
        ret, frame = self._get_cap().read()
        if not ret:
            raise RuntimeError("Failed to read frame.")
        # Switch color order if desired
        if self.color_order == ColorOrder.RGB:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # move channel dim
        frame = np.moveaxis(frame, -1, self.channel_dim)
        return self.transform(frame) if self.transform is not None else frame


class VideoFileDataset(Dataset, VideoDataset):
    """
    Video dataset from file.

    This class requires that OpenCV be installed.
    """

    def __init__(self, *args, **kwargs) -> None:
        VideoDataset.__init__(self, *args, **kwargs)
        num_frames = self.get_num_frames()
        if self.max_num_frames is None or num_frames < self.max_num_frames:
            self.max_num_frames = num_frames

    @staticmethod
    def get_available_codecs() -> dict[str, str]:
        """Try different codecs, see which are available.
        Returns a dictionary with of available codecs with codecs as keys and file extensions as values."""
        if not has_cv2:
            return {}
        all_codecs = {"mp4v": ".mp4", "X264": ".avi", "H264": ".mp4", "MP42": ".mp4", "MJPG": ".mjpeg", "DIVX": ".avi"}
        codecs = {}
        with SuppressStderr():
            with tempfile.TemporaryDirectory() as tmp_dir:
                for codec, ext in all_codecs.items():
                    writer = cv2.VideoWriter()
                    fname = os.path.join(tmp_dir, f"test{ext}")
                    fourcc = cv2.VideoWriter_fourcc(*codec)  # type: ignore[attr-defined]
                    noviderr = writer.open(fname, fourcc, 1, (10, 10))
                    if noviderr:
                        codecs[codec] = ext
                    writer.release()
        return codecs

    def get_num_frames(self) -> int:
        """
        Return the number of frames in a video file.

        Raises:
            RuntimeError: no frames found.
        """
        num_frames = int(self._get_cap().get(cv2.CAP_PROP_FRAME_COUNT))
        if num_frames == 0:
            raise RuntimeError("0 frames found")
        return num_frames

    def __len__(self):
        return self.max_num_frames

    def __getitem__(self, index: int) -> Any:
        """
        Fetch single data item from index.
        """
        if self.max_num_frames is not None and index >= self.max_num_frames:
            raise IndexError
        self._get_cap().set(cv2.CAP_PROP_POS_FRAMES, index)
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
        if not has_cv2:
            return 0
        num_devices = 0
        while True:
            cap = cv2.VideoCapture(num_devices)
            if not cap.read()[0]:
                break
            num_devices += 1
            cap.release()
        return num_devices

    def __iter__(self):
        frame_count = 0
        while True:
            frame = self.get_frame()
            frame_count += 1
            yield frame
            if self.max_num_frames is not None:
                if frame_count == self.max_num_frames:
                    break
