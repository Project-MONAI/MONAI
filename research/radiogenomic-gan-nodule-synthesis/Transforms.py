
import numpy as np
from monai.transforms import Transform, MapTransform
from monai.config.type_definitions import KeysCollection

class ClipIntensity(Transform):
    def __init__(
        self,
        min: int,
        max: int
    ) -> None:
        """
            Clip (limit) the values in a np array. Sets values outside min/max to limit, keeps values inbetween bounds the same.
            Args:
                max: upper bound of values in array
                min: lower bound of values in array
        """
        super().__init__()
        self.min = min
        self.max = max

    def __call__(self, data):
        return np.clip(data, self.min, self.max)

class ClipIntensityd(MapTransform):
    def __init__(
        self,
        keys: KeysCollection,
        min: int,
        max: int
    ) -> None:
        """
        Clip (limit) the values in a np array. Sets values outside min/max to limit, keeps values inbetween bounds the same.
            Args:
                keys: keys of images to clip
                max: upper bound of values in array
                min: lower bound of values in array
        """
        super().__init__(keys)
        self.clipper = ClipIntensity(min, max)

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            d[key] = self.clipper(d[key])
        return d

class CropWithBoundingBoxd(MapTransform):
    def __init__(
        self,
        keys: KeysCollection,
        bbox_key: str
    ) -> None:
        """
        Args:
            keys: keys of items to crop using bounding box values.
            bbox_key: key of bounding box datapoint in dict. 
                Should be array of integers with values [x-left, y-top, width, height]
        """
        super().__init__(keys)
        self.bbox_key = bbox_key
        self.cropper = CropWithBoundingBox()

    def __call__(self, data):
        d = dict(data)
        img_shape = d['image_meta_dict']['spatial_shape']
        bbox_values = d[self.bbox_key]
        for key in self.keys:
            d[key] = self.cropper(d[key], bbox_values, img_shape)
        return d

class CropWithBoundingBox(Transform):
    def __init__(
        self
    ) -> None:
        """
        Args:
            data: image to apply crop
            bbox_values: list-like obj with int values [x-left, y-top, width, height]
            image_shape: (width, height)
        """
        super().__init__()

    def _bbox_to_crop_coords(self, bbox_values, img_width, img_height):
        """
        bbox_values = [x-left, y-top, width, height]
        return: indexes to crop
        """
        x_left = bbox_values[0]
        y_top = bbox_values[1]
        roi_width = bbox_values[2]
        roi_height = bbox_values[3]
        r = max(roi_width, roi_height)
        center_x = int((2 * x_left + roi_width) / 2)
        center_y = int((2 * y_top + roi_height) / 2)
        y1 = max(0, center_y - r)
        y2 = min(img_height, center_y + r)
        x1 = max(0, center_x - r)
        x2 = min(img_width, center_x + r)
        return x1, x2, y1, y2

    def _crop_image_with_coords(self, image, x1, x2, y1, y2):
        return image[x1:x2, y1:y2]

    def __call__(self, data, bbox_values, image_shape):
        x1, x2, y1, y2 = self._bbox_to_crop_coords(bbox_values, image_shape[0], image_shape[1])
        img = self._crop_image_with_coords(data, x1, x2, y1, y2)
        return img
