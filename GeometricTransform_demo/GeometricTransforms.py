import os
import random
from typing import List, Dict, Optional, Sequence, Union, Tuple
from typing import Any, Callable, Hashable, Mapping, Dict, List, Union, Sequence, Optional
import cv2
import json
import numpy as np
from monai.transforms import Compose, ScaleIntensity, EnsureChannelFirst, \
    Flip, Rotate90, RepeatChannel, AsChannelLast, LoadImage, ToNumpy, ToTensor, Transpose, \
    Resize, RandAdjustContrast, GaussianSmooth


def sanity_check(image , labels_list: List[np.ndarray], line_width: int = 20):
    """
    plot the label on the image to make sure the label is correct
    Args:
        image_np: [nx, ny, 3] is a numpy array
        labels: List of labels
    Returns:
    """
    img = np.ascontiguousarray(image, dtype=np.uint8)
    for labels in labels_list:
        # random color
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        if labels.shape[0] > 2:
            new_labels = np.vstack([labels, labels[0]])
            for i in range(0, len(new_labels)-1):
                img = cv2.line(img, (int(new_labels[i][0]), int(new_labels[i][1])),
                                 (int(new_labels[i+1][0]), int(new_labels[i+1][1])),
                              color, line_width)
        else:
            img = cv2.line(img, (int(labels[0][0]), int(labels[0][1])),
                             (int(labels[1][0]), int(labels[1][1])),
                             color, line_width)
    return img


class ImageIntensityAndAnnotation(object):
    def __init__(self, gamma=2.0, prob=0.1, mode='RandAdjustContrast', sigma=1):
        self.gamma = gamma
        self.prob = prob
        self.mode = mode
        self.sigma = sigma

    def __call__(self, sample):
        image = sample['images']
        annotation = sample['labels']
        if self.mode == 'RandAdjustContrast':
            image_transform_list = Compose([Transpose([2, 0, 1]),
                                            RandAdjustContrast(prob=self.prob, gamma=self.gamma), AsChannelLast()])

        if self.mode == 'ScaleIntensity':
            image_transform_list = Compose([Transpose([2, 0, 1]),
                                            ScaleIntensity(minv=0.0, maxv=1.0, factor=None), AsChannelLast()])

        if self.mode == 'GaussianSmooth':
            image_transform_list = Compose([Transpose([2, 0, 1]),
                                            GaussianSmooth(sigma=self.sigma), AsChannelLast()])

        image_np = image_transform_list(image)
        return {'images': image_np, 'labels': annotation}


class LoadImageAndAnnotations(object):
    def __init__(self, labels: List[str]):
        self.labels = labels

    def _convert_2_points_to_bbox(self, points: np.ndarray):
        """
        convert the points to bounding box
        Args:
            points: [2,2] array
        Returns:
            bbox: [4,2] array
        """
        bbox = np.zeros((4, 2))
        bbox[0] = points[0]
        bbox[2] = points[1]
        bbox[1] = points[1][0], points[0][1]
        bbox[3] = points[0][0], points[1][1]
        return bbox

    def __call__(self, sample):
        self.image_name = sample['images']
        self.labelfile = sample['labels']
        image_transform_list = Compose([LoadImage(image_only=True), EnsureChannelFirst(),
                                        Flip(spatial_axis=1), Rotate90(), RepeatChannel(repeats=3), AsChannelLast()])
        image = image_transform_list(self.image_name)
        fid = open(self.labelfile, 'r')
        anno_dict_list = json.load(fid)['shapes']
        all_labels = []
        for _lab in self.labels:
            for _dict in anno_dict_list:
                if _dict['label'] == _lab:
                    if _lab == 'box':
                        all_labels.append(self._convert_2_points_to_bbox(np.asarray(_dict['points'])))
                    else:
                        all_labels.append(np.asarray(_dict['points']))
        return {'images': image, 'labels': all_labels}


class RotateImageAndAnnotations(object):
    def __init__(self, angle: float = 10.0):
        self.angle = angle

    def __call__(self, sample, *args, **kwargs):
        self.image = sample['images']
        self.label = sample['labels']
        rotated_image, rotated_points = self._rotate_image_points()
        return {'images': rotated_image, 'labels': rotated_points}

    def _rotate_image_points(self):
        h, w = self.image.shape[0], self.image.shape[1]
        # print(h,w)
        cX, cY = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D((cX, cY), -self.angle, 1.0)
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])

        nW = int((h * sin) + (w * cos))
        nH = int((h * cos) + (w * sin))
        M[0, 2] += (nW / 2) - cX
        M[1, 2] += (nH / 2) - cY

        calculated_point_list = []

        for _points in self.label:
            corners = _points.reshape(-1, 2)
            corners = np.hstack((corners, np.ones((corners.shape[0], 1), dtype=type(corners[0][0]))))
            new_points = np.dot(M, corners.T).T
            calculated_point_list.append(new_points)

        _image = ToNumpy()(self.image) # Ask about this added this to use warp affine
        rotated_image = cv2.warpAffine(_image, M, (nW, nH), flags=cv2.INTER_LINEAR)
        _rotated_image = ToTensor()(rotated_image)
        return _rotated_image, calculated_point_list


class FlipImageAndAnnotations(object):
    def __init__(self, flip_axes: int = 1):
        self.flip_axes = flip_axes

    def __call__(self, sample, *args, **kwargs):
        image = sample['images']
        self.label = sample['labels']
        h_center = image.shape[0] // 2
        w_center = image.shape[1] // 2
        self.img_center = (h_center, w_center)

        transform_list = Compose([Transpose([2, 0, 1]), Flip(spatial_axis=self.flip_axes),
                                  AsChannelLast()])
        flipped_image = transform_list(image)
        new_label_list = []
        for _lab in self.label:
            new_pt_list = []
            for _pt in _lab:
                _new_pt = self._flippoints(_pt)
                new_pt_list.append(_new_pt)
            new_label_list.append(np.asarray(new_pt_list))
        return {'images': flipped_image, 'labels': new_label_list}

    def _flippoints(self, points):
        new_point = [0] * 2
        if self.flip_axes == 1:
            new_point[0] = points[0] + 2 * (self.img_center[1] - points[0])
            new_point[1] = points[1]

        if self.flip_axes == 0:
            new_point[1] = points[1] + 2 * (self.img_center[0] - points[1])
            new_point[0] = points[0]

        return np.asarray(new_point)

class ResampleImageAndAnnotations(object):
    def __init__(self, output_size: Tuple[int, ...]):
        self.output_size = output_size
        self.new_width = output_size[0]
        self.new_height = output_size[1]

    def __call__(self, sample, *args, **kwargs):
        self.image = sample['images']
        self.label = sample['labels']

        self.old_width = self.image.shape[0]
        self.old_height = self.image.shape[1]

        transform_list = Compose([Transpose([2, 0, 1]), Resize(self.output_size), AsChannelLast()])
        image_np = transform_list(self.image)
        new_labels = []
        for _pt_list in self.label:
            _label = self._resample(_pt_list)
            new_labels.append(_label)

        return {'images': image_np, 'labels': new_labels}

    def _resample(self, _pt_arr):

        new_pt_list = []
        for i in range(len(_pt_arr)):
            x1 = _pt_arr[i][0]
            y1 = _pt_arr[i][1]

            new_x1 = x1 * self.new_height / self.old_height
            new_y1 = y1 * self.new_width / self.old_width
            new_pt_list.append([new_x1, new_y1])
        new_pt_arr = np.asarray(new_pt_list)
        return new_pt_arr


def main():
    sample_data = {'images': './sample_data/Mammo_194_LMLO_N_080839.jpg'
                   , 'labels': './sample_data/Mammo_194_LMLO_N_080839.json'}

    t = LoadImageAndAnnotations(labels=['pent'])(sample_data)

    transform = Compose([LoadImageAndAnnotations(labels=['pent', 'line']),
                         RotateImageAndAnnotations(angle=10.0),
                         ResampleImageAndAnnotations(output_size=(512, 512)),
                         ])

    #sanity_check(img, labels)
    x = transform(sample_data)
    img = x['images'].numpy()
    labels = x['labels']

    print("Labels: ", labels)
    for _lab in labels:
        img = sanity_check(img, _lab)
    cv2.imwrite('test.jpg', img)


def demo_flywheel():
    sample_data = {'images': './sample_data/Mammo_194_LMLO_N_080839.jpg'
        , 'labels': './sample_data/Mammo_194_LMLO_N_080839.json'}

    t = LoadImageAndAnnotations(labels=['pent', 'line', 'box'])(sample_data)
    img = sanity_check(t['images'], t['labels'])
    from matplotlib import pyplot as plt
    plt.imshow(img)

    plt.axis('off')
    plt.show()
    #cv2.imwrite('./test.jpg', img)
    print("Done")

if __name__ == '__main__':
    demo_flywheel()
