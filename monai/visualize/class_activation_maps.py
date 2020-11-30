# Copyright 2020 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Callable, Dict

import numpy as np
import torch
import torch.nn.functional as F

from monai.transforms import ScaleIntensity
from monai.utils import InterpolateMode, ensure_tuple


class ModelWithHooks:
    """
    A model wrapper to run model forward/backward steps and storing some intermediate feature/gradient information.
    """

    def __init__(self, model, target_layer_names, register_forward: bool = False, register_backward: bool = False):
        """

        Args:
            model: the model to be wrapped.
            target_layer_names: the names of the layer to
            register_forward: whether to cache the forward pass output corresponding to `target_layer_names`.
            register_backward: whether to cache the backward pass output corresponding to `target_layer_names`.
        """
        self.model = model
        self.target_layers = ensure_tuple(target_layer_names)

        self.gradients: Dict[str, torch.Tensor] = {}
        self.activations: Dict[str, torch.Tensor] = {}
        self.register_backward = register_backward
        self.register_forward = register_forward

        for name, mod in model.named_modules():
            if name not in self.target_layers:
                continue
            if self.register_backward:
                mod.register_backward_hook(self.backward_hook(name))
            if self.register_forward:
                mod.register_forward_hook(self.forward_hook(name))

    def backward_hook(self, name):
        def _hook(_module, _grad_input, grad_output):
            self.gradients[name] = grad_output[0]

        return _hook

    def forward_hook(self, name):
        def _hook(_module, _input, output):
            self.activations[name] = output

        return _hook

    def class_score(self, logits, class_idx=None):
        if class_idx is not None:
            return logits[:, class_idx].squeeze()
        return logits[:, logits.argmax(1)].squeeze()

    def __call__(self, x, class_idx=None, retain_graph=False):
        logits = self.model(x)
        acti, grad = None, None
        if self.register_forward:
            acti = tuple(self.activations[layer] for layer in self.target_layers)
        if self.register_backward:
            score = self.class_score(logits, class_idx)
            self.model.zero_grad()
            score.backward(retain_graph=retain_graph)
            grad = tuple(self.gradients[layer] for layer in self.target_layers)
        return logits, acti, grad


def default_upsampler(spatial_size) -> Callable[[torch.Tensor], torch.Tensor]:
    """
    A linear interpolation method for upsampling the feature map.
    The output of this function is a callable `func`,
    such that `func(activation_map)` returns an upsampled tensor.
    """

    def up(acti_map):
        linear_mode = [InterpolateMode.LINEAR, InterpolateMode.BILINEAR, InterpolateMode.TRILINEAR]
        interp_mode = linear_mode[len(spatial_size) - 1]
        return F.interpolate(acti_map, size=spatial_size, mode=str(interp_mode.value), align_corners=False)

    return up


def default_normalizer(acti_map) -> np.ndarray:
    """
    A linear intensity scaling by mapping the (min, max) to (1, 0).
    """
    if isinstance(acti_map, torch.Tensor):
        acti_map = acti_map.detach().cpu().numpy()
    scaler = ScaleIntensity(minv=1.0, maxv=0.0)
    acti_map = [scaler(x) for x in acti_map]
    return np.stack(acti_map, axis=0)


class CAM:
    def __init__(
        self,
        model,
        target_layers,
        fc_weights=lambda m: m.fc.weight,
        upsampler=default_upsampler,
        postprocessing: Callable = default_normalizer,
    ):
        """

        Args:
            model: the model to be visualised
            target_layers: name of the model layer to generate the feature map.
            fc_weights: a callable used to get fully-connected weights to compute activation map
                from the target_layers (without pooling). The default is `lambda m: m.fc.weight`, that is
                getting the fully-connected layer by `model.fc.weight`.
            upsampler: an upsampling method to upsample the feature map.
            postprocessing: a callable that applies on the upsampled feature map.
        """
        self.net = ModelWithHooks(model, target_layers, register_forward=True)
        self.upsampler = upsampler
        self.postprocessing = postprocessing
        self.fc_weights = fc_weights

    def compute_map(self, x, class_idx=None):
        logits, acti, _ = self.net(x)
        acti = acti[0]
        if class_idx is None:
            class_idx = torch.argmax(logits, dim=1)
        b, c, *spatial = acti.shape
        weights = self.fc_weights(self.net.model)
        weights = weights[class_idx].view(c, *[1 for _ in spatial])
        return (weights * acti).sum(1, keepdim=True)

    def feature_map_size(self, input_size, device="cpu"):
        return self.compute_map(torch.zeros(*input_size, device=device)).shape

    def __call__(self, x, class_idx=None):
        acti_map = self.compute_map(x, class_idx)
        if self.upsampler:
            img_spatial = x.shape[2:]
            acti_map = self.upsampler(img_spatial)(acti_map)
        if self.postprocessing:
            acti_map = self.postprocessing(acti_map)
        return acti_map


class GradCAM:
    def __init__(self, model, target_layers, upsampler=default_upsampler, postprocessing=default_normalizer):
        """

        Args:
            model: the model to be visualised
            target_layers: name of the model layer to generate the feature map.
            upsampler: an upsampling method to upsample the feature map.
            postprocessing: a callable that applies on the upsampled feature map.
        """
        self.net = ModelWithHooks(model, target_layers, register_forward=True, register_backward=True)
        self.upsampler = upsampler
        self.postprocessing = postprocessing

    def compute_map(self, x, class_idx=None, retain_graph=False):
        logits, acti, grad = self.net(x, class_idx=class_idx, retain_graph=retain_graph)
        acti, grad = acti[0], grad[0]
        b, c, *spatial = grad.shape
        grad_ave = grad.view(b, c, -1).mean(2)
        weights = grad_ave.view(b, c, *[1 for _ in spatial])
        acti_map = (weights * acti).sum(1, keepdim=True)
        return F.relu(acti_map)

    def feature_map_size(self, input_size, device="cpu"):
        return self.compute_map(torch.zeros(*input_size, device=device)).shape

    def __call__(self, x, class_idx=None, retain_graph=False):
        acti_map = self.compute_map(x, class_idx=class_idx, retain_graph=retain_graph)
        if self.upsampler:
            img_spatial = x.shape[2:]
            acti_map = self.upsampler(img_spatial)(acti_map)
        if self.postprocessing:
            acti_map = self.postprocessing(acti_map)
        return acti_map


# if __name__ == "__main__":
#     import argparse
#     import glob
#
#     import cv2
#     import numpy as np
#     import PIL
#     from matplotlib import pyplot as plt
#     from torchvision import transforms
#
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--device", default="0", type=str)
#     parser.add_argument(
#         "--target_layer",
#         default="layer4",
#         type=str,
#         help="Specify the name of the target layer (before the pooling layer)",
#     )
#     parser.add_argument(
#         "--final_layer", default="fc", type=str, help="Specify the name of the last classification layer"
#     )
#     args = parser.parse_args()
#     device = torch.device("cuda:" + args.device) if torch.cuda.is_available() else torch.device("cpu")
#     model = torch.load("temp/resnet-cam.pt", map_location=device)
#     # print(model)
#     if torch.cuda.is_available():
#         model.cuda()
#     model.eval()
#     # cam_computer = CAM(model, target_layers=[args.target_layer, args.final_layer])
#     cam_computer = GradCAM(model, target_layers=args.target_layer)
#     resize_param = (224, 224)
#     norm_mean = [0.5528, 0.5528, 0.5528]
#     norm_std = [0.1583, 0.1583, 0.1583]
#     disp_size = 10
#     preprocess = transforms.Compose(
#         [transforms.Resize(resize_param), transforms.ToTensor(), transforms.Normalize(norm_mean, norm_std)]
#     )
#     plt.figure(figsize=(disp_size, disp_size))
#     for i, file in enumerate(glob.glob("./temp/test_images/*")):
#         image = PIL.Image.open(file)
#         h, w, b = np.shape(np.array(image))
#         img_tensor = preprocess(image).unsqueeze(0).to(device)
#         cam_img = cam_computer(img_tensor)[0, 0]
#         img = np.array(image)
#         cam_img = cv2.resize(cam_img, (h, w), interpolation=cv2.INTER_CUBIC)
#         cam_img = np.uint8(cam_img * 255)
#         height, width, _ = img.shape
#         heatmap = cv2.applyColorMap(cam_img, cv2.COLORMAP_JET)
#         result = heatmap * 0.3 + img * 0.6
#         plt.subplot(2, 1, i + 1)
#         plt.imshow(result.astype(np.int))
#
#     plt.show()
