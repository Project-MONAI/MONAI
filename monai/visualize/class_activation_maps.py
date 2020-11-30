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

import torch

from monai.utils import ensure_tuple


class ModelWithHooks:
    """
    A model wrapper to run model forward and store intermediate forward, backward information.
    """

    def __init__(self, model, target_layers, register_forward: bool = False, register_backward: bool = False):
        self.model = model
        self.target_layers = ensure_tuple(target_layers)

        self.gradients = {}
        self.activations = {}
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


class CAM:
    def __init__(self, model, target_layers):
        self.net = ModelWithHooks(model, target_layers, register_forward=True)

    def norm_features(self, feature):
        feature -= feature.min()
        feature /= feature.max()
        return 1.0 - feature

    def __call__(self, x, class_idx=None):
        logits, acti, _ = self.net(x)
        acti = acti[0]
        if class_idx is None:
            class_idx = torch.argmax(logits, dim=1)
        b, c, *spatial = acti.shape
        weight = self.net.model.fc.weight[class_idx].view(c, *[1 for _ in spatial])
        map = [self.norm_features((weight * a).sum(0)) for a in acti]
        return torch.stack(map, dim=0)


class GradCAM:
    def __init__(self, model, target_layers):
        self.net = ModelWithHooks(model, target_layers, register_forward=True, register_backward=True)

    def norm_features(self, feature):
        feature -= feature.min()
        feature /= feature.max()
        return 1.0 - feature

    def __call__(self, x):
        logits, acti, grad = self.net(x)
        acti = acti[0]
        grad = grad[0]
        b, c, *spatial = grad.shape
        grad_ave = grad.view(b, c, -1).mean(2)
        weights = grad_ave.view(b, c, 1, 1)
        map = (weights * acti).sum(1)
        return self.norm_features(map)


# if __name__ == "__main__":
#     from torchvision import transforms
#     import glob
#     import numpy as np
#     import PIL
#     import cv2
#     from matplotlib import pyplot as plt
#     import argparse
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
#         cam_img = cam_computer(img_tensor)[0].detach().cpu().numpy()
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
