# Copyright 2020 - 2021 MONAI Consortium
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
import unittest
import warnings
from urllib.error import ContentTooShortError, HTTPError

import numpy as np
import torch
from torch.utils.data import DataLoader

import monai
from monai.apps import download_and_extract
from monai.metrics import compute_roc_auc
from monai.networks import eval_mode
from monai.networks.nets import DenseNet121
from monai.transforms import (
    AddChannel,
    Compose,
    LoadImage,
    RandFlip,
    RandRotate,
    RandZoom,
    ScaleIntensity,
    ToTensor,
    Activations,
    AsDiscrete,
)
from monai.utils import set_determinism
from tests.testing_data.integration_answers import test_integration_value
from tests.utils import DistTestCase, TimedCall, skip_if_quick

TEST_DATA_URL = "https://www.dropbox.com/s/5wwskxctvcxiuea/MedNIST.tar.gz?dl=1"
MD5_VALUE = "0bc7306e7427e00ad1c5526a6677552d"
TASK = "integration_classification_2d"


class MedNISTDataset(torch.utils.data.Dataset):
    def __init__(self, image_files, labels, transforms):
        self.image_files = image_files
        self.labels = labels
        self.transforms = transforms

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        return self.transforms(self.image_files[index]), self.labels[index]


def run_training_test(root_dir, train_x, train_y, val_x, val_y, device="cuda:0", num_workers=10):

    monai.config.print_config()
    # define transforms for image and classification
    train_transforms = Compose(
        [
            LoadImage(image_only=True),
            AddChannel(),
            ScaleIntensity(),
            RandRotate(range_x=np.pi / 12, prob=0.5, keep_size=True),
            RandFlip(spatial_axis=0, prob=0.5),
            RandZoom(min_zoom=0.9, max_zoom=1.1, prob=0.5),
            ToTensor(),
        ]
    )
    train_transforms.set_random_state(1234)
    val_transforms = Compose([LoadImage(image_only=True), AddChannel(), ScaleIntensity(), ToTensor()])
    act = Activations(softmax=True)
    to_onehot = AsDiscrete(to_onehot=True, n_classes=len(np.unique(train_y)))

    # create train, val data loaders
    train_ds = MedNISTDataset(train_x, train_y, train_transforms)
    train_loader = DataLoader(train_ds, batch_size=300, shuffle=True, num_workers=num_workers)

    val_ds = MedNISTDataset(val_x, val_y, val_transforms)
    val_loader = DataLoader(val_ds, batch_size=300, num_workers=num_workers)

    model = DenseNet121(spatial_dims=2, in_channels=1, out_channels=len(np.unique(train_y))).to(device)
    loss_function = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), 1e-5)
    epoch_num = 4
    val_interval = 1

    # start training validation
    best_metric = -1
    best_metric_epoch = -1
    epoch_loss_values = []
    metric_values = []
    model_filename = os.path.join(root_dir, "best_metric_model.pth")
    for epoch in range(epoch_num):
        print("-" * 10)
        print(f"Epoch {epoch + 1}/{epoch_num}")
        model.train()
        epoch_loss = 0
        step = 0
        for batch_data in train_loader:
            step += 1
            inputs, labels = batch_data[0].to(device), batch_data[1].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        epoch_loss /= step
        epoch_loss_values.append(epoch_loss)
        print(f"epoch {epoch + 1} average loss:{epoch_loss:0.4f}")

        if (epoch + 1) % val_interval == 0:
            with eval_mode(model):
                y_pred = torch.tensor([], dtype=torch.float32, device=device)
                y = torch.tensor([], dtype=torch.long, device=device)
                for val_data in val_loader:
                    val_images, val_labels = val_data[0].to(device), val_data[1].to(device)
                    y_pred = torch.cat([y_pred, model(val_images)], dim=0)
                    y = torch.cat([y, val_labels], dim=0)

                # compute accuracy
                acc_value = torch.eq(y_pred.argmax(dim=1), y)
                acc_metric = acc_value.sum().item() / len(acc_value)
                # compute AUC
                y_pred = act(y_pred)
                y = to_onehot(y)
                auc_metric = compute_roc_auc(y_pred, y)
                metric_values.append(auc_metric)
                if auc_metric > best_metric:
                    best_metric = auc_metric
                    best_metric_epoch = epoch + 1
                    torch.save(model.state_dict(), model_filename)
                    print("saved new best metric model")
                print(
                    f"current epoch {epoch +1} current AUC: {auc_metric:0.4f} "
                    f"current accuracy: {acc_metric:0.4f} best AUC: {best_metric:0.4f} at epoch {best_metric_epoch}"
                )
    print(f"train completed, best_metric: {best_metric:0.4f}  at epoch: {best_metric_epoch}")
    return epoch_loss_values, best_metric, best_metric_epoch


def run_inference_test(root_dir, test_x, test_y, device="cuda:0", num_workers=10):
    # define transforms for image and classification
    val_transforms = Compose([LoadImage(image_only=True), AddChannel(), ScaleIntensity(), ToTensor()])
    val_ds = MedNISTDataset(test_x, test_y, val_transforms)
    val_loader = DataLoader(val_ds, batch_size=300, num_workers=num_workers)

    model = DenseNet121(spatial_dims=2, in_channels=1, out_channels=len(np.unique(test_y))).to(device)

    model_filename = os.path.join(root_dir, "best_metric_model.pth")
    model.load_state_dict(torch.load(model_filename))
    y_true = []
    y_pred = []
    with eval_mode(model):
        for test_data in val_loader:
            test_images, test_labels = test_data[0].to(device), test_data[1].to(device)
            pred = model(test_images).argmax(dim=1)
            for i in range(len(pred)):
                y_true.append(test_labels[i].item())
                y_pred.append(pred[i].item())
    tps = [np.sum((np.asarray(y_true) == idx) & (np.asarray(y_pred) == idx)) for idx in np.unique(test_y)]
    return tps


@skip_if_quick
class IntegrationClassification2D(DistTestCase):
    def setUp(self):
        set_determinism(seed=0)
        self.data_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "testing_data")
        data_dir = os.path.join(self.data_dir, "MedNIST")
        dataset_file = os.path.join(self.data_dir, "MedNIST.tar.gz")

        if not os.path.exists(data_dir):
            try:
                download_and_extract(TEST_DATA_URL, dataset_file, self.data_dir, MD5_VALUE)
            except (ContentTooShortError, HTTPError, RuntimeError) as e:
                print(str(e))
                if isinstance(e, RuntimeError):
                    # FIXME: skip MD5 check as current downloading method may fail
                    self.assertTrue(str(e).startswith("md5 check"))
                return  # skipping this test due the network connection errors

        assert os.path.exists(data_dir)

        class_names = sorted((x for x in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, x))))
        image_files = [
            [os.path.join(data_dir, class_name, x) for x in sorted(os.listdir(os.path.join(data_dir, class_name)))]
            for class_name in class_names
        ]
        image_file_list, image_classes = [], []
        for i, _ in enumerate(class_names):
            image_file_list.extend(image_files[i])
            image_classes.extend([i] * len(image_files[i]))

        # split train, val, test
        valid_frac, test_frac = 0.1, 0.1
        self.train_x, self.train_y = [], []
        self.val_x, self.val_y = [], []
        self.test_x, self.test_y = [], []
        for i in range(len(image_classes)):
            rann = np.random.random()
            if rann < valid_frac:
                self.val_x.append(image_file_list[i])
                self.val_y.append(image_classes[i])
            elif rann < test_frac + valid_frac:
                self.test_x.append(image_file_list[i])
                self.test_y.append(image_classes[i])
            else:
                self.train_x.append(image_file_list[i])
                self.train_y.append(image_classes[i])

        self.device = "cuda:0" if torch.cuda.is_available() else "cpu:0"

    def tearDown(self):
        set_determinism(seed=None)
        try:
            os.remove(os.path.join(self.data_dir, "best_metric_model.pth"))
        except FileNotFoundError:
            warnings.warn("not found best_metric_model.pth, training skipped?")
            pass

    def train_and_infer(self, idx=0):
        results = []
        if not os.path.exists(os.path.join(self.data_dir, "MedNIST")):
            # skip test if no MedNIST dataset
            return results

        set_determinism(seed=0)
        losses, best_metric, best_metric_epoch = run_training_test(
            self.data_dir, self.train_x, self.train_y, self.val_x, self.val_y, device=self.device
        )
        infer_metric = run_inference_test(self.data_dir, self.test_x, self.test_y, device=self.device)

        print(f"integration_classification_2d {losses}")
        print("best metric", best_metric)
        print("infer metric", infer_metric)
        # check training properties
        self.assertTrue(test_integration_value(TASK, key="losses", data=losses, rtol=1e-2))
        self.assertTrue(test_integration_value(TASK, key="best_metric", data=best_metric, rtol=1e-4))
        np.testing.assert_allclose(best_metric_epoch, 4)
        model_file = os.path.join(self.data_dir, "best_metric_model.pth")
        self.assertTrue(os.path.exists(model_file))
        # check inference properties
        self.assertTrue(test_integration_value(TASK, key="infer_prop", data=np.asarray(infer_metric), rtol=1))
        results.extend(losses)
        results.append(best_metric)
        results.extend(infer_metric)
        return results

    def test_training(self):
        repeated = []
        for i in range(2):
            results = self.train_and_infer(i)
            repeated.append(results)
        np.testing.assert_allclose(repeated[0], repeated[1])

    @TimedCall(seconds=1000, skip_timing=not torch.cuda.is_available(), daemon=False)
    def test_timing(self):
        self.train_and_infer()


if __name__ == "__main__":
    unittest.main()
