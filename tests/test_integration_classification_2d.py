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

import os
import shutil
import subprocess
import tarfile
import tempfile
import unittest

import numpy as np
import torch
from torch.utils.data import DataLoader

import monai
from monai.metrics import compute_roc_auc
from monai.networks.nets import densenet121
from monai.transforms import AddChannel, Compose, LoadPNG, RandFlip, RandRotate, RandZoom, ScaleIntensity, ToTensor
from monai.utils import set_determinism
from tests.utils import skip_if_quick

TEST_DATA_URL = "https://www.dropbox.com/s/5wwskxctvcxiuea/MedNIST.tar.gz"


class MedNISTDataset(torch.utils.data.Dataset):
    def __init__(self, image_files, labels, transforms):
        self.image_files = image_files
        self.labels = labels
        self.transforms = transforms

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        return self.transforms(self.image_files[index]), self.labels[index]


def run_training_test(root_dir, train_x, train_y, val_x, val_y, device=torch.device("cuda:0")):

    monai.config.print_config()
    # define transforms for image and classification
    train_transforms = Compose(
        [
            LoadPNG(image_only=True),
            AddChannel(),
            ScaleIntensity(),
            RandRotate(range_x=15, prob=0.5, keep_size=True),
            RandFlip(spatial_axis=0, prob=0.5),
            RandZoom(min_zoom=0.9, max_zoom=1.1, prob=0.5),
            ToTensor(),
        ]
    )
    train_transforms.set_random_state(1234)
    val_transforms = Compose([LoadPNG(image_only=True), AddChannel(), ScaleIntensity(), ToTensor()])

    # create train, val data loaders
    train_ds = MedNISTDataset(train_x, train_y, train_transforms)
    train_loader = DataLoader(train_ds, batch_size=300, shuffle=True, num_workers=10)

    val_ds = MedNISTDataset(val_x, val_y, val_transforms)
    val_loader = DataLoader(val_ds, batch_size=300, num_workers=10)

    model = densenet121(spatial_dims=2, in_channels=1, out_channels=len(np.unique(train_y))).to(device)
    loss_function = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), 1e-5)
    epoch_num = 4
    val_interval = 1

    # start training validation
    best_metric = -1
    best_metric_epoch = -1
    epoch_loss_values = list()
    metric_values = list()
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
            model.eval()
            with torch.no_grad():
                y_pred = torch.tensor([], dtype=torch.float32, device=device)
                y = torch.tensor([], dtype=torch.long, device=device)
                for val_data in val_loader:
                    val_images, val_labels = val_data[0].to(device), val_data[1].to(device)
                    y_pred = torch.cat([y_pred, model(val_images)], dim=0)
                    y = torch.cat([y, val_labels], dim=0)
                auc_metric = compute_roc_auc(y_pred, y, to_onehot_y=True, softmax=True)
                metric_values.append(auc_metric)
                acc_value = torch.eq(y_pred.argmax(dim=1), y)
                acc_metric = acc_value.sum().item() / len(acc_value)
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


def run_inference_test(root_dir, test_x, test_y, device=torch.device("cuda:0")):
    # define transforms for image and classification
    val_transforms = Compose([LoadPNG(image_only=True), AddChannel(), ScaleIntensity(), ToTensor()])
    val_ds = MedNISTDataset(test_x, test_y, val_transforms)
    val_loader = DataLoader(val_ds, batch_size=300, num_workers=10)

    model = densenet121(spatial_dims=2, in_channels=1, out_channels=len(np.unique(test_y))).to(device)

    model_filename = os.path.join(root_dir, "best_metric_model.pth")
    model.load_state_dict(torch.load(model_filename))
    model.eval()
    y_true = list()
    y_pred = list()
    with torch.no_grad():
        for test_data in val_loader:
            test_images, test_labels = test_data[0].to(device), test_data[1].to(device)
            pred = model(test_images).argmax(dim=1)
            for i in range(len(pred)):
                y_true.append(test_labels[i].item())
                y_pred.append(pred[i].item())
    tps = [np.sum((np.asarray(y_true) == idx) & (np.asarray(y_pred) == idx)) for idx in np.unique(test_y)]
    return tps


class IntegrationClassification2D(unittest.TestCase):
    def setUp(self):
        set_determinism(seed=0)
        self.data_dir = tempfile.mkdtemp()

        # download
        subprocess.call(["wget", "-nv", "-P", self.data_dir, TEST_DATA_URL])
        dataset_file = os.path.join(self.data_dir, "MedNIST.tar.gz")
        assert os.path.exists(dataset_file)

        # extract tarfile
        datafile = tarfile.open(dataset_file)
        datafile.extractall(path=self.data_dir)
        datafile.close()

        # find image files and labels
        data_dir = os.path.join(self.data_dir, "MedNIST")
        class_names = sorted([x for x in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, x))])
        image_files = [
            [os.path.join(data_dir, class_name, x) for x in sorted(os.listdir(os.path.join(data_dir, class_name)))]
            for class_name in class_names
        ]
        image_file_list, image_classes = [], []
        for i, class_name in enumerate(class_names):
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

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu:0")

    def tearDown(self):
        set_determinism(seed=None)
        shutil.rmtree(self.data_dir)

    @skip_if_quick
    def test_training(self):
        repeated = []
        for i in range(2):
            torch.manual_seed(0)

            repeated.append([])
            losses, best_metric, best_metric_epoch = run_training_test(
                self.data_dir, self.train_x, self.train_y, self.val_x, self.val_y, device=self.device
            )

            # check training properties
            np.testing.assert_allclose(
                losses, [0.7797081090842083, 0.16179659706392105, 0.07446704363557184, 0.045996826011568875], rtol=1e-3
            )
            repeated[i].extend(losses)
            print("best metric", best_metric)
            np.testing.assert_allclose(best_metric, 0.9999268330306007, rtol=1e-4)
            repeated[i].append(best_metric)
            np.testing.assert_allclose(best_metric_epoch, 4)
            model_file = os.path.join(self.data_dir, "best_metric_model.pth")
            self.assertTrue(os.path.exists(model_file))

            infer_metric = run_inference_test(self.data_dir, self.test_x, self.test_y, device=self.device)

            # check inference properties
            np.testing.assert_allclose(np.asarray(infer_metric), [1031, 895, 981, 1033, 960, 1047], atol=1)
            repeated[i].extend(infer_metric)

        np.testing.assert_allclose(repeated[0], repeated[1])


if __name__ == "__main__":
    unittest.main()
