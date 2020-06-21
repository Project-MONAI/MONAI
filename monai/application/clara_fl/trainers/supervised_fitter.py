import torch
from monai.utils import set_determinism
from monai.transforms import Compose, LoadNiftid, AddChanneld, ScaleIntensityRanged, CropForegroundd, \
    RandCropByPosNegLabeld, RandShiftIntensityd, ToTensord, Activationsd, AsDiscreted
from monai.data import load_decathalon_datalist, CacheDataset, DataLoader
from monai.networks.nets import UNet
from monai.networks.layers import Norm
from monai.losses import DiceLoss
from monai.handlers import StatsHandler, TensorBoardStatsHandler, ValidationHandler, \
    LrScheduleHandler, CheckpointSaver, MeanDice
from monai.engines import SupervisedTrainer, SupervisedEvaluator
from monai.inferers import SimpleInferer, SlidingWindowInferer
from ignite.metrics import Accuracy


class SupervisedFitter:
    def __init__(self, num_epochs):
        self.num_epochs = num_epochs
        self.multi_gpu = False
        self.net = UNet(dimensions=3, in_channels=1, out_channels=2, channels=(16, 32, 64, 128, 256),
                        strides=(2, 2, 2, 2), num_res_units=2, norm=Norm.BATCH)

        class TrainContext:
            def __init__(self):
                self.my_rank = 0
                self.global_round = 0
                self.total_steps = 0

        self.train_ctx = TrainContext()
        # define transforms for training and validation
        train_transforms = Compose([
            LoadNiftid(keys=("image", "label")),
            AddChanneld(keys=("image", "label")),
            ScaleIntensityRanged(keys="image", a_min=-57, a_max=164, b_min=0.0, b_max=1.0, clip=True),
            CropForegroundd(keys=("image", "label"), source_key="image"),
            # randomly crop out patch samples from big image based on pos / neg ratio
            # the image centers of negative samples must be in valid image area
            RandCropByPosNegLabeld(keys=("image", "label"), label_key="label", size=(96, 96, 96), pos=1,
                                   neg=1, num_samples=4, image_key="image", image_threshold=0),
            RandShiftIntensityd(keys="image", offsets=0.1, prob=0.5),
            ToTensord(keys=("image", "label"))
        ])
        val_transforms = Compose([
            LoadNiftid(keys=("image", "label")),
            AddChanneld(keys=("image", "label")),
            ScaleIntensityRanged(keys="image", a_min=-57, a_max=164, b_min=0.0, b_max=1.0, clip=True),
            CropForegroundd(keys=("image", "label"), source_key="image"),
            ToTensord(keys=("image", "label"))
        ])

        # define dataset and dataloader
        data_list = "/workspace/data/medical/spleen/dataset_0.json"
        data_root = "/workspace/data/medical/spleen"
        train_datalist = load_decathalon_datalist(data_list, True, "training", data_root)
        train_ds = CacheDataset(train_datalist, train_transforms, 32, 0.0, 4)
        # use batch_size=2 to load images and use RandCropByPosNegLabeld to generate 2 x 4 images for network training
        self.train_loader = DataLoader(train_ds, batch_size=2, shuffle=True, num_workers=4)

        val_datalist = load_decathalon_datalist(data_list, True, "validation", data_root)
        val_ds = CacheDataset(val_datalist, val_transforms, 9, 0.0, 4)
        self.val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=4)

    def get_train_context(self):
        return self.train_ctx

    def tf_reset(self):
        pass

    def fit(self):
        set_determinism(seed=0)
        self.train_ctx.global_round += 1

        # define training components
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net = self.net.to(device)
        loss_function = DiceLoss(to_onehot_y=True, softmax=True)
        optimizer = torch.optim.Adam(self.net.parameters(), 1e-4)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5000, gamma=0.1)

        # define post transforms for model output
        val_post_transforms = Compose(
            [
                Activationsd(keys="pred", softmax=True),
                AsDiscreted(keys=("pred", "label"), argmax=(True, False), to_onehot=True, n_classes=2)
            ]
        )
        # define event-handlers for engine
        val_handlers = [
            StatsHandler(output_transform=lambda x: None)
        ]
        evaluator = SupervisedEvaluator(
            device=device,
            val_data_loader=self.val_loader,
            network=self.net,
            inferer=SlidingWindowInferer(roi_size=(160, 160, 160), sw_batch_size=2, overlap=0.5),
            post_transform=val_post_transforms,
            key_val_metric={"val_mean_dice": MeanDice(output_transform=lambda x: (x["pred"], x["label"]))},
            additional_metrics={"val_acc": Accuracy(output_transform=lambda x: (x["pred"], x["label"]))},
            val_handlers=val_handlers
        )
        train_post_transforms = Compose(
            [
                Activationsd(keys="pred", softmax=True),
                AsDiscreted(keys=("pred", "label"), argmax=(True, False), to_onehot=True, n_classes=2)
            ]
        )
        train_handlers = [
            LrScheduleHandler(lr_scheduler=lr_scheduler, print_lr=True),
            ValidationHandler(validator=evaluator, interval=1, epoch_level=True),
            StatsHandler(tag_name="train_loss", output_transform=lambda x: x["loss"])
        ]
        self.trainer = SupervisedTrainer(
            device=device,
            max_epochs=self.num_epochs,
            train_data_loader=self.train_loader,
            network=self.net,
            optimizer=optimizer,
            loss_function=loss_function,
            inferer=SimpleInferer(),
            post_transform=train_post_transforms,
            key_train_metric={"train_acc": Accuracy(output_transform=lambda x: (x["pred"], x["label"]))},
            train_handlers=train_handlers,
        )
        self.train_ctx.total_steps = self.trainer.state.epoch_length
        self.trainer.run()
        return True
