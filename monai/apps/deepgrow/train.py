import os
import tempfile
import time

import torch

from monai.apps.deepgrow import (
    AddInitialSeedPoint,
    AddGuidanceSignal,
    InteractionFindDiscrepancyRegions,
    InteractionAddRandomGuidance,
    InteractionAddGuidanceSignal,
    Interaction,
    DeepgrowStatsHandler,
    DeepgrowDataset
)
from monai.data.dataloader import DataLoader
from monai.engines import SupervisedEvaluator
from monai.engines import SupervisedTrainer
from monai.handlers import (
    StatsHandler,
    TensorBoardStatsHandler,
    ValidationHandler,
    LrScheduleHandler,
    CheckpointSaver,
    MeanDice)
from monai.inferers import SimpleInferer
from monai.losses import DiceLoss
from monai.networks.nets import BasicUnet
from monai.transforms import (
    Compose,
    LoadNumpyd,
    AddChanneld,
    ToTensord,
    ToNumpyd,
    NormalizeIntensityd,
    Activationsd,
    AsDiscreted,
)

directory = os.environ.get("MONAI_DATA_DIRECTORY", '/workspace/Data/deepgrow_spleen')
root_dir = tempfile.mkdtemp() if directory is None else directory
os.makedirs(root_dir, exist_ok=True)
print(root_dir)

network = BasicUnet(dimensions=2, in_channels=3, out_channels=1, features=(64, 128, 256, 512, 1024, 64))

pre_transforms = Compose([
    LoadNumpyd(keys=('image', 'label')),
    AddChanneld(keys=('image', 'label')),
    AddInitialSeedPoint(
        label_field='label',
        positive_guidance_field='positive_guidance',
        negative_guidance_field='negative_guidance'),
    NormalizeIntensityd(keys='image', subtrahend=208.0, divisor=388.0),
    AddGuidanceSignal(
        field='image',
        positive_guidance_field='positive_guidance',
        negative_guidance_field='negative_guidance'),
    ToTensord(keys=['image', 'label'])
])

interaction_transforms = Compose([
    Activationsd(keys='pred', sigmoid=True),
    ToNumpyd(keys=['image', 'label', 'pred', 'positive_guidance', 'negative_guidance', 'p_interact']),
    InteractionFindDiscrepancyRegions(
        prediction_field='pred',
        label_field='label',
        positive_disparity_field='positive_disparity',
        negative_disparity_field='negative_disparity'),
    InteractionAddRandomGuidance(
        label_field='label',
        positive_guidance_field='positive_guidance',
        negative_guidance_field='negative_guidance',
        positive_disparity_field='positive_disparity',
        negative_disparity_field='negative_disparity',
        p_interact_field='p_interact'),
    InteractionAddGuidanceSignal(
        field='image',
        positive_guidance_field='positive_guidance',
        negative_guidance_field='negative_guidance'),
    ToTensord(keys=['image', 'label'])
])

post_transforms = Compose([
    Activationsd(keys='pred', sigmoid=True),
    AsDiscreted(keys='pred', threshold_values=True, logit_thresh=0.5)
])

train_ds = DeepgrowDataset(
    dimension=2,
    pixdim=[1.0, 1.0, 1.0],
    spatial_size=[512, 512],
    root_dir=root_dir,
    transform=pre_transforms,
    section="training",
    cache_num=0,
    limit=1,
    task="Task09_Spleen",
    download=True
)
train_loader = DataLoader(train_ds, batch_size=4, shuffle=True, num_workers=4)

val_ds = DeepgrowDataset(
    dimension=2,
    pixdim=[1.0, 1.0, 1.0],
    spatial_size=[512, 512],
    root_dir=root_dir,
    transform=pre_transforms,
    section="validation",
    cache_num=0,
    limit=1,
    task="Task09_Spleen"
)
val_loader = DataLoader(val_ds, batch_size=4, shuffle=False, num_workers=4)

device = torch.device("cuda")
network = network.to(device)

output = os.path.join(root_dir, 'output')
os.makedirs(output, exist_ok=True)

save_interval = 5
max_val_interactions = 5
max_train_interactions = 15
learning_rate = 0.0001
epochs = 5
amp = False

# define event-handlers for engine
val_handlers = [
    StatsHandler(output_transform=lambda x: None),
    TensorBoardStatsHandler(log_dir=output, output_transform=lambda x: None),
    DeepgrowStatsHandler(log_dir=output, tag_name='val_dice'),
    CheckpointSaver(save_dir=output, save_dict={"net": network}, save_key_metric=True, save_final=True,
                    save_interval=save_interval, final_filename='model.pt')
]

evaluator = SupervisedEvaluator(
    device=device,
    val_data_loader=val_loader,
    network=network,
    iteration_update=Interaction(
        transforms=interaction_transforms,
        max_interactions=max_val_interactions,
        train=False),
    inferer=SimpleInferer(),
    post_transform=post_transforms,
    key_val_metric={
        "val_dice": MeanDice(
            include_background=False,
            output_transform=lambda x: (x["pred"], x["label"])
        )
    },
    val_handlers=val_handlers
)

loss_function = DiceLoss(sigmoid=True, squared_pred=True)
optimizer = torch.optim.Adam(network.parameters(), learning_rate)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5000, gamma=0.1)

train_handlers = [
    LrScheduleHandler(lr_scheduler=lr_scheduler, print_lr=True),
    ValidationHandler(validator=evaluator, interval=1, epoch_level=True),
    StatsHandler(tag_name="train_loss", output_transform=lambda x: x["loss"]),
    TensorBoardStatsHandler(log_dir=output, tag_name="train_loss", output_transform=lambda x: x["loss"]),
    CheckpointSaver(save_dir=output, save_dict={"net": network, "opt": optimizer, "lr": lr_scheduler},
                    save_interval=save_interval, save_final=True, final_filename='checkpoint.pt'),
]
trainer = SupervisedTrainer(
    device=device,
    max_epochs=epochs,
    train_data_loader=train_loader,
    network=network,
    iteration_update=Interaction(
        transforms=interaction_transforms,
        max_interactions=max_train_interactions,
        train=True),
    optimizer=optimizer,
    loss_function=loss_function,
    inferer=SimpleInferer(),
    post_transform=post_transforms,
    amp=amp,
    key_train_metric={
        "train_dice": MeanDice(
            include_background=False,
            output_transform=lambda x: (x["pred"], x["label"])
        )
    },
    train_handlers=train_handlers,
)

start_time = time.time()
trainer.run()
end_time = time.time()

print('Total Training Time {}'.format(end_time - start_time))
