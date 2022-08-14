#!/usr/bin/env python

import argparse
import json
import monai
import os
import torch
import torch.distributed as dist
import yaml

from monai.data import (
    Dataset,
    DataLoader,
    decollate_batch,
    partition_dataset,
)
from monai.inferers import sliding_window_inference
from monai.transforms import (
    Activationsd,
    AsDiscreted,
    Compose,
    EnsureChannelFirstd,
    LoadImaged,
    NormalizeIntensityd,
    SaveImaged,
    ScaleIntensityRanged,
    CropForegroundd,
    Invertd,
    EnsureTyped,
    CastToTyped,
    Orientationd,
    Spacingd,
    KeepLargestConnectedComponentd,
)
from torch.nn.parallel import DistributedDataParallel


class SupervisedEvaluator:
    def __init__(
        self,
        arch_ckpt,
        ckpt_path,
        data_list_key,
        data_stat,
        input_info,
        output_dir,
        repo_root,
        ts_path,
        if_keeping_largest_connected_component=False,
        if_saving_masks=False,
    ):
        self.output_dir = output_dir
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir, exist_ok=True)

        # setting for different datasets
        with open(data_stat) as f_data_stat:
            data_stat = yaml.full_load(f_data_stat)

        with open(input_info) as f_input_info:
            input_info = yaml.full_load(f_input_info)

        data_root = input_info["dataroot"]
        self.input_channels = int(
            data_stat["stats_summary"]["image_stats"]["channels"]["max"]
        )
        self.output_classes = len(data_stat["stats_summary"]["label_stats"]["labels"])

        patch_size = [96, 96, 96]
        max_shape = data_stat["stats_summary"]["image_stats"]["shape"]["max"][0]
        for _k in range(3):
            patch_size[_k] = (
                max(32, max_shape[_k] // 32 * 32)
                if max_shape[_k] < patch_size[_k]
                else patch_size[_k]
            )
        self.patch_size_valid = patch_size
        print("self.patch_size_valid", self.patch_size_valid)

        nomalizing_transform = None
        if input_info["modality"].lower() == "ct":
            intensity_upper_bound = float(
                data_stat["stats_summary"]["image_foreground_stats"]["intensity"][
                    "percentile_99_5"
                ][0]
            )
            intensity_lower_bound = float(
                data_stat["stats_summary"]["image_foreground_stats"]["intensity"][
                    "percentile_00_5"
                ][0]
            )
            # print("[info] intensity_upper_bound", intensity_upper_bound)
            # print("[info] intensity_lower_bound", intensity_lower_bound)
            nomalizing_transform = [
                ScaleIntensityRanged(
                    keys=["image"],
                    a_min=intensity_lower_bound,
                    a_max=intensity_upper_bound,
                    b_min=0.0,
                    b_max=1.0,
                    clip=True,
                ),
                CropForegroundd(keys=["image"], source_key="image"),
            ]
        else:
            spacing = data_stat["stats_summary"]["image_stats"]["spacing"]["median"]

            nomalizing_transform = [
                NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
            ]

        # define pre transforms
        pre_transforms = Compose(
            [
                LoadImaged(keys=["image"]),
                EnsureChannelFirstd(keys=["image"]),
                Orientationd(keys=["image"], axcodes="RAS"),
                Spacingd(
                    keys=["image"],
                    pixdim=(1.0, 1.0, 1.0),
                    mode=("bilinear",),
                    align_corners=(True,),
                ),
                EnsureTyped(keys=["image"]),
                CastToTyped(keys=["image"], dtype=(torch.float32)),
            ]
            + nomalizing_transform
        )

        if torch.cuda.device_count() > 1:
            # initialize the distributed training process, every GPU runs in a process
            dist.init_process_group(backend="nccl", init_method="env://")

            # dist.barrier()
            world_size = dist.get_world_size()
        else:
            world_size = 1

        # load data list (.json)
        with open(os.path.join(repo_root, input_info["datalist"])) as f:
            json_data = json.load(f)
        list_test = json_data[data_list_key]

        files = []
        for _i in range(len(list_test)):
            str_img = os.path.join(data_root, list_test[_i]["image"])

            if not os.path.exists(str_img):
                continue

            files.append({"image": str_img})
        test_files = files

        if torch.cuda.device_count() > 1:
            test_files = partition_dataset(
                data=test_files,
                shuffle=False,
                num_partitions=world_size,
                even_divisible=False,
            )[dist.get_rank()]
        print("test_files:", len(test_files))

        # define dataset and dataloader
        dataset = Dataset(data=test_files, transform=pre_transforms)
        self.dataloader = DataLoader(dataset, batch_size=1, num_workers=8)

        if torch.cuda.device_count() > 1:
            self.device = torch.device(f"cuda:{dist.get_rank()}")
        else:
            self.device = torch.device(f"cuda:0")
        torch.cuda.set_device(self.device)
        print("device", self.device)

        post_transforms = [
            EnsureTyped(keys="pred"),
            Activationsd(keys="pred", softmax=True),
            Invertd(
                keys="pred",
                transform=pre_transforms,
                orig_keys="image",
                meta_keys="pred_meta_dict",
                nearest_interp=False,
                to_tensor=True,
                device=self.device,
            ),
            monai.transforms.SplitChanneld(
                keys="pred",
                output_postfixes=[
                    "prob" + str(_k) for _k in range(self.output_classes)
                ],
                channel_dim=0,
            ),
            AsDiscreted(keys="pred", argmax=True),
            monai.transforms.Lambdad(
                keys=["pred_prob" + str(_k) for _k in range(1, self.output_classes)],
                func=lambda x: torch.ceil(x * 255.0),
                overwrite=True,
            ),
            CastToTyped(
                keys=["pred_prob" + str(_k) for _k in range(1, self.output_classes)],
                dtype=(torch.uint8),
            ),
        ]
        if if_keeping_largest_connected_component:
            post_transforms.append(
                KeepLargestConnectedComponentd(
                    keys="pred",
                    applied_labels=[_k for _k in range(1, self.output_classes)],
                    independent=False,
                )
            )
        if if_saving_masks:
            post_transforms.append(
                SaveImaged(
                    keys="pred",
                    meta_keys="pred_meta_dict",
                    output_dir=output_dir,
                    output_postfix="",
                    resample=False,
                )
            )
            for _j in range(1, self.output_classes):
                post_transforms.append(
                    SaveImaged(
                        keys="pred_prob" + str(_j),
                        meta_keys="pred_meta_dict",
                        output_dir=output_dir,
                        output_postfix="prob" + str(_j),
                        resample=False,
                    )
                )
        self.post_transforms = Compose(post_transforms)

        pretrained_ckpt = torch.load(ckpt_path, map_location=self.device)

        # network architecture
        ckpt = torch.load(arch_ckpt)
        node_a = ckpt["node_a"]
        arch_code_a = ckpt["code_a"]
        arch_code_c = ckpt["code_c"]

        dints_space = monai.networks.nets.TopologyInstance(
            channel_mul=1.0,
            num_blocks=12,
            num_depths=4,
            use_downsample=True,
            arch_code=[arch_code_a, arch_code_c],
            device=self.device,
        )

        self.net = monai.networks.nets.DiNTS(
            dints_space=dints_space,
            in_channels=self.input_channels,
            num_classes=self.output_classes,
            use_downsample=True,
            node_a=node_a,
        )
        self.net = self.net.to(self.device)

        if torch.cuda.device_count() > 1:
            if dist.get_rank() == 0:
                print("Let's use", torch.cuda.device_count(), "GPUs!")

            self.net = DistributedDataParallel(
                self.net, device_ids=[self.device], find_unused_parameters=True
            )

        if ts_path is None:
            if torch.cuda.device_count() > 1:
                # self.net.module.load_state_dict(pretrained_ckpt["state_dict"])
                self.net.load_state_dict(pretrained_ckpt)
            else:
                self.net.load_state_dict(pretrained_ckpt["state_dict"])

    def run(self):
        self.net.eval()
        with torch.no_grad():
            metric = torch.zeros(
                (self.output_classes - 1) * 2, dtype=torch.float, device=self.device
            )

            _index = 0
            for d in self.dataloader:
                torch.cuda.empty_cache()

                images = d["image"].to(self.device)

                # customized for msd
                print(
                    "d[image_meta_dict][filename_or_obj]",
                    d["image_meta_dict"]["filename_or_obj"],
                )
                # customized for kits'19
                d["image_meta_dict"]["filename_or_obj"][0] = d["image_meta_dict"]["filename_or_obj"][0].replace("/imaging.nii.gz", ".nii.gz")
                d["image_meta_dict"]["filename_or_obj"][0] = d["image_meta_dict"]["filename_or_obj"][0].replace("case_", "prediction_")
                print("d[image_meta_dict][filename_or_obj]", d["image_meta_dict"]["filename_or_obj"])

                # define sliding window size and batch size for windows inference
                with torch.cuda.amp.autocast():
                    d["pred"] = sliding_window_inference(
                        images,
                        self.patch_size_valid,
                        2,
                        lambda x: self.net(x),
                        mode="gaussian",
                        overlap=0.625,
                    )

                # decollate the batch data into a list of dictionaries, then execute postprocessing transforms
                d = [self.post_transforms(i) for i in decollate_batch(d)]

        return


def main():
    parser = argparse.ArgumentParser(description="inference")
    parser.add_argument(
        "--arch_ckpt",
        action="store",
        required=True,
        help="data root",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="checkpoint full path",
    )
    parser.add_argument(
        "--data_stat",
        action="store",
        required=True,
        help="data stat",
    )
    parser.add_argument(
        "--fold",
        action="store",
        required=True,
        help="fold index in N-fold cross-validation",
    )
    parser.add_argument(
        "--input_info",
        action="store",
        required=True,
        help="input information",
    )
    parser.add_argument(
        "--json_key",
        action="store",
        required=True,
        help="selected key in .json data list",
    )
    parser.add_argument(
        "--local_rank",
        required=int,
        help="local process rank",
    )
    parser.add_argument(
        "--num_folds",
        action="store",
        required=True,
        help="number of folds in cross-validation",
    )
    parser.add_argument(
        "--output_root",
        action="store",
        required=True,
        help="output root",
    )
    parser.add_argument(
        "--repo_root",
        action="store",
        required=True,
        help="repository root",
    )
    args = parser.parse_args()

    fold = int(args.fold)
    num_folds = int(args.num_folds)

    evaluator = SupervisedEvaluator(
        arch_ckpt=args.arch_ckpt,
        ckpt_path=args.checkpoint,
        data_list_key=args.json_key,
        data_stat=args.data_stat,
        input_info=args.input_info,
        output_dir=args.output_root,
        repo_root=args.repo_root,
        ts_path=None,
        if_keeping_largest_connected_component=False,
        if_saving_masks=True,
    )

    evaluator.run()

    return


if __name__ == "__main__":
    main()
