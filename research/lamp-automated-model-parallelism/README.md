# LAMP: Large Deep Nets with Automated Model Parallelism for Image Segmentation

<p>
<img src="./fig/acc_speed_han_0_5hor.png" alt="LAMP on Head and Neck Dataset" width="500"/>
</p>


> If you use this work in your research, please cite the paper.

A reimplementation of the LAMP system originally proposed by:

Wentao Zhu, Can Zhao, Wenqi Li, Holger Roth, Ziyue Xu, and Daguang Xu (2020)
"LAMP: Large Deep Nets with Automated Model Parallelism for Image Segmentation."
MICCAI 2020 (Early Accept, paper link: TBC)


## To run the demo:

### Prerequisites
- install the latest version of MONAI: `git clone https://github.com/Project-MONAI/MONAI` and `pip install -e .`.
- `pip install torchgpipe`

### Data
Head and Neck CT dataset
please download and unzip the images into `./data` folder.

- `HaN.zip`: https://drive.google.com/file/d/1A2zpVlR3CkvtkJPvtAF3-MH0nr1WZ2Mn/view?usp=sharing

Please find more details at https://github.com/wentaozhu/AnatomyNet-for-anatomical-segmentation.git


### Hardware
U-Net-32 2x16G, U-Net-64 4x16G, U-Net-128 2x32G


### Commands
The number of features in the first block (--n_feat) can be 32, 64, or 128.
```bash
    mkdir ./log;
    python train.py --n_feat=128 --crop_size='64,64,64' --bs=16 --ep=4800 > ./log/YOURLOG.log
    python train.py --n_feat=128 --crop_size='128,128,128' --bs=4 --ep=1200 --pretrain='./model/BESTMODELFROM64,64,64' > ./log/YOURLOG.log
    python train_1.py --n_feat=128 --crop_size='-1,-1,-1' --bs=1 --ep=300 --pretrain='./model/BESTMODELFROM128,128,128' > ./log/YOURLOG.log
```
