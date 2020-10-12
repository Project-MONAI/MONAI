# Radiogenomic GAN for End-To-End Nodule Image Generation and Radiogenomic Map Learning

> If you use this work in your research, please cite the paper.

A re-implementation of the Radiogenomic-GAN originally proposed by:

Ziyue Xu, X. Wang, H. Shin, D. Yang, H. Roth, F. Milletari, L. Zhang, D. Xu (2019) "Correlation via synthesis: end-to-end nodule image generation and radiogenomic map learning based on generative adversarial network. 2020. DOI: [1907.03728](https://arxiv.org/pdf/1907.03728.pdf)

This research prototype network is adapted from:

- [MC-GAN Code Repository](https://github.com/HYOJINPARK/MC_GAN)

## Running the demo:

### NSCLC Radiogenomics dataset

The original research uses the [NSCLC-Radiogenomics dataset](https://wiki.cancerimagingarchive.net/display/Public/NSCLC+Radiogenomics), which contains PET/CT scans, semantic tumor annotations, RNA sequences from surgically excised tissue, and survival outcomes from 211 patients with Non-Small Cell Lung Cancer. From the dataset authors: 

>  This dataset was created to facilitate the discovery of the underlying relationship between genomic and medical image features, as well as the development and evaluation of prognostic medical image biomarkers.
from

### Preparing the data

1. Create an input folder called `data` in this research directory. (You can use an alternate location and specify it later.)
2. Visit the dataset link above and download these files:
  - Images and Segmentations (DICOM, 97.6 GB)
     - Use the [NBIA Data Retriever](https://wiki.cancerimagingarchive.net/display/NBIA/Downloading+TCIA+Images) to parse the `.tcia` file and download the patient data into a `NSCLC Radiogenomic` folder. This will take some time. After the download completes, place the `NSCLC Radiogenomic` folder into your input data folder. We will only need the `R01` patients.
  - RNA sequence data (`GSE103584_R01_NSCLC_RNAseq.txt.gz`)
     - Copy the gz file into the input data directory.


### Train the Model

1. Install the latest version of MONAI: [Installation Guide](https://docs.monai.io/en/latest/installation.html)
2. Complete the data preperation steps above.
3. Execute `python run_training.py` to start training, the generation results will be saved in `./ModelOut`.

#### Example execution

Users can control the training script with args to tune network architecture and training hyperparameters.

```bash
# Default training with CLI output saved in text file
python run_training.py > rggan_train_log.txt
# Changing the input data directory
python run_training.py --input /datapool/NSCLC_data > log.txt
# Reducing CacheDataset and DataLoader computer load
python run_training.py --cr 0.5 --workers 1
# Adjusting training loss coefficients and increase the number of image featuremaps.
python run_training.py --c_i 1.5 --c_is 15.0 --c_isc 8.0 --g_n_feat 64 --d_n_feat 128
```

#### Training arguments

Viewable in terminal with `python run_training.py --help`.

| Arg Flag        | Description                                            | Default    |
| --------------- | ------------------------------------------------------ | ---------- |
| --device        | device type string. e.g. 'cuda:0' or 'cpu'.            | cuda:0     |
| --ep            | Number of training epochs.                             | 500        |
| --bs            | Batch size.                                            | 16         |
| --seed          | Random seed.                                           | 12345      |
| --input         | Location of data workspace.                            | ./data     |
| --output        | Model output folder.                                   | ./ModelOut |
| --save_interval | Save checkpoints every N epochs.                       | 50         |
| --g_lr          | Loss rate for G ADAM optimizer                         | 0.0001     |
| --d_lr          | Loss rate for D ADAM optimizer                         | 0.0001     |
| --g_n_feat      | Number of image feature maps for G.                    | 32         |
| --d_n_feat      | Number of image feature maps for D.                    | 64         |
| --embed_dim     | Size of rna_sequence embedding from g Encoder Network. | 128        |
| --g_ls          | Size of random latent input for GenNet.                | 10         |
| --code_kernel   | Size of kernel for evaluating embedding and net codes. | 8          |
| --c_i           | Image loss coefficient.                                | 8.0        |
| --c_is          | Image-Segment loss coefficient.                        | 1.0        |
| --c_isc         | Image-Segment-Code loss coefficient.                   | 10.0       |
| --c_bg          | BgReconstruction loss coefficient.                     | 100        |
| --workers       | Number of processing units for DataLoader.             | 4          |
| --cr            | Cache ratio for MONAI CacheDataset.                    | 1.0        |

### Running Model Inference

- TODO: Improve evaluation script.
