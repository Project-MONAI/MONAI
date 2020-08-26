# Radiogenomic GAN for End-To-End Nodule Image Generation and Radiogenomic Map Learning

> If you use this work in your research, please cite the paper.

A re-implementation of the Radiogenomic-GAN originally proposed by:

Z. Xu, X. Wang, H. Shin, D. Yang, H. Roth, F.
Milletari, L. Zhang, D. Xu (2019) "Correlation via synthesis: end-to-end nodule image generation and radiogenomic map learning based on generative adversarial network. 2020. DOI: [1907.03728](https://arxiv.org/pdf/1907.03728.pdf)

This research prototype neural network is adapted from:

- [MC-GAN Code Repository](https://github.com/HYOJINPARK/MC_GAN)

**Downloading the Data**

The research was done with the [NSCLC-Radiogenomics dataset](https://wiki.cancerimagingarchive.net/display/Public/NSCLC+Radiogenomics), which contains CT scans, PET/CT scans, semantic tumor annotations, gene mutation, RNA sequences from excised tissue, and survival outcomes from 211 patients with Non-Small Cell Lung Cancer.

To train the model, visit the link above and download:

- Images and Segmentations (DICOM, 97.6 GB)	
- Clinical Data (csv)	
- RNA sequence data (web)

**Preparing the Dataset**

- Todo

**Training the Model**

- Install the latest version of MONAI: [Installation Guide](https://docs.monai.io/en/latest/installation.html)
- Complete the data preperation steps above.
- Place cleaned data in `./data` or adjust path in training script. 
- Execute `python run_training.py` to start RGGAN training, the generation results will be saved in `./ModelOut`.

**Running Model Inference** 

- Todo: Improve evaluation script.
