# GAN for End-To-End Nodule Image Generation and Radiogenomic Map Learned

> If you use this work in your research, please cite the paper.

A re-implementation of the Radiogenomic-GAN originally proposed by:

Z. Xu, X. Wang, H. Shin, D. Yang, H. Roth, F.
Milletari, L. Zhang, D. Xu (2019) "Correlation via synthesis: end-to-end nodule image generation and radiogenomic map learning based on generative adversarial network. 2020. DOI: [1907.03728](https://arxiv.org/pdf/1907.03728.pdf)

This research prototype neural network is adapted from:

- [MC-GAN Code Repository](https://github.com/HYOJINPARK/MC_GAN)

**Downloading the Data**

The network uses the [NSCLC-Radiogenomics dataset](https://wiki.cancerimagingarchive.net/display/Public/NSCLC+Radiogenomics), which contains CT scans, PET/CT scans, semantic tumor annotations, gene mutation, RNA sequences from excised tissue, and survival outcomes from 211 patients with Non-Small Cell Lung Cancer. 

**Preparing the Dataset**

- todo

**Training the Model**

- install the latest version of MONAI: `git clone https://github.com/Project-MONAI/MONAI` and `pip install -e .`.
- extract the open source dataset to `./images`.
- run `python run_training.py` and the generation results will be saved at `./output`.
- todo

**Running Model Inference** 

- todo
