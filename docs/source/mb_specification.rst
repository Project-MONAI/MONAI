
==========================
MONAI Bundle Specification
==========================

Overview
========

This is the specification for the MONAI Bundle (MB) format of portable described deep learning models. The objective of a MB is to define a packaged network or model which includes the critical information necessary to allow users and  programs to understand how the model is used and for what purpose. A bundle includes the stored weights of a model as a pickled state dictionary and/or a Torchscript object. Additional JSON files are included to store metadata about the model, information for constructing training, inference, and post-processing transform sequences, plain-text description, legal information, and other data the model creator wishes to include.

This specification defines the directory structure a bundle must have and the necessary files it must contain. Additional files may be included and the directory packaged into a zip file or included as extra files directly in a Torchscript file.

Directory Structure
===================

A MONAI Bundle is defined primarily as a directory with a set of specifically named subdirectories containing the model and metadata files. The root directory should be named for the model, given as "ModelName" in this exmaple, and should contain the following structure:

::

  ModelName
  ┣━ configs
  ┃  ┗━ metadata.json
  ┣━ models
  ┃  ┣━ model.pt
  ┃  ┗━ model.ts
  ┗━ docs
     ┣━ README.md
     ┗━ license.txt


These files mostly are required to be present with the given names for the directory to define a valid bundle:

* **metadata.json**: metadata information in JSON format relating to the type of model, definition of input and output tensors, versions of the model and used software, and other information described below.
* **model.pt**: the state dictionary of a saved model, the information to instantiate the model must be found in the metadata file.
* **model.ts**: the Torchscript saved model if the model is compatible with being saved correctly in this format.
* **README.md**: plain-language information on the model, how to use it, author information, etc. in Markdown format.
* **license.txt**: software license attached to the model, can be left blank if no license needed.

Archive Format
==============

The bundle directory and its contents can be compressed into a zip file to constitute a single file package. When unzipped into a directory this file will reproduce the above directory structure, and should itself also be named after the model it contains.

The Torchscript file format is also just a zip file with a specific structure. When creating such an archive with `save_net_with_metadata` a MB-compliant Torchscript file can be created by including the contents of `metadata.json` as the `meta_values` argument of the function, and other files included as `more_extra_files` entries. These will be stored in a `extras` directory in the zip file and can be retrieved with `load_net_with_metadata` or with any other library/tool that can read zip data. In this format the `model.*` files are obviously not needed by `README.md` and `license.txt` can be added as more extra files.

metadata.json File
==================

This file contains the metadata information relating to the model, including what the shape and format of inputs and outputs are, what the meaning of the outputs are, what type of model is present, and other information. The JSON structure is a dictionary containing a defined set of keys with additional user-specified keys. The mandatory keys are as follows:

* **version**: version of the stored model, this allows multiple versions of the same model to be differentiated.
* **monai_version**: version of MONAI the bundle was generated on, later versions expected to work.
* **pytorch_version**: version of Pytorch the bundle was generated on, later versions expected to work.
* **numpy_version**: version of Numpy the bundle was generated on, later versions expected to work.
* **optional_packages_version**: dictionary relating optional package names to their versions, these packages are not needed but are recommended to be installed with this stated minimum version.
* **task**: plain-language description of what the model is meant to do.
* **description**: longer form plain-language description of what the model is, what it does, etc.
* **authorship**: state author(s) of the model.
* **copyright**: state model copyright.
* **network_data_format**: defines the format, shape, and meaning of inputs and outputs to the model, contains keys "inputs" and "outputs" relating named inputs/outputs to their format specifiers (defined below).

Tensor format specifiers are used to define input and output tensors and their meanings, and must be a dictionary containing at least these keys:

* **type**: what sort of data the tensor represents: "image", "label", etc.
* **format**: what format of information is stored: "magnitude", "hounsfield", "kspace", "segmentation", "multiclass", etc.
* **num_channels**: number of channels the tensor has, assumed channel dimension first
* **spatial_shape**: shape of the spatial dimensions of the form "[H]", "[H, W]", or "[H, W, D]", see below for possible values of H, W, and D
* **dtype**: data type of tensor, eg. "float32", "int32"
* **value_range**: minimum and maximum values the input data is expected to have of the form "[MIN, MAX]" or "[]" if not known
* **is_patch_data**: "true" if the data is a patch of an input/output tensor or the entirely of the tensor, "false" otherwise
* **channel_def**: dictionary relating channel indices to plain-language description of what the channel contains

Optional keys:

* **changelog**: dictionary relating previous version names to strings describing the version.
* **intended_use**: what the model is to be used for, ie. what task it accomplishes.
* **data_source**: description of where training/validation can be sourced.
* **data_type**: type of source data used for training/validation.
* **references**: list of published referenced relating to the model.

Spatial shape definition can be complex for models accepting inputs of varying shapes, especially if there are specific conditions on what those shapes can be. Shapes are specified as lists of either positive integers for fixed sizes or strings containing expressions defining the condition a size depends on. This can be "*" to mean any size, or use an expression with Python mathematical operators and one character variables to represent dependence on an unknown quantity. For example, "2**n" represents a size which must be a power of 2, "2**n*m" must be a multiple of a power of 2. Variables are shared between dimension expressions, so a spatial shape of `["2**n", "2**n"]` states that the dimensions must be the same powers of 2 given by `n`.

A JSON schema for this file can be found at https://github.com/Project-MONAI/MONAI/blob/3049e280f2424962bb2a69261389fcc0b98e0036/monai/apps/mmars/schema/metadata.json

An example JSON metadata file:

::

  {
      "version": "0.1.0",
      "changelog": {
          "0.1.0": "complete the model package",
          "0.0.1": "initialize the model package structure"
      },
      "monai_version": "0.8.0",
      "pytorch_version": "1.10.0",
      "numpy_version": "1.21.2",
      "optional_packages_version": {"nibabel": "3.2.1"},
      "task": "Decathlon spleen segmentation",
      "description": "A pre-trained model for volumetric (3D) segmentation of the spleen from CT image",
      "authorship": "MONAI team",
      "copyright": "Copyright (c) MONAI Consortium",
      "data_source": "Task09_Spleen.tar from http://medicaldecathlon.com/",
      "data_type": "dicom",
      "dataset_dir": "/workspace/data/Task09_Spleen",
      "image_classes": "single channel data, intensity scaled to [0, 1]",
      "label_classes": "single channel data, 1 is spleen, 0 is everything else",
      "pred_classes": "2 channels OneHot data, channel 1 is spleen, channel 0 is background",
      "eval_metrics": {
          "mean_dice": 0.96
      },
      "intended_use": "This is an example, not to be used for diagnostic purposes",
      "references": [
          "Xia, Yingda, et al. '3D Semi-Supervised Learning with Uncertainty-Aware Multi-View Co-Training.' arXiv preprint arXiv:1811.12506 (2018). https://arxiv.org/abs/1811.12506.",
          "Kerfoot E., Clough J., Oksuz I., Lee J., King A.P., Schnabel J.A. (2019) Left-Ventricle Quantification Using Residual U-Net. In: Pop M. et al. (eds) Statistical Atlases and Computational Models of the Heart. Atrial Segmentation and LV Quantification Challenges. STACOM 2018. Lecture Notes in Computer Science, vol 11395. Springer, Cham. https://doi.org/10.1007/978-3-030-12029-0_40"
      ],
      "network_data_format":{
          "inputs": {
              "image": {
                  "type": "image",
                  "format": "magnitude",
                  "num_channels": 1,
                  "spatial_shape": [160, 160, 160],
                  "dtype": "float32",
                  "value_range": [0, 1],
                  "is_patch_data": false,
                  "channel_def": {0: "image"}
              }
          },
          "outputs":{
              "pred": {
                  "type": "image",
                  "format": "segmentation",
                  "num_channels": 2,
                  "spatial_shape": [160, 160, 160],
                  "dtype": "float32",
                  "value_range": [0, 1],
                  "is_patch_data": false,
                  "channel_def": {0: "background", 1: "spleen"}
              }
          }
      }
  }
