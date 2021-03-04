# GeodisTK: Geodesic Distance Transform Toolkit for 2D and 3D Images
Geodesic transformation of images can be implementated with two approaches: fast marching and raster scan. Fast marching is based on the iterative propagation of a pixel front with velocity F [1]. Raster scan is based on kernel operations that are sequentially applied over the image in multiple passes [2][3]. In GeoS [4], the authors proposed to use a 3x3 kernel for forward and backward passes for efficient geodesic distance transform, which was used for image segmentation. 

![ranster scan](./data/ranster_scan.png)
Raster scan for geodesic distance transform. Image from [4].

DeepIGeoS [5] proposed to combine geodesic distance transforms with convolutional neural networks for efficient interactive segmentation of 2D and 3D images. 

* [1] Sethian, James A. "Fast marching methods." SIAM review 41, no. 2 (1999): 199-235.
* [2] Borgefors, Gunilla. "Distance transformations in digital images." CVPR, 1986
* [3] Toivanen, Pekka J. "New geodesic distance transforms for gray-scale images." Pattern Recognition Letters 17, no. 5 (1996): 437-450.
* [4] Criminisi, Antonio, Toby Sharp, and Andrew Blake. "Geos: Geodesic image segmentation." ECCV, 2008.
* [5] Wang, Guotai, et al. "[`DeepIGeoS: A deep interactive geodesic framework for medical image segmentation`](https://ieeexplore.ieee.org/document/8370732)."  TPAMI, 2018. 

![2D example](./data/2d_example.png)

A comparison of fast marching and ranster scan for 2D geodesic distance transform. (d) shows the Euclidean distance and (e) is a mixture of Geodesic and Euclidean distance.

This toolkit provides a cpp implementation of fast marching and raster scan for 2D/3D geodesic and
Euclidean distance transforms and a mixture of them, and proivdes a python interface to use it. This is part of the work of DeepIGeoS [5]. [`If you use our code, please cite this paper.`](https://ieeexplore.ieee.org/document/8370732)


### How to install
1. Install this toolkit easily by typing [`pip install GeodisTK`](https://pypi.org/project/GeodisTK/)

2. Alternatively, if you want to build from source files, download this package and run the following:
```bash
python setup.py build
python setup.py install
```

### How to use
1. See a 2D example, run `python demo2d.py`

2. See a 3D example, run `python demo3d.py`
