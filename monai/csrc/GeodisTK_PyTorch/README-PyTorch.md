# README-Ala.md


### This file shows how to build the C++ files to create a library file. This allows porting the GeodisTK C++ functions into PyTorch.


## How to Build
There are two methods for building the library file.


## Method 1: Using setup.py

#### 1. open setup.py and replace "/home/.../torch/include/" line with the path of the include folder of torch.
    You can find the torch installation folder using:  \
	  $ import torch \
	  $ print(torch.__file__) 

#### 2. Run this command
    $ python setup.py build 

    note: you might need to install clang before running this command: 
    $ sudo apt-get install clang

  You will get a library file as below:
#### 3. build/lib.linux-x86_64-3.8/GeodisTKcustom.cpython-38-x86_64-linux-gnu.so
    Note the library name may differ depending on your environment. The above is based on my WSL/Windows machine.



## How to Use

    Add these lines at the top of your python script:
      ```
      import torch
      torch.ops.load_library("../build/lib.linux-x86_64-3.8/GeodisTKcustom.cpython-38-x86_64-linux-gnu.so")
      ```
Please see cpp/test1.py for an example.

## Method 2: Using cmake

#### 1. cd to GeodisTK/cpp.

#### 2. mkdir build; cd build.

#### 3. Edit CMakeLists.txt - set your include file location by INCLUDE_DIRECTORIES() 

    It needs numpy include files.
    For example, my WSL/Ubuntu Python include directories are these:
      #in case of WSL/Ubuntu:
        INCLUDE_DIRECTORIES("//wsl/Ubuntu-20.04/usr/include/python3.8")
        INCLUDE_DIRECTORIES("//wsl/Ubuntu-20.04/home/ala/.local/lib/python3.8/site-packages/numpy/core/include/numpy")
      #in case of macosx:
        INCLUDE_DIRECTORIES("/Library/Frameworks/Python.framework/Versions/3.8/include/python3.8/")
        INCLUDE_DIRECTORIES("/Library/Frameworks/Python.framework/Versions/3.8/lib/python3.8/site-packages/numpy/core/include/")

#### 5. run this command (you are at cpp/build directory).
    Make sure you have cmake installed.
      $ cmake -DCMAKE_PREFIX_PATH="$(python -c 'import torch.utils; print(torch.utils.cmake_prefix_path)')" ..

#### 6. run make
    $ make -j

#### 7. you will have libgeodistk.so libgeodistk.dylib (MacOS) in the cpp/build directory.

#### 8. now you are ready to use the GeodisTK functions. See cpp/test1.py for an usage example.
    Simply, add this line at the top of your python source.
      ```
      import torch
      torch.ops.load_library("your-lib-location/libgeodistk.so")
      ```

## Interface to GeodisTk functions

```
geodesic2d_fast_marching_pytorch(torch::Tensor I,  /*img float* */
				 torch::Tensor S,  /* seeds uint8* */
				 torch::Tensor chann /* int value */
				 )
geodesic3d_fast_marching_pytorch(torch::Tensor I,  /*img float* */
				 torch::Tensor S,  /* seeds uint8* */
				 torch::Tensor spacing, /* float vec */
				 torch::Tensor chann  /* int value */
				 )
geodesic2d_raster_scan_pytorch(torch::Tensor I,  /* img float* [depth,width,height,channel]*/
			       torch::Tensor S,  /* seeds uint8* */
			       torch::Tensor lamb,  /* float */
			       torch::Tensor chann,  /* int */
			       torch::Tensor iter   /* int */
			       )
geodesic2d_raster_scan_pytorch(torch::Tensor I,  /* img float* [depth,width,height,channel]*/
			       torch::Tensor S,  /* seeds uint8* */
			       torch::Tensor lamb,  /* float */
			       torch::Tensor chann,  /* int */
			       torch::Tensor iter   /* int */
			       )
```
Please also refer to cpp/test1.py
