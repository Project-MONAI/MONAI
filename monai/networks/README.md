
# Networks

This module contains network definitions, loss functions, the components they are constructed from, and ancillary utilities.
All components are defined in pure Pytorch which are all usable without any other facilities provided by this framework
other than a few functions from the utilities module. The submodule breakdown is as such:

* **layers**: Contains the definitions of various network layer components, for example `Convolution` which encapsulates
a 1/2/3D convolution with normalization, dropout, and activation components. These are implemented as `nn.Module` subclasses.

* **losses**: Classes defining loss functions not provided by Pytorch.

* **nets**: Full network definitions, eg. UNet, which are composed from the classes in **layers**. The expectation is
that the definitions here rely on no other infrastructure besides some utility functions so can be used on generic 
Pytorch code.

* **initializers**: Contains visitor functions for initializing networks using their `apply()` methods.

* **utils**: Utility functions implemented with Pytorch, kept out of the utilities submodule which should be all non-Pytorch.

