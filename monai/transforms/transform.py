# Copyright 2020 - 2021 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
A collection of generic interfaces for MONAI transforms.
"""

import warnings
from abc import ABC, abstractmethod
from typing import Any, Hashable, Optional, Tuple
import torch
import numpy as np
from itertools import chain

from monai.config import KeysCollection
from monai.utils import MAX_SEED, ensure_tuple
from monai.utils import optional_import

sitk, has_sitk = optional_import("SimpleITK")
vtk, has_vtk = optional_import("vtk")
vtk_numpy_support, _ = optional_import("vtk.util.numpy_support")

__all__ = ["Randomizable", "Transform", "MapTransform", "InvertibleTransform", "NonRigidTransform"]


class Randomizable(ABC):
    """
    An interface for handling random state locally, currently based on a class variable `R`,
    which is an instance of `np.random.RandomState`.
    This is mainly for randomized data augmentation transforms. For example::

        class RandShiftIntensity(Randomizable):
            def randomize():
                self._offset = self.R.uniform(low=0, high=100)
            def __call__(self, img):
                self.randomize()
                return img + self._offset

        transform = RandShiftIntensity()
        transform.set_random_state(seed=0)

    """

    R: np.random.RandomState = np.random.RandomState()

    def __init__(self, prob):
        self._do_transform = False
        self.prob = prob

    def set_random_state(
        self, seed: Optional[int] = None, state: Optional[np.random.RandomState] = None
    ) -> "Randomizable":
        """
        Set the random state locally, to control the randomness, the derived
        classes should use :py:attr:`self.R` instead of `np.random` to introduce random
        factors.

        Args:
            seed: set the random state with an integer seed.
            state: set the random state with a `np.random.RandomState` object.

        Raises:
            TypeError: When ``state`` is not an ``Optional[np.random.RandomState]``.

        Returns:
            a Randomizable instance.

        """
        if seed is not None:
            _seed = id(seed) if not isinstance(seed, (int, np.integer)) else seed
            _seed = _seed % MAX_SEED
            self.R = np.random.RandomState(_seed)
            return self

        if state is not None:
            if not isinstance(state, np.random.RandomState):
                raise TypeError(f"state must be None or a np.random.RandomState but is {type(state).__name__}.")
            self.R = state
            return self

        self.R = np.random.RandomState()
        return self

    @abstractmethod
    def randomize(self, data: Any) -> None:
        """
        Within this method, :py:attr:`self.R` should be used, instead of `np.random`, to introduce random factors.

        all :py:attr:`self.R` calls happen here so that we have a better chance to
        identify errors of sync the random state.

        This method can generate the random factors based on properties of the input data.

        Raises:
            NotImplementedError: When the subclass does not override this method.

        """
        raise NotImplementedError(f"Subclass {self.__class__.__name__} must implement this method.")


class Transform(ABC):
    """
    An abstract class of a ``Transform``.
    A transform is callable that processes ``data``.

    It could be stateful and may modify ``data`` in place,
    the implementation should be aware of:

        #. thread safety when mutating its own states.
           When used from a multi-process context, transform's instance variables are read-only.
        #. ``data`` content unused by this transform may still be used in the
           subsequent transforms in a composed transform.
        #. storing too much information in ``data`` may not scale.

    See Also

        :py:class:`monai.transforms.Compose`
    """

    @abstractmethod
    def __call__(self, data: Any):
        """
        ``data`` is an element which often comes from an iteration over an
        iterable, such as :py:class:`torch.utils.data.Dataset`. This method should
        return an updated version of ``data``.
        To simplify the input validations, most of the transforms assume that

        - ``data`` is a Numpy ndarray, PyTorch Tensor or string
        - the data shape can be:

          #. string data without shape, `LoadImage` transform expects file paths
          #. most of the pre-processing transforms expect: ``(num_channels, spatial_dim_1[, spatial_dim_2, ...])``,
             except that `AddChannel` expects (spatial_dim_1[, spatial_dim_2, ...]) and
             `AsChannelFirst` expects (spatial_dim_1[, spatial_dim_2, ...], num_channels)
          #. most of the post-processing transforms expect
             ``(batch_size, num_channels, spatial_dim_1[, spatial_dim_2, ...])``

        - the channel dimension is not omitted even if number of channels is one

        This method can optionally take additional arguments to help execute transformation operation.

        Raises:
            NotImplementedError: When the subclass does not override this method.

        """
        raise NotImplementedError(f"Subclass {self.__class__.__name__} must implement this method.")


class MapTransform(Transform):
    """
    A subclass of :py:class:`monai.transforms.Transform` with an assumption
    that the ``data`` input of ``self.__call__`` is a MutableMapping such as ``dict``.

    The ``keys`` parameter will be used to get and set the actual data
    item to transform.  That is, the callable of this transform should
    follow the pattern:

        .. code-block:: python

            def __call__(self, data):
                for key in self.keys:
                    if key in data:
                        # update output data with some_transform_function(data[key]).
                    else:
                        # do nothing or some exceptions handling.
                return data

    Raises:
        ValueError: When ``keys`` is an empty iterable.
        TypeError: When ``keys`` type is not in ``Union[Hashable, Iterable[Hashable]]``.

    """

    def __init__(self, keys: KeysCollection) -> None:
        self.keys: Tuple[Hashable, ...] = ensure_tuple(keys)
        if not self.keys:
            raise ValueError("keys must be non empty.")
        for key in self.keys:
            if not isinstance(key, Hashable):
                raise TypeError(f"keys must be one of (Hashable, Iterable[Hashable]) but is {type(keys).__name__}.")

    @abstractmethod
    def __call__(self, data):
        """
        ``data`` often comes from an iteration over an iterable,
        such as :py:class:`torch.utils.data.Dataset`.

        To simplify the input validations, this method assumes:

        - ``data`` is a Python dictionary
        - ``data[key]`` is a Numpy ndarray, PyTorch Tensor or string, where ``key`` is an element
          of ``self.keys``, the data shape can be:

          #. string data without shape, `LoadImaged` transform expects file paths
          #. most of the pre-processing transforms expect: ``(num_channels, spatial_dim_1[, spatial_dim_2, ...])``,
             except that `AddChanneld` expects (spatial_dim_1[, spatial_dim_2, ...]) and
             `AsChannelFirstd` expects (spatial_dim_1[, spatial_dim_2, ...], num_channels)
          #. most of the post-processing transforms expect
             ``(batch_size, num_channels, spatial_dim_1[, spatial_dim_2, ...])``

        - the channel dimension is not omitted even if number of channels is one

        Raises:
            NotImplementedError: When the subclass does not override this method.

        returns:
            An updated dictionary version of ``data`` by applying the transform.

        """
        raise NotImplementedError(f"Subclass {self.__class__.__name__} must implement this method.")


class InvertibleTransform(ABC):
    """Classes for invertible transforms.

    This class exists so that an ``invert`` method can be implemented. This allows, for
    example, images to be cropped, rotated, padded, etc., during training and inference,
    and after be returned to their original size before saving to file for comparison in
    an external viewer.

    When the `__call__` method is called, a serialization of the class is stored. When
    the `inverse` method is called, the serialization is then removed. We use last in,
    first out for the inverted transforms.
    """

    def append_applied_transforms(self, data: dict, key: Hashable, idx: int = 0, extra_info: Optional[dict] = None, orig_size: Optional[Tuple] = None) -> None:
        """Append to list of applied transforms for that key."""
        key_transform = str(key) + "_transforms"
        info = {}
        info["class"] = type(self)
        info["init_args"] = self.get_input_args(key, idx)
        info["orig_size"] = orig_size or data[key].shape[1:]
        info["extra_info"] = extra_info
        # If class is randomizable, store whether the transform was actually performed (based on `prob`)
        if isinstance(self, Randomizable):
            info["do_transform"] = self._do_transform
        # If this is the first, create list
        if key_transform not in data:
            data[key_transform] = []
        data[key_transform].append(info)

    def check_transforms_match(self, transform: dict, key: Hashable) -> None:
        explanation = "Should inverse most recently applied invertible transform first"
        # Check transorms are of same type.
        if transform["class"] != type(self):
            raise RuntimeError(explanation)

        t1 = transform["init_args"]
        t2 = self.get_input_args(key)

        if t1.keys() != t2.keys():
            raise RuntimeError(explanation)
        for k in t1.keys():
            if np.any(t1[k] != t2[k]):
                raise RuntimeError(explanation)

    def get_most_recent_transform(self, data: dict, key: Hashable) -> dict:
        """Get most recent transform."""
        transform = dict(data[str(key) + "_transforms"][-1])
        self.check_transforms_match(transform, key)
        return transform

    @staticmethod
    def remove_most_recent_transform(data: dict, key: Hashable) -> None:
        """Remove most recent transform."""
        data[str(key) + "_transforms"].pop()

    def get_input_args(self, key: Hashable, idx: int = 0) -> dict:
        """Get input arguments for a single key."""
        raise NotImplementedError(f"Subclass {self.__class__.__name__} must implement this method.")

    def inverse(self, data: dict):
        """
        Inverse of ``__call__``.

        Raises:
            NotImplementedError: When the subclass does not override this method.

        """
        raise NotImplementedError(f"Subclass {self.__class__.__name__} must implement this method.")

class NonRigidTransform(ABC):
    @staticmethod
    def _get_disp_to_def_arr(shape, spacing):
        def_to_disp = np.mgrid[[slice(0, i) for i in shape]].astype(np.float64)
        for idx, i in enumerate(shape):
            # shift for origin (in MONAI, center of image)
            def_to_disp[idx] -= (i - 1) / 2
            # if supplied, account for spacing (e.g., for control point grids)
            if spacing is not None:
                def_to_disp[idx] *= spacing[idx]
        return def_to_disp

    @staticmethod
    def _inv_disp_w_sitk(fwd_disp, num_iters):
        fwd_disp_sitk = sitk.GetImageFromArray(fwd_disp, isVector=True)
        inv_disp_sitk = sitk.InvertDisplacementField(fwd_disp_sitk, num_iters)
        inv_disp = sitk.GetArrayFromImage(inv_disp_sitk)
        return inv_disp

    @staticmethod
    def _inv_disp_w_vtk(fwd_disp):
        while fwd_disp.shape[-1] < 3:
            fwd_disp = np.append(fwd_disp, np.zeros(fwd_disp.shape[:-1] + (1, )), axis=-1)
            fwd_disp = fwd_disp[..., None, :]
        # fwd_disp_vtk = vtk.vtkImageImport()
        # # The previously created array is converted to a string of chars and imported.
        # data_string = fwd_disp.tostring()
        # fwd_disp_vtk.CopyImportVoidPointer(data_string, len(data_string))
        # # The type of the newly imported data is set to unsigned char (uint8)
        # fwd_disp_vtk.SetDataScalarTypeToUnsignedChar()
        # fwd_disp_vtk.SetNumberOfScalarComponents(3)
        extent = list(chain.from_iterable(zip([0, 0, 0], fwd_disp.shape[:-1])))
        # fwd_disp_vtk.SetWholeExtent(extent)
        # fwd_disp_vtk.SetDataExtentToWholeExtent()
        # fwd_disp_vtk.Update()
        # fwd_disp_vtk = fwd_disp_vtk.GetOutput()

        fwd_disp_flattened = fwd_disp.flatten()  # need to keep this in memory
        vtk_data_array = vtk_numpy_support.numpy_to_vtk(fwd_disp_flattened)

        # Generating the vtkImageData
        fwd_disp_vtk = vtk.vtkImageData()
        fwd_disp_vtk.SetOrigin(0, 0, 0)
        fwd_disp_vtk.SetSpacing(1, 1, 1)
        fwd_disp_vtk.SetDimensions(*fwd_disp.shape[:-1])

        fwd_disp_vtk.AllocateScalars(vtk_numpy_support.get_vtk_array_type(fwd_disp.dtype), 3)
        fwd_disp_vtk.SetExtent(extent)
        fwd_disp_vtk.GetPointData().AddArray(vtk_data_array)

        # # create b-spline coefficients for the displacement grid
        # bspline_filter = vtk.vtkImageBSplineCoefficients()
        # bspline_filter.SetInputData(fwd_disp_vtk)
        # bspline_filter.Update()

        # # use these b-spline coefficients to create a transform
        # bspline_transform = vtk.vtkBSplineTransform()
        # bspline_transform.SetCoefficientData(bspline_filter.GetOutput())
        # bspline_transform.Update()

        # # invert the b-spline transform onto a new grid
        # grid_maker = vtk.vtkTransformToGrid()
        # grid_maker.SetInput(bspline_transform.GetInverse())
        # grid_maker.SetGridOrigin(fwd_disp_vtk.GetOrigin())
        # grid_maker.SetGridSpacing(fwd_disp_vtk.GetSpacing())
        # grid_maker.SetGridExtent(fwd_disp_vtk.GetExtent())
        # grid_maker.SetGridScalarTypeToFloat()
        # grid_maker.Update()

        # # Get inverse displacement as an image
        # inv_disp_vtk = grid_maker.GetOutput()

        # from vtk.util.numpy_support import vtk_to_numpy
        # inv_disp = vtk_numpy_support.vtk_to_numpy(inv_disp_vtk.GetPointData().GetScalars())
        inv_disp = vtk_numpy_support.vtk_to_numpy(fwd_disp_vtk.GetPointData().GetArray(0))
        inv_disp = inv_disp.reshape(fwd_disp.shape)

        return inv_disp


    @staticmethod
    def compute_inverse_deformation(num_spatial_dims, fwd_def_orig, spacing=None, num_iters: int = 100, use_package: str = "sitk"):
        """Package can be vtk or sitk."""
        if use_package.lower() == "vtk" and not has_vtk:
            warnings.warn("Please install VTK to estimate inverse of non-rigid transforms. Data has not been modified")
            return None
        if use_package.lower() == "sitk" and not has_sitk:
            warnings.warn("Please install SimpleITK to estimate inverse of non-rigid transforms. Data has not been modified")
            return None

        # Convert to numpy if necessary
        if isinstance(fwd_def_orig, torch.Tensor):
            fwd_def_orig = fwd_def_orig.cpu().numpy()
        # Remove any extra dimensions (we'll add them back in at the end)
        fwd_def = fwd_def_orig[:num_spatial_dims]
        # Def -> disp
        def_to_disp = NonRigidTransform._get_disp_to_def_arr(fwd_def.shape[1:], spacing)
        fwd_disp = fwd_def - def_to_disp
        # move tensor component to end (T,H,W,[D])->(H,W,[D],T)
        fwd_disp = np.moveaxis(fwd_disp, 0, -1)

        # If using vtk...
        if use_package.lower() == "vtk":
            inv_disp = NonRigidTransform._inv_disp_w_vtk(fwd_disp)
        # If using sitk...
        else:
            inv_disp = NonRigidTransform._inv_disp_w_sitk(fwd_disp, num_iters)


        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, 2)
        for i, direc1 in enumerate(["x", "y"]):
            for j, (im, direc2) in enumerate(zip([fwd_disp, inv_disp], ["fwd", "inv"])):
                ax = axes[i, j]
                im_show = ax.imshow(im[..., i])
                ax.set_title(f"{direc2}{direc1}", fontsize=25)
                ax.axis("off")
                fig.colorbar(im_show, ax=ax)
        plt.show()
        # move tensor component back to beginning
        inv_disp = np.moveaxis(inv_disp, -1, 0)
        # Disp -> def
        inv_def = inv_disp + def_to_disp
        # Add back in any removed dimensions
        ndim_in = fwd_def_orig.shape[0]
        ndim_out = inv_def.shape[0]
        inv_def = np.concatenate([inv_def, fwd_def_orig[ndim_out:ndim_in]])

        return inv_def
