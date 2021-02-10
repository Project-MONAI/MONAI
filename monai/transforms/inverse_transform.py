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

import warnings
from abc import ABC
from itertools import chain
from typing import Any, Dict, Hashable, Optional, Tuple
import numpy as np
import torch

from monai.utils import optional_import
from monai.transforms.transform import Randomizable

sitk, has_sitk = optional_import("SimpleITK")
vtk, has_vtk = optional_import("vtk")
vtk_numpy_support, _ = optional_import("vtk.util.numpy_support")

__all__ = ["InvertibleTransform", "NonRigidTransform"]

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

    def append_applied_transforms(
        self,
        data: dict,
        key: Hashable,
        idx: int = 0,
        extra_info: Optional[dict] = None,
        orig_size: Optional[Tuple] = None,
    ) -> None:
        """Append to list of applied transforms for that key."""
        key_transform = str(key) + "_transforms"
        info: Dict[str, Any] = {}
        info["id"] = id(self)
        info["orig_size"] = orig_size or data[key].shape[1:]
        if extra_info is not None:
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
        if transform["id"] != id(self):
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
            fwd_disp = np.append(fwd_disp, np.zeros(fwd_disp.shape[:-1] + (1,)), axis=-1)
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
    def compute_inverse_deformation(
        num_spatial_dims, fwd_def_orig, spacing=None, num_iters: int = 100, use_package: str = "sitk"
    ):
        """Package can be vtk or sitk."""
        if use_package.lower() == "vtk" and not has_vtk:
            warnings.warn("Please install VTK to estimate inverse of non-rigid transforms. Data has not been modified")
            return None
        if use_package.lower() == "sitk" and not has_sitk:
            warnings.warn(
                "Please install SimpleITK to estimate inverse of non-rigid transforms. Data has not been modified"
            )
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

        if False:
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
