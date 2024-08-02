import torch

from monai.data.meta_obj import get_track_meta
from monai.data.meta_tensor import MetaTensor
from monai.transforms.utils import convert_to_tensor, create_translate, spatial_dims_from_tensorlike
from monai.transforms.inverse import TraceableTransform

def traced_no_op(data, lazy, transform_info):
    data = convert_to_tensor(data, track_meta=get_track_meta())
    spatial_dims = spatial_dims_from_tensorlike(data)
    meta_info = TraceableTransform.track_transform_meta(
        data,
        sp_size=None,
        affine=create_translate(spatial_dims, [0] * spatial_dims),
        extra_info={},
        orig_size=None,
        transform_info=transform_info,
        lazy=lazy,
    )
    # TODO: what is this for? If it is needed, it should be part of utility transforms,
    # not spatial transforms
    # out = _maybe_new_metatensor(data)
    out = data
    if lazy:
        return out.copy_meta_from(meta_info) if isinstance(out, MetaTensor) else meta_info
    out = torch.clone(data)
    return out.copy_meta_from(meta_info) if isinstance(out, MetaTensor) else out
