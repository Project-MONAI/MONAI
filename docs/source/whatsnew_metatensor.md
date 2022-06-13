# What's new -- `MetaTensor`

- New class `MetaTensor`. Stores meta data, image affine, and stack of transforms that have been applied to an image.
- Meta data will now be stored in `MetaTensor`, as opposed to using a dictionary in an adjacent key of the dictionary. This keeps the meta data more closely associated with the principal data, and allows array transforms to be aware (and update) meta data, not just our dictionary transforms.
- Previously, MONAI was fairly agnostic to the use of NumPy arrays and PyTorch tensors in its transforms. With the addition of `MetaTensor`, 
Transforms now expect `torch.Tensor` as input. 

## Disabling `MetaTensor`

This should ideally be a last resort, but if you are experiencing problems due to `MetaTensor`, `set_track_meta(False)` can be used.

Output will be returned as `torch.Tensor` instead of `MetaTensor`. This won't necessarily match prevoius functionality, as more data will be converted from `numpy.ndarray` to `torch.Tensor`.