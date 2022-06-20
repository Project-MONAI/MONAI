# What's new -- `MetaTensor`

- New class `MetaTensor`. Stores meta data, image affine, and stack of transforms that have been applied to an image.
- Meta data will now be stored in `MetaTensor`, as opposed to using a dictionary in an adjacent key of the dictionary. This keeps the meta data more closely associated with the principal data, and allows array transforms to be aware (and update) meta data, not just our dictionary transforms.
- Previously, MONAI was fairly agnostic to the use of NumPy arrays and PyTorch tensors in its transforms. With the addition of `MetaTensor`,
Transforms largely use `torch.Tensor`. Input will be converted to `MetaTensor` by default.

## Manipulating `MetaTensor`

A `MetaTensor` can be created with e.g., `img=MetaTensor(torch.ones((1,4,5,6))`, which will use the default meta data (empty), and default affine transformation matrix (identity). These can be altered with input arguments.

With the `MetaTensor` created, the extra information can be accessed as follows:

- Meta data: `img.meta`,
- Affine: `img.affine`, and
- Applied operations (normally the traced/invertible transforms): `img.applied_operations`.

## Inverse array transforms

Previously, only dictionary transforms were invertible. Now, array transforms are, too!

```python
tr = Compose([LoadImage(), AddChannel(), Orientation(), Spacing()])
im = MetaTensor(...)
im_fwd = tr(im)
im_fwd_inv = tr.inverse(im_fwd)
print(im_fwd.applied_operations)  # prints list of invertible transforms
print(im_fwd_inv.applied_operations)  # should be back to 0
```

## Converting to and from `MetaTensor`

Users may, for example, want to use their own transforms which they developed prior to these changes. In a chain of transforms, you may have previously had something like this:

```python
transforms = Compose([
    LoadImaged(), AddChanneld(), MyOwnTransformd(), Spacingd(),
])
```

If `MyOwnTransformd` expects the old type of data structure, then the transform stack can be modified to this:

```python
transforms = Compose([
    LoadImaged(), AddChanneld(), FromMetaTensord(),
    MyOwnTransformd(), ToMetaTensord(), Spacingd(),
])
```

That is to say, you can use `FromMetaTensord` to convert from e.g., `{"img": MetaTensor(...)}` to `{"img": torch.Tensor(...), "img_meta_dict: {...}` and `ToMetaTensord` will do the opposite.

## Batches of `MetaTensor`

The end user should not really need to modify this logic, it is here for interest.

We use a flag inside of the meta data to determine whether a `MetaTensor` is in fact a batch of multiple images. This logic is contained in our `default_collate`:

```python
im1, im2 = MetaTensor(...), MetaTensor(...)
print(im1.meta.is_batch)  # False
batch = default_collate([im1, im2])
print(batch.meta.is_batch)  # True
```

Similar functionality can be seen with the `DataLoader`:
```python
ds = Dataset([im1, im2])
print(ds[0].meta.is_batch)  # False
dl = DataLoader(ds, batch_size=2)
batch = next(iter(dl))
print(batch.meta.is_batch)  # True
```

**We recommend using MONAI's Dataset where possible, as this will use the correct collation method and ensure that MONAI is made aware of when a batch of data is being used or just a single image.**

## Disabling `MetaTensor`

This should ideally be a last resort, but if you are experiencing problems due to `MetaTensor`, `set_track_meta(False)` can be used.

Output will be returned as `torch.Tensor` instead of `MetaTensor`. This won't necessarily match prevoius functionality, as the meta data will no longer be present and so won't be used or stored. Further, more data will be converted from `numpy.ndarray` to `torch.Tensor`.
