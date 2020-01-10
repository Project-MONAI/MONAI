
# Data

This implements the data streams classes and contains a few example datasets. Data streams are iterables which produce
single data items or batches thereof from source iterables (usually). Chaining these together is how data pipelines are
implemented in the framework. Data augmentation routines are also provided here which can applied to data items as they
pass through the stream, either singly or in parallel. 

For example, the following stream reads image/segmentation pairs from `imSrc` (any iterable), applies the augmentations
to convert the array format and apply simple augmentations (rotation, transposing, flipping, shifting) using mutliple
threads, and wraps the whole stream in a buffering thread stream:

```
def normalizeImg(im,seg):
    im=utils.arrayutils.rescaleArray(im)
    im=im[None].astype(np.float32)
    seg=seg[None].astype(np.int32)
    return im, seg

augs=[
    normalizeImg,
    augments.rot90, 
    augments.transpose, 
    augments.flip,
    partial(augments.shift,dimFract=5,order=0,nonzeroIndex=1),
]

src=data.augments.augmentstream.ThreadAugmentStream(imSrc,200,augments=augs)
src=data.streams.ThreadBufferStream(src)
```

In this code, `src` is now going to yield batches of 200 images in a separate thread when iterated over. This can be
fed directly into a `NetworkManager` class as its `src` parameter. 

Module breakdown:

* **augments**: Contains definitions and stream types for doing data augmentation. An augment is simply a callable which
accepts one or more Numpy arrays and returns the augmented result. The provided decorators are for adding probability
and other facilities to a function.

* **readers**: Subclasses of `DataStream` for reading data from arrays and various file formats.

* **streams**: Contains the definitions of the stream classes which implement a number of operations on streams. The
root of the stream classes is `DataStream` which provides a very simple iterable facility. It iterates over its `src`
member, passes each item into its `generate()` generator method and yields each resulting value. This allows subclasses
to implement `generate` to modify data as it moves through the stream. The `streamgen` decorator is provided to simplify
this by being applied to a generator function to fill this role in a new object. Other subclasses implement buffering,
batching, merging from multiple sources, cycling between sources, prefetching, and fetching data from the source in a
separate thread.

