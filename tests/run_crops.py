import numpy as np

import torch

import monai.transforms.croppad.old_array as old_ca
import monai.transforms.croppad.old_dictionary as old_cd
import monai.transforms.croppad.array as ca
import monai.transforms.croppad.dictionary as cd
import monai.transforms.spatial.old_array as old_sa
import monai.transforms.spatial.old_dictionary as old_sd
import monai.transforms.spatial.array as sa
import monai.transforms.spatial.dictionary as sd


def get_img(size, dtype=torch.float32, offset=0):
    img = torch.zeros(size, dtype=dtype)
    if len(size) == 2:
        for j in range(size[0]):
            for i in range(size[1]):
                img[j, i] = i + j * size[0] + offset
    else:
        for k in range(size[0]):
            for j in range(size[1]):
                for i in range(size[2]):
                    img[k, j, i] = i + j * size[0] + k * size[0] * size[1]
    return np.expand_dims(img, 0)


class RNGWrapper(np.random.RandomState):

    def __init__(self, tag, rng):
        self.tag = tag
        self.rng = rng
        self.calls = 0

    def rand(self, *args, **kwargs):
        self.calls += 1
        value = self.rng.rand(*args, **kwargs)
        print(self.tag, self.calls, value)
        return value

    def randint(self, *args, **kwargs):
        self.calls += 1
        value = self.rng.randint(*args, **kwargs)
        print(self.tag, self.calls, value)
        return value


data = get_img((32, 32))
data[0, 7:9, 7:9] = 1096
data[0, 15:17, :] = 1160
data[0, 0, :] = 1224
data[0, :, 0] = 1286
print(data.shape)


def do_pads():
    def old_pad(shape):
        z1 = old_ca.Pad(shape, mode="constant", constant_values=0)
        z2 = old_ca.Pad(shape, mode="reflect")
        z3 = old_ca.Pad(shape, mode="replicate")
        z4 = old_ca.Pad(shape, mode="circular")

        def _inner(imgs):
            datas = []
            datas.append(imgs)
            data1 = z1(imgs)
            datas.append(data1)
            data2 = z2(imgs)
            datas.append(data2)
            data3 = z3(imgs)
            datas.append(data3)
            data4 = z4(imgs)
            datas.append(data4)
            return datas

        return _inner


    def new_pad(shape):
        print("this is still the old pad!")
        z1 = ca.Pad(shape, mode="zeros", value=0, lazy_evaluation=False)
        z2 = ca.Pad(shape, mode="reflection", lazy_evaluation=False)
        z3 = ca.Pad(shape, mode="border", lazy_evaluation=False)
        z4 = ca.Pad(shape, mode="wrap", lazy_evaluation=False)

        def _inner(imgs):
            datas = []
            datas.append(imgs)
            data1 = z1(imgs)
            datas.append(data1)
            data2 = z2(imgs)
            datas.append(data2)
            data3 = z3(imgs)
            datas.append(data3)
            data4 = z4(imgs)
            datas.append(data4)
            return datas

        return _inner


    pad_shape = [(0, 0), (1, 1), (1, 1)]
    old_results = old_pad(pad_shape)(data)
    new_results = new_pad(pad_shape[1:])(data)

    diffs = []
    for o, n in zip(old_results, new_results):
        diffs.append(n - o)


def do_zooms():
    def old_zoom(zooms):
        print("old_zooms")
        z1 = old_sa.Zoom(zoom=zooms)
        z2 = old_sa.Zoom(zoom=zooms)
        z3 = old_sa.Zoom(zoom=zooms)
        z4 = old_sa.Zoom(zoom=zooms)

        def _inner(imgs):
            datas = []
            datas.append(imgs)
            data1 = z1(imgs)
            datas.append(data1)
            data2 = z2(imgs)
            datas.append(data2)
            data3 = z3(imgs)
            datas.append(data3)
            data4 = z4(imgs)
            datas.append(data4)
            return datas

        return _inner

    def new_zoom(zooms):
        print("new_zooms")
        z1 = sa.Zoom(zoom=zooms, lazy_evaluation=False)
        z2 = sa.Zoom(zoom=zooms, lazy_evaluation=False)
        z3 = sa.Zoom(zoom=zooms, lazy_evaluation=False)
        z4 = sa.Zoom(zoom=zooms, lazy_evaluation=False)

        def _inner(imgs):
            datas = []
            datas.append(imgs)
            data1 = z1(imgs)
            datas.append(data1)
            data2 = z2(imgs)
            datas.append(data2)
            data3 = z3(imgs)
            datas.append(data3)
            data4 = z4(imgs)
            datas.append(data4)
            return datas

        return _inner

    zooms = 2.0
    old_results = old_zoom(zooms)(data)
    new_results = new_zoom(zooms)(data)

    diffs = []
    for o, n in zip(old_results, new_results):
        diffs.append(n - o)


def do_rand_spatial_crop():
    img = torch.rand(1, 32, 32, 16)
    data = {'img': img}

    t = old_cd.RandSpatialCropd(('img',), (16, 16, 8), random_center=True, random_size=False)
    t.set_random_state(state=RNGWrapper("trad", np.random.RandomState(1540998321)))
    data = t(data)
    print([d.shape for k, d in data.items()])

    for j in ((1, 0, 0, 1), (1, 1, 0, 1), (1, 0, 1, 1), (1, 1, 1, 1),
              (0, 0, 0, 0, 1), (1, 0, 0, 0, 1), (1, 1, 0, 0, 1), (1, 1, 1, 1, 1)):
        print(j)
        ring = np.random.RandomState(1540998321)
        for i in range(len(j)):
            if j[i] == 0:
                print(i, ring.randint(0, 10, dtype=np.int64))
            else:
                print(i, ring.rand())

# do_zooms()
do_rand_spatial_crop()
