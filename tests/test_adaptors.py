# Copyright 2020 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import unittest
import itertools

from monai.data.transforms.adaptors import adaptor, FunctionSignature


class TestAdaptors(unittest.TestCase):

    # def test_check_signature(self):
    #     def foo(image, label=None, *a):
    #         pass
    #
    #     print(check_signature(foo))
    #     import inspect
    #     ps = inspect.signature(foo).parameters.values()
    #     for p in ps:
    #         print(dir(p))
    #         print(p, p.kind, p.default is p.empty)
    #
    #     print(dict)
    #
    #     print(dict(v, len(v) < 3) for v in ['a', 'ab', 'abc', 'abcd'])


    def test_function_signature(self):

        def foo(image, label=None, *a, **kw):
            pass

        f = FunctionSignature(foo)
        print(f)



    def test_single_in_single_out(self):
        def foo(image):
            return image * 2

        it = itertools.product(
            ['image', ['image']],
            [None, 'image', ['image'], {'image': 'image'}]
        )
        for i in it:
            d = {'image': 2}
            dres = adaptor(foo, i[0], i[1])(d)
            self.assertEqual(dres['image'], 4)

        d = {'image': 2}
        dres = adaptor(foo, 'image')(d)
        self.assertEqual(dres['image'], 4)

        d = {'image': 2}
        dres = adaptor(foo, 'image', 'image')(d)
        self.assertEqual(dres['image'], 4)

        d = {'image': 2}
        dres = adaptor(foo, 'image', {'image': 'image'})(d)
        self.assertEqual(dres['image'], 4)

        d = {'img': 2}
        dres = adaptor(foo, 'img', {'img': 'image'})(d)
        self.assertEqual(dres['img'], 4)

        d = {'img': 2}
        dres = adaptor(foo, ['img'], {'img': 'image'})(d)
        self.assertEqual(dres['img'], 4)
