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

from monai.transforms.compose import Compose, Randomizable


class TestCompose(unittest.TestCase):

    def test_empty_compose(self):
        c = Compose()
        i = 1
        self.assertEqual(c(i), 1)

    def test_non_dict_compose(self):

        def a(i):
            return i + 'a'

        def b(i):
            return i + 'b'

        c = Compose([a, b, a, b])
        self.assertEqual(c(''), 'abab')

    def test_dict_compose(self):

        def a(d):
            d = dict(d)
            d['a'] += 1
            return d

        def b(d):
            d = dict(d)
            d['b'] += 1
            return d

        c = Compose([a, b, a, b, a])
        self.assertDictEqual(c({'a': 0, 'b': 0}), {'a': 3, 'b': 2})

    def test_random_compose(self):

        class _Acc(Randomizable):
            self.rand = 0.0

            def randomize(self):
                self.rand = self.R.rand()

            def __call__(self, data):
                self.randomize()
                return self.rand + data

        c = Compose([_Acc(), _Acc()])
        self.assertNotAlmostEqual(c(0), c(0))
        c.set_random_state(123)
        self.assertAlmostEqual(c(1), 2.39293837)
        c.set_random_state(223)
        c.randomize()
        self.assertAlmostEqual(c(1), 2.57673391)

    def test_randomize_warn(self):

        class _RandomClass(Randomizable):

            def randomize(self, foo):
                pass

        c = Compose([_RandomClass(), _RandomClass()])
        with self.assertWarns(Warning):
            c.randomize()


if __name__ == '__main__':
    unittest.main()
