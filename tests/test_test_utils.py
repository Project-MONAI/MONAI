from __future__ import annotations

import unittest

from tests.test_utils import dict_product


class TestTestUtils(unittest.TestCase):
    def setUp(self):
        TEST_CASE_PATCHEMBEDDINGBLOCK = []
        for dropout_rate in (0.5,):
            for in_channels in [1, 4]:
                for hidden_size in [96, 288]:
                    for img_size in [32, 64]:
                        for patch_size in [8, 16]:
                            for num_heads in [8, 12]:
                                for proj_type in ["conv", "perceptron"]:
                                    for pos_embed_type in ["none", "learnable", "sincos"]:
                                        # for classification in (False, True):  # TODO: add classification tests
                                        for nd in (2, 3):
                                            test_case = [
                                                {
                                                    "in_channels": in_channels,
                                                    "img_size": (img_size,) * nd,
                                                    "patch_size": (patch_size,) * nd,
                                                    "hidden_size": hidden_size,
                                                    "num_heads": num_heads,
                                                    "proj_type": proj_type,
                                                    "pos_embed_type": pos_embed_type,
                                                    "dropout_rate": dropout_rate,
                                                    "spatial_dims": nd
                                                },
                                                (2, in_channels, *([img_size] * nd)),
                                                (2, (img_size // patch_size) ** nd, hidden_size),
                                            ]
                                            TEST_CASE_PATCHEMBEDDINGBLOCK.append(test_case)

        self.test_case_patchembeddingblock = TEST_CASE_PATCHEMBEDDINGBLOCK

    def test_case_patchembeddingblock(self):
        test_case_patchembeddingblock = dict_product(
            dropout_rate=[0.5],
            in_channels=[1, 4],
            hidden_size=[96, 288],
            img_size=[32, 64],
            patch_size=[8, 16],
            num_heads=[8, 12],
            proj_type=["conv", "perceptron"],
            pos_embed_type=["none", "learnable", "sincos"],
            nd=[2, 3],
        )
        test_case_patchembeddingblock = [
                [
                    params,
                    (2, params["in_channels"], *([params["img_size"]] * params["nd"])),
                    (2, (params["img_size"] // params["patch_size"]) ** params["nd"], params["hidden_size"]),
                ]
                for params in test_case_patchembeddingblock
        ]

        self.assertIsInstance(test_case_patchembeddingblock, list)
        self.assertGreater(len(test_case_patchembeddingblock), 0)
        self.assertEqual(
            len(test_case_patchembeddingblock),
            len(self.test_case_patchembeddingblock),
        )
        self.assertEqual(
            len(test_case_patchembeddingblock[0]),
            len(self.test_case_patchembeddingblock[0]),
        )
        self.assertEqual(
            len(test_case_patchembeddingblock[0][0]),
            len(self.test_case_patchembeddingblock[0][0]),
        )
        self.assertEqual(
            test_case_patchembeddingblock[0][0]["in_channels"],
            self.test_case_patchembeddingblock[0][0]["in_channels"],
        )
        self.assertEqual(
            test_case_patchembeddingblock[0][1],
            self.test_case_patchembeddingblock[0][1],
        )
        self.assertEqual(
            test_case_patchembeddingblock[0][2],
            self.test_case_patchembeddingblock[0][2],
        )

if __name__ == "__main__":
    unittest.main()
