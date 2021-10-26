import unittest

import torch
from parameterized import parameterized

from monai.networks import eval_mode
from monai.networks.nets.vit_ae import ViT_AE

TEST_CASE_Vit = []
for dropout_rate in [0.6]:
    for in_channels in [4]:
        for hidden_size in [768]:
            for img_size in [64, 96, 128]:
                for patch_size in [16]:
                    for num_heads in [12]:
                        for mlp_dim in [3072]:
                            for num_layers in [4]:
                                for num_classes in [8]:
                                    for pos_embed in ["conv"]:
                                        for classification in [False]:
                                            for same_as_input_size in [True]:
                                                for nd in (2, 3):
                                                    test_case = [
                                                        {
                                                            "in_channels": in_channels,
                                                            "img_size": (img_size,) * nd,
                                                            "patch_size": (patch_size,) * nd,
                                                            "hidden_size": hidden_size,
                                                            "mlp_dim": mlp_dim,
                                                            "num_layers": num_layers,
                                                            "num_heads": num_heads,
                                                            "pos_embed": pos_embed,
                                                            "classification": classification,
                                                            "num_classes": num_classes,
                                                            "dropout_rate": dropout_rate,
                                                            "same_as_input_size": same_as_input_size
                                                        },
                                                        (2, in_channels, *([img_size] * nd)),
                                                        (2, (img_size // patch_size) ** nd, hidden_size),
                                                    ]
                                                    if nd == 2:
                                                        test_case[0]["spatial_dims"] = 2  # type: ignore
                                                    if test_case[0]["classification"]:  # type: ignore
                                                        test_case[2] = (2, test_case[0]["num_classes"])  # type: ignore
                                                    TEST_CASE_Vit.append(test_case)


class TestPatchEmbeddingBlock(unittest.TestCase):
    @parameterized.expand(TEST_CASE_Vit)
    def test_shape(self, input_param, input_shape, expected_shape):
        net = ViT_AE(**input_param)
        with eval_mode(net):
            result, _ = net(torch.randn(input_shape))
            self.assertEqual(result.shape, expected_shape)

    def test_ill_arg(self):
        with self.assertRaises(ValueError):
            ViT_AE(
                in_channels=1,
                img_size=(128, 128, 128),
                patch_size=(16, 16, 16),
                hidden_size=128,
                mlp_dim=3072,
                num_layers=12,
                num_heads=12,
                pos_embed="conv",
                classification=False,
                dropout_rate=5.0,
            )

        with self.assertRaises(ValueError):
            ViT_AE(
                in_channels=1,
                img_size=(32, 32, 32),
                patch_size=(64, 64, 64),
                hidden_size=512,
                mlp_dim=3072,
                num_layers=12,
                num_heads=8,
                pos_embed="perceptron",
                classification=False,
                dropout_rate=0.3,
            )

        with self.assertRaises(ValueError):
            ViT_AE(
                in_channels=1,
                img_size=(96, 96, 96),
                patch_size=(8, 8, 8),
                hidden_size=512,
                mlp_dim=3072,
                num_layers=12,
                num_heads=14,
                pos_embed="conv",
                classification=False,
                dropout_rate=0.3,
            )

        with self.assertRaises(ValueError):
            ViT_AE(
                in_channels=1,
                img_size=(97, 97, 97),
                patch_size=(4, 4, 4),
                hidden_size=768,
                mlp_dim=3072,
                num_layers=12,
                num_heads=8,
                pos_embed="perceptron",
                classification=True,
                dropout_rate=0.3,
            )

        with self.assertRaises(ValueError):
            ViT_AE(
                in_channels=4,
                img_size=(96, 96, 96),
                patch_size=(16, 16, 16),
                hidden_size=768,
                mlp_dim=3072,
                num_layers=12,
                num_heads=12,
                pos_embed="perc",
                classification=False,
                dropout_rate=0.3,
            )

if __name__ == "__main__":
    unittest.main()
