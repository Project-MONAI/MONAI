from __future__ import annotations
import unittest
import test_torchscript_utils
from parameterized import parameterized 

from monai.networks import eval_mode
from monai.networks.nets import Quicknat 
from tests.utils import test_script_save

TEST_CASES = [
    # params, input_shape, expected_shape 
    [{"num_classes": 1,"num_channels": 1, "num_filters": 1, "se_block" : None}, (), ()],
    [{"num_classes": 1,"num_channels": 1, "num_filters": 4, "se_block" : None}, (), ()],
    [{"num_classes": 1,"num_channels": 1, "num_filters": 64, "se_block" : None}, (), ()],
    [{"num_classes": 4,"num_channels": 1, "num_filters": 64, "se_block" : None}, (), ()],
    [{"num_classes": 33,"num_channels": 1, "num_filters": 64, "se_block" : None}, (), ()],
    [{"num_classes": 1,"num_channels": 1, "num_filters": 64, "se_block" : "CSE"}, (), ()],
    [{"num_classes": 1,"num_channels": 1, "num_filters": 64, "se_block" : "SSE"}, (), ()],
    [{"num_classes": 1,"num_channels": 1, "num_filters": 64, "se_block" : "CSSE"}, (), ()]
]

class TestQuicknat(unittest.TestCase): 
    @parameterized.expand(TEST_CASES)
    def test_shape(self, input_param, input_shape, expected_shape): 
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(input_param)
        net = Quicknat(**input_param).to(device)
        with eval_mode(net): 
            result = net(torch.randn(input_shape).to(device))
        self.assertEqual(result.shape, expected_shape)
    def test_script(self): 
        net = Quicknat(in_channels = 1, out_channels = 1)
        test_data = torch.randn(16,1,32,32)
        test_script_save(net, test_data)

if __name__ == "__main__":
    unittest.main()