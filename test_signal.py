from parameterized import parameterized
import unittest
import numpy as np
from monai.transforms.signal.array import SignalResample

VALID_CASES = [('interpolation',500,250,2000),('poly',500,250,2000)]
INVALID_CASES = [('wrongmethod',500,250,ValueError)]


class TestSignalResample(unittest.TestCase):
    
    def setUp(self):
        self.test_sig = np.array([np.math.sin(x) for x in np.arange(0,100,0.1)])
        
    @parameterized.expand(VALID_CASES)
    def test_correct_parameters(self,method,current_sample_rate,target_sample_rate,_):
        self.assertIsInstance(SignalResample(method,current_sample_rate,target_sample_rate),SignalResample)
   
    @parameterized.expand(VALID_CASES)
    def test_correct_results(self,method,current_sample_rate,target_sample_rate,length):
        
        self.assertAlmostEqual(len(SignalResample(method,current_sample_rate,target_sample_rate)(self.test_sig)),2000)
    
if __name__ == '__main__':
    unittest.main()
