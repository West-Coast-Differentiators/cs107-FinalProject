import unittest
import numpy as np

from WestCoastAD import Initializer

np.random.seed(seed=2020)

class InitializerUnitTest(unittest.TestCase):
    
    def test_Zeros(self):
        initializer = Initializer.Zeros()
        var_init = initializer(5)
        
        self.assertEqual(np.zeros(5), var_init)
   
    def test_Ones(self):
        initializer = Initializer.Ones()
        var_init = initializer(10)
        
        self.assertEqual(np.ones(10), var_init)
          
    def test_Constant(self):
        value = -1.33
        initializer = Initializer.Constant(value)
        var_init = initializer(10)
        
        self.assertEqual(value*np.ones(10), var_init)
        self.assertEqual({'value': value}, initializer.get_config)
        
    def test_RandomUniform(self):
        maxval = 5
        minval = 2.5
        initializer = Initializer.RandomUniform(minval, maxval)
        var_init = initializer(7)
        
        self.assertEqual(np.random.uniform(2.5,5,7), var_init)
        self.assertEqual({"min value": minval, "max value": maxval}, initializer.get_config)
        
    def test_RandomNormal(self):
        mean = 1
        stddev = 0.25
        initializer = Initializer.RandomNormal(mean, stddev)
        var_init = initializer(100)
        
        self.assertEqual(np.random.normal(1, 0.25, 100), var_init)
        self.assertEqual({"mean": mean, "stddev": stddev}, initializer.get_config)
   