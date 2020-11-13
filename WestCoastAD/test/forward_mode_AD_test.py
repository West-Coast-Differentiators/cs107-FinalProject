import unittest
import numpy as np

from WestCoastAD import Variable

class VariableUnitTest(unittest.TestCase):

    def test__add__scalar_two_variable_objects(self):
        var1 = Variable(10.1, 2.1)
        var2 = Variable(9, 3)
        summation = var1 + var2
        summation_reverse_order = var2 + var1
        
        self.assertEqual(19.1, summation.value)
        self.assertEqual(19.1, summation_reverse_order.value)
        self.assertEqual(5.1, summation.derivative)
        self.assertEqual(5.1, summation_reverse_order.derivative)
    
    def test__add__scalar_one_variable_one_constant(self):
        var = Variable(3, 17)
        summation = var + 4
        summation2 = 5 + var
        
        self.assertEqual(7, summation.value)
        self.assertEqual(8, summation2.value)
        self.assertEqual(17, summation.derivative)
        self.assertEqual(17, summation2.derivative)
    
    def test_sin_scalar(self):
        var = Variable(9, 2)
        result = np.sin(var)
        
        self.assertEqual(np.sin(9), result.value)
        self.assertEqual(np.cos(9)*2, result.derivative)
    
    def test_tan_scalar(self):
        var = Variable(-0.1, -2)
        result = np.tan(var)

        self.assertEqual(np.tan(-0.1), result.value)
        self.assertEqual(-2/(np.cos(-0.1)**2), result.derivative)

    def test_sinh_scalar(self):
        var = Variable(-.5, 1.2)
        result = np.sinh(var)

        self.assertEqual(np.sinh(-.5), result.value)
        self.assertEqual(np.cosh(-.5)*1.2, result.derivative)

    def test_arcsin_scalar(self):
        var = Variable(.4, -2)
        result = np.arcsin(var)

        self.assertEqual(np.arcsin(.4), result.value)
        self.assertEqual(-2/np.sqrt(1-.4**2), result.derivative)

if __name__ == '__main__':
    unittest.main()