import unittest
import numpy as np

from WestCoastAD import Variable

class VariableUnitTest(unittest.TestCase):
    
    def test_value_setter_string(self):
        with self.assertRaises(TypeError) as e:
            var = Variable('s', 1)
        self.assertEqual('Input value should be numerical.', str(e.exception))
            
    def test_value_setter_float_and_int(self):
        var1 = Variable(1.2, 1)
        var2 = Variable(1, 1.3)
        var1.value = 2
        var2.value = 1.1
        
    def test_derivative_setter_string(self):
        with self.assertRaises(TypeError) as e:
            var = Variable(1.2, 'string')
        self.assertEqual('Input derivative seed should be numerical.', str(e.exception))
    
    def test_derivative_setter_float_and_int(self):
        var1 = Variable(1.2, 1)
        var2 = Variable(1, 1.3)
        var1.derivative = 1.5
        var2.derivative = 6
        
    def test__add__scalar_two_variable_objects(self):
        var1 = Variable(10.1, 2.1)
        var2 = Variable(9, 3)
        summation = var1 + var2
        summation_reverse_order = var2 + var1
        
        self.assertEqual(19.1, summation.value)
        self.assertEqual(19.1, summation_reverse_order.value)
        self.assertEqual(5.1, summation.derivative)
        self.assertEqual(5.1, summation_reverse_order.derivative)

    def test__repr__(self):
        var = Variable(2, 3)
        self.assertEqual(str(var), "Variable(value: 2, derivative: 3)")
    
    def test__add__scalar_one_variable_one_constant(self):
        var = Variable(3, 17)
        summation = var + 4
        summation2 = 5 + var
        
        self.assertEqual(7, summation.value)
        self.assertEqual(8, summation2.value)
        self.assertEqual(17, summation.derivative)
        self.assertEqual(17, summation2.derivative)
        
    def test_sub_scalar_two_variable_objects(self):
        var1 = Variable(10.1, 2.1)
        var2  = Variable(9, 3)
        substraction = var1 - var2
        substraction_reverse_order = var2 - var1
        
        self.assertEqual(1.1, round(substraction.value, 1))
        self.assertEqual(-1.1, round(substraction_reverse_order.value, 1))
        self.assertEqual(-0.9, round(substraction.derivative, 1))
        self.assertEqual(0.9, round(substraction_reverse_order.derivative, 1))
        
    def test__sub__scalar_one_variable_one_constant(self):
        var = Variable(3, 17)
        substraction = var - 4
        substraction2 = 5 - var
        
        self.assertEqual(-1, substraction.value)
        self.assertEqual(2, substraction2.value)
        self.assertEqual(17, substraction.derivative)
        self.assertEqual(-17, substraction2.derivative)
          
    def test_neg(self):
        var1 = Variable(9, 2)
        negation1 = - var1
        var2 = Variable(-8, -1)
        negation2 = - var2
        
        self.assertEqual(-9, negation1.value)
        self.assertEqual(-2, negation1.derivative)
        self.assertEqual(8, negation2.value)
        self.assertEqual(1, negation2.derivative)
        
    def test_sin_scalar(self):
        var = Variable(9, 2)
        result = np.sin(var)
        
        self.assertEqual(np.sin(9), result.value)
        self.assertEqual(np.cos(9)*2, result.derivative)
        
    def test_cos_scalar(self):
        var = Variable(8, 3)
        result = np.cos(var)
        
        self.assertEqual(np.cos(8), result.value)
        self.assertEqual(-np.sin(8)*3, result.derivative)
    
    def test_tan_scalar(self):
        var = Variable(-0.1, -2)
        result = np.tan(var)

        self.assertEqual(np.tan(-0.1), result.value)
        self.assertEqual(-2/(np.cos(-0.1)**2), result.derivative)
    
    def test_tan_scalar_invalid_value(self):
        with self.assertRaises(ValueError) as e:
            var = Variable(-5*np.pi/2, 1)
            np.tan(var)
        self.assertEqual("Inputs to tan should not be odd multiples of pi/2", str(e.exception))

    def test_sinh_scalar(self):
        var = Variable(-.5, 1.2)
        result = np.sinh(var)

        self.assertEqual(np.sinh(-.5), result.value)
        self.assertEqual(np.cosh(-.5)*1.2, result.derivative)
    
    def test_cosh_scalar(self):
        var = Variable(8, 1.3)
        result = np.cosh(var)
        
        self.assertEqual(np.cosh(8), result.value)
        self.assertEqual(np.sinh(8)*1.3, result.derivative)

    def test_log_scalar(self):
        var = Variable(5, 1.5)
        result = np.log(var)

        self.assertEqual(np.log(5), result.value)
        self.assertEqual((1/5)*1.5, result.derivative)

    def test_exp_scalar(self):
        var = Variable(5, 1.5)
        result = np.exp(var)

        self.assertEqual(np.exp(5), result.value)
        self.assertEqual(np.exp(5)*1.5, result.derivative)

    def test_arcsin_scalar(self):
        var = Variable(.4, -2)
        result = np.arcsin(var)

        self.assertEqual(np.arcsin(.4), result.value)
        self.assertEqual(-2/np.sqrt(1-.4**2), result.derivative)
    
    def test_arcsin_scalar_invalid_value(self):
        with self.assertRaises(ValueError) as e:
            var = Variable(-20, 1)
            np.arcsin(var)
        self.assertEqual("Inputs to arcsin should be in [-1, 1].", str(e.exception))

        with self.assertRaises(ValueError) as e:
            var = Variable(20, 1)
            np.arcsin(var)
        self.assertEqual("Inputs to arcsin should be in [-1, 1].", str(e.exception))

    def test_arccos_scalar(self):
        var = Variable(.8, -1.2)
        result = np.arccos(var)

        self.assertEqual(np.arccos(.8), result.value)
        self.assertEqual((-1)*(-1.2)/np.sqrt(1-.8**2), result.derivative)
    
    def test_arccos_scalar_invalid_value(self):
        with self.assertRaises(ValueError) as e:
            var = Variable(18, 2)
            np.arccos(var)
        self.assertEqual("Inputs to arccos should be in [-1, 1].", str(e.exception))

        with self.assertRaises(ValueError) as e:
            var = Variable(-18, 2)
            np.arccos(var)
        self.assertEqual("Inputs to arccos should be in [-1, 1].", str(e.exception))
        
class VariableIntegrationTest(unittest.TestCase):

    def test_sum_and_sin_scalar(self):
        value = np.pi /3
        var = Variable(value, 1)
        equation = var + np.sin(var)
        
        self.assertEqual(value+ np.sin(value), equation.value)
        self.assertEqual(1+np.cos(value), equation.derivative)
    
    def test_sinh_of_sin_scalar(self):
        value = 0.34
        var = Variable(value, 1)
        equation = np.sinh(np.sin(var))
        
        self.assertEqual(np.sinh(np.sin(value)), equation.value)
        expected_derivative = np.cosh(np.sin(value))*np.cos(value)
        self.assertEqual(expected_derivative, equation.derivative)

if __name__ == '__main__':
    unittest.main()