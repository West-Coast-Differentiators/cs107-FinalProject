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
        result = var.log()

        self.assertEqual(np.log(5), result.value)
        self.assertEqual((1/5)*1.5, result.derivative)

    def test_exp_scalar(self):
        var = Variable(5, 1.5)
        result = var.exp()

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

    def test_sqrt_scalar(self):
        var = Variable(4, -1)
        result = var.sqrt()

        self.assertEqual(np.sqrt(4), result.value)
        self.assertEqual((-1 * 0.5) * np.power(4, -0.5), result.derivative)

    def test__pow__scalar(self):
        var = Variable(4, 3)
        result = var ** 3
        reverse_result = 3 ** var
        combined_result = var ** Variable(5, 0.7)

        self.assertEqual(4 ** 3, result.value)
        self.assertEqual(var.derivative * 3 * (4**2), result.derivative)
        self.assertEqual(3 ** 4, reverse_result.value)
        self.assertAlmostEqual(var.derivative * np.log(3) * (3 ** var.value), reverse_result.derivative)
        self.assertEqual(4 ** 5, combined_result.value)
        self.assertAlmostEqual(4 ** 5 * ((5 * 3 / 4) + (np.log(4) * 0.7)), combined_result.derivative)

    def test_pow_exception(self):
        with self.assertRaises(TypeError):
            result = Variable(6, 6.7) ** 'string'
        with self.assertRaises(ValueError):
            result = Variable(-1, 6.7) ** 45.7
        with self.assertRaises(ValueError):
            result = Variable(0, 6) ** -2
        with self.assertRaises(ValueError):
            result = (-0.7) ** Variable(2, 4)
        with self.assertRaises(ValueError):
            result = 0 ** Variable(-2, 0.45)

    def test_log_exception(self):
        with self.assertRaises(ValueError) as e:
            np.log(Variable(-23, 9))

    def test_abs_scalar(self):
        var = Variable(-12, 9)
        zero_var = Variable(0, 0.5)
        result = abs(var)

        self.assertEqual(12, result.value)
        self.assertEqual(var.derivative * -1, result.derivative)
        with self.assertRaises(ValueError) as e:
            abs(zero_var)
    
    def test_arccos_scalar_invalid_value(self):
        with self.assertRaises(ValueError) as e:
            var = Variable(18, 2)
            np.arccos(var)
        self.assertEqual("Inputs to arccos should be in [-1, 1].", str(e.exception))

        with self.assertRaises(ValueError) as e:
            var = Variable(-18, 2)
            np.arccos(var)
        self.assertEqual("Inputs to arccos should be in [-1, 1].", str(e.exception))

    def test_arccos_scalar_invalid_value(self):
        with self.assertRaises(ValueError) as e:
            var = Variable(18, 2)
            np.arccos(var)
        self.assertEqual("Inputs to arccos should be in [-1, 1].", str(e.exception))

        with self.assertRaises(ValueError) as e:
            var = Variable(-18, 2)
            np.arccos(var)
        self.assertEqual("Inputs to arccos should be in [-1, 1].", str(e.exception))

    def test_arctan_scalar(self):
        var = Variable(.5, .75)
        result = np.arctan(var)

        self.assertEqual(np.arctan(.5), result.value)
        self.assertEqual((1)/(1 + .5**2)*.75, result.derivative)

    def test_tanh_scalar(self):
        var = Variable(.5, .75)
        result = np.tanh(var)

        self.assertEqual(np.tanh(.5), result.value)
        self.assertEqual((1)/(np.cosh(.5)**2)*.75, result.derivative)

    def test__mul__scalar_two_variable_objects(self):
        var1 = Variable(5.0, 1.0)
        var2 = Variable(2.0, 2.0)
        mult = var1 * var2
        mult_reverse_order = var2 * var1
        
        self.assertEqual(10.0, mult.value)
        self.assertEqual(12, mult.derivative)
        self.assertEqual(10.0, mult_reverse_order.value)
        self.assertEqual(12, mult_reverse_order.derivative)
    
    def test__mul__scalar_one_variable_one_constant(self):
        var = Variable(5.0, 2.0)
        multiply = var * 4
        multiply2 = 5 * var
        self.assertEqual(20, multiply.value)
        self.assertEqual(8, multiply.derivative)
        self.assertEqual(25, multiply2.value)
        self.assertEqual(10, multiply2.derivative)

    def test__truediv__scalar_two_variable_objects(self):
        var1 = Variable(20.0, 2.0)
        var2 = Variable(10.0, 5.0)
        divided = var1 / var2
        divided_reverse_order = var2 / var1
        
        self.assertEqual(2, divided.value)
        self.assertEqual(-0.8, divided.derivative)
        self.assertEqual(.5, divided_reverse_order.value)
        self.assertEqual(.2, divided_reverse_order.derivative)
    
    def test_truediv_scalar_zero_value(self):
        with self.assertRaises(ZeroDivisionError) as e:
            var = Variable(20.0, 2.0)
            divided = var / 0
        self.assertEqual("You cannot use a value of Zero.", str(e.exception))

        with self.assertRaises(ZeroDivisionError) as e:
            var1 = Variable(20.0, 2.0)
            var2 = Variable(0.0, 5.0)
            divided = var1 / var2
        self.assertEqual("You cannot use a value of Zero.", str(e.exception))
    
    def test__truediv__scalar_one_variable_one_constant(self):
        var = Variable(20.0, 2.0)
        divided = var / 4
        divided2 = 100 / var
        
        self.assertEqual(5, divided.value)
        self.assertEqual(0.5, divided.derivative)
        self.assertEqual(5, divided2.value)
        self.assertEqual(-0.5, divided2.derivative)

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
        
    def test_sub_and_sin_scalar(self):
        value = np.pi / 4
        var = Variable(value, 1)
        equation = var - np.sin(var)
        
        self.assertEqual(value - np.sin(value), equation.value)
        self.assertEqual(1 - np.cos(value), equation.derivative)
        
    def test_sum_and_cos_scalar(self):
        value = np.pi /3
        var = Variable(value, 1)
        equation = var + np.cos(var)
        
        self.assertEqual(value + np.cos(value), equation.value)
        self.assertEqual(1 - np.sin(value), equation.derivative)
    
    
    def test_sub_and_cos_scalar(self):
        value = np.pi /3
        var = Variable(value, 1)
        equation = var - np.cos(var)
        
        self.assertEqual(value - np.cos(value), equation.value)
        self.assertEqual(1 + 1*np.sin(value), equation.derivative)
    
    def test_cosh_of_cos_scalar(self):
        value = 0.23
        var = Variable(value, 1)
        equation = np.cosh(np.cos(var))
        expected_derivative = np.sinh(np.cos(value))*(-np.sin(value))
        
        self.assertEqual(np.cosh(np.cos(value)), equation.value)
        self.assertEqual(expected_derivative, equation.derivative)

        
    def test_cosh_of_sin_scalar(self):
        value = 0.55
        var = Variable(value, 1)
        equation = np.cosh(np.sin(var))
        expected_derivative = np.sinh(np.sin(value))*(np.cos(value))
        
        self.assertEqual(np.cosh(np.sin(value)), equation.value)
        self.assertEqual(expected_derivative, equation.derivative)

    def test_sinh_of_cos_scalar(self):
        value = 0.92
        var = Variable(value, 1)
        equation = np.sinh(np.cos(var))
        expected_derivative = np.cosh(np.cos(value))*(-np.sin(value))
        
        self.assertEqual(np.sinh(np.cos(value)), equation.value)
        self.assertEqual(expected_derivative, equation.derivative)
        
    def test_add_cos_and_arccos_scalar(self):
        value1 = 0.92
        value2 = 1.42
        var1 = Variable(value1, 4)
        var2 = Variable(value2, 2)
        equation = np.arccos(var1) + np.cos(var2)
        expected_derivative = (-1) * 4/np.sqrt(1-value1**2) - 2 * np.sin(value2)
        
        self.assertEqual(np.arccos(value1) + np.cos(value2), equation.value)
        self.assertEqual(expected_derivative, equation.derivative)
        
    def test_sub_sin_and_arcsin_scalar(self):
        value1 = -0.10
        value2 = 1.51
        var1 = Variable(value1, 5)
        var2 = Variable(value2, 3)
        equation = np.arcsin(var1) - np.sin(var2)
        expected_derivative = 1 * 5/np.sqrt(1-value1**2) - 3 * np.cos(value2)
        
        self.assertEqual(np.arcsin(value1) - np.sin(value2), equation.value)
        self.assertEqual(expected_derivative, equation.derivative)
    
    def test_mx_plus_b_scalar(self):
        m, alpha, beta = 2, 2.0, 3.0
        x = Variable(m, 1)
        equation1 = alpha * x + beta
        equation2 = x * alpha + beta
        equation3 = beta + alpha * x
        equation4 = beta + x * alpha
        
        self.assertEqual(7, equation1.value)
        self.assertEqual(2, equation1.derivative)
        self.assertEqual(7, equation2.value)
        self.assertEqual(2, equation2.derivative)
        self.assertEqual(7, equation3.value)
        self.assertEqual(2, equation3.derivative)
        self.assertEqual(7, equation4.value)
        self.assertEqual(2, equation4.derivative)

    def test_truediv_and_tanh_scalar(self):
        value = np.pi / 4
        var = Variable(value, 1)
        equation = var / np.tanh(var)
        equation2 = np.tanh(var) / var

        self.assertEqual(value / np.tanh(value), equation.value)
        expected_derivative = (np.tanh(value) * 1 - value * (1 / (np.cosh(value) **2))) / (np.tanh(value) **2)
        self.assertEqual(expected_derivative, equation.derivative)

        # Test for rtruediv
        self.assertEqual(np.tanh(value) / value, equation2.value)
        expected_derivative2 = (value * (1 / (np.cosh(value) **2) * 1) - np.tanh(value)*1) / (value **2)
        self.assertEqual(expected_derivative2, equation2.derivative)
    
    def test_mul_and_tanh_scalar(self):
        value = np.pi / 3
        var = Variable(value, 1)
        equation = var * np.tanh(var)
        
        self.assertEqual(value * np.tanh(value), equation.value)
        expected_derivative = value * ((1 / (np.cosh(value)**2) * 1)) + np.tanh(value) * 1
        self.assertEqual(expected_derivative, equation.derivative)

    def test_mul_and_tan_scalar(self):
        value = np.pi /3
        var = Variable(value, 1)
        equation = var * np.tan(var)
        
        self.assertEqual(value * np.tan(value), equation.value)
        expected_derivative = value * (1 / (np.cos(value) **2)) + np.tan(value) * 1
        self.assertEqual(expected_derivative, equation.derivative)

    def test_truediv_and_tan_scalar(self):
        value = np.pi / 4
        var = Variable(value, 1)
        equation = var / np.tan(var)
        equation2 = np.tan(var) / var

        self.assertEqual(value / np.tan(value), equation.value)
        expected_derivative = (np.tan(value) * 1 - value * (1 / (np.cos(value) **2))) / (np.tan(value) **2)
        self.assertEqual(expected_derivative, equation.derivative)

        # Test for rtruediv
        self.assertEqual(np.tan(value) / value, equation2.value)
        expected_derivative2 = (value * (1 / (np.cos(value) **2)) - np.tan(value)*1) / (value **2)
        self.assertEqual(expected_derivative2, equation2.derivative)

    def test_mul_and_cos_scalar(self):
        value = np.pi /3
        var = Variable(value, 1)
        equation = var * np.cos(var)
        
        self.assertEqual(value * np.cos(value), equation.value)
        expected_derivative = value * -np.sin(value) + np.cos(value) * 1
        self.assertEqual(expected_derivative, equation.derivative)

    def test_truediv_and_cos_scalar(self):
        value = np.pi / 4
        var = Variable(value, 1)
        equation = var / np.cos(var)
        equation2 = np.cos(var) / var

        self.assertEqual(value / np.cos(value), equation.value)
        expected_derivative = (np.cos(value)*1 - value * -np.sin(value)) / (np.cos(value) **2)
        self.assertEqual(expected_derivative, equation.derivative)

        # Test for rtruediv
        self.assertEqual(np.cos(value) / value, equation2.value)
        expected_derivative2 = (value * -np.sin(value) -np.cos(value)*1) / (value **2)
        self.assertEqual(expected_derivative2, equation2.derivative)

    def test_mul_and_cos_scalar(self):
        value = np.pi /3
        var = Variable(value, 1)
        equation = var * np.cos(var)
        
        self.assertEqual(value * np.cos(value), equation.value)
        expected_derivative = value * -np.sin(value) + np.cos(value) * 1
        self.assertEqual(expected_derivative, equation.derivative)

    def test_mul_and_sin_scalar(self):
        value = np.pi / 4
        var = Variable(value, 1)
        equation = var * np.sin(var)

        self.assertEqual(value * np.sin(value), equation.value)
        expected_derivative = value * np.cos(value) + np.sin(value) * 1
        self.assertEqual(expected_derivative, equation.derivative)

    def test_truediv_and_sin_scalar(self):
        value = np.pi / 4
        var = Variable(value, 1)
        equation = var / np.sin(var)
        equation2 = np.sin(var) / var

        self.assertEqual(value / np.sin(value), equation.value)
        expected_derivative = (np.sin(value)*1 - value * np.cos(value)) / (np.sin(value) **2)
        self.assertEqual(expected_derivative, equation.derivative)

        # Test for rtruediv
        self.assertEqual(np.sin(value) / value, equation2.value)
        expected_derivative2 = (value * np.cos(value) - np.sin(value)*1) / (value **2)
        self.assertEqual(expected_derivative2, equation2.derivative)

    def test_mul_tanh_and_arctan_scalar(self):
        value1 = 5.0
        value2 = 2.0
        var1 = Variable(value1, 1)
        var2 = Variable(value2, 2)
        equation = np.arctan(var1) * np.tanh(var2)
        expected_derivative = np.arctan(value1) * (1 / (np.cosh(value2)**2) * 2) + np.tanh(value2) * (1 / (1 + value1**2) * 1)
        
        self.assertEqual(np.arctan(value1) * np.tanh(value2), equation.value)
        self.assertEqual(expected_derivative, equation.derivative)

    def test_truediv_tanh_and_arctan_scalar(self):
        value1 = 5.0
        value2 = 2.0
        var1 = Variable(value1, 1)
        var2 = Variable(value2, 2)
        equation = np.arctan(var1) / np.tanh(var2)

        self.assertEqual(np.arctan(value1) / np.tanh(value2), equation.value)
        expected_derivative = (np.tanh(value2) * (1 / (1 + value1**2) * 1) - np.arctan(value1) * (1 / (np.cosh(value2)**2) * 2)) / (np.tanh(value2) **2)
        self.assertEqual(expected_derivative, equation.derivative)

    def test_mul_and_arctan_scalar(self):
        value = np.pi / 3
        var = Variable(value, 1)
        equation = var * np.arctan(var)
        
        self.assertEqual(value * np.arctan(value), equation.value)
        expected_derivative = value * (1 / (1 + value**2)) + np.arctan(value) * 1
        self.assertEqual(expected_derivative, equation.derivative)

    def test_truediv_and_arctan_scalar(self):
        value = np.pi / 4
        var = Variable(value, 1)
        equation = var / np.arctan(var)
        equation2 = np.arctan(var) / var

        self.assertEqual(value / np.arctan(value), equation.value)
        expected_derivative = (np.arctan(value) * 1 - value * (1 / (1 + value**2) * 1)) / (np.arctan(value) **2)
        self.assertEqual(expected_derivative, equation.derivative)

        # Test for rtruediv
        self.assertEqual(np.arctan(value) / value, equation2.value)
        expected_derivative2 = (value * (1 / (1 + value**2) * 1) - np.arctan(value)*1) / (value **2)
        self.assertEqual(expected_derivative2, equation2.derivative)
    
    def test_mul_and_arctan_scalar(self):
        value = np.pi / 3
        var = Variable(value, 1)
        equation = var * np.arctan(var)
        
        self.assertEqual(value * np.arctan(value), equation.value)
        expected_derivative = value * (1 / (1 + value**2)) + np.arctan(value) * 1
        self.assertEqual(expected_derivative, equation.derivative)

if __name__ == '__main__':
    unittest.main()