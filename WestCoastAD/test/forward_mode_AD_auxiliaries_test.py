import unittest
import numpy as np

from WestCoastAD import Variable
from WestCoastAD import vector_function_jacobian, vector_function_value, multivariate_dimension_check

class AuxiliariesUnitTest(unittest.TestCase):

    def test_vector_function_polynomial(self):
        x = Variable(2, np.array([1, 0, 0]))
        y = Variable(3, np.array([0, 1, 0]))
        z = Variable(1, np.array([0, 0, 1]))

        f_1 = x + y + z
        f_2 = x**2*y

        f = np.array([f_1, f_2])
        f_value = vector_function_value(f)
        f_jac = vector_function_jacobian(f)

        x = 2
        y = 3
        z = 1

        self.assertEqual(f_value[0], x + y + z)
        self.assertEqual(f_value[1], x**2*y)
        np.testing.assert_array_equal(f_jac, np.array([[1, 1, 1], [2*x*y, x**2, 0]]))
    

    def test_multivariate_dimension_check(self):
        x = Variable(2, np.array([1, 0, 0]))
        y = Variable(3, np.array([0, 1, 1]))
        z = Variable(1, np.array([0, 0, 1, 0]))
        self.assertFalse(multivariate_dimension_check([x,y,z]))

        x = Variable(2, np.array([1, 0]))
        y = Variable(3, np.array([0, 1, 1]))
        z = Variable(1, np.array([0, 0, 1]))
        self.assertFalse(multivariate_dimension_check([x,y,z]))

        x = Variable(2, np.array([1, 0]))
        y = Variable(3, np.array([0, 1]))
        z = Variable(1, np.array([0, 1]))
        self.assertTrue(multivariate_dimension_check([x,y,z]))

        x = Variable(2, np.array([1, 0]))
        self.assertTrue(multivariate_dimension_check([x]))
    
    
    def test_multivariate_dimension_check_exception(self):
        with self.assertRaises(ValueError) as e:
            multivariate_dimension_check([])
        self.assertEqual('variable_list must have at least one variable.', str(e.exception))



if __name__ == '__main__':
    unittest.main()