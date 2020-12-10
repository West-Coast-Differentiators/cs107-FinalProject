import unittest
import numpy as np

from WestCoastAD import Optimizer


class VariableUnitTest(unittest.TestCase):
    
    def test_x_squared_optimization(self):
        def objective_func(x):
            return x**2

        var_init = np.array([2])
        optimizer = Optimizer(objective_func, 1, var_init)
        min_value, var_value = optimizer.gd_optimize(tolerance=0.0000001, num_iterations=1000, verbose=False)
        self.assertAlmostEqual(min_value, 0, places=5)
        self.assertAlmostEqual(var_value[0], 0, places=5)
    
    def test_x_y_z_squared_optimization(self):
        def objective_func(x, y, z):
            return x**2+y**2+z**2

        var_init = np.array([-15, 100, -20])
        optimizer = Optimizer(objective_func, 3, var_init)
        min_value, var_value = optimizer.gd_optimize(tolerance=0.0000001, num_iterations=1000, verbose=False)
        self.assertAlmostEqual(min_value, 0, places=5)
        self.assertAlmostEqual(var_value[0], 0, places=5)
        self.assertAlmostEqual(var_value[1], 0, places=5)
        self.assertAlmostEqual(var_value[2], 0, places=5)

if __name__ == '__main__':
    unittest.main()