import unittest
import numpy as np

from WestCoastAD import Optimizer


class VariableUnitTest(unittest.TestCase):
    
    def test_x_squared_optimization(self):
        def objective_func(x):
            return x**2

        var_init = np.array([2])
        optimizer = Optimizer(objective_func, var_init)
        min_value, var_value = optimizer.gd_optimize(tolerance=0.0000001, num_iterations=1000, verbose=False)
        self.assertAlmostEqual(min_value, 0, places=5)
        self.assertAlmostEqual(var_value[0], 0, places=5)
    
    def test_x_y_z_squared_optimization(self):
        def objective_func(x, y, z):
            return x**2+y**2+z**2

        var_init = np.array([-15, 100, -20])
        optimizer = Optimizer(objective_func, var_init)
        min_value, var_value = optimizer.gd_optimize(tolerance=0.0000001, num_iterations=1000, verbose=False)
        self.assertAlmostEqual(min_value, 0, places=5)
        self.assertAlmostEqual(var_value[0], 0, places=5)
        self.assertAlmostEqual(var_value[1], 0, places=5)
        self.assertAlmostEqual(var_value[2], 0, places=5)

        def objective_func(x):
            return x[0]**2+x[1]**2+x[2]**2

        var_init = np.array([-15, 100, -20])
        optimizer = Optimizer(objective_func, var_init, scalar=False)
        min_value, var_value = optimizer.gd_optimize(tolerance=0.0000001, num_iterations=1000, verbose=False)
        self.assertAlmostEqual(min_value, 0, places=5)
        self.assertAlmostEqual(var_value[0], 0, places=5)
        self.assertAlmostEqual(var_value[1], 0, places=5)
        self.assertAlmostEqual(var_value[2], 0, places=5)

    def test_univariate_scalar_adam_optimize(self):
        def objective_func(x):
            return np.exp(-2.0 * np.sin(4.0*x)*np.sin(4.0*x))

        var_init = np.array([2])
        optimizer = Optimizer(objective_func, var_init)
        min_value, var_value = optimizer.adam_optimize(learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, num_iterations=1000)
        self.assertEqual(min_value, 0.17914695868777564)
        self.assertEqual(var_value[0], 1.86753937)

    def test_x_y_z_squared_adam_optimize(self):
        def objective_func(x, y, z):
            return x**2+y**2+z**2

        var_init = np.array([-15, 100, -20])
        optimizer = Optimizer(objective_func, var_init)
        min_value, var_value = optimizer.adam_optimize(learning_rate=0.01, beta1=0.9, beta2=0.999, epsilon=1e-8, num_iterations=1000, tolerance=0.0000001)
        self.assertAlmostEqual(min_value, 0, places=5)
        self.assertAlmostEqual(var_value[0], 0, places=5)
        self.assertAlmostEqual(var_value[1], 0, places=5)
        self.assertAlmostEqual(var_value[2], 0, places=5)

        def objective_func(x):
            return x[0]**2+x[1]**2+x[2]**2

        var_init = np.array([-15, 100, -20])
        optimizer = Optimizer(objective_func, var_init, scalar=False)
        min_value, var_value = optimizer.adam_optimize(tolerance=0.0000001,learning_rate=0.01, beta1=0.9, beta2=0.999, epsilon=1e-8, num_iterations=1000)
        self.assertAlmostEqual(min_value, 0, places=5)
        self.assertAlmostEqual(var_value[0], 0, places=5)
        self.assertAlmostEqual(var_value[1], 0, places=5)
        self.assertAlmostEqual(var_value[2], 0, places=5)

    def test_x_y_exp_func_adam_optimize(self):
        def objective_func(x, y):
            return x*y + np.exp(x*y)

        var_init = np.array([2, 2])
        optimizer = Optimizer(objective_func, var_init)
        min_value, var_value = optimizer.adam_optimize(learning_rate=0.01, beta1=0.9, beta2=0.999, epsilon=1e-8, num_iterations=1000, tolerance=1.0e-08)
        self.assertAlmostEqual(min_value, 1, places=5)
        self.assertAlmostEqual(var_value[0], 0, places=4)
        self.assertAlmostEqual(var_value[1], 0, places=4)

if __name__ == '__main__':
    unittest.main()
