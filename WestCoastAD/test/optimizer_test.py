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

    def test_univariate_scalar_momentum_optimization(self):
        def objective_func(x):
            return x**6 - 2*x

        var_init = np.array([2])
        optimizer = Optimizer(objective_func, var_init)
        min_value, var_value = optimizer.momentum_optimize(tolerance=0.0000001, num_iterations=1000, verbose=False)
        self.assertAlmostEqual(min_value, -1.3, places=1)
        self.assertAlmostEqual(var_value[0], 0.8, places=1)

    def test_multivariate_scalar_momentum_optimization(self):
        def objective_func(x, y):
            return x**2 + x*y + y**2

        var_init = np.array([0.2, 0.5])
        optimizer = Optimizer(objective_func, var_init)
        min_value, var_value = optimizer.momentum_optimize(tolerance=0.0000001, num_iterations=1000, verbose=False)
        self.assertAlmostEqual(min_value, 0, places=1)
        self.assertAlmostEqual(var_value[0], 0, places=1)
        self.assertAlmostEqual(var_value[1], 0, places=1)


    def test_multivariate_vector_momentum_optimization(self):
        def objective_func(x):
            return x[0]**2 + x[0]*x[1] + x[1]**2

        var_init = np.array([0.2, 0.5])
        optimizer = Optimizer(objective_func, var_init, scalar=False)
        min_value, var_value = optimizer.momentum_optimize(tolerance=0.0000001, num_iterations=1000, verbose=False)
        self.assertAlmostEqual(min_value, 0, places=1)
        self.assertAlmostEqual(var_value[0], 0, places=1)
        self.assertAlmostEqual(var_value[1], 0, places=1)

    def test_beta_exception(self):
        def objective_func(x):
            return x
        with self.assertRaises(ValueError) as e:
            var_init = np.array([0.2])
            optimizer = Optimizer(objective_func, var_init)
            optimizer.momentum_optimize(beta=54, num_iterations=1000, verbose=False)
        self.assertAlmostEqual("The value of beta (sample weight) should be between 0 and 1.", str(e.exception))

    def test_univariate_scalar_adagrad_optimization(self):
        def objective_func(x):
            return x * np.log(x)

        var_init = np.array([2])
        optimizer = Optimizer(objective_func, var_init)
        min_value, var_value = optimizer.adagrad_optimize(tolerance=0.0000001, num_iterations=100000, verbose=False)
        self.assertAlmostEqual(min_value, -0.36, places=1)
        self.assertAlmostEqual(var_value[0], 0.36, places=1)

    def test_multivariate_scalar_adagrad_optimization(self):
        def objective_func(x, y):
            return x**2 + x*y + y**2

        var_init = np.array([0.2, 0.5])
        optimizer = Optimizer(objective_func, var_init)
        min_value, var_value = optimizer.adagrad_optimize(tolerance=0.0000001, num_iterations=10000, verbose=False)
        self.assertAlmostEqual(min_value, 0, places=1)
        self.assertAlmostEqual(var_value[0], 0, places=1)
        self.assertAlmostEqual(var_value[1], 0, places=1)


    def test_multivariate_vector_adagrad_optimization(self):
        def objective_func(x):
            return x[0]**2 + x[0]*x[1] + x[1]**2

        var_init = np.array([0.2, 0.5])
        optimizer = Optimizer(objective_func, var_init, scalar=False)
        min_value, var_value = optimizer.adagrad_optimize(tolerance=0.0000001, num_iterations=10000, verbose=False)
        self.assertAlmostEqual(min_value, 0, places=1)
        self.assertAlmostEqual(var_value[0], 0, places=1)
        self.assertAlmostEqual(var_value[1], 0, places=1)

if __name__ == '__main__':
    unittest.main()