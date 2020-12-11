"""
This file provides the definition of an optimizer class that can be used for optimizing multi and uni variate
scalar functions
"""

import numpy as np 

from WestCoastAD import differentiate

class Optimizer():
    """
    This is an optimizer class used for minimizing functions defined in terms of WestCoastAD variables.
    """

    def __init__(self, objective_function, variable_initialization, scalar=True):
        """
        constructor for the Optimizer class.

        INPUTS
        =======
        - objective_function: a python function that takes as input a single vector or one or more scalars and
                returns a 1D numpy array of functions if objective_function is a vector function or a single function
                if objective_function is a scalar function.
        - variable_initialization: a 1D numpy array of floats/ints containing initial values for the inputs to the 
                objective function
        - scalar: True if the inputs to objective_function are one or more scalars otherwise False; Default is True

        RETURNS
        ========
        None

        NOTES
        =====
        Pre:
         - objective_function must only use operations that are supported by WestCoastAD's Variable class
         - variable_values has the same length as the number of inputs to objective_fuctnion if objective_function takes 
                scalar inputs, or the length of the vector input to objective_function.
         - variable_values must be in the same order as the inputs to func

        EXAMPLES
        =========
        
        # multivariate function with scalars as input
        >>> import numpy as np
        >>> f = lambda x, y: x**2 + y**2
        >>> op = Optimizer(f, np.array([1, -1]))

        # multivariate function with a vector as input
        >>> import numpy as np
        >>> f = lambda x: x[0]**2 + x[1]**2
        >>> op = Optimizer(f, np.array([1, -1]), scalar=False)

        # univariate function with scalar as input
        >>> import numpy as np
        >>> f = lambda x: x**2
        >>> op = Optimizer(f, np.array([1]))
        
        """

        self.objective_function = objective_function
        self.scalar = scalar
        self.variable_initialization = variable_initialization

    def _print_updates(self, verbose_flag, index, value):
        """
        Private helper method to print out the iteration count and objective function value on all optimizer methods.

        INPUTS
        =======
        - verbose_flag: a boolean specifying whether updates about the optimization process will be printed
                to the console.
        - index: An int specifying the iteration count.
        - value: a float value, specifying the value of the objective function.

        RETURNS
        ========
        - None

        EXAMPLES
        =========

        # Print the value to stdout when verbose_flag is true
        >>> f = lambda x: x
        >>> op = Optimizer(f, np.array([1]))
        >>> op._print_updates(True, 1, 2)
        iteration: 1, objective function value: 2

        # Print nothing when verbose_flag is false
        >>> f = lambda x: x
        >>> op = Optimizer(f, np.array([1]))
        >>> op._print_updates(False, 1, 2)

        """
        if verbose_flag:
            print("iteration: {}, objective function value: {}".format(index, value))

    def _tolerance_check(self, tolerance, value):
        """
        Private helper method to check if the a given value is lesser than the tolerance threshold.

        INPUTS
        =======
        - tolerance: a float value, specifying the threshold against which we need to .
        - value: a float value, specifying the value of the objective function.

        RETURNS
        ========
        - True if the L2 norm of the given value is less than tolerance.

        EXAMPLES
        =========

        # Print the value to stdout when verbose_flag is true
        >>> f = lambda x: x
        >>> op = Optimizer(f, np.array([1]))
        >>> print(op._tolerance_check(0.01, 1))
        None

        # Print nothing when verbose_flag is false
        >>> f = lambda x: x
        >>> op = Optimizer(f, np.array([1]))
        >>> print(op._tolerance_check(1, .05))
        Variable update tolerance was reached. Terminating Search.
        True

        """
        if tolerance != None and np.linalg.norm(value) < tolerance:
            print("Variable update tolerance was reached. Terminating Search.")
            return True

    def gd_optimize(self, num_iterations=100, learning_rate=0.01, tolerance=None, verbose=False):
        """
        method that performs gradient descent optimization of the objective function

        INPUTS
        =======
        - num_iterations: an int specifying the maximum number of iterations of gradient descent; Default is 100
        - learning_rate: a float/int specifying the learning rate for gradient descent; Default is 0.01
        - tolerance: a float specifying the smallest tolerance for the updates to the variables. If the L2 norm
                of the update step is smaller than this value, gradient descent will terminate; Default is None 
                (no tolerance check is used) 
        - verbose: a boolean specifying whether updates about the optimization process will be printed
                to the console. Default is False

        RETURNS
        ========
        - val: the minimum value of the objective_function that was found (float)
        - cur_variable_values: the values for the inputs to objective_function that gave the
                minimum objective_value found. (1D array of floats with the same size as the number of
                inputs to the objective function)


        EXAMPLES
        =========

        # multivariate function with scalars as input
        >>> import numpy as np
        >>> f = lambda x, y: x**2 + y**2
        >>> op = Optimizer(f, np.array([1, -1]))
        >>> op.gd_optimize(num_iterations=1000, learning_rate=0.1)
        (3.026941164608489e-194, array([ 1.23023192e-97, -1.23023192e-97]))

        # multivariate function with a vector as input
        >>> import numpy as np
        >>> f = lambda x: x[0]**2 + x[1]**2
        >>> op = Optimizer(f, np.array([1, -1]), scalar=False)
        >>> op.gd_optimize(num_iterations=1000, learning_rate=0.1)
        (3.026941164608489e-194, array([ 1.23023192e-97, -1.23023192e-97]))

        # univariate function with scalar as input
        >>> import numpy as np
        >>> f = lambda x: x**2
        >>> op = Optimizer(f, np.array([1]))
        >>> op.gd_optimize(num_iterations=1000, learning_rate=0.1)
        (1.5134705823042444e-194, array([1.23023192e-97]))

        """

        cur_variable_values = self.variable_initialization
        val, der = differentiate(self.objective_function, cur_variable_values, self.scalar)
        
        for i in range(num_iterations):
            
            delta_var = learning_rate * der
            cur_variable_values = cur_variable_values - delta_var
            val, der = differentiate(self.objective_function, cur_variable_values, self.scalar)

            self._print_updates(verbose, i, val)

            if self._tolerance_check(tolerance, delta_var):
                break

        return val, cur_variable_values

    def momentum_optimize(self, num_iterations=100, learning_rate=0.01, beta=0.9, tolerance=None, verbose=False):
        """
        Method that performs momentum gradient descent optimization of the objective function. It does so by factoring a
        momentum term during learning, which is an exponential moving average of current and past gradients.

        INPUTS
        =======
        - num_iterations: an int specifying the maximum number of iterations of gradient descent; Default is 100
        - learning_rate: a float/int specifying the learning rate for gradient descent; Default is 0.01
        - beta: A float ranging between 0 and 1 specifying the sample weight for exponential average of weights; Default
                is 0.9
        - tolerance: a float specifying the smallest tolerance for the updates to the variables. If the L2 norm
                       of the update step is smaller than this value, gradient descent will terminate; Default is None
                       (no tolerance check is used)
        - verbose: a boolean specifying whether updates about the optimization process will be printed
                       to the console. Default is False

        RETURNS
        ========
        - objective_value: the minimum value of the objective_function that was found (float)
        - cur_variable_values: the values for the inputs to objective_function that gave the
                       minimum objective_value found. (1D array of floats with the same size as the number of
                       inputs to the objective function)


        EXAMPLES
        =========
        # Univariate objective function with scalar inputs.
        >>> import numpy as np
        >>> g = lambda x: x**4 - x
        >>> op = Optimizer(g, np.array([1]))
        >>> op.momentum_optimize(num_iterations=1000, learning_rate=0.01)
        (-0.4724703937105774, array([0.62996052]))

        # Multivariate objective function with scalar inputs.
        >>> import numpy as np
        >>> g = lambda x, y: x**3 + 2*y**2 + 12
        >>> op = Optimizer(g, np.array([0.5, 0.88]))
        >>> op.momentum_optimize(num_iterations=10000, learning_rate=0.01)
        (12.00002667493136, array([2.98791178e-02, 1.51990528e-23]))

        # Multivariate objective function with vector inputs.
        >>> import numpy as np
        >>> g = lambda x: x[0]**3 + 2*x[1]**2 + 12
        >>> op = Optimizer(g, np.array([0.5, 0.88]), scalar=False)
        >>> op.momentum_optimize(num_iterations=1000, learning_rate=0.01)
        (12.00002667493136, array([2.98791178e-02, 1.51990528e-23]))

        """
        if not 0 <= beta <= 1:
            raise ValueError("The value of beta (sample weight) should be between 0 and 1.")
        cur_variable_values = self.variable_initialization
        val, der = differentiate(self.objective_function, cur_variable_values, self.scalar)
        _current_momentum = 0

        for i in range(num_iterations):
            _current_momentum = (beta * _current_momentum) + ((1 - beta) * der)
            delta_var = learning_rate * _current_momentum
            cur_variable_values = cur_variable_values - delta_var
            val, der = differentiate(self.objective_function, cur_variable_values, self.scalar)

            self._print_updates(verbose, i, val)

            if self._tolerance_check(tolerance, delta_var):
                break

        return val, cur_variable_values

    def adagrad_optimize(self, num_iterations=10000, learning_rate=0.01, fuzz_factor=0.0000001, tolerance=None, verbose=False):
        """
        Method that performs adaptive gradient descent optimization of the objective function.Adagrad adjusts the learning rate         alpha by dividing it by the square root of the cumulative sum of current and past squared gradients.

        INPUTS
        =======
        - num_iterations: an int specifying the maximum number of iterations of gradient descent; Default is 100
        - learning_rate: a float/int specifying the learning rate for gradient descent; Default is 0.01
        - fuzz_factor: A float to prevent division by zero during optimization; Default is 0.00000001
        - tolerance: a float specifying the smallest tolerance for the updates to the variables. If the L2 norm
                       of the update step is smaller than this value, gradient descent will terminate; Default is None
                       (no tolerance check is used)
        - verbose: a boolean specifying whether updates about the optimization process will be printed
                       to the console. Default is False

        RETURNS
        ========
        - objective_value: the minimum value of the objective_function that was found (float)
        - cur_variable_values: the values for the inputs to objective_function that gave the
                       minimum objective_value found. (1D array of floats with the same size as the number of
                       inputs to the objective function)


        EXAMPLES
        =========
        # Univariate objective function with scalar inputs.
        >>> import numpy as np
        >>> g = lambda x: x**4 - x
        >>> op = Optimizer(g, np.array([1]))
        >>> op.adagrad_optimize(num_iterations=1000, learning_rate=0.01)
        (-0.4705616040471904, array([0.65786042]))

        # Multivariate objective function with scalar inputs.
        >>> import numpy as np
        >>> g = lambda x, y: x**2 + y**2 + 12
        >>> op = Optimizer(g, np.array([0.5, 0.88]))
        >>> op.adagrad_optimize(num_iterations=10000, learning_rate=0.01)
        (12.000013226920059, array([8.13318093e-08, 3.63688329e-03]))

        # Multivariate objective function with vector inputs.
        >>> import numpy as np
        >>> g = lambda x: x[0]**2 + 2*x[1]**2 + 12
        >>> op = Optimizer(g, np.array([0.5, 0.88]), scalar=False)
        >>> op.adagrad_optimize(num_iterations=10000, learning_rate=0.01)
        (12.000026453839908, array([8.13318093e-08, 3.63688327e-03]))

        """
        cur_variable_values = self.variable_initialization
        val, der = differentiate(self.objective_function, cur_variable_values, self.scalar)
        _cumsum_gradient = 0

        for i in range(num_iterations):
            _cumsum_gradient = _cumsum_gradient + (der**2)
            delta_var = (learning_rate * der) / np.sqrt(_cumsum_gradient + fuzz_factor)
            cur_variable_values = cur_variable_values - delta_var
            val, der = differentiate(self.objective_function, cur_variable_values, self.scalar)

            self._print_updates(verbose, i, val)

            if self._tolerance_check(tolerance, delta_var):
                break

        return val, cur_variable_values

    def rmsprop_optimize(self, num_iterations=10000, learning_rate=0.001, fuzz_factor=0.0000001, beta=0.9, tolerance=None, verbose=False):
        """
        Method that performs RMSProp gradient descent optimization of the objective function.
        This is an enhancement to Adagrad and adjusts the learning rate alpha by dividing it by the
        exponential moving averages of gradients.

        INPUTS
        =======
        - num_iterations: an int specifying the maximum number of iterations of gradient descent; Default is 100
        - learning_rate: a float/int specifying the learning rate for gradient descent; Default is 0.001
        - fuzz_factor: A float to prevent division by zero during optimization; Default is 0.0000001
        - beta: A float ranging between 0 and 1 specifying the sample weight for exponential average of weights; Default
                is 0.9
        - tolerance: a float specifying the smallest tolerance for the updates to the variables. If the L2 norm
                       of the update step is smaller than this value, gradient descent will terminate; Default is None
                       (no tolerance check is used)
        - verbose: a boolean specifying whether updates about the optimization process will be printed
                       to the console. Default is False

        RETURNS
        ========
        - objective_value: the minimum value of the objective_function that was found (float)
        - cur_variable_values: the values for the inputs to objective_function that gave the
                       minimum objective_value found. (1D array of floats with the same size as the number of
                       inputs to the objective function)


        EXAMPLES
        =========
        # Univariate objective function with scalar inputs.
        >>> import numpy as np
        >>> g = lambda x: x**4 - x
        >>> op = Optimizer(g, np.array([1]))
        >>> op.rmsprop_optimize(num_iterations=1000, learning_rate=0.01)
        (-0.47214233786026005, array([0.61814907]))

        # Multivariate objective function with scalar inputs.
        >>> import numpy as np
        >>> g = lambda x, y: x**2 + y**2 + 12
        >>> op = Optimizer(g, np.array([0.5, 0.88]))
        >>> op.rmsprop_optimize(num_iterations=10000, learning_rate=0.01)
        (12.000499499999068, array([-0.01580348, -0.01580348]))

        # Multivariate objective function with vector inputs.
        >>> import numpy as np
        >>> g = lambda x: x[0]**2 + 2*x[1]**2 + 12
        >>> op = Optimizer(g, np.array([0.5, 0.88]), scalar=False)
        >>> op.rmsprop_optimize(num_iterations=10000, learning_rate=0.01)
        (12.000746582986158, array([-0.01580348, -0.01576123]))

        """
        cur_variable_values = self.variable_initialization
        val, der = differentiate(self.objective_function, cur_variable_values, self.scalar)
        _exp_average_gradient = 0

        for i in range(num_iterations):
            _exp_average_gradient = (beta * _exp_average_gradient) + ((1 - beta) * der**2)
            delta_var = (learning_rate * der) / np.sqrt(_exp_average_gradient + fuzz_factor)
            cur_variable_values = cur_variable_values - delta_var
            val, der = differentiate(self.objective_function, cur_variable_values, self.scalar)

            self._print_updates(verbose, i, val)

            if self._tolerance_check(tolerance, delta_var):
                break

        return val, cur_variable_values