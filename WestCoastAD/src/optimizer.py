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

            if verbose:
                print("iteration: {}, objective function value: {}".format(i, val))

            if tolerance!=None and np.linalg.norm(delta_var) < tolerance:
                print("Variable update tolerance was reached. Terminating Search.")
                break
        
        return val, cur_variable_values

class MomentumOptimizer(Optimizer):
    """
    This extends the base optimizer class for minimizing the objective functions expressed as Variable
    using a momentum, which is an exponential moving average of current and past gradients.
    """
    def gd_optimize(self, num_iterations=100, learning_rate=0.01, beta=0.9, tolerance=None, verbose=False):
        """
        method that performs gradient descent optimization of the objective function

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
        >>> op = MomentumOptimizer(g, np.array([1]))
        >>> op.gd_optimize(num_iterations=1000, learning_rate=0.01)
        (-0.4724703937105774, array([0.62996052]))

        # Multivariate objective function with scalar inputs.
        >>> import numpy as np
        >>> g = lambda x, y: x**3 + 2*y**2 + 12
        >>> op = MomentumOptimizer(g, np.array([0.5, 0.88]))
        >>> op.gd_optimize(num_iterations=1000, learning_rate=0.01)
        (12.00002667493136, array([2.98791178e-02, 1.51990528e-23]))

        # Multivariate objective function with vector inputs.
        >>> import numpy as np
        >>> g = lambda x: x[0]**3 + 2*x[1]**2 + 12
        >>> op = MomentumOptimizer(g, np.array([0.5, 0.88]), scalar=False)
        >>> op.gd_optimize(num_iterations=1000, learning_rate=0.01)
        (12.00002667493136, array([2.98791178e-02, 1.51990528e-23]))

        """
        if not 0 <= beta <= 1:
            raise ValueError("The value of beta (sample weight) should be between 0 and 1.")
        cur_variable_values = self.variable_initialization
        val, der = differentiate(self.objective_function, cur_variable_values, self.scalar)
        _momentum = []

        for i in range(num_iterations):
            _current_momentum = (beta * (0 if i == 0 else _momentum[i-1])) + ((1 - beta) * der)
            delta_var = learning_rate * _current_momentum
            cur_variable_values = cur_variable_values - delta_var
            _momentum.append(_current_momentum)
            val, der = differentiate(self.objective_function, cur_variable_values, self.scalar)

            if verbose:
                print("iteration: {}, objective function value: {}, delta: {}".format(i, val, delta_var))

            if tolerance != None and np.linalg.norm(delta_var) < tolerance:
                print("Variable update tolerance was reached. Terminating Search.")
                break

        return val, cur_variable_values
