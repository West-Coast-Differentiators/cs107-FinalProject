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
