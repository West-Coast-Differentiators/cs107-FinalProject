import numpy as np 

from WestCoastAD import Variable

class Optimizer():
    """
    This is an optimizer class used for minimizing functions defined in terms of WestCoastAD variables.
    """

    def __init__(self, objective_function, num_variables, variable_initialization):
        """
        constructor for the Optimizer class.
        INPUTS
        =======
        - objective_function: a python function with one or more WestCoastAD Variable instances as inputs.
                The function has to return a single WestCoastAD Variable instance that stores the derivative
                and value of the objective function. 
        - num_variables: number of inputs to the objective function. The function will be optimized with
                respect to these inputs
        - variable_initialization: a 1D numpy array of floats/ints containing initial values for the inputs to the 
                objective function
        RETURNS
        ========
        None
        NOTES
        =====
        Pre:
         - the number of values in variable_initialization must equal num_variables otherwise an AssertionError will be thrown
         - the initialization values must be given in the same order as the objective_function inputs
         - the number of inputs to objective function must equal num_variables
        EXAMPLES
        =========
        
        >>> import numpy as np
        >>> from WestCoastAD import Optimizer
        >>> f = lambda x, y: x**2 + y**2
        >>> op = Optimizer(f, 2, np.array([1, -1]))
        
        """
        assert len(variable_initialization) == num_variables, "Length of variable_initialization should equal the number of variables."
        self.objective_function = objective_function
        self.num_variables = num_variables
        self.variable_initialization = variable_initialization


    def _generate_seed_derivative(self, index):
        """
        This is a private method used for generating standard unit vectors to represent seed derivatives of
        the variables of the objective function.
        """
        seed_derivative = np.zeros(self.num_variables)
        seed_derivative[index] = 1.0
        return seed_derivative


    def _array_to_variable_class(self, variable_array):
        """
        This is a private method used for converting a 1D array of values for the objective function variables
        into WestCoastAD variables
        """
        return [Variable(variable_array[i], self._generate_seed_derivative(i)) for i in range(self.num_variables)]


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
        - objective_value: the minimum value of the objective_function that was found (float)
        - cur_variable_values: the values for the inputs to objective_function that gave the
                minimum objective_value found. (1D array of floats with the same size as the number of
                inputs to the objective function)
        EXAMPLES
        =========
        
        >>> import numpy as np
        >>> from WestCoastAD import Optimizer
        >>> f = lambda x, y: x**2 + y**2
        >>> op = Optimizer(f, 2, np.array([1, -1]))
        >>> op.gd_optimize(num_iterations=1000, learning_rate=0.1)
        (4.7295955697007643e-194, array([ 1.23023192e-97, -1.23023192e-97]))
        """

        cur_variable_values = self.variable_initialization
        
        for i in range(num_iterations):

            objective = self.objective_function(*self._array_to_variable_class(cur_variable_values))
            delta_var = learning_rate * objective.derivative
            cur_variable_values = cur_variable_values - delta_var

            if verbose:
                print("iteration: {}, objective function value: {}".format(i, objective.value))

            if tolerance!=None and np.linalg.norm(delta_var) < tolerance:
                print("Variable update tolerance was reached. Terminating Search. {}".format(i))
                break
        
        return objective.value, cur_variable_values

    def adam_optimizer(self, num_iterations=100, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, tolerance=None, verbose=False):
        """
        method that performs Adaptive Moment Estimation(adam) optimization of the objective function

        INPUTS
        =======
        Default parameters follow those provided in the original paper.
        - num_iterations: an int specifying the maximum number of iterations; Default is 100
        - learning_rate: a float/int specifying the learning rate for gradient descent; Default value follow those provided in the original paper.
        - beta1: Exponential decay hyperparameter for the first moment estimates. Default value follow those provided in the original paper.
        - beta2: Exponential decay hyperparameter for the second moment estimates. Default value follow those provided in the original paper.
        - epsilon: Hyperparameter preventing division by zero. Default value follow those provided in the original paper. Default value follow those provided in the original paper.
        - tolerance: a float specifying the smallest tolerance for the updates to the variables. If the L2 norm
                of the update step is smaller than this value, the adam_optimizer will terminate; Default is None 
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
        >>> import numpy as np
        >>> from WestCoastAD import Optimizer
        >>> f = lambda x, y: x**2 + y**2
        >>> op = Optimizer(f, 2, np.array([1, -1]))
        >>> op.adam_optimizer(num_iterations=1000, learning_rate=0.01, beta1=0.9, beta2=0.999, epsilon=1e-8)
        (5.330320183722531e-54, array([-1.37088662e-27,  1.37088662e-27]))

        """
        cur_variable_values = self.variable_initialization
        v, s, v_corrected, s_corrected,t = 0,0,0,0,0
        for l in range(num_iterations):
            t += 1
            # Compute the gradient
            objective = self.objective_function(*self._array_to_variable_class(cur_variable_values))
            delta_var = learning_rate * objective.derivative
            cur_variable_values = cur_variable_values - delta_var
            # Compute the moving average of the gradients.
            v = beta1 * v + (1 - beta1) * cur_variable_values
            # Compute bias-corrected first moment estimate.
            v_corrected = v / (1 - np.power(beta1, t))
            # Moving average of the squared gradients.
            s = beta2 * s + (1 - beta2) * np.power(cur_variable_values, 2)
            # Compute bias-corrected second raw moment estimate.
            s_corrected = s / (1 - np.power(beta2, t))
            # Update the derivatives.
            cur_variable_values = cur_variable_values - learning_rate * v_corrected / np.sqrt(s_corrected + epsilon)
            if verbose:
                print("iteration: {}, objective function value: {}".format(l, objective.value))
            if tolerance!=None and np.linalg.norm(delta_var) < tolerance:
                print("Variable update tolerance was reached. Terminating Search. {}".format(l))
                return objective.value, cur_variable_values
        return objective.value, cur_variable_values
