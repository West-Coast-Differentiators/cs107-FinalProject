import numpy as np 

from WestCoastAD import Variable

class Optimizer():
    """
    This is an optimizer class used for minimizing functions defined in terms of WestCoastAD variables.
    """

    def __init__(self, objective_function, num_variables, variable_initialization, objective_function_prime=None):
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
        self.objective_function_prime = objective_function_prime
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
        return [Variable(float(variable_array[i]), self._generate_seed_derivative(i)) for i in range(self.num_variables)]

    


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
            #print(cur_variable_values)
            if verbose:
                print("iteration: {}, objective function value: {}".format(i, objective.value))

            if tolerance!=None and np.linalg.norm(delta_var) < tolerance:
                print("Variable update tolerance was reached. Terminating Search.")
                break
        
        return objective.value, cur_variable_values
    
    def newton_raphson(self, num_iterations=100, tolerance=None, verbose=False):
        """
        method that performs Neton's method optimization of the objective function
        INPUTS
        =======
        - num_iterations: an int specifying the maximum number of iterations of Newton's method; Default is 100
        - tolerance: a float specifying the smallest tolerance for the updates to the variables. 
        - verbose: a boolean specifying whether updates about the optimization process will be printed
                to the console. Default is False

        RETURNS
        ========
        - f.value: the minimum value of the objective_function that was found (float)
        - xnew: the values for the inputs to objective_function that gave the
                minimum objective_value found. (1D array of floats with the same size as the number of
                inputs to the objective function)

        EXAMPLES
        =========
        >>> f = lambda x: x**2 - 12*x + 4
        >>> fprime = lambda x: 2*x-12
        >>> op = Optimizer(f, 1, np.array([1]),fprime)
        >>> op.newton_raphson(x0=np.array([12]), num_iterations=100, tolerance =1e-3, verbose=True)
        (-32.0, array([6.]))
        """
        if self.objective_function_prime is None:
            raise ValueError("Newton's method requires the first derivative function of the objective function.")
     
        x = self.variable_initialization
        
        for i in range(num_iterations):
            
            f = self.objective_function(*self._array_to_variable_class(x))
            f_prime = self.objective_function_prime(*self._array_to_variable_class(x))
            
            xnew = x - f_prime.value/f_prime.derivative          
            print(xnew)
            if verbose:
                print("iteration: {}, objective function value: {}".format(i, f.value))
            stepsize = np.linalg.norm(x-xnew)
            if tolerance!=None and stepsize < tolerance:
                print("Variable update tolerance was reached. Terminating Search.")
                return f.value, xnew
            
            x = xnew

        return f.value, xnew
    

if __name__ == "__main__":
    import doctest
    doctest.testmod()
    