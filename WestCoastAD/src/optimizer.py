import numpy as np 

from WestCoastAD import Variable

class Optimizer():

    def __init__(self, objective_function, num_variables, variable_initialization):
        assert len(variable_initialization) == num_variables, "Length of variable_initialization should equal the number of variables."
        self.objective_function = objective_function
        self.num_variables = num_variables
        self.variable_initialization = variable_initialization


    def _generate_seed_derivative(self, index):
        seed_derivative = np.zeros(self.num_variables)
        seed_derivative[index] = 1.0
        return seed_derivative


    def _array_to_variable_class(self, variable_array):
        return [Variable(variable_array[i], self._generate_seed_derivative(i)) for i in range(self.num_variables)]


    def gd_optimize(self, num_iterations=100, learning_rate=0.01, tolerance=None, verbose=True):

        cur_variable_values = self.variable_initialization
        for i in range(num_iterations):

            objective = self.objective_function(*self._array_to_variable_class(cur_variable_values))
            delta_var = learning_rate * objective.derivative
            cur_variable_values = cur_variable_values - delta_var

            if verbose:
                print("iteration: {}, objective function value: {}".format(i, objective.value))

            if tolerance!=None and np.linalg.norm(delta_var) < tolerance:
                print("Variable update tolerance was reached. Terminating Search.")
                break
        
        return objective.value, cur_variable_values
    
