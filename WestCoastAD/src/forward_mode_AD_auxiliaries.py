import numpy as np

from WestCoastAD import Variable

def vector_function_jacobian(vector_function):
    """
    Function that returns the jacobian of the given vector valued function as a 
    single numpy array

    INPUTS
    =======
    vector_function: a 1D numpy array (size n) of multi or uni variate functions defined in
    terms of WestCoastAD variables

    RETURNS
    ========
    The Jacobian of the vector function as an n by d matrix where n is the size of the
    input array and d is the dimensionality of the derivative of individual Variables in
    the input array. 

    NOTES
    =====
    PRE:
     - vector_function elements must be of the type WestCoastAD.Variable class
     - vector_function elements must have the same dimensionality (ie. their derivative dimensions must match).
       multivariate_dimension_check can be used to check the dimensionality constraint is satisfied 
    
    EXAMPLES
    =========

    >>> import numpy as np
    >>> from WestCoastAD import Variable
    >>> x = Variable(4, np.array([1, 0]))
    >>> y = Variable(3, np.array([0, 1]))
    >>> f = np.array([x+y, y**2, x*y])
    >>> vector_function_jacobian(f)
    array([[1., 1.],
           [0., 6.],
           [3., 4.]])

    """
    return np.array([func.derivative for func in vector_function])


def vector_function_value(vector_function):
    """
    Function that returns the value of the given vector valued function as a 
    single numpy array

    INPUTS
    =======
    vector_function: a 1D numpy array (size n) of multi or uni variate functions defined in
    terms of WestCoastAD variables

    RETURNS
    ========
    The  value of the vector function as an array of size n

    NOTES
    =====
    PRE:
     - vector_function elements must be of the type WestCoastAD.Variable class
     - vector_function elements must have the same dimensionality (ie. their derivative dimensions must match).
       multivariate_dimension_check can be used to check the dimensionality constraint is satisfied 
    
    EXAMPLES
    =========

    >>> import numpy as np
    >>> from WestCoastAD import Variable
    >>> x = Variable(4, np.array([1, 0]))
    >>> y = Variable(3, np.array([0, 1]))
    >>> f = np.array([x+y, y**2, x*y])
    >>> vector_function_value(f)
    array([ 7,  9, 12])

    """
    return np.array([func.value for func in vector_function])


def multivariate_dimension_check(variables):
    """
    Function that checks whether the derivatives of the variable classes have the same
    dimensionality

    INPUTS
    =======
    variables: a 1D numpy array or a list (size n) of WestCoastAD variable instances

    RETURNS
    ========
    True if the dimensions of all the variables match, else False

    NOTES
    =====
    PRE:
     - variables' elements must be of the type WestCoastAD.Variable class
     - variables must have length greater than zero otherwise a ValueError will be raised
    
    EXAMPLES
    =========

    # mismatched dimensions with list input
    >>> from WestCoastAD import Variable
    >>> x = Variable(4, np.array([1, 0]))
    >>> y = Variable(3, np.array([0, 1, 0]))
    >>> multivariate_dimension_check([x, y])
    False

    # matched dimensions with array input
    >>> import numpy as np
    >>> from WestCoastAD import Variable
    >>> x = Variable(4, np.array([1, 0, 0]))
    >>> y = Variable(3, np.array([0, 1, 0]))
    >>> multivariate_dimension_check(np.array([x, y]))
    True

    """
    
    if len(variables) < 1:
        raise ValueError("variable_list must have at least one variable.")
    derivative_dim = len(variables[0].derivative)
    for variable in variables:
        if len(variable.derivative) != derivative_dim:
            return False
    return True


if __name__ == "__main__":
    import doctest
    doctest.testmod()