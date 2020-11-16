import numpy as np 

class Variable:
    """
    This is a custom variable class with elementary function and operation overloading
    to perform automatic differentiation.

    EXAMPLES
    =========

    # Derivative computation for a single input scalar function
    >>> x = Variable(4, 1)
    >>> f = 3*x**2 + 3
    >>> f.value
    51.0
    >>> f.derivative
    24.0

    """

    def __init__(self, value, derivative_seed):
        """ 
        constructor for the Variable class

        INPUTS
        =======
        value: An int or float giving the value of the variable
        derivative_seed: An int or float giving a seed value for the variable derivative

        """

        self.value = value
        self.derivative = derivative_seed
    
    
    def __repr__(self):
        return "Variable(value: {}, derivative: {})".format(self.value, self.derivative)
        
    @property
    def value(self):
        """ 
        getter method for getting the value attribute of the Variable object

        INPUTS
        =======
        None

        RETURNS
        ========
        the value attribute of the Variable object

        """
        return self._value
    
    @property
    def derivative(self):
        """ 
        getter method for getting the derivative attribute of the Variable object

        INPUTS
        =======
        None

        RETURNS
        ========
        the derivative attribute of the Variable object

        """
        return self._derivative
   
    @value.setter
    def value(self, value):
        """ 
        setter method for setting the value attribute of Variable object

        INPUTS
        =======
        value: An int or float giving the value of the variable

        RETURNS
        ========
        None

        """
        if not isinstance(value, (int, float)):
            raise TypeError('Input value should be numerical.')
        else:
            self._value = value
        
    @derivative.setter
    def derivative(self,derivative_seed):
        """ 
        setter method for setting the derivative attribute of Variable object

        INPUTS
        =======
        derivative_seed: An int or float giving a seed value for the variable derivative

        RETURNS
        ========
        None

        """
        if not isinstance(derivative_seed, (int, float)):
            raise TypeError('Input derivative seed should be numerical.')
        else:
            self._derivative = derivative_seed
        
    def __add__(self, other):
        """ 
        Dunder method for overloading the "+" operator. 
        Computes the value and the derivative of the summation operation

        INPUTS
        =======
        other: a Variable object, an int, or a float

        RETURNS
        ========
        a Variable object with the derivative and value of the summation operation.

        NOTES
        =====
        POST:
         - self is not changed by this function

        """
        try:
            val = self.value + other.value
            der = self.derivative + other.derivative
        except AttributeError:
            val = self.value + other
            der = self.derivative
        
        return Variable(val, der)


    def __radd__(self, other):
        """ 
        Dunder method for overloading the "+" operator. 
        Computes the value and derivative of the summation operation

        INPUTS
        =======
        other: a Variable object, an int, or a float

        RETURNS
        ========
        a Variable object with the derivative and value of the summation operation.

        NOTES
        =====
        POST:
         - self is not changed by this function

        """
        return self.__add__(other)
        
    
    def __mul__(self, other):
        pass
    
    def __rmul__(self, other):
        pass
        
    def __sub__(self, other):
        """ 
        Dunder method for overloading the "-" operator. 
        Computes the value and derivative of the substraction operation

        INPUTS
        =======
        other: a Variable object, an int, or a float

        RETURNS
        ========
        a Variable object with the derivative and value of the substraction operation.

        NOTES
        =====
        POST:
         - self is not changed by this function

        """
        return self + (-other)
    
    def __rsub__(self, other):
        """ 
        Dunder method for overloading the "-" operator. 
        Computes the value and derivative of the substraction operation

        INPUTS
        =======
        other: a Variable object, an int, or a float

        RETURNS
        ========
        a Variable object with the derivative and value of the substraction operation.

        NOTES
        =====
        POST:
         - self is not changed by this function

        """
        return (-self) + other
    
    def __truediv__(self, other):
        pass
    
    def __rtruediv__(self, other):
        pass
    
    def __pow__(self, other):
        pass

    def __rpow__(self, other):
        pass
    
    def __neg__(self):
        """ 
        This method is called using '-' operator.
        Computes the value and derivative of the negation operation

        INPUTS
        =======
        None

        RETURNS
        ========
        a Variable object with the derivative and value of the negation operation.

        NOTES
        =====
        POST:
         - self is not changed by this function

        """
        val = (-1) * self.value 
        der = (-1) * self.derivative
        return Variable(val, der)
    
    def log(self):
        val = np.log(self.value)
        der = self.derivative * (1/self.value)
        return Variable(val, der)
    
    def exp(self):
        val = np.exp(self.value)
        der = self.derivative * np.exp(self.value)
        return Variable(val, der)
    
    def sqrt(self):
        pass
    
    def sin(self):
        """ 
        Computes the value and derivative of the sin function

        INPUTS
        =======
        None

        RETURNS
        ========
        a Variable object with the derivative and value of the sin function.

        NOTES
        =====
        POST:
         - self is not changed by this function

        """

        val = np.sin(self.value)
        der = np.cos(self.value) * self.derivative
        return Variable(val, der)
    
    def cos(self):
        """ 
        Computes the value and derivative of the cos function

        INPUTS
        =======
        None

        RETURNS
        ========
        a Variable object with the derivative and value of the cos function.

        NOTES
        =====
        POST:
         - self is not changed by this function

        """
        val = np.cos(self.value)
        der = -np.sin(self.value) * self.derivative
        return Variable(val, der)
    
    
    def tan(self):
        """ 
        Computes the value and derivative of the tan function

        INPUTS
        =======
        None

        RETURNS
        ========
        a Variable object with the derivative and value of the tan function.

        NOTES
        =====
        PRE:
         -  self.value is not an odd multiple of pi/2 otherwise a ValueError will be raised
        POST:
         - self is not changed by this function

        """
        if (self.value / (np.pi/2)) % 2 == 1:
            raise ValueError("Inputs to tan should not be odd multiples of pi/2")

        val = np.tan(self.value)
        der = self.derivative / np.cos(self.value)**2
        return Variable(val, der)
    
    ## Other functions beyond the basics

    def sinh(self):
        """ 
        Computes the value and derivative of the sinh function

        INPUTS
        =======
        None

        RETURNS
        ========
        a Variable object with the derivative and value of the sinh function.

        NOTES
        =====
        POST:
         - self is not changed by this function

        """
        val = np.sinh(self.value)
        der = np.cosh(self.value) * self.derivative
        return Variable(val, der)
    
    def cosh(self):
        """ 
        Computes the value and derivative of the cosh function

        INPUTS
        =======
        None

        RETURNS
        ========
        a Variable object with the derivative and value of the cosh function.

        NOTES
        =====
        POST:
         - self is not changed by this function

        """
        val = np.cosh(self.value)
        der = np.sinh(self.value) * self.derivative
        return Variable(val, der)
    
    def tanh(self):
        pass
    
    def arcsin(self):
        """ 
        Computes the value and derivative of the arcsin function

        INPUTS
        =======
        None

        RETURNS
        ========
        a Variable object with the derivative and value of the arcsin function.

        NOTES
        =====
        PRE:
         - self.value should be in [-1, 1], otherwise a ValueError will be raised
        POST:
         - self is not changed by this function

        """

        if self.value > 1 or self.value < -1:
            raise ValueError("Inputs to arcsin should be in [-1, 1].")
        
        val = np.arcsin(self.value)
        der = self.derivative / np.sqrt(1-self.value**2)
        return Variable(val, der)
    
    def arccos(self):
        """ 
        Computes the value and derivative of the arccos function

        INPUTS
        =======
        None

        RETURNS
        ========
        a Variable object with the derivative and value of the arccos function.

        NOTES
        =====
        PRE:
         - self.value should be in [-1, 1], otherwise a ValueError will be raised
        POST:
         - self is not changed by this function

        """
        if self.value > 1 or self.value < -1:
            raise ValueError("Inputs to arccos should be in [-1, 1].")
        val = np.arccos(self.value)
        der = (-1) * self.derivative / np.sqrt(1-self.value**2)
        return Variable(val,der)
    
    def arctan(self):
        pass
    
    def abs(self):
        pass