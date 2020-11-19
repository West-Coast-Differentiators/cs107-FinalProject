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
        """ 
        Dunder method for overloading the "*" operator. 
        Computes the value and derivative of the multiplication operation

        INPUTS
        =======
        other: a Variable object, an int, or a float

        RETURNS
        ========
        a Variable object with the derivative and value of the multiplication operation.

        NOTES
        =====
        POST:
         - self is not changed by this function
        """
        try:
            return Variable(self.value * other.value, self.value * other.derivative + other.value * self.derivative)
        except AttributeError:
            other = Variable(other, 0)
            return Variable(self.value * other.value, self.value * other.derivative + other.value * self.derivative)
    

    def __rmul__(self, other):
        """ 
        Dunder method for overloading the "*" operator. 
        Computes the value and derivative of the multiplication operation

        INPUTS
        =======
        other: a Variable object, an int, or a float

        RETURNS
        ========
        a Variable object with the derivative and value of the multiplication operation.

        NOTES
        =====
        POST:
         - self is not changed by this function
        """
        return self.__mul__(other)


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
        """ 
        Dunder method for overloading the "/" operator. 
        Computes the value and derivative of the divison operation

        INPUTS
        =======
        other: a Variable object, an int, or a float

        RETURNS
        ========
        a Variable object with the derivative and value of the divison operation.

        NOTES
        =====
        PRE:
         -  other cannot be Zero a ZeroDivisionError will be raised.
        POST:
         - self is not changed by this function
        """
        try:
            if other.value == 0:
                raise ZeroDivisionError("Division by zero encountered")
            return Variable(self.value / other.value, (other.value *  self.derivative - self.value * other.derivative) / (other.value ** 2))
        except AttributeError:
            if other == 0:
                raise ZeroDivisionError("Division by zero encountered")
            other = Variable(other, 0)
            return Variable(self.value / other.value, (other.value *  self.derivative - self.value * other.derivative) / (other.value ** 2))
    

    def __rtruediv__(self, other):
        """ 
        Dunder method for overloading the "/" operator. 
        Computes the value and derivative of the divison operation

        INPUTS
        =======
        other: a Variable object, an int, or a float

        RETURNS
        ========
        a Variable object with the derivative and value of the divison operation.

        NOTES
        =====
        PRE:
         -  self.value cannot be Zero a ZeroDivisionError will be raised.
        POST:
         - self is not changed by this function
        """
        try:
            if self.value == 0:
                raise ZeroDivisionError("Division by zero encountered")
            return Variable(other.value / self.value, (self.value * other.derivative - other.value * self.derivative) / (self.value ** 2))
        except AttributeError:
            other = Variable(other, 0)
            return Variable(other.value / self.value, (self.value * other.derivative - other.value * self.derivative) / (self.value ** 2))
    

    def __pow__(self, other):
        """
        Dunder method for overloading the "**" operator.
        Computes the value and derivative of the power operation.

        INPUTS
        =======
        other: a Variable object, an int, or a float

        RETURNS
        ========
        a Variable object with the derivative and value of the power operation.

        NOTES
        =====
        POST:
         - self is not changed by this function

        """
        try:
            val = self.value ** other.value
            intermediate_value = other * np.log(self)
            der = val * intermediate_value.derivative
            return Variable(val, der)
        except AttributeError:
            if self.value < 0 and other % 1 != 0:
                raise ValueError("Cannot raise a negative number to the power of a non-integer value.")
            if self.value == 0 and other < 1:
                raise ValueError("Power function does not have a derivative at 0 if the exponent is less than 1.")
            val = self.value ** other
            der = self.derivative * other * self.value ** (other - 1)
            return Variable(val, der)


    def __rpow__(self, other):
        """
        Dunder method for overloading the "**" operator.
        Computes the value and derivative of the power operation.

        INPUTS
        =======
        other: a Variable object, an int, or a float

        RETURNS
        ========
        a Variable object with the derivative and value of the power operation.

        NOTES
        =====
        POST:
         - self is not changed by this function

        """
        if other == 0 and self.value <= 0:
            raise ValueError("Derivative of 0^x is undefined for non-positive x values")
        if other < 0:
            raise ValueError("Values and derivatives of a^x for a<0 are not defined in the real number domain")
        val = other ** self.value
        der = np.log(other) * val * self.derivative
        return Variable(val, der)
    

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
        """
        Computes the value and derivative of log function

        INPUTS
        =======
        None

        RETURNS
        ========
        a Variable object with the derivative and value of the log function.

        NOTES
        =====
        POST:
         - self is not changed by this function

        """
        if self.value <= 0.0:
            raise ValueError('Values for log should be greater than or equal to zero.')
        val = np.log(self.value)
        der = self.derivative * (1/self.value)
        return Variable(val, der)
    

    def exp(self):
        """
        Computes the value and derivative of exp function

        INPUTS
        =======
        None

        RETURNS
        ========
        a Variable object with the derivative and value of the exp function.

        NOTES
        =====
        POST:
         - self is not changed by this function

        """
        val = np.exp(self.value)
        der = self.derivative * np.exp(self.value)
        return Variable(val, der)
    

    def sqrt(self):
        """
        Computes the value and derivative of sqrt function

        INPUTS
        =======
        None

        RETURNS
        ========
        a Variable object with the derivative and value of the sqrt function.

        NOTES
        =====
        POST:
         - self is not changed by this function

        """
        return self.__pow__(0.5)
    

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
        """ 
        Computes the value and derivative of the tanh function

        INPUTS
        =======
        None

        RETURNS
        ========
        a Variable object with the derivative and value of the tanh function.

        NOTES
        =====
        POST:
         - self is not changed by this function
        """
        val = np.tanh(self.value)
        der = 1 / (np.cosh(self.value)**2) * self.derivative
        return Variable(val, der)
    

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
         - self.value should be in (-1, 1), otherwise a ValueError will be raised
        POST:
         - self is not changed by this function

        """

        if self.value >= 1 or self.value <= -1:
            raise ValueError("Inputs to arcsin should be in (-1, 1) for the derivative to be defined.")
        
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
         - self.value should be in (-1, 1), otherwise a ValueError will be raised
        POST:
         - self is not changed by this function

        """
        if self.value >= 1 or self.value <= -1:
            raise ValueError("Inputs to arccos should be in (-1, 1) for the derivative to be defined.")
        val = np.arccos(self.value)
        der = (-1) * self.derivative / np.sqrt(1-self.value**2)
        return Variable(val,der)


    def arctan(self):
        """ 
        Computes the value and derivative of the arctan function

        INPUTS
        =======
        None

        RETURNS
        ========
        a Variable object with the derivative and value of the arctan function.

        NOTES
        =====
        POST:
         - self is not changed by this function
        """
        val = np.arctan(self.value)
        der = 1 / (1 + self.value**2) * self.derivative
        return Variable(val, der)


    def __abs__(self):
        """
        Dunder method for overloading the abs function.
        Computes the value and derivative of abs function

        INPUTS
        =======
        None

        RETURNS
        ========
        a Variable object with the derivative and value of the abs function.

        NOTES
        =====
        POST:
         - self is not changed by this function

        """
        if self.value != 0.0:
            val = abs(self.value)
            der = self.derivative * (self.value / val)
            return Variable(val, der)
        else:
            raise ValueError('Abs function derivative does not exist at 0')

value = 0.75
var = Variable(value, 1)
equation = np.tanh(np.sin(var))

print(np.tanh(np.sin(value)))
print(equation.value)
# 1 / (np.cosh(self.value)**2) * self.derivative
expected_derivative = 1/((np.cosh(value)**2) * 1) *(np.cos(value)*1)

print(expected_derivative) 
print(equation.derivative)
