import numpy as np 

class Variable:

    def __init__(self, value, derivative_seed):
        self.value = value
        self.derivative = derivative_seed
        
    @property
    def value(self):
        return self._value
    
    @property
    def derivative(self):
        return self._derivative
   
    @value.setter
    def value(self, value):
        if not isinstance(value, (int, float)):
            raise TypeError('Input value should be numerical.')
        else:
            self._value = value
        
    @derivative.setter
    def derivative(self,derivative_seed):
        if not isinstance(derivative_seed, (int, float)):
            raise TypeError('Input derivative seed should be numerical.')
        else:
            self._derivative = derivative_seed
        
        

    def __add__(self, other):
        try:
            val = self.value + other.value
            der = self.derivative + other.derivative
        except AttributeError:
            val = self.value + other
            der = self.derivative
        
        return Variable(val, der)

    def __radd__(self, other):
        return self.__add__(other)
        
    
    def __mul__(self, other):
        pass
    
    def __rmul__(self, other):
        pass
    
    # def __sub__(self, other):
        # try:
        #     val = self.value - other.value
        #     der = self.derivative - other.derivative
        # except AttributeError:
        #     val = self.value - other
        #     der = self.derivative
    
        
    def __sub__(self, other):
        return self + (-other)
    
    def __rsub__(self, other):
        return (-self) + other
    
    def __truediv__(self, other):
        pass
    
    def __rtruediv__(self, other):
        pass
    
    def __pow__(self, other):
        if not isinstance(other, (int, float)):
            raise TypeError('Exponent should be numerical.')
        else:
            val = self.value ** other
            der = self.derivative * other * self.value ** (other - 1)
            return Variable(val, der)

    def __rpow__(self, other):
        return self.__pow__(other)
    
    def __neg__(self):
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
        val = np.power(self.value, 0.5)
        der = self.derivative * 0.5 * np.power(self.value, -0.5)
        return Variable(val, der)
    
    def sin(self):
        val = np.sin(self.value)
        der = np.cos(self.value) * self.derivative
        return Variable(val, der)
    
    def cos(self):
        val = np.cos(self.value)
        der = -np.sin(self.value) * self.derivative
        return Variable(val, der)
    
    
    def tan(self):
        val = np.tan(self.value)
        der = self.derivative / np.cos(self.value)**2
        return Variable(val, der)
    
    ## Other functions beyond the basics

    def sinh(self):
        val = np.sinh(self.value)
        der = np.cosh(self.value) * self.derivative
        return Variable(val, der)
    
    def cosh(self):
        val = np.cosh(self.value)
        der = np.sinh(self.value) * self.derivative
        return Variable(val, der)
    
    def tanh(self):
        pass
    
    def arcsin(self):
        val = np.arcsin(self.value)
        der = self.derivative / np.sqrt(1-self.value**2)
        return Variable(val, der)
    
    def arccos(self):
        val = np.arccos(self.value)
        der = (-1) * self.derivative / np.sqrt(1-self.value**2)
        return Variable(val,der)
    
    def arctan(self):
        pass
    
    def abs(self):
        if self.value != 0.0:
            val = abs(self.value)
            der = self.derivative * (self.value / abs(val))
            return Variable(val, der)
        else:
            raise ValueError('Abs function derivative does not exist at 0')