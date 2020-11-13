import numpy as np 

class Variable:

    def __init__(self, value, derivative_seed):
        self.value = value
        self.derivative = derivative_seed

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
    
    def __sub__(self, other):
        pass
    
    def __rsub__(self, other):
        pass
    
    def __truediv__(self, other):
        pass
    
    def __rtruediv__(self, other):
        pass
    
    def __pow__(self, other):
        pass

    def __rpow__(self, other):
        pass
    
    def __neg__(self):
        pass
    
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
        val = np.sin(self.value)
        der = np.cos(self.value) * self.derivative
        return Variable(val, der)
    
    def cos(self):
        pass
    
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
        pass

    def tanh(self):
        pass
    
    def arcsin(self):
        val = np.arcsin(self.value)
        der = self.derivative / np.sqrt(1-self.value**2)
        return Variable(val, der)
    
    def arccos(self):
        pass
    
    def arctan(self):
        pass
    
    def abs(self):
        pass
    

    


    
