import numpy as np

class Initializer(object):
    """
    This is an initializer class used for setting the initial value of WestCoastAD variables.
    
    """
    def __call__(self, shape):
        """
        Returns a numpy array initialized as specified by the initializer.
        
        INPUTS
        =======
        - shape: int, number of Variable object

        """
        raise NotImplementedError()
    
    def get_config(self):
        """
        Returns the configuration of the initializer as a JSON-serializable dict.

        RETURNS
        ========
        A JSON-serializable Python dict.

        """
        return {}

class Zeros(Initializer):
    """
    Initializer that sets the initial value of WestCoastAD variables to zeros.
    
    EXAMPLES
    =========

    >>> import numpy as np
    >>> from WestCoastAD import Zeros
    >>> initializer = Zeros()
    >>> initializer(5)
    array([0., 0., 0., 0., 0.])
    
    """
    def __call__(self, shape):
        """
        Returns a numpy array initialized to 0.

        INPUTS
        =======
        - shape : int, number of Variable object

        """
        return np.zeros(shape)
    
class Ones(Initializer):
    """
    Initializer that sets the initial value of WestCoastAD variables to ones.
    
    EXAMPLES
    =========

    >>> import numpy as np
    >>> from WestCoastAD import Ones
    >>> initializer = Ones()
    >>> initializer(4)
    array([1., 1., 1., 1.])
    """
    def __call__(self, shape):
        """
        Returns a numpy array initialized to 1.

        INPUTS
        =======
        - shape : int, number of Variable object

        """
        return np.ones(shape)

class Constant(Initializer):
    """
    Initializer that sets the initial value of WestCoastAD variables to an input constant.
    
    EXAMPLES
    =========
    >>> import numpy as np
    >>> from WestCoastAD import Constant
    >>> initializer = Constant(-6.5)
    >>> initializer(10)
    array([-6.5, -6.5, -6.5, -6.5, -6.5, -6.5, -6.5, -6.5, -6.5, -6.5])
    >>> initializer.get_config()
    {'value': -6.5}
    
    """
    def __init__(self, value=0):
        """
        INPUTS
        =======
        - value : a scalar. A value used for initialization.
        
        RETURNS
        ========
        None

        """
        self.value = value
        
    
    def __call__(self, shape):
        """
        Returns a numpy array initialized to a constant.

        INPUTS
        =======
        - shape : int, number of Variable object

        """
        return self.value*np.ones(shape)
    
    def get_config(self):
        """
        Returns the configuration of the initializer as a JSON-serializable dict.

        RETURNS
        ========
        A JSON-serializable Python dict.

        """
        return {'value': self.value}
    
class RandomUniform(Initializer):
    """
    Initializer that sets the initial value of WestCoastAD variables according to a uniform distribution.
    
    EXAMPLES
    =========

    >>> import numpy as np
    >>> from WestCoastAD import RandomUniform
    >>> initializer = RandomUniform(1,4)
    >>> initializer.get_config()
    {'min value': 1, 'max value': 4}
    
    """
    def __init__(self, minval, maxval):
        """
        INPUTS
        =======
        - minval : a scalar. Lower bound of the range of random values to generate (inclusive).
        - maxval : a scalar. Upper bound of the range of random values to generate (exclusive).
        
        RETURNS
        ========
        None
        """
        self.minval = minval
        self.maxval = maxval
        
    def __call__(self, shape):
        """
        Returns a numpy array initialized according to a uniform distribution.

        INPUTS
        =======
        - shape : int, number of Variable object

        """
        return np.random.uniform(low=self.minval, high=self.maxval, size=shape)
    
    def get_config(self):  # To support serialization
        """
        Returns the configuration of the initializer as a JSON-serializable dict.

        RETURNS
        ========
        A JSON-serializable Python dict.

        """
        return {"min value": self.minval, "max value": self.maxval}
    
class RandomNormal(Initializer):
    """
    Initializer that sets the initial value of WestCoastAD variables according to a normal distribution.

    EXAMPLES
    =========

    >>> import numpy as np
    >>> from WestCoastAD import RandomNormal
    >>> initializer = RandomNormal(1,0.2)
    >>> initializer.get_config()
    {'mean': 1, 'stddev': 0.2}
    
    """
    def __init__(self, mean, stddev):
        """
        INPUTS
        =======
        - mean : a scalar. Mean of the random values to generate.
        - stddev : a scalar. Standard deviation of the random values to generate.
        
        RETURNS
        ========
        None
        """
        self.mean = mean
        self.stddev = stddev
    
    def __call__(self, shape):
        """
        Returns a numpy array initialized according to a normal distribution.

        INPUTS
        =======
        - shape : int, number of Variable object
        
        """
        return np.random.normal(loc=self.mean, scale=self.stddev, size=shape)
    
    def get_config(self):  # To support serialization
        """
        Returns the configuration of the initializer as a JSON-serializable dict.

        RETURNS
        ========
        A JSON-serializable Python dict.

        """
        return {"mean": self.mean, "stddev": self.stddev}

if __name__ == "__main__":
    import doctest
    doctest.testmod()
    