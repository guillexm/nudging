from abc import ABCMeta, abstractmethod, abstractproperty
class model(object, metaclass=ABCMeta):
    def __init__(self):
        pass

    @abstractmethod
    def run(nsteps, W, X0, X1):
        """
        nsteps - number of timesteps
        W - an nsteps x k array containing random numbers
        X0 - a Firedrake Function containing the initial condition
        X1- a Firedrake Function to copy the result into
        """
        pass
    
