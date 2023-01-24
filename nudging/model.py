from abc import ABCMeta, abstractmethod, abstractproperty
from functools import cached_property
from firedrake import Function, FunctionSpaec

class base_model(object, metaclass=ABCMeta):
    def __init__(self):
        pass

    @abstractmethod
    def setup(self, comm):
        """
        comm - the MPI communicator used to build the mesh object

        This method should build the mesh and everything else that
        needs the mesh
        """
        pass

    @abstractmethod
    def run(self, nsteps, W, X0, X1):
        """
        nsteps - number of timesteps
        W - an nsteps x k array containing random numbers
        X0 - a Firedrake Function containing the initial condition
        X1- a Firedrake Function to copy the result into
        """
        pass

    @abstractmethod
    def obs(self,X0):
        """
        Observation operator

        X0 - a Firedrake Function containing the initial condition

        returns

        y - a k-dimensional numpy array of the observations
        """
        pass

    @abstractmethod
    def allocate(self):
        """
        Allocate a function to store a model state
        
        returns
        X - a Function of the required type
        """
        pass
    

    @cached_property
    def U(self):
        """
        An R space function to deal with uniform random numbers
        for resampling
        """
        R = FunctionSpace(self.mesh, "R", 0)
        U = Function(R)
        return U
