from abc import ABCMeta, abstractmethod, abstractproperty
from functools import cached_property
from firedrake import Function, FunctionSpace, PCG64, RandomGenerator
import firedrake as fd

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
    def run(self, X0, X1):
        """
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

    @abstractmethod
    def randomize(self, c1=0, c2=1):
        """
        replace dW_o <- c1*dW_o + c2*dW_n
        where dW_o is the old noise and 
        dW_n is the new noise
        """
        pass
    
    @cached_property
    def R(self):
        """
        An R space to deal with uniform random numbers
        for resampling etc
        """
        R = FunctionSpace(self.mesh, "R", 0)
        return R

    @cached_property
    def U(self):
        """
        An R space function to deal with uniform random numbers
        for resampling
        """
        U = Function(self.R)
        return U

    @cached_property
    def rg(self):
        pcg = PCG64(seed=self.seed)
        return RandomGenerator(pcg)

    def copy(self, Xin, Xout):
        for i in range(len(Xin)):
            Xout[i].assign(Xin[i])
