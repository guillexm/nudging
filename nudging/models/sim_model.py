from firedrake import *
from firedrake.petsc import PETSc
from pyop2.mpi import MPI
from nudging.model import *
import numpy as np


class SimModel(base_model):

    def __init__(self):
        pass

    def setup(self, comm=MPI.COMM_WORLD):
        self.mesh = UnitSquareMesh(20,20, comm = comm)
        self.V = FunctionSpace(self.mesh, "CG", 1)

    def run(self, X):
        X.assign(self.ensemble_rank)

    def obs(self):
        return np.random.normal(0,1)

    def allocate(self):
        return Function(self.V)

