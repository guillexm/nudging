import firedrake as fd
from pyop2.mpi import MPI
from nudging.model import base_model
import numpy as np


class SimModel(base_model):

    def __init__(self, seed=64534):
        self.seed = seed

    def setup(self, comm=MPI.COMM_WORLD):
        self.mesh = fd.UnitSquareMesh(20, 20, comm=comm)
        self.V = fd.FunctionSpace(self.mesh, "CG", 1)

    def run(self, X):
        pass

    def obs(self):
        return np.random.normal(0, 1)

    def allocate(self):
        return [fd.Function(self.V)]

    def randomize(self):
        pass

    def controls(self):
        pass
