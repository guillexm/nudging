from firedrake import *
from firedrake_adjoint import *
from firedrake.petsc import PETSc
from pyop2.mpi import MPI
from nudging.model import *
import numpy as np

class LGModel(base_model):

    def __init__(self, n, nsteps, xpoints, scale,  seed ,   lambdas=False):
        self.n = n
        self.nsteps = nsteps
        self.seed = seed
        self.xpoints = xpoints
        self.A = scale
        self.lambdas = lambdas 

    def setup(self, comm=MPI.COMM_WORLD):
        self.mesh = PeriodicIntervalMesh(self.n, 40.0, comm = comm)

        self.R = FunctionSpace(self.mesh, "R", 0)
        self.u = Function(self.R)
        self.dW = Function(self.R)

        # state for controls
        self.X = self.allocate()

        # vertex only mesh for observations
        x_obs =np.linspace(0, 40, num=self.xpoints, endpoint=False) 
        x_obs_list = []
        for i in x_obs:
            x_obs_list.append([i])
        self.VOM = VertexOnlyMesh(self.mesh, x_obs_list)
        self.VVOM = FunctionSpace(self.VOM, "DG", 0)

    def run(self, X0, X1):
            
            for i in range(len(X0)):
                self.X[i].assign(X0[i])

            self.u.assign(self.X[0])

            for step in range(self.nsteps):
                self.dW.assign(self.X[step+1])
                self.u.assign(self.A*self.u + self.dW)    
            #PETSc.Sys.Print('X value', (1/40)*assemble(self.u*dx), 'dw value', (1/40)*assemble(self.dW*dx) )
            X1[0].assign(self.u)

    def controls(self):
        controls_list = []
        for i in range(len(self.X)):
            controls_list.append(Control(self.X[i]))
        return controls_list   

    def obs(self):
        u = self.u
        Y = Function(self.VVOM)
        Y.interpolate(u)
        return Y


    def allocate(self):
        particle = [Function(self.R)]
        for i in range(self.nsteps):
            dW = Function(self.R)
            particle.append(dW)
        if self.lambdas:
            for i in range(self.nsteps):
                dW = Function(self.R)
                particle.append(dW)
        return particle

    def randomize(self, X, c1=0, c2=1, gscale=None, g=None):
        rg = self.rg
        count = 0
        for i in range(self.nsteps):
            count += 1
            X[count].assign(c1*X[count] + c2*rg.normal(
                self.R, 0., 0.25))
            if g:
                X[count] += gscale*g[count]

    def lambda_functional(self):
        pass