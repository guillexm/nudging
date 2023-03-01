from firedrake import *
from firedrake_adjoint import *
from firedrake.petsc import PETSc
from pyop2.mpi import MPI
from nudging.model import *
import numpy as np

class euler(base_model):
    def __init__(self, n, nsteps, dt = 0.01, seed=12353):

        self.n = n
        self.nsteps = nsteps
        self.dt = dt
        self.seed = seed

    def setup(self, comm = MPI.COMM_WORLD):
        self.mesh = UnitSquareMesh(self.n, 40.0, comm = comm) # mesh need to be setup in parallel
        self.x = SpatialCoordinate(self.mesh)

        #FE spaces
        self.Vcg = FunctionSpace(self.mesh, "CG", 1)
        self.Vdg = FunctionSpace(self.mesh, "CG", 1)
        self.q0 = Function(self.Vdg)
        self.q1 = Function(self.Vdg)

        #streamfunction solver
        p = TestFunction(self.Vcg)
        psi = TrialFunction(self.Vcg)

        a = inner(grad(p), grad(psi))*dx
        L = p*q1*dx

        self.psi = Function(self.Vcg)
        bcs = [DirichetBC(self.Vcg, 0., "on_boundary")]
        psi_prob = LinearVariationalProblem(a, L, self.psi, bcs=bcs)
        params = {"ksp_type":"preonly",
                  "pc_type":"lu"}
        self.psi_solver = LinearVariationalSolver(psi_prob,
                                                  solver_parameters=params)

        # noise variables
        self.w1 = Function(self.W)
        self.w1.assign(self.w0)
        self.m1, self.u1 = split(self.w1)   # for n+1 the  time
        self.m0, self.u0 = split(self.w0)   # for n th time 
        
        #Adding extra term included random number
        self.fx1 = Function(self.V)
        self.fx2 = Function(self.V)
        self.fx3 = Function(self.V)
        self.fx4 = Function(self.V)

        self.fx1.interpolate(0.1*sin(pi*self.x/8.))
        self.fx2.interpolate(0.1*sin(2.*pi*self.x/8.))
        self.fx3.interpolate(0.1*sin(3.*pi*self.x/8.))
        self.fx4.interpolate(0.1*sin(4.*pi*self.x/8.))

        # with added term
        self.R = FunctionSpace(self.mesh, "R", 0)
        self.dW = []
        for i in range(self.nsteps):
            subdW = []
            for j in range(4):
                subdW.append(Function(self.R))
            self.dW.append(subdW)

        self.dW1 = Function(self.R)
        self.dW2 = Function(self.R)
        self.dW3 = Function(self.R)
        self.dW4 = Function(self.R)
        self.Ln = self.fx1*self.dW1+self.fx2*self.dW2+self.fx3*self.dW3+self.fx4*self.dW4
        
        # finite element linear functional 
        Dt = Constant(self.dt)
        self.mh = 0.5*(self.m1 + self.m0)
        self.uh = 0.5*(self.u1 + self.u0)
        self.v = self.uh*Dt+self.Ln*Dt**0.5

        self.L = ((self.q*self.u1 + alphasq*self.q.dx(0)*self.u1.dx(0) - self.q*self.m1)*dx +(self.p*(self.m1-self.m0) + (self.p*self.v.dx(0)*self.mh -self.p.dx(0)*self.v*self.mh))*dx)

        #def Linearfunc

        # solver

        self.uprob = NonlinearVariationalProblem(self.L, self.w1)
        self.usolver = NonlinearVariationalSolver(self.uprob, solver_parameters={'mat_type': 'aij', 'ksp_type': 'preonly','pc_type': 'lu'})

        # Data save
        self.m0, self.u0 = self.w0.split()
        self.m1, self.u1 = self.w1.split()

        # state for controls
        self.X = self.allocate()

        # vertex only mesh for observations
        x_obs = np.arange(0.5,40.0)
        x_obs_list = []
        for i in x_obs:
            x_obs_list.append([i])
        self.VOM = VertexOnlyMesh(self.mesh, x_obs_list)
        self.VVOM = FunctionSpace(self.VOM, "DG", 0)

    def run(self, X0, X1):
        for i in range(len(X0)):
            self.X[i].assign(X0[i])
        self.w0.assign(self.X[0])
        self.msolve.solve()
        for step in range(self.nsteps):
            self.dW1.assign(self.X[4*step+1])
            self.dW2.assign(self.X[4*step+2])
            self.dW3.assign(self.X[4*step+3])
            self.dW4.assign(self.X[4*step+4])

            self.usolver.solve()
            self.w0.assign(self.w1)
        X1[0].assign(self.w0) # save sol at the nstep th time 

    def controls(self):
        controls_list = []
        for i in range(len(self.X)):
            controls_list.append(Control(self.X[i]))
        return controls_list
        
    def obs(self):
        m, u = self.w0.split()
        Y = Function(self.VVOM)
        Y.interpolate(u)
        return Y

    def allocate(self):
        particle = [Function(self.W)]
        for i in range(self.nsteps):
            for j in range(4):
                dW = Function(self.R)
                dW.assign(self.rg.normal(self.R, 0., 1.0))
                particle.append(dW) 
        return particle 


    def randomize(self, X, c1=0, c2=1, gscale=None, g=None):
        rg = self.rg
        count = 0
        for i in range(self.nsteps):
            for j in range(4):
                count += 1
                X[count].assign(c1*X[count] + c2*rg.normal(self.R, 0., 1.0))
                if g:
                    X[count] += gscale*g[count]
