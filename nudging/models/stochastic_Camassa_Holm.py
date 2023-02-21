from firedrake import *
from firedrake.petsc import PETSc
from pyop2.mpi import MPI
from nudging.model import *
import numpy as np

class Camsholm(base_model):
    def __init__(self, n, nsteps, dt = 0.01, alpha=1.0, seed=12353):

        self.n = n
        self.nsteps = nsteps
        self.alpha = alpha
        self.dt = dt
        self.seed = seed

    def setup(self, comm = MPI.COMM_WORLD):
        self.mesh = PeriodicIntervalMesh(self.n, 40.0, comm = comm) # mesh need to be setup in parallel
        self.x, = SpatialCoordinate(self.mesh)

        #FE spaces
        self.V = FunctionSpace(self.mesh, "CG", 1)
        self.W = MixedFunctionSpace((self.V, self.V))
        self.w0 = Function(self.W)
        self.m0, self.u0 = self.w0.split()       

        #Interpolate the initial condition

        #Solve for the initial condition for m.
        alphasq = self.alpha**2
        self.p = TestFunction(self.V)
        self.m = TrialFunction(self.V)
        
        self.am = self.p*self.m*dx
        self.Lm = (self.p*self.u0 + alphasq*self.p.dx(0)*self.u0.dx(0))*dx
        mprob = LinearVariationalProblem(self.am, self.Lm, self.m0)
        solver_parameters={'ksp_type': 'preonly', 'pc_type': 'lu'}
        self.msolve = LinearVariationalSolver(mprob,
                                              solver_parameters=solver_parameters)
        
        #Build the weak form of the timestepping algorithm. 

        self.p, self.q = TestFunctions(self.W)

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
    
    def run(self, X0, X1):
        self.w0.assign(X0['State']
        self.msolve.solve()
        for step in range(self.nsteps):
            self.dW1.assign(X0['Noise'+str(step)+'_'+str(0)])
            self.dW2.assign(X0['Noise'+str(step)+'_'+str(1)])
            self.dW3.assign(X0['Noise'+str(step)+'_'+str(2)])
            self.dW4.assign(X0['Noise'+str(step)+'_'+str(3)])

            self.usolver.solve()
            self.w0.assign(self.w1)
        X1['State'].assign(self.w0) # save sol at the nstep th time 


    def obs(self, X0):
        m, u = X0.split()
        x_obs = np.arange(0.0,40.0)
        return np.array(u.at(x_obs))


    def allocate(self):
        particle = [Function(self.W)]
        for i in range(self.nsteps):
            for j in range(4):
                dW = Function(self.R)
                dW.assign(rg.normal(self.R, 0., 1.0))
                particle.extend(dW) 

    def randomize(self, X):
        rg = self.rg
        count = 0
        for i in range(self.nsteps):
            for j in range(4):
                count += 1
                X[count].assign(
                    rg.normal(self.R, 0., 1.0))
