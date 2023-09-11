from firedrake import *
from firedrake_adjoint import *
from firedrake.petsc import PETSc
from pyop2.mpi import MPI
from nudging.model import *
import numpy as np
from operator import mul
from functools import reduce

class Camsholm(base_model):
    def __init__(self, n, nsteps, xpoints, lambdas=False,
                 dt = 0.025, alpha=1.0, mu=0.01,  seed=12353):

        self.n = n
        self.nsteps = nsteps
        self.alpha = alpha
        self.mu = mu
        self.dt = dt
        self.seed = seed
        self.xpoints = xpoints
        self.lambdas = lambdas # include lambdas in allocate

    def setup(self, comm = MPI.COMM_WORLD):
        self.mesh = PeriodicIntervalMesh(self.n, 40.0, comm = comm) # mesh need to be setup in parallel, width =4 and cell = self.n
        x, = SpatialCoordinate(self.mesh)

        #FE spaces
        V = FunctionSpace(self.mesh, "CG", 1)
        self.W = MixedFunctionSpace((V, V))
        self.w0 = Function(self.W)
        m0, u0 = self.w0.split()       
        One = Function(V).assign(1.0)
        self.Area = assemble(One*dx)
        
        #Interpolate the initial condition

        #Solve for the initial condition for m.
        alphasq = self.alpha**2
        p = TestFunction(V)
        m = TrialFunction(V)

        am = p*m*dx
        Lm = (p*u0 + alphasq*p.dx(0)*u0.dx(0))*dx
        mprob = LinearVariationalProblem(am, Lm, m0)
        sp={'ksp_type': 'preonly', 'pc_type': 'lu'}
        self.msolve = LinearVariationalSolver(mprob,
                                              solver_parameters=sp)

        #Build the weak form of the timestepping algorithm. 

        p, q = TestFunctions(self.W)
        self.w1 = Function(self.W)
        self.w1.assign(self.w0)
        m1, u1 = split(self.w1)   # for n+1 the  time
        m0, u0 = split(self.w0)   # for n th time 
        

        #Setup noise term using Matern formula
        self.W_F = FunctionSpace(self.mesh, "DG", 0) 
        self.dW = Function(self.W_F) 
        self.alpha_w = CellVolume(self.mesh)
        dphi = TestFunction(V)
        du = TrialFunction(V)
        
        self.dU = Function(V)
        kappa_isq = 0.01
        a_w = (dphi*du + kappa_isq*dphi.dx(0)*du.dx(0))*dx
        L_w = self.alpha_w*dphi*self.dW*dx
        w_prob = LinearVariationalProblem(a_w, L_w, self.dU)
        self.wsolver = LinearVariationalSolver(w_prob,
                                              solver_parameters=sp)     
        
        #finite element linear functional 
        Dt = self.dt
        mh = 0.5*(m1 + m0)
        uh = 0.5*(u1 + u0)
        v = uh*Dt+self.dU*Dt**0.5

        L = ((q*u1 + alphasq*q.dx(0)*u1.dx(0) - q*m1)*dx +
             (p*(m1-m0)+ (p*v.dx(0)*mh -p.dx(0)*v*mh)+self.mu*Dt*p.dx(0)*mh.dx(0))*dx)

        # timestepping solver
        uprob = NonlinearVariationalProblem(L, self.w1)
        self.usolver = NonlinearVariationalSolver(uprob,
                                                  solver_parameters={'mat_type': 'aij', 'ksp_type': 'preonly','pc_type': 'lu'})

        # state for controls
        self.X = self.allocate()

        # vertex only mesh for observations
        #x_obs =np.linspace(0, 40,num=self.xpoints, endpoint=False) # This is better choice
        x_obs = np.arange(0.5,self.xpoints)
        x_obs_list = []
        for i in x_obs:
            x_obs_list.append([i])
        self.VOM = VertexOnlyMesh(self.mesh, x_obs_list)
        self.VVOM = FunctionSpace(self.VOM, "DG", 0)

    def run(self, X0, X1, diagonastic = False):
        # copy input into model variables for taping
        for i in range(len(X0)):
            self.X[i].assign(X0[i])

        # copy initial condition into model variable
        self.w0.assign(self.X[0])

        # ensure momentum and velocity are syncronised
        self.msolve.solve()

        # do the timestepping
        for step in range(self.nsteps):
            # get noise variables and lambdas
            self.dW.assign(self.X[step+1])
            if self.lambdas:
                self.dW += self.X[step+1+self.nsteps]*(self.dt)**0.5
            # solve  dW --> dU
            self.wsolver.solve()
            # advance in time
            self.usolver.solve()
            # copy output to input
            self.w0.assign(self.w1)

            #exact callbacks
            if diagonastic:
                self.data_store(self.w0)

        # return outputs
        X1[0].assign(self.w0) # save sol at the nstep th time 

    # not efficient but works need to replace in future
    def forcast_data_allstep(self, X0):
        # copy input into model variables for taping
        for i in range(len(X0)):
            self.X[i].assign(X0[i])

        # copy initial condition into model variable
        self.w0.assign(self.X[0])

        # ensure momentum and velocity are syncronised
        self.msolve.solve()

        # do the timestepping
        # save time steps 
        w0_list = []
        for step in range(self.nsteps):
            # get noise variables and lambdas
            self.dW.assign(self.X[step+1])
            if self.lambdas:
                self.dW += self.X[step+1+self.nsteps]*(self.dt)**0.5
            # solve  dW --> dU
            self.wsolver.solve()
            # advance in time
            self.usolver.solve()
            # copy output to input
            self.w0.assign(self.w1)
            m, u = self.w0.split()
            Y = Function(self.VVOM)
            Y.interpolate(u)
            w0_list.append(Y.dat.data[:])
        # return outputs
        return w0_list # save sol at the nstep th time 



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
            dW = Function(self.W_F)
            particle.append(dW)
        if self.lambdas:
            for i in range(self.nsteps):
                dW = Function(self.W_F)
                particle.append(dW)
        return particle

    def randomize(self, X, c1=0, c2=1, gscale=None, g=None):
        rg = self.rg
        count = 0
        for i in range(self.nsteps):
            count += 1
            X[count].assign(c1*X[count] + c2*rg.normal(
                self.W_F, 0., 1.0))
            if g:
                X[count] += gscale*g[count]

    def lambda_functional(self):
        nsteps = self.nsteps
        dt = self.dt
        for step in range(nsteps):
            lambda_step = self.X[nsteps + 1 + step]
            dW_step = self.X[1 + step]
            
            dlfunc = assemble((1/self.alpha_w)*lambda_step**2*dt/2*dx
                - (1/self.alpha_w)*lambda_step*dW_step*dt**0.5*dx)
            if step == 0:
                lfunc = dlfunc
            else:
                lfunc += dlfunc
        return lfunc
