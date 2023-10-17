from firedrake import *
from firedrake_adjoint import *
from firedrake.petsc import PETSc
from pyop2.mpi import MPI
from nudging.model import *
import numpy as np
from operator import mul
from functools import reduce

class Euler_SD(base_model):
    def __init__(self, n_xy_pts, nsteps, dt = 0.5, lambdas=False,seed=12353):

        self.n = n_xy_pts
        self.nsteps = nsteps
        self.dt = dt
        self.lambdas = lambdas # include lambdas in allocate
        self.seed = seed

    def setup(self, comm = MPI.COMM_WORLD):
        r = 0.01
        self.Lx = 2.0*pi  # Zonal length
        self.Ly = 2.0*pi  # Meridonal length
        self.mesh = PeriodicRectangleMesh(self.n, self.n, self.Lx, self.Ly, direction="x", quadrilateral=True, comm = comm)
        x = SpatialCoordinate(self.mesh)

        #FE spaces
        self.Vcg = FunctionSpace(self.mesh, "CG", 1) # Streamfunctions
        self.Vdg = FunctionSpace(self.mesh, "DQ", 1) # potential vorticity (PV)
        
        self.q0 = Function(self.Vdg)
        self.q1 = Function(self.Vdg)
        # Define function to store the fields
        self.dq1 = Function(self.Vdg)  # PV fields for different time steps
       
        ##################################  Bilinear form for Stream function ##########################################
        # Define the weakfunction for stream functions
        psi = TrialFunction(self.Vcg)  
        phi = TestFunction(self.Vcg)
        self.psi0 = Function(self.Vcg) 
        

        # Build the weak form for the inversion
        Apsi = (inner(grad(psi), grad(phi)) +  psi*phi) * dx
        Lpsi = -self.q1 * phi * dx

        bc1 = DirichletBC(self.Vcg, 0.0, (1, 2))

        psi_problem = LinearVariationalProblem(Apsi, Lpsi, self.psi0, bcs=bc1, constant_jacobian=True)
        self.psi_solver = LinearVariationalSolver(psi_problem, solver_parameters={"ksp_type": "cg", "pc_type": "hypre"})


        #####################################   Bilinear form  for noise variable  ################################################
        self.W_F = FunctionSpace(self.mesh, "DG", 0)
        self.dW = Function(self.W_F)

        dW_phi = TestFunction(self.Vcg)
        dw = TrialFunction(self.Vcg)
        self.alpha_w = CellVolume(self.mesh)

        # to store noise data
        du_w = Function(self.Vcg)

        #### Define Bilinear form with Dirichlet BC 
        bcs_dw = DirichletBC(self.Vcg,  zero(), ("on_boundary"))
        a_dW = inner(grad(dw), grad(dW_phi))*dx + dw*dW_phi*dx
        L_dW = self.dW*dW_phi*dx
        
        #make a solver 
        dW_problem = LinearVariationalProblem(a_dW, L_dW, du_w, bcs=bcs_dw)
        self.dW_solver = LinearVariationalSolver(dW_problem, solver_parameters={"ksp_type": "cg", "pc_type": "hypre"})

        ################################### Setup for stcohastic  velocity #####################################################################
        # Add noise with stream  fucntion to get stcohastic velocity 
        self.psi_mod = self.psi0+du_w

        self.gradperp = lambda u: as_vector((-u.dx(1), u.dx(0)))
        # upwinding terms
        n_F = FacetNormal(self.mesh)
        un = 0.5 * (dot(self.gradperp(self.psi_mod), n_F) + abs(dot(self.gradperp(self.psi_mod), n_F)))

        #####################################   Bilinear form  for PV  ################################################
        q = TrialFunction(self.Vdg)
        p = TestFunction(self.Vdg)
        Q = Function(self.Vdg).interpolate(0.1 * sin(8*pi*x[0]))

        a_mass = p*q*dx
        a_int = (dot(grad(p), -q*self.gradperp(self.psi_mod)) -p*(Q-r*q)) * dx  # with stream function
        a_flux = (dot(jump(p), un("+") * q("+") - un("-") * q("-")))*dS        # with velocity 
        arhs = a_mass - self.dt*(a_int+ a_flux) 

        #print(type(action(arhs, self.q1)), 'action')
        q_prob = LinearVariationalProblem(a_mass, action(arhs, self.q1), self.dq1)
        self.q_solver = LinearVariationalSolver(q_prob,
                                   solver_parameters={"ksp_type": "preonly",
                                                      "pc_type": "bjacobi",
                                                      "sub_pc_type": "ilu"})

        ############################################ state for controls  ###################################
        self.X = self.allocate()
        ################################ Setup VVOM for vectorfunctionspace ###################################
        x_point = np.linspace(0.0, self.Lx, self.n+1 )
        y_point = np.linspace(0.0, self.Ly, self.n+1 )
        xv, yv  = np.meshgrid(x_point, y_point)
        x_obs_list = np.vstack([xv.ravel(), yv.ravel()]).T.tolist()
        VOM = VertexOnlyMesh(self.mesh, x_obs_list)
        self.VVOM = VectorFunctionSpace(VOM, "DG", 0)

    def run(self, X0, X1):
        for i in range(len(X0)):
            self.X[i].assign(X0[i])
            
        self.q0.assign(self.X[0])
        for step in range(self.nsteps):
            #compute the noise term
            self.dW.assign(self.X[step+1])
            if self.lambdas:
                self.dW += self.X[step+1+self.nsteps]*(self.dt)**0.5
            self.dW_solver.solve()

            # Compute the streamfunction for the known value of q0
            self.q1.assign(self.q0)
            self.psi_solver.solve()
            self.q_solver.solve()

            # Find intermediate solution q^(1)
            self.q1.assign(self.dq1)
            self.psi_solver.solve()
            self.q_solver.solve()

            # Find intermediate solution q^(2)
            self.q1.assign(0.75 * self.q0 + 0.25 * self.dq1)
            self.psi_solver.solve()
            self.q_solver.solve()

            # Find new solution q^(n+1)
            self.q0.assign(self.q0 / 3 + 2*self.dq1 /3)
        X1[0].assign(self.q0) # save sol at the nstep th time 


    # control PV 
    def controls(self):
        controls_list = []
        for i in range(len(self.X)):
            controls_list.append(Control(self.X[i]))
        return controls_list
        

    def obs(self):
        self.q1.assign(self.q0) # assigned at time t+1
        self.psi_solver.solve() # solved at t+1 for psi
        u  = self.gradperp(self.psi0) # evaluated velocity at time t+1
        Y = Function(self.VVOM)
        Y.interpolate(u)
        return Y

    # fIX allocation method
    def allocate(self):
        particle = [Function(self.Vdg)]
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


