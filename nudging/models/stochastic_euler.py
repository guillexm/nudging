from firedrake import *
from firedrake_adjoint import *
from firedrake.petsc import PETSc
from pyop2.mpi import MPI
from nudging.model import *
import numpy as np

class Euler_SD(base_model):
    def __init__(self, n, nsteps, dt = 0.01, seed=12353):

        self.n = n
        self.nsteps = nsteps
        self.dt = dt
        self.seed = seed

    def setup(self, comm = MPI.COMM_WORLD):

        alpha = Constant(1.0)
        beta = Constant(0.1)
        self.Lx = 2.0*pi  # Zonal length
        self.Ly = 2.0*pi  # Meridonal length
        self.mesh = PeriodicRectangleMesh(self.n, self.n, self.Lx, self.Ly, direction="x", quadrilateral=True, comm = comm)
        self.x = SpatialCoordinate(self.mesh)

        #FE spaces
        self.V = FunctionSpace(self.mesh, "CG", 1) #  noise term
        self.Vcg = FunctionSpace(self.mesh, "CG", 1) # Streamfunctions
        self.Vdg = FunctionSpace(self.mesh, "DQ", 1) # potential velocity (PV)
        self.q0 = Function(self.Vdg)
        self.q1 = Function(self.Vdg)
        self.Vu = FunctionSpace(self.mesh, "DQ", 0) # velocity
        self.u0 = Function(self.Vu)

        # Define function to store the fields
        self.dq1 = Function(self.Vdg)  # PV fields for different time steps
        self.q1 = Function(self.Vdg)
       

        # Define the weakfunction for stream 
        self.psi0 = Function(self.Vcg) 
        self.psi = TrialFunction(self.Vcg)  
        self.phi = TestFunction(self.Vcg)
        

        # Build the weak form for the inversion
        self.Apsi = (inner(grad(self.psi), grad(self.phi)) +  self.psi * self.phi) * dx
        Lpsi = -self.q1 * self.phi * dx
        #print(type(Lpsi), "type lpsi")

        bc1 = DirichletBC(self.Vcg, 0.0, (1, 2))

        self.psi_problem = LinearVariationalProblem(self.Apsi, Lpsi, self.psi0, bcs=bc1, constant_jacobian=True)
        self.psi_solver = LinearVariationalSolver(self.psi_problem, solver_parameters={"ksp_type": "cg", "pc_type": "hypre"})

        # setup the second equation
        self.gradperp = lambda u: as_vector((-u.dx(1), u.dx(0)))
        # upwinding terms
        self.n_F = FacetNormal(self.mesh)
        self.un = 0.5 * (dot(self.gradperp(self.psi0), self.n_F) + abs(dot(self.gradperp(self.psi0), self.n_F)))
        
        
        ##############################################################################################
        # For noise variable
        self.dW = TrialFunction(self.V)
        self.dW_phi = TestFunction(self.V)
        # Fix the white noise
        self.dXi = Function(self.V)
        self.dXi.assign(self.rg.normal(self.V, 0., 1))
        # Bilinear form
        bcs_dw = DirichletBC(self.V,  zero(), ("on_boundary",))
        self.a_dW = inner(grad(self.dW), grad(self.dW_phi))*dx + self.dW*self.dW_phi*dx
        L_dW = self.dXi*self.dW_phi*dx
        #print(type(L_dW), "type L_dw")
        # Solve for noise 
        self.dW_n = Function(self.V)
        #make a solver 
        self.dW_problem = LinearVariationalProblem(self.a_dW, L_dW, self.dW_n, bcs=bcs_dw)
        self.dW_solver = LinearVariationalSolver(self.dW_problem, solver_parameters={"ksp_type": "cg", "pc_type": "hypre"})
        ################################################################################################

        #Bilinear form for PV 
        
        self.q = TrialFunction(self.Vdg)
        self.p = TestFunction(self.Vdg)

        a_mass = self.p * self.q * dx
        a_int = (dot(grad(self.p), -self.gradperp(self.psi0) * self.q) + beta * self.p * self.psi0.dx(0)) * dx
        a_flux = (dot(jump(self.p), self.un("+") * self.q("+") - self.un("-") * self.q("-")))*dS
        a_noise = self.p*self.dW_n *dx
        arhs = a_mass - self.dt*(a_int+ a_flux+a_noise) 
        #arhs = a_mass - self.dt*(a_int+ a_flux) 
        
        #a_mass = a_mass + a_noise
        #print(type(action(arhs, self.q1)), 'action')
        self.q_prob = LinearVariationalProblem(a_mass, action(arhs, self.q1), self.dq1)
        self.q_solver = LinearVariationalSolver(self.q_prob,
                                   solver_parameters={"ksp_type": "preonly",
                                                      "pc_type": "bjacobi",
                                                      "sub_pc_type": "ilu"})

        # state for controls
        self.X = self.allocate()

        # need modification w.r.t 2D
        # vertex only mesh for observations  
        x_point = np.linspace(0.0, 2.0*pi, self.n+1 )
        y_point = np.linspace(0.0, 2.0*pi, self.n+1 )
        xv, yv  = np.meshgrid(x_point, y_point)
        x_obs_list = np.vstack([xv.ravel(), yv.ravel()]).T.tolist()
        self.VOM = VertexOnlyMesh(self.mesh, x_obs_list)
        self.VVOM = FunctionSpace(self.VOM, "DG", 0)

    def run(self, X0, X1):
        for i in range(len(X0)):
            self.X[i].assign(X0[i])
            
        self.q0.assign(self.X[0])
        for step in range(self.nsteps):
            self.dXi.assign(self.X[step+1])
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

    def controls(self):
        controls_list = []
        for i in range(len(self.X)):
        #for i in range(1):
            controls_list.append(Control(self.X[i]))
        return controls_list
        
    #only for velocity
    def obs(self):
        self.q1.assign(self.q0)
        self.psi_solver.solve()
        u  = self.gradperp(self.psi0)
        #print(type(self.u[0]))
        Y_1 = Function(self.VVOM)
        #Y_2 = Function(self.VVOM)
        Y_1.interpolate(u[0])
        #Y_2.interpolate(u[1])
        return Y_1


    # memory allocation for PV
    def allocate(self):
        particle = [Function(self.Vdg)]
        for i in range(self.nsteps):
            dW_star = Function(self.V)
            particle.append(dW_star) 
        return particle 



    # fix randomize
    def randomize(self, X, c1=0, c2=1, gscale=None, g=None):
        #return
        rg = self.rg
        count = 0
        self.dXi = Function(self.V)
        for i in range(self.nsteps):
               self.dXi.assign(self.rg.normal(self.V, 0., 0.05))
               self.dW_solver.solve()
               count += 1
               X[count].assign(c1*X[count] + c2*self.dW_n)
               if g:
                    X[count] += gscale*g[count]
               
