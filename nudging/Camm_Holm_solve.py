#!/usr/bin/env python
# coding: utf-8

# In[1]:


import math
from firedrake import*
import matplotlib.pyplot as plt
import numpy as np


# In[10]:


#n = 100
#m=20

alpha = 1.0
alphasq = Constant(alpha**2)
dt = 0.01
Dt = Constant(dt)


class Camsholm:
    
    def __init__(self,n,m):
        self.n = n
        self.m = m
        self.mesh = PeriodicIntervalMesh(n, 40.0)
        self.V = FunctionSpace(self.mesh, "CG", 1)
        self.W = MixedFunctionSpace((self.V, self.V))
        self.w0 = Function(self.W)
        self.m0, self.u0 = self.w0.split()
        self.x, = SpatialCoordinate(self.mesh)
        
        #Interpolate the initail condition

        self.u0.interpolate(0.2*2/(exp(self.x-403./15.) + exp(-self.x+403./15.))+ 0.5*2/(exp(self.x-203./15.)+exp(-self.x+203./15.)))
        #Solve for the initial condition for m.

        self.p = TestFunction(self.V)
        self.m = TrialFunction(self.V)
        
        self.am = self.p*self.m*dx
        self.Lm = (self.p*self.u0 + alphasq*self.p.dx(0)*self.u0.dx(0))*dx
        solve(self.am == self.Lm, self.m0, solver_parameters={'ksp_type': 'preonly', 'pc_type': 'lu'})
        
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

        self.fx1.interpolate(0.1*sin(math.pi*self.x/8.))
        self.fx2.interpolate(0.1*sin(2.*math.pi*self.x/8.))
        self.fx3.interpolate(0.1*sin(3.*math.pi*self.x/8.))
        self.fx4.interpolate(0.1*sin(4.*math.pi*self.x/8.))

        # with added term
        self.dW1 = Constant(0)
        self.dW2 = Constant(0)
        self.dW3 = Constant(0)
        self.dW4 = Constant(0)


        self.Ln = self.fx1*self.dW1+self.fx2*self.dW2+self.fx3*self.dW3+self.fx4*self.dW4
        
        # finite element linear functional 


        self.mh = 0.5*(self.m1 + self.m0)
        self.uh = 0.5*(self.u1 + self.u0)
        self.v = self.uh*Dt+self.Ln


        self.L = ((self.q*self.u1 + alphasq*self.q.dx(0)*self.u1.dx(0) - self.q*self.m1)*dx +(self.p*(self.m1-self.m0) + (self.p*self.v.dx(0)*self.mh -self.p.dx(0)*self.v*self.mh))*dx)

        #def Linearfunc

        # solver

        self.uprob = NonlinearVariationalProblem(self.L, self.w1)
        self.usolver = NonlinearVariationalSolver(self.uprob, solver_parameters={'mat_type': 'aij', 'ksp_type': 'preonly','pc_type': 'lu'})

        # Data save
        self.m0, self.u0 = self.w0.split()
        self.m1, self.u1 = self.w1.split()
        
        
        
    
  
    
    def runmethod(self, T):
        
        self.t = 0.0
        while (self.t <  T - 0.5*dt):
            self.t += dt
            self.dW1.assign(np.random.randn()*sqrt(dt))   
            self.dW2.assign(np.random.randn()*sqrt(dt))  
            self.dW3.assign(np.random.randn()*sqrt(dt))
            self.dW4.assign(np.random.randn()*sqrt(dt))
            
            self.usolver.solve()
            return self.w0.assign(self.w1)
        
    


# In[11]:


newobj = Camsholm(100,20)


              
newobj.runmethod(10)


# In[ ]:




